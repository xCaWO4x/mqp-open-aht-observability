"""
GPL training on Level-Based Foraging.

Trains GPL under a stationary uniform composition distribution.
to establish the baseline performance. Uses Algorithm 5 (online synchronous
training) from Rahman et al. 2023, with support for parallel environments
matching the paper's 16-env synchronous data collection.

Usage:
    python experiments/train_gpl.py
    python experiments/train_gpl.py --config configs/gpl_lbf.yaml
    python experiments/train_gpl.py --smoke-test          # 5 episodes, no logging
    python experiments/train_gpl.py --n-episodes 500      # override episode count
    python experiments/train_gpl.py --n-envs 16           # parallel envs (paper default)
    python experiments/train_gpl.py --eval-episodes 25    # stabler periodic eval IQM
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from lbforaging.foraging.environment import ForagingEnv

from agents.gpl.gpl_agent import GPLAgent
from envs.env_utils import preprocess_lbf
from eval.logger import Logger


# ======================================================================
# Environment creation
# ======================================================================

def make_lbf_env(cfg: dict, seed: int = 0) -> ForagingEnv:
    """Create an LBF ForagingEnv with full level range for obs space.

    The env is created with min_level=1, max_level=K so the observation
    space can accommodate any agent/food level. Actual levels are set
    per-episode by the stationary training loop.
    """
    env_cfg = cfg["env"]
    types_cfg = cfg["types"]
    food_cfg = cfg["food"]
    K = types_cfg["K"]
    n_agents = env_cfg["n_agents"]
    n_food = food_cfg["n_food"]
    grid = env_cfg["grid_size"]

    env = ForagingEnv(
        players=n_agents,
        min_player_level=np.ones(n_agents, dtype=int),
        max_player_level=np.full(n_agents, K, dtype=int),
        field_size=(grid, grid),
        max_num_food=n_food,
        min_food_level=np.ones(n_food, dtype=int),
        max_food_level=np.full(n_food, K, dtype=int),
        sight=env_cfg.get("sight", grid),
        max_episode_steps=env_cfg["max_steps"],
        force_coop=env_cfg.get("force_coop", False),
        observe_agent_levels=env_cfg.get("observe_agent_levels", True),
    )
    env.np_random = np.random.default_rng(seed)
    return env


# ======================================================================
# Stationary composition sampling
# ======================================================================

def sample_stationary_composition(
    n_agents: int,
    K: int,
    rng: np.random.Generator,
    mu: np.ndarray = None,
) -> list:
    """Sample agent levels from stationary distribution (uniform by default)."""
    if mu is None:
        mu = np.ones(K) / K
    types = rng.choice(K, size=n_agents, p=mu)
    return (types + 1).tolist()  # type 0 -> level 1


def sample_food_levels(n_food: int, rng: np.random.Generator, probs: dict = None) -> list:
    """Sample food levels from fixed distribution."""
    if probs is None:
        probs = {2: 0.6, 3: 0.4}
    levels = list(probs.keys())
    p = np.array([probs[l] for l in levels])
    p = p / p.sum()
    return rng.choice(levels, size=n_food, p=p).tolist()


def inject_levels(env: ForagingEnv, agent_levels: list, food_levels: list):
    """Set exact agent and food levels on the env before reset."""
    env.min_player_level = np.array(agent_levels)
    env.max_player_level = np.array(agent_levels)
    env.min_food_level = np.array(food_levels)
    env.max_food_level = np.array(food_levels)


# ======================================================================
# Gym/Gymnasium compatibility helpers
# ======================================================================

def _unpack_reset(reset_out):
    """Handle gymnasium (obs, info) vs gym (obs) reset returns."""
    if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
        return reset_out[0]
    return reset_out


def _unpack_step(step_out):
    """Handle gymnasium 5-tuple vs gym 4-tuple step returns."""
    if len(step_out) == 5:
        obs, rewards, terminated, truncated, info = step_out
        if isinstance(terminated, (list, tuple)):
            done = all(terminated) or all(truncated)
        else:
            done = bool(terminated) or bool(truncated)
    else:
        obs, rewards, dones_out, info = step_out
        if isinstance(dones_out, (list, tuple)):
            done = all(dones_out)
        else:
            done = bool(dones_out)
    # Extract scalar reward (learner = agent 0)
    if isinstance(rewards, (list, tuple)):
        reward = float(rewards[0])
    else:
        reward = float(rewards)
    return obs, reward, done, info


# ======================================================================
# Evaluation
# ======================================================================

def evaluate(
    agent: GPLAgent,
    env: ForagingEnv,
    n_episodes: int,
    n_agents: int,
    n_food: int,
    K: int,
    rng: np.random.Generator,
    food_probs: dict = None,
    hidden_dim: int = 100,
    device: str = "cpu",
    action_dim: int = 6,
    observe_agent_levels: bool = True,
) -> dict:
    """Evaluate agent for n_episodes under stationary composition, no training."""
    returns = []
    lengths = []

    for _ in range(n_episodes):
        agent_levels = sample_stationary_composition(n_agents, K, rng)
        food_levels = sample_food_levels(n_food, rng, food_probs)
        inject_levels(env, agent_levels, food_levels)

        obs = _unpack_reset(env.reset())
        agent.reset()

        ep_return = 0.0
        ep_len = 0
        done = False

        while not done:
            B, _, _ = preprocess_lbf(
                obs, n_agents, n_food,
                hidden_dim=hidden_dim, device=device,
                observe_agent_levels=observe_agent_levels,
            )
            B_np = B.cpu().numpy()
            action = agent.act(B_np, learner_idx=0, epsilon=0.0)
            # Advance hidden states (act() no longer updates them)
            agent.advance_hidden(B_np)

            joint_action = [rng.integers(0, action_dim) for _ in range(n_agents)]
            joint_action[0] = action

            obs, reward, done, _ = _unpack_step(env.step(joint_action))
            ep_return += reward
            ep_len += 1

        returns.append(ep_return)
        lengths.append(ep_len)

    returns = np.array(returns)
    q25, q75 = np.percentile(returns, [25, 75])
    mask = (returns >= q25) & (returns <= q75)
    iqm = returns[mask].mean() if mask.sum() > 0 else returns.mean()

    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "iqm_return": float(iqm),
        "mean_length": float(np.mean(lengths)),
    }


# ======================================================================
# Training loop — Algorithm 5 with parallel environments
# ======================================================================

def train(cfg: dict, smoke_test: bool = False):
    """Main training loop with N parallel environments.

    Paper: 16 parallel envs, synchronous data collection (A3C-style).
    Each "parallel step" produces N transitions. Gradients accumulate
    over t_update parallel steps before applying.
    """
    # --- Config ---
    env_cfg = cfg["env"]
    types_cfg = cfg["types"]
    food_cfg = cfg["food"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    eps_cfg = cfg["epsilon"]
    eval_cfg = cfg["eval"]
    log_cfg = cfg["logging"]

    n_agents = env_cfg["n_agents"]
    n_food = food_cfg["n_food"]
    K = types_cfg["K"]
    obs_dim = cfg["preprocess"]["obs_dim"]
    action_dim = model_cfg["action_dim"]
    hidden_dim = model_cfg["hidden_dim"]
    max_steps = env_cfg["max_steps"]
    observe_agent_levels = env_cfg.get("observe_agent_levels", True)
    n_envs = train_cfg.get("n_envs", 1)
    n_episodes = 5 if smoke_test else train_cfg["n_episodes"]

    if "device" in cfg:
        device = cfg["device"]
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    seed = cfg.get("seed", 42)

    food_probs = food_cfg.get("fixed_level_probs", {2: 0.6, 3: 0.4})
    food_probs = {int(k): v for k, v in food_probs.items()}

    # --- RNG ---
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # --- Parallel environments ---
    envs = [make_lbf_env(cfg, seed=seed + i) for i in range(n_envs)]
    eval_env = envs[0]  # reuse first env for evaluation

    # --- Agent ---
    agent = GPLAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        type_dim=model_cfg["type_dim"],
        hidden_dim=hidden_dim,
        n_gnn_layers=model_cfg["n_gnn_layers"],
        pairwise_rank=model_cfg["pairwise_rank"],
        lr=train_cfg["lr"],
        gamma=train_cfg["gamma"],
        tau=cfg.get("tau", None),
        t_update=train_cfg["t_update"],
        t_targ_update=train_cfg["t_targ_update"],
        polyak_tau=train_cfg.get("polyak_tau", None),
        device=device,
    )

    # --- Logger ---
    logger = None
    if not smoke_test:
        logger = Logger(
            log_dir=log_cfg.get("log_dir", "runs/gpl_lbf"),
            use_wandb=False,
            use_tensorboard=log_cfg.get("tensorboard", True),
        )

    # --- Checkpoint dir ---
    results_dir = log_cfg.get("results_dir", "results/gpl_lbf")
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Epsilon schedule (step-based, matching paper) ---
    eps_init = eps_cfg["init"]
    eps_final = eps_cfg["final"]
    # Paper: epsilon decays over 4.8M env steps.
    # Support both step-based and episode-based decay.
    eps_decay_steps = eps_cfg.get("decay_steps", None)
    eps_decay_episodes = eps_cfg.get("decay_episodes", 1500)

    def get_epsilon_by_step(step: int) -> float:
        if eps_decay_steps is not None:
            frac = min(step / max(eps_decay_steps, 1), 1.0)
        else:
            # Approximate: convert episodes to steps
            approx_steps = eps_decay_episodes * max_steps * n_envs
            frac = min(step / max(approx_steps, 1), 1.0)
        return eps_init + (eps_final - eps_init) * frac

    # --- Per-env state ---
    # Each env has: current obs, done flag, episode return, episode length,
    # and LSTM hidden states (h_q, h_agent, h_q_target)
    env_obs = [None] * n_envs
    env_done = [True] * n_envs     # start as done to trigger initial reset
    env_ep_return = [0.0] * n_envs
    env_ep_len = [0] * n_envs
    # Hidden states per env: (hidden_q, hidden_agent, hidden_q_target)
    env_hidden = [(None, None, None)] * n_envs

    # --- Training ---
    print(f"Training GPL on LBF: {n_episodes} episodes, "
          f"K={K}, n_agents={n_agents}, n_food={n_food}, obs_dim={obs_dim}")
    print(f"Config: lr={train_cfg['lr']}, gamma={train_cfg['gamma']}, "
          f"t_update={train_cfg['t_update']}, t_targ_update={train_cfg['t_targ_update']}, "
          f"n_envs={n_envs}")

    all_returns = []    # completed episode returns
    global_step = 0
    completed_episodes = 0
    t_start = time.time()
    last_metrics = None

    pbar = tqdm(
        total=n_episodes,
        desc="GPL LBF",
        unit="ep",
        disable=smoke_test,
    )

    while completed_episodes < n_episodes:
        epsilon = get_epsilon_by_step(global_step)

        # --- Process all N envs for one step ---
        for env_idx in range(n_envs):
            if completed_episodes >= n_episodes:
                break

            # --- Auto-reset if episode ended ---
            if env_done[env_idx]:
                agent_levels = sample_stationary_composition(n_agents, K, rng)
                food_levels = sample_food_levels(n_food, rng, food_probs)
                inject_levels(envs[env_idx], agent_levels, food_levels)

                env_obs[env_idx] = _unpack_reset(envs[env_idx].reset())
                env_done[env_idx] = False
                env_ep_return[env_idx] = 0.0
                env_ep_len[env_idx] = 0
                # Reset hidden states for this env
                env_hidden[env_idx] = (None, None, None)

            # --- Load this env's hidden states into agent ---
            agent._hidden_q = env_hidden[env_idx][0]
            agent._hidden_agent = env_hidden[env_idx][1]
            agent._hidden_q_target = env_hidden[env_idx][2]

            # --- PREPROCESS ---
            obs = env_obs[env_idx]
            B, _, _ = preprocess_lbf(
                obs, n_agents, n_food,
                hidden_dim=hidden_dim, device=device,
                observe_agent_levels=observe_agent_levels,
            )
            B_np = B.cpu().numpy()

            # --- Action selection (epsilon-greedy) ---
            action = agent.act(B_np, learner_idx=0, epsilon=epsilon)

            # --- Joint action: learner at 0, teammates random ---
            joint_action = [rng.integers(0, action_dim) for _ in range(n_agents)]
            joint_action[0] = action

            # --- Step environment ---
            next_obs, reward, done, _ = _unpack_step(
                envs[env_idx].step(joint_action)
            )

            # --- PREPROCESS next state ---
            B_next, _, _ = preprocess_lbf(
                next_obs, n_agents, n_food,
                hidden_dim=hidden_dim, device=device,
                observe_agent_levels=observe_agent_levels,
            )
            B_next_np = B_next.cpu().numpy()

            # --- Train step (Algorithm 5): accumulates gradients ---
            metrics = agent.train_step_online(
                B_np, np.array(joint_action), reward, B_next_np, done,
                learner_idx=0,
            )
            if metrics is not None:
                last_metrics = metrics

            # --- Save this env's hidden states ---
            env_hidden[env_idx] = (
                agent._hidden_q,
                agent._hidden_agent,
                agent._hidden_q_target,
            )

            # --- Update per-env episode tracking ---
            env_ep_return[env_idx] += reward
            env_ep_len[env_idx] += 1
            env_obs[env_idx] = next_obs
            env_done[env_idx] = done
            global_step += 1

            # --- Episode completed ---
            if done:
                ep_return = env_ep_return[env_idx]
                ep_len = env_ep_len[env_idx]
                all_returns.append(ep_return)
                completed_episodes += 1
                pbar.update(1)

                recent_n = min(50, len(all_returns))
                train_avg = float(np.mean(all_returns[-recent_n:]))

                if not smoke_test:
                    pbar.set_postfix(
                        ret=f"{ep_return:.2f}",
                        avg50=f"{train_avg:.2f}",
                        eps=f"{epsilon:.3f}",
                        steps=global_step,
                        refresh=False,
                    )

                # --- Logging ---
                if logger is not None:
                    logger.log_episode(completed_episodes - 1, ep_return, ep_len, {
                        "epsilon": epsilon,
                        "global_step": global_step,
                    })
                    if last_metrics is not None:
                        logger.log_scalars("train", last_metrics, completed_episodes - 1)

                # --- Periodic eval ---
                save_every = eval_cfg.get("save_every", 50)
                eval_every = eval_cfg.get("eval_every", 50)

                if completed_episodes % eval_every == 0:
                    # Save/restore agent hidden state for eval
                    saved_hidden = (agent._hidden_q, agent._hidden_agent, agent._hidden_q_target)
                    eval_result = evaluate(
                        agent, eval_env, eval_cfg.get("eval_episodes", 5),
                        n_agents, n_food, K, rng, food_probs, hidden_dim, device,
                        action_dim, observe_agent_levels,
                    )
                    agent._hidden_q, agent._hidden_agent, agent._hidden_q_target = saved_hidden

                    recent = all_returns[-min(50, len(all_returns)):]
                    elapsed = time.time() - t_start
                    eps_per_sec = completed_episodes / elapsed if elapsed > 0 else 0
                    steps_per_sec = global_step / elapsed if elapsed > 0 else 0

                    msg = (
                        f"[Ep {completed_episodes:5d}/{n_episodes}] "
                        f"train_avg={np.mean(recent):.3f} "
                        f"eval_iqm={eval_result['iqm_return']:.3f} "
                        f"eps={epsilon:.3f} "
                        f"steps={global_step:,} "
                        f"ep/s={eps_per_sec:.1f} "
                        f"step/s={steps_per_sec:.0f}"
                    )
                    tqdm.write(msg)

                    if logger is not None:
                        logger.log_scalars("eval", eval_result, completed_episodes - 1)

                # --- Checkpoint ---
                if completed_episodes % save_every == 0:
                    path = os.path.join(ckpt_dir, f"gpl_ep{completed_episodes}.pt")
                    agent.save(path)

    pbar.close()

    # Final checkpoint
    agent.save(os.path.join(ckpt_dir, "gpl_final.pt"))

    # Summary
    elapsed = time.time() - t_start
    print(f"\nTraining complete: {completed_episodes} episodes, "
          f"{global_step:,} env steps in {elapsed:.1f}s")
    print(f"Final avg return (last 50): {np.mean(all_returns[-50:]):.3f}")
    print(f"Throughput: {completed_episodes/elapsed:.1f} ep/s, "
          f"{global_step/elapsed:.0f} step/s")

    if logger is not None:
        logger.close()

    return agent, all_returns


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GPL on LBF")
    parser.add_argument("--config", default="configs/gpl_lbf.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 5 episodes with no logging for quick verification")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Override number of training episodes")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel environments (paper: 16)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--eval-episodes", type=int, default=None,
        help="Override eval episodes per checkpoint (default from config).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device (cpu/mps/cuda).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.n_episodes is not None:
        cfg["training"]["n_episodes"] = args.n_episodes
    if args.n_envs is not None:
        cfg["training"]["n_envs"] = args.n_envs
    if args.eval_episodes is not None:
        cfg.setdefault("eval", {})["eval_episodes"] = args.eval_episodes
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device

    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
