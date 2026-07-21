"""
GPL training with an auxiliary inference head.

Same as train_gpl.py but uses GPLAgentInf, which adds an auxiliary level prediction head trained with privileged teammate-level labels. Compare against Q3_rw to isolate the auxiliary objective.

Usage:
    python experiments/train_gpl_inf.py --config configs/gpl_lbf_q3_inf_aux.yaml
    python experiments/train_gpl_inf.py --config configs/gpl_lbf_q3_inf_aux.yaml --smoke-test
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

from agents.gpl.gpl_agent_inf import GPLAgentInf
from envs.env_utils import preprocess_lbf
from eval.logger import Logger

# Reuse shared utilities from train_gpl
from experiments.train_gpl import (
    make_lbf_env,
    sample_stationary_composition,
    sample_food_levels,
    inject_levels,
    _unpack_reset,
    _unpack_step,
)


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_inf(
    agent: GPLAgentInf,
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
    """Evaluate GPLAgentInf under stationary composition."""
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

            action = agent.act_inf(B_np, learner_idx=0, epsilon=0.0)
            agent.advance_hidden_inf(B_np)

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
# Training loop — Algorithm 5 + auxiliary inference
# ======================================================================

def train(cfg: dict, smoke_test: bool = False):
    """Training loop with auxiliary inference head."""
    # --- Config ---
    env_cfg = cfg["env"]
    types_cfg = cfg["types"]
    food_cfg = cfg["food"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    eps_cfg = cfg["epsilon"]
    eval_cfg = cfg["eval"]
    log_cfg = cfg["logging"]
    inf_cfg = cfg.get("inference", {})

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

    # Auxiliary-head config
    aux_weight = inf_cfg.get("aux_weight", 0.1)
    aux_n_classes = inf_cfg.get("aux_n_classes", K)

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
    eval_env = envs[0]

    # --- Agent (GPLAgentInf with aux head) ---
    agent = GPLAgentInf(
        obs_dim=obs_dim,
        aux_n_classes=aux_n_classes,
        aux_weight=aux_weight,
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
            log_dir=log_cfg.get("log_dir", "runs/q3_inf_aux_rw_stationary"),
            use_wandb=False,
            use_tensorboard=log_cfg.get("tensorboard", True),
        )

    # --- Checkpoint dir ---
    results_dir = log_cfg.get("results_dir", "results/q3_inf_aux_rw_stationary")
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Epsilon schedule ---
    eps_init = eps_cfg["init"]
    eps_final = eps_cfg["final"]
    eps_decay_steps = eps_cfg.get("decay_steps", None)
    eps_decay_episodes = eps_cfg.get("decay_episodes", 1500)

    def get_epsilon_by_step(step: int) -> float:
        if eps_decay_steps is not None:
            frac = min(step / max(eps_decay_steps, 1), 1.0)
        else:
            approx_steps = eps_decay_episodes * max_steps * n_envs
            frac = min(step / max(approx_steps, 1), 1.0)
        return eps_init + (eps_final - eps_init) * frac

    # --- Per-env state ---
    env_obs = [None] * n_envs
    env_done = [True] * n_envs
    env_ep_return = [0.0] * n_envs
    env_ep_len = [0] * n_envs
    env_hidden = [(None, None, None)] * n_envs
    # Track agent levels per env (for auxiliary loss)
    env_agent_levels = [None] * n_envs

    # --- Training ---
    print(f"Training GPL-aux on LBF: {n_episodes} episodes, "
          f"K={K}, n_agents={n_agents}, obs_dim={obs_dim}, "
          f"aux_weight={aux_weight}")

    all_returns = []
    global_step = 0
    completed_episodes = 0
    t_start = time.time()
    last_metrics = None

    pbar = tqdm(
        total=n_episodes,
        desc="GPL-aux LBF",
        unit="ep",
        disable=smoke_test,
    )

    while completed_episodes < n_episodes:
        epsilon = get_epsilon_by_step(global_step)

        for env_idx in range(n_envs):
            if completed_episodes >= n_episodes:
                break

            # --- Auto-reset ---
            if env_done[env_idx]:
                agent_levels = sample_stationary_composition(n_agents, K, rng)
                food_levels = sample_food_levels(n_food, rng, food_probs)
                inject_levels(envs[env_idx], agent_levels, food_levels)

                env_obs[env_idx] = _unpack_reset(envs[env_idx].reset())
                env_done[env_idx] = False
                env_ep_return[env_idx] = 0.0
                env_ep_len[env_idx] = 0
                env_hidden[env_idx] = (None, None, None)
                env_agent_levels[env_idx] = agent_levels

            # --- Load hidden states ---
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

            # --- Action selection ---
            action = agent.act_inf(B_np, learner_idx=0, epsilon=epsilon)

            # --- Joint action ---
            joint_action = [rng.integers(0, action_dim) for _ in range(n_agents)]
            joint_action[0] = action

            # --- Step environment ---
            next_obs, reward, done, _ = _unpack_step(
                envs[env_idx].step(joint_action)
            )

            # --- PREPROCESS next state (raw) ---
            B_next, _, _ = preprocess_lbf(
                next_obs, n_agents, n_food,
                hidden_dim=hidden_dim, device=device,
                observe_agent_levels=observe_agent_levels,
            )
            B_next_np = B_next.cpu().numpy()

            # --- Train step with auxiliary loss ---
            metrics = agent.train_step_online_inf(
                B_np, np.array(joint_action), reward, B_next_np, done,
                agent_levels=env_agent_levels[env_idx],
                learner_idx=0,
            )
            if metrics is not None:
                last_metrics = metrics

            # --- Save hidden states ---
            env_hidden[env_idx] = (
                agent._hidden_q,
                agent._hidden_agent,
                agent._hidden_q_target,
            )

            # --- Update tracking ---
            env_ep_return[env_idx] += reward
            env_ep_len[env_idx] += 1
            env_obs[env_idx] = next_obs
            env_done[env_idx] = done
            global_step += 1

            # --- Episode completed ---
            if done:
                ep_return = env_ep_return[env_idx]
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
                    log_metrics = {
                        "epsilon": epsilon,
                        "global_step": global_step,
                    }
                    if last_metrics and "aux_loss" in last_metrics:
                        log_metrics["aux_loss"] = last_metrics["aux_loss"]
                    logger.log_episode(completed_episodes - 1, ep_return,
                                       env_ep_len[env_idx], log_metrics)
                    if last_metrics is not None:
                        logger.log_scalars("train", last_metrics,
                                           completed_episodes - 1)

                # --- Periodic eval ---
                save_every = eval_cfg.get("save_every", 50)
                eval_every = eval_cfg.get("eval_every", 50)

                if completed_episodes % eval_every == 0:
                    saved_hidden = (agent._hidden_q, agent._hidden_agent,
                                    agent._hidden_q_target)
                    eval_result = evaluate_inf(
                        agent, eval_env, eval_cfg.get("eval_episodes", 5),
                        n_agents, n_food, K, rng, food_probs, hidden_dim,
                        device, action_dim, observe_agent_levels,
                    )
                    agent._hidden_q, agent._hidden_agent, agent._hidden_q_target = saved_hidden

                    recent = all_returns[-min(50, len(all_returns)):]
                    elapsed = time.time() - t_start
                    eps_per_sec = completed_episodes / elapsed if elapsed > 0 else 0

                    aux_str = ""
                    if last_metrics and "aux_loss" in last_metrics:
                        aux_str = f" aux={last_metrics['aux_loss']:.4f}"

                    msg = (
                        f"[Ep {completed_episodes:5d}/{n_episodes}] "
                        f"train_avg={np.mean(recent):.3f} "
                        f"eval_iqm={eval_result['iqm_return']:.3f} "
                        f"eps={epsilon:.3f}{aux_str} "
                        f"steps={global_step:,} "
                        f"ep/s={eps_per_sec:.1f}"
                    )
                    tqdm.write(msg)

                    if logger is not None:
                        logger.log_scalars("eval", eval_result,
                                           completed_episodes - 1)

                # --- Checkpoint ---
                if completed_episodes % save_every == 0:
                    path = os.path.join(ckpt_dir,
                                        f"gpl_ep{completed_episodes}.pt")
                    agent.save(path)

    pbar.close()

    # Final checkpoint
    agent.save(os.path.join(ckpt_dir, "gpl_final.pt"))

    elapsed = time.time() - t_start
    print(f"\nTraining complete: {completed_episodes} episodes, "
          f"{global_step:,} env steps in {elapsed:.1f}s")
    print(f"Final avg return (last 50): {np.mean(all_returns[-50:]):.3f}")

    if logger is not None:
        logger.close()

    return agent, all_returns


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train GPL with auxiliary inference on LBF"
    )
    parser.add_argument("--config", default="configs/gpl_lbf_q3_inf_aux.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
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
