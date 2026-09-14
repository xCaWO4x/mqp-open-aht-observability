"""
Unified Benchmark Runner for LBF and Wolfpack Observability Sweeps.

Usage:
    python experiments/run_benchmark.py --env lbf --sight 4 --teammate_type random --seed 42
    python experiments/run_benchmark.py --env wolfpack --sight 4 --teammate_type greedy --seed 42 --smoke-test
"""

import argparse
import csv
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.gpl.gpl_agent import GPLAgent
from agents.teammate_policies import get_teammate_policy
from envs.env_utils import preprocess_lbf, preprocess_wolfpack
from eval.logger import Logger


def make_benchmark_env(env_name: str, sight: int, seed: int = 0, **kwargs):
    env_lower = env_name.lower()
    if "lbf" in env_lower or "foraging" in env_lower:
        from lbforaging.foraging.environment import ForagingEnv
        grid_size = kwargs.get("grid_size", 8)
        n_agents = kwargs.get("n_agents", 3)
        n_food = kwargs.get("n_food", 3)
        max_steps = kwargs.get("max_steps", 50)
        K = kwargs.get("K", 3)
        observe_agent_levels = kwargs.get("observe_agent_levels", False)

        env = ForagingEnv(
            players=n_agents,
            min_player_level=np.ones(n_agents, dtype=int),
            max_player_level=np.full(n_agents, K, dtype=int),
            field_size=(grid_size, grid_size),
            max_num_food=n_food,
            min_food_level=np.ones(n_food, dtype=int),
            max_food_level=np.full(n_food, K, dtype=int),
            sight=sight,
            max_episode_steps=max_steps,
            force_coop=kwargs.get("force_coop", False),
            observe_agent_levels=observe_agent_levels,
        )
        env.np_random = np.random.default_rng(seed)
        return env
    elif "wolfpack" in env_lower:
        from envs.wolfpack_env import WolfpackEnv
        return WolfpackEnv(
            grid_size=kwargs.get("grid_size", 10),
            n_wolves=kwargs.get("n_wolves", 3),
            n_prey=kwargs.get("n_prey", 2),
            max_steps=kwargs.get("max_steps", 50),
            sight=sight,
            coop_radius=kwargs.get("coop_radius", 1),
            min_coop_wolves=kwargs.get("min_coop_wolves", 2),
            capture_reward=kwargs.get("capture_reward", 5.0),
            respawn_prey=kwargs.get("respawn_prey", True),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown benchmark environment: {env_name}")


def unpack_reset(reset_out):
    if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
        return reset_out[0]
    return reset_out


def unpack_step(step_out):
    if len(step_out) == 5:
        obs, rewards, terminated, truncated, info = step_out
        done = (all(terminated) if isinstance(terminated, (list, tuple)) else bool(terminated)) or \
               (all(truncated) if isinstance(truncated, (list, tuple)) else bool(truncated))
    else:
        obs, rewards, dones_out, info = step_out
        done = all(dones_out) if isinstance(dones_out, (list, tuple)) else bool(dones_out)

    reward = float(rewards[0]) if isinstance(rewards, (list, tuple)) else float(rewards)
    return obs, reward, done, info


def sample_lbf_levels(n_agents: int, K: int, n_food: int, rng: np.random.Generator, food_probs: dict = None):
    agent_levels = (rng.choice(K, size=n_agents) + 1).tolist()
    if food_probs is None:
        food_probs = {2: 0.6, 3: 0.4}
    levels = list(food_probs.keys())
    p = np.array([food_probs[l] for l in levels], dtype=float)
    p = p / p.sum()
    food_levels = rng.choice(levels, size=n_food, p=p).tolist()
    return agent_levels, food_levels


def run_benchmark(
    env_name: str,
    sight: int,
    teammate_type: str = "random",
    seed: int = 42,
    n_episodes: Optional[int] = None,
    output_dir: Optional[str] = None,
    smoke_test: bool = False,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env_is_lbf = "lbf" in env_name.lower() or "foraging" in env_name.lower()

    if output_dir is None:
        output_dir = f"results/{env_name}_sight{sight}_{teammate_type}_seed{seed}"
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if env_is_lbf:
        n_agents = 3
        n_food = 3
        K = 3
        obs_dim = 11  # 2 + 3 * 3 (levels hidden)
        action_dim = 6
        hidden_dim = 100
        max_steps = 50
        env = make_benchmark_env("lbf", sight=sight, seed=seed, grid_size=8, n_agents=n_agents, n_food=n_food, max_steps=max_steps, K=K, observe_agent_levels=False)
        preprocess_fn = lambda obs, dev: preprocess_lbf(obs, n_agents=n_agents, n_food=n_food, hidden_dim=hidden_dim, device=dev, observe_agent_levels=False, from_ego_perspective=True)[0]
    else:
        n_agents = 3
        n_prey = 2
        obs_dim = 8   # 2 + 3 * 2
        action_dim = 5
        hidden_dim = 100
        max_steps = 50
        env = make_benchmark_env("wolfpack", sight=sight, seed=seed, grid_size=10, n_wolves=n_agents, n_prey=n_prey, max_steps=max_steps)
        preprocess_fn = lambda obs, dev: preprocess_wolfpack(obs, n_wolves=n_agents, n_prey=n_prey, hidden_dim=hidden_dim, device=dev, from_ego_perspective=True)[0]

    # Teammate policy
    teammate_policy = get_teammate_policy(env_name, teammate_type, rng=rng)

    # GPL Agent
    agent = GPLAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        type_dim=100,
        hidden_dim=hidden_dim,
        n_gnn_layers=2,
        pairwise_rank=5,
        lr=2.5e-4,
        gamma=0.99,
        t_update=4,
        t_targ_update=1,
        polyak_tau=1.0e-3,
        device=device,
    )

    logger = None
    if not smoke_test:
        logger = Logger(log_dir=os.path.join(output_dir, "tb_logs"), use_wandb=False, use_tensorboard=True)

    target_episodes = 5 if smoke_test else (n_episodes if n_episodes is not None else 1000)
    eps_init = 1.0
    eps_final = 0.05
    decay_steps = 200000

    metrics_file = os.path.join(output_dir, "metrics.csv")
    csv_writer = None
    csv_f = None
    if not smoke_test:
        csv_f = open(metrics_file, "w", newline="")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["episode", "return", "length", "epsilon", "q_loss", "agent_loss"])

    print(f"=== Starting Benchmark ===")
    print(f"Env: {env_name} | Sight: {sight} | Teammates: {teammate_type} | Seed: {seed}")
    print(f"Episodes: {target_episodes} | Device: {device} | Output: {output_dir}")

    global_step = 0
    all_returns = []

    pbar = tqdm(total=target_episodes, desc=f"{env_name}_s{sight}_{teammate_type}")
    for ep in range(target_episodes):
        if env_is_lbf:
            ag_levels, fd_levels = sample_lbf_levels(n_agents, K, n_food, rng)
            env.min_player_level = np.array(ag_levels)
            env.max_player_level = np.array(ag_levels)
            env.min_food_level = np.array(fd_levels)
            env.max_food_level = np.array(fd_levels)

        obs = unpack_reset(env.reset())
        agent.reset()
        ep_return = 0.0
        ep_len = 0
        done = False
        last_metrics = {}

        while not done:
            frac = min(global_step / max(decay_steps, 1), 1.0)
            eps = eps_init + (eps_final - eps_init) * frac

            B = preprocess_fn(obs, device)
            B_np = B.cpu().numpy()

            ego_action = agent.act(B_np, learner_idx=0, epsilon=eps)

            # Joint action
            joint_actions = [ego_action]
            for tm_idx in range(1, n_agents):
                tm_act = teammate_policy.select_action(tm_idx, env=env, obs=obs)
                joint_actions.append(tm_act)

            next_obs, reward, done, _ = unpack_step(env.step(joint_actions))

            B_next = preprocess_fn(next_obs, device)
            B_next_np = B_next.cpu().numpy()

            step_metrics = agent.train_step_online(
                B_np, np.array(joint_actions), reward, B_next_np, done, learner_idx=0
            )
            if step_metrics is not None:
                last_metrics = step_metrics

            obs = next_obs
            ep_return += reward
            ep_len += 1
            global_step += 1

        all_returns.append(ep_return)
        pbar.update(1)
        pbar.set_postfix(ret=f"{ep_return:.2f}", avg=f"{np.mean(all_returns[-50:]):.2f}")

        if logger is not None:
            logger.log_episode(ep, ep_return, ep_len, {"epsilon": eps})
            if last_metrics:
                logger.log_scalars("train", last_metrics, ep)

        if csv_writer is not None:
            q_loss = last_metrics.get("q_loss", float("nan"))
            ag_loss = last_metrics.get("agent_model_loss", float("nan"))
            csv_writer.writerow([ep, ep_return, ep_len, eps, q_loss, ag_loss])
            csv_f.flush()

        if not smoke_test and (ep + 1) % 50 == 0:
            agent.save(os.path.join(ckpt_dir, f"gpl_ep{ep + 1}.pt"))

    pbar.close()
    if csv_f is not None:
        csv_f.close()
    if logger is not None:
        logger.close()

    if not smoke_test:
        agent.save(os.path.join(ckpt_dir, "gpl_final.pt"))

    print(f"Finished! Mean return: {np.mean(all_returns):.3f}")
    return all_returns


def main():
    parser = argparse.ArgumentParser(description="Unified Observability Benchmark Runner")
    parser.add_argument("--env", type=str, required=True, choices=["lbf", "wolfpack"])
    parser.add_argument("--sight", type=int, required=True)
    parser.add_argument("--teammate_type", type=str, default="random", choices=["random", "greedy"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_benchmark(
        env_name=args.env,
        sight=args.sight,
        teammate_type=args.teammate_type,
        seed=args.seed,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        device=args.device,
    )


if __name__ == "__main__":
    main()
