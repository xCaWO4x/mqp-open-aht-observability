#!/usr/bin/env python
"""Stationary greedy evaluation for trained GPL checkpoints."""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
from lbforaging.foraging.environment import ForagingEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.gpl.gpl_agent import GPLAgent
from agents.gpl.gpl_agent_inf import GPLAgentInf
from envs.env_utils import preprocess_lbf
from experiments.train_gpl import (
    _unpack_reset,
    _unpack_step,
    inject_levels,
    sample_food_levels,
    sample_stationary_composition,
)


def make_lbf_env(cfg: dict, seed: int = 0) -> ForagingEnv:
    env_cfg = cfg["env"]
    types_cfg = cfg["types"]
    food_cfg = cfg["food"]
    k = types_cfg["K"]
    n_agents = env_cfg["n_agents"]
    n_food = food_cfg["n_food"]
    grid = env_cfg["grid_size"]
    env = ForagingEnv(
        players=n_agents,
        min_player_level=np.ones(n_agents, dtype=int),
        max_player_level=np.full(n_agents, k, dtype=int),
        field_size=(grid, grid),
        max_num_food=n_food,
        min_food_level=np.ones(n_food, dtype=int),
        max_food_level=np.full(n_food, k, dtype=int),
        sight=env_cfg.get("sight", grid),
        max_episode_steps=env_cfg["max_steps"],
        force_coop=env_cfg.get("force_coop", False),
        observe_agent_levels=env_cfg.get("observe_agent_levels", True),
    )
    env.np_random = np.random.default_rng(seed)
    return env


def compute_iqm(returns: np.ndarray) -> float:
    if returns.size == 0:
        return float("nan")
    q25, q75 = np.percentile(returns, [25, 75])
    mask = (returns >= q25) & (returns <= q75)
    return float(returns[mask].mean()) if mask.any() else float(returns.mean())


def build_agent(cfg: dict, device: str):
    model_cfg = cfg["model"]
    common = dict(
        action_dim=model_cfg["action_dim"],
        type_dim=model_cfg["type_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_gnn_layers=model_cfg["n_gnn_layers"],
        pairwise_rank=model_cfg["pairwise_rank"],
        lr=cfg["training"]["lr"],
        gamma=cfg["training"]["gamma"],
        tau=cfg.get("tau", None),
        t_update=cfg["training"]["t_update"],
        t_targ_update=cfg["training"]["t_targ_update"],
        polyak_tau=cfg["training"].get("polyak_tau", None),
        device=device,
    )
    inf_cfg = cfg.get("inference")
    if inf_cfg is None:
        return GPLAgent(obs_dim=cfg["preprocess"]["obs_dim"], **common)
    return GPLAgentInf(
        obs_dim=cfg["preprocess"]["obs_dim"],
        aux_n_classes=inf_cfg.get("aux_n_classes", 3),
        aux_weight=inf_cfg.get("aux_weight", 0.1),
        **common,
    )


def evaluate(agent, cfg: dict, n_episodes: int, seed: int):
    env_cfg = cfg["env"]
    types_cfg = cfg["types"]
    food_cfg = cfg["food"]
    model_cfg = cfg["model"]
    n_agents = env_cfg["n_agents"]
    n_food = food_cfg["n_food"]
    k = types_cfg["K"]
    hidden_dim = model_cfg["hidden_dim"]
    action_dim = model_cfg["action_dim"]
    observe_agent_levels = env_cfg.get("observe_agent_levels", True)
    food_probs = {int(level): prob for level, prob in food_cfg.get("fixed_level_probs", {2: 0.6, 3: 0.4}).items()}
    rng = np.random.default_rng(seed)
    env = make_lbf_env(cfg, seed=seed)
    is_inf = isinstance(agent, GPLAgentInf)
    rows = []
    for episode in range(n_episodes):
        agent_levels = sample_stationary_composition(n_agents, k, rng)
        food_levels = sample_food_levels(n_food, rng, food_probs)
        inject_levels(env, agent_levels, food_levels)
        obs = _unpack_reset(env.reset())
        agent.reset()
        ep_return = 0.0
        ep_len = 0
        done = False
        while not done:
            b, _, _ = preprocess_lbf(
                obs, n_agents, n_food, hidden_dim=hidden_dim,
                device=str(agent.device), observe_agent_levels=observe_agent_levels,
            )
            b_np = b.cpu().numpy()
            if is_inf:
                action = agent.act_inf(b_np, learner_idx=0, epsilon=0.0)
                agent.advance_hidden_inf(b_np)
            else:
                action = agent.act(b_np, learner_idx=0, epsilon=0.0)
                agent.advance_hidden(b_np)
            joint_action = [rng.integers(0, action_dim) for _ in range(n_agents)]
            joint_action[0] = action
            obs, reward, done, _ = _unpack_step(env.step(joint_action))
            ep_return += reward
            ep_len += 1
        rows.append({
            "episode": episode,
            "return": ep_return,
            "length": ep_len,
            "agent_levels": ";".join(str(x) for x in agent_levels),
            "food_levels": ";".join(str(x) for x in food_levels),
        })
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/gpl_lbf.yaml")
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    n_episodes = 3 if args.smoke_test else args.n_episodes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = build_agent(cfg, device)
    agent.load(args.checkpoint)
    rows = evaluate(agent, cfg, n_episodes=n_episodes, seed=args.seed)
    returns = np.array([row["return"] for row in rows], dtype=np.float64)
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, f"stationary_eval_seed{args.seed}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "length", "agent_levels", "food_levels"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")
    print(f"mean_return={returns.mean():.4f} iqm_return={compute_iqm(returns):.4f} n={len(rows)}")


if __name__ == "__main__":
    main()
