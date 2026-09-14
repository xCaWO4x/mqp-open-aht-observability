"""
Verification test script for Multi-Agent Observability Benchmark.

Verifies:
1. 100 steps in LBF at r=4 and r=8 under both 'random' and 'greedy' teammates.
2. 100 steps in Wolfpack at r=4 and r=10 under both 'random' and 'greedy' teammates.
3. Observation masking: entities outside sight radius are masked to -1.
4. 1 batch gradient update of GPL-Q on both environments (CPU and GPU if available).
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.wolfpack_env import WolfpackEnv
from envs.env_utils import preprocess_lbf, preprocess_wolfpack
from agents.gpl.gpl_agent import GPLAgent
from agents.teammate_policies import get_teammate_policy
from experiments.run_benchmark import make_benchmark_env, unpack_reset, unpack_step, sample_lbf_levels


def run_env_steps(env_name: str, sight: int, teammate_type: str, n_steps: int = 100, seed: int = 42):
    print(f"--> Testing {env_name.upper()} | sight={sight} | teammates={teammate_type} ({n_steps} steps)...")
    rng = np.random.default_rng(seed)

    if env_name == "lbf":
        n_agents = 3
        env = make_benchmark_env("lbf", sight=sight, seed=seed, grid_size=8, n_agents=n_agents, n_food=3, max_steps=50, K=3, observe_agent_levels=False)
    else:
        n_agents = 3
        env = make_benchmark_env("wolfpack", sight=sight, seed=seed, grid_size=10, n_wolves=n_agents, n_prey=2, max_steps=50, capture_reward=1.0)

    teammate_policy = get_teammate_policy(env_name, teammate_type, rng=rng)

    step_count = 0
    episodes = 0
    total_rewards = 0.0

    while step_count < n_steps:
        if env_name == "lbf":
            ag_levels, fd_levels = sample_lbf_levels(n_agents, 3, 3, rng)
            env.min_player_level = np.array(ag_levels)
            env.max_player_level = np.array(ag_levels)
            env.min_food_level = np.array(fd_levels)
            env.max_food_level = np.array(fd_levels)

        obs = unpack_reset(env.reset())
        episodes += 1
        done = False

        while not done and step_count < n_steps:
            # Action selection: agent 0 chooses random action, teammates use policy
            actions = [int(rng.integers(0, 6 if env_name == "lbf" else 5))]
            for tm_idx in range(1, n_agents):
                act = teammate_policy.select_action(tm_idx, env=env, obs=obs)
                actions.append(act)

            obs, reward, done, info = unpack_step(env.step(actions))
            total_rewards += reward
            step_count += 1

    print(f"    PASSED: {step_count} steps across {episodes} episodes | total_reward={total_rewards:.2f}")


def test_observation_masking():
    print("--> Testing observation masking logic...")
    # Wolfpack: check that sight=3 masks entities > 3 cells away
    env = WolfpackEnv(grid_size=10, n_wolves=3, n_prey=2, sight=3, seed=123)
    obs, _ = env.reset()

    # Place wolf 0 at (0, 0), wolf 1 at (0, 8), prey 0 at (8, 8)
    env.wolf_positions[0] = [0, 0]
    env.wolf_positions[1] = [0, 8]
    env.prey_positions[0] = [8, 8]

    masked_obs = env._get_obs()[0]
    # Obs layout: [prey_0(3), prey_1(3), self(2), other_wolves...]
    # prey 0 is at (8, 8), distance from (0, 0) is 8 > 3 -> should be masked to -1
    assert masked_obs[0] == -1.0 and masked_obs[1] == -1.0, f"Prey 0 not masked: {masked_obs[:3]}"
    # wolf 1 is at (0, 8), distance 8 > 3 -> should be masked to -1
    # wolf 1 is at index 6+2 = 8, 9
    assert masked_obs[8] == -1.0 and masked_obs[9] == -1.0, f"Wolf 1 not masked: {masked_obs[8:10]}"
    # self is at index 6, 7 -> should be (0, 0)
    assert masked_obs[6] == 0.0 and masked_obs[7] == 0.0, f"Self position incorrect: {masked_obs[6:8]}"

    # Full sight: check sight=10 unmasks
    env_full = WolfpackEnv(grid_size=10, n_wolves=3, n_prey=2, sight=10, seed=123)
    env_full.reset()
    env_full.wolf_positions[0] = [0, 0]
    env_full.wolf_positions[1] = [0, 8]
    env_full.prey_positions[0] = [8, 8]
    full_obs = env_full._get_obs()[0]
    assert full_obs[0] == 8.0 and full_obs[1] == 8.0, f"Prey 0 should be visible: {full_obs[:3]}"
    assert full_obs[8] == 0.0 and full_obs[9] == 8.0, f"Wolf 1 should be visible: {full_obs[8:10]}"
    print("    PASSED: Observation masking validated successfully.")


def test_gpl_training_iteration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Testing GPL-Q single training update iteration (device={device})...")

    # 1. LBF update
    print("    [1/2] GPL-Q on LBF (obs_dim=11, action_dim=6)...")
    agent_lbf = GPLAgent(obs_dim=11, action_dim=6, type_dim=32, hidden_dim=64, n_gnn_layers=2, pairwise_rank=4, t_update=1, device=device)
    B_t_lbf = np.random.randn(3, 11).astype(np.float32)
    actions_lbf = np.array([1, 2, 0])
    reward_lbf = 1.0
    B_next_lbf = np.random.randn(3, 11).astype(np.float32)
    done_lbf = False

    metrics_lbf = agent_lbf.train_step_online(B_t_lbf, actions_lbf, reward_lbf, B_next_lbf, done_lbf, learner_idx=0)
    assert metrics_lbf is not None, "LBF metrics should not be None when t_update=1"
    assert "q_loss" in metrics_lbf and "agent_model_loss" in metrics_lbf, f"Missing loss keys: {metrics_lbf}"
    assert np.isfinite(metrics_lbf["q_loss"]), f"q_loss not finite: {metrics_lbf['q_loss']}"
    assert np.isfinite(metrics_lbf["agent_model_loss"]), f"agent_model_loss not finite: {metrics_lbf['agent_model_loss']}"
    print(f"    LBF update passed! q_loss={metrics_lbf['q_loss']:.4f}, agent_loss={metrics_lbf['agent_model_loss']:.4f}")

    # 2. Wolfpack update
    print("    [2/2] GPL-Q on Wolfpack (obs_dim=8, action_dim=5)...")
    agent_wp = GPLAgent(obs_dim=8, action_dim=5, type_dim=32, hidden_dim=64, n_gnn_layers=2, pairwise_rank=4, t_update=1, device=device)
    B_t_wp = np.random.randn(3, 8).astype(np.float32)
    actions_wp = np.array([2, 1, 4])
    reward_wp = 1.0
    B_next_wp = np.random.randn(3, 8).astype(np.float32)
    done_wp = False

    metrics_wp = agent_wp.train_step_online(B_t_wp, actions_wp, reward_wp, B_next_wp, done_wp, learner_idx=0)
    assert metrics_wp is not None, "Wolfpack metrics should not be None when t_update=1"
    assert "q_loss" in metrics_wp and "agent_model_loss" in metrics_wp, f"Missing loss keys: {metrics_wp}"
    assert np.isfinite(metrics_wp["q_loss"]), f"q_loss not finite: {metrics_wp['q_loss']}"
    assert np.isfinite(metrics_wp["agent_model_loss"]), f"agent_model_loss not finite: {metrics_wp['agent_model_loss']}"
    print(f"    Wolfpack update passed! q_loss={metrics_wp['q_loss']:.4f}, agent_loss={metrics_wp['agent_model_loss']:.4f}")

    print("    PASSED: GPL-Q training updates validated on both environments.")


def main():
    print("==================================================")
    print("STARTING MULTI-AGENT OBSERVABILITY BENCHMARK TESTS")
    print("==================================================")

    # 1. Observation masking test
    test_observation_masking()

    # 2. LBF tests
    run_env_steps("lbf", sight=4, teammate_type="random", n_steps=100)
    run_env_steps("lbf", sight=4, teammate_type="greedy", n_steps=100)
    run_env_steps("lbf", sight=8, teammate_type="random", n_steps=100)
    run_env_steps("lbf", sight=8, teammate_type="greedy", n_steps=100)

    # 3. Wolfpack tests
    run_env_steps("wolfpack", sight=4, teammate_type="random", n_steps=100)
    run_env_steps("wolfpack", sight=4, teammate_type="greedy", n_steps=100)
    run_env_steps("wolfpack", sight=10, teammate_type="random", n_steps=100)
    run_env_steps("wolfpack", sight=10, teammate_type="greedy", n_steps=100)

    # 4. Training update integration test
    test_gpl_training_iteration()

    print("==================================================")
    print("ALL VERIFICATION SUITE CHECKS PASSED (100%)")
    print("==================================================")


if __name__ == "__main__":
    main()
