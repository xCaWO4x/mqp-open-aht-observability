"""
Unified Modular Training Entrypoint for Observability Sweeps.

Supports:
    --algo: transformer_q, ippo_gru, gpl
    --env: lbf, wolfpack
    --sight_radius: integer (e.g. 3, 4, 6, 8, 10) or string aliases ("restricted", "intermediate", "full")
    --teammate_type: greedy, random
    --seed: integer

Usage Examples:
    python train.py --algo transformer_q --env lbf --sight_radius 4 --seed 1
    python train.py --algo ippo_gru --env lbf --sight_radius full --seed 1
    python train.py --algo transformer_q --env wolfpack --sight_radius restricted --seed 1 --smoke-test
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

if "OMP_NUM_THREADS" not in os.environ and torch.get_num_threads() > 4:
    torch.set_num_threads(4)

from agents.gpl.gpl_agent import GPLAgent
from agents.ippo_gru.ippo_gru_agent import IPPOAgent
from agents.teammate_policies import get_teammate_policy
from agents.transformer_q.transformer_q_agent import TransformerQAgent
from envs.entity_adapter import extract_entities
from envs.env_utils import preprocess_lbf, preprocess_wolfpack
from experiments.run_benchmark import make_benchmark_env, sample_lbf_levels, unpack_reset


def resolve_sight_radius(env_name: str, sight_arg: str) -> int:
    """Resolve sight radius argument into concrete integer."""
    sight_str = str(sight_arg).strip().lower()
    env_lower = env_name.lower()
    is_lbf = "lbf" in env_lower or "foraging" in env_lower

    if sight_str == "full":
        return 8 if is_lbf else 10
    elif sight_str == "restricted":
        # Strongest restricted radius identified in benchmark
        return 3 if is_lbf else 4
    elif sight_str == "intermediate":
        return 4 if is_lbf else 6
    else:
        try:
            return int(sight_str)
        except ValueError:
            raise ValueError(f"Cannot parse sight_radius '{sight_arg}' for {env_name}.")


def unpack_step_info(step_out) -> Tuple[object, float, bool, bool, bool, dict]:
    """Unpack step output distinguishing true termination from time-limit truncation."""
    if len(step_out) == 5:
        obs, rewards, terminated, truncated, info = step_out
        term = all(terminated) if isinstance(terminated, (list, tuple)) else bool(terminated)
        trunc = all(truncated) if isinstance(truncated, (list, tuple)) else bool(truncated)
        done = term or trunc
    else:
        obs, rewards, dones_out, info = step_out
        done = all(dones_out) if isinstance(dones_out, (list, tuple)) else bool(dones_out)
        trunc = False
        if isinstance(info, dict):
            trunc = bool(info.get("TimeLimit.truncated", False) or info.get("truncated", False))
        term = done and not trunc

    reward = float(rewards[0]) if isinstance(rewards, (list, tuple)) else float(rewards)
    return obs, reward, done, term, trunc, info


def run_evaluation(
    algo: str,
    agent,
    env_name: str,
    sight: int,
    teammate_policy,
    n_agents: int,
    n_eval_episodes: int = 10,
    device: str = "cpu",
    seed: int = 9999,
) -> Tuple[float, float, float]:
    """Run greedy evaluation episodes and return (mean_return, success_rate, mean_length)."""
    eval_env = make_benchmark_env(
        env_name,
        sight=sight,
        seed=seed,
        grid_size=8 if "lbf" in env_name else 10,
        n_agents=n_agents,
        n_food=3,
        n_wolves=n_agents,
        n_prey=2,
        max_steps=50,
        observe_agent_levels=False,
    )
    rng = np.random.default_rng(seed)

    returns = []
    lengths = []
    successes = []

    eval_hidden = None
    if algo == "ippo_gru":
        eval_hidden = agent.ac.get_initial_hidden(batch_size=1, device=device)

    for _ in range(n_eval_episodes):
        if "lbf" in env_name:
            ag_levels, fd_levels = sample_lbf_levels(n_agents, 3, 3, rng)
            eval_env.min_player_level = np.array(ag_levels)
            eval_env.max_player_level = np.array(ag_levels)
            eval_env.min_food_level = np.array(fd_levels)
            eval_env.max_food_level = np.array(fd_levels)

        obs = unpack_reset(eval_env.reset())
        if algo == "ippo_gru":
            eval_hidden.zero_()
        elif algo == "gpl":
            agent.reset()

        ep_ret = 0.0
        ep_len = 0
        done = False

        while not done:
            # Select greedy action
            if algo == "transformer_q":
                eb = extract_entities(obs, env_name, device=device)
                ego_action = agent.act(eb, epsilon=0.0)
            elif algo == "ippo_gru":
                eb = extract_entities(obs, env_name, device=device)
                with torch.no_grad():
                    _, _, _, eval_hidden = agent.ac.step(eb.flat_obs, eval_hidden)
                    logits = agent.ac.actor(eval_hidden.squeeze(0))
                    ego_action = int(logits.argmax(dim=-1).item())
            elif algo == "gpl":
                if "lbf" in env_name:
                    B = preprocess_lbf(obs, n_agents=n_agents, n_food=3, hidden_dim=100, device=device, observe_agent_levels=False, from_ego_perspective=True)[0]
                else:
                    B = preprocess_wolfpack(obs, n_wolves=n_agents, n_prey=2, hidden_dim=100, device=device, from_ego_perspective=True)[0]
                ego_action = agent.act(B.cpu().numpy(), learner_idx=0, epsilon=0.0)

            # Teammate actions
            joint_actions = [ego_action]
            for tm_idx in range(1, n_agents):
                tm_act = teammate_policy.select_action(tm_idx, env=eval_env, obs=obs)
                joint_actions.append(tm_act)

            next_obs, rew, done, _, _, _ = unpack_step_info(eval_env.step(joint_actions))
            obs = next_obs
            ep_ret += rew
            ep_len += 1

        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(1.0 if ep_ret > 0.0 else 0.0)

    return float(np.mean(returns)), float(np.mean(successes)), float(np.mean(lengths))


def train(
    algo: str,
    env_name: str,
    sight_radius: Union[int, str],
    teammate_type: str = "greedy",
    seed: int = 1,
    train_steps: int = 2000000,
    n_envs: int = 16,
    eval_interval: int = 50000,
    n_eval_episodes: int = 10,
    output_dir: Optional[str] = None,
    smoke_test: bool = False,
    device: Optional[str] = None,
):
    start_time = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sight = resolve_sight_radius(env_name, sight_radius)
    env_is_lbf = "lbf" in env_name.lower() or "foraging" in env_name.lower()

    if output_dir is None:
        output_dir = f"results/{env_name}_{algo}_sight{sight}_{teammate_type}_seed{seed}"
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Seeds
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Environment parameters
    if env_is_lbf:
        n_agents = 3
        n_food = 3
        K = 3
        action_dim = 6
        grid_size = 8
        max_steps = 50
        obs_dim = 11
        flat_obs_dim = 18  # 6 entities * 3 features
        envs = [
            make_benchmark_env(
                "lbf",
                sight=sight,
                seed=seed + i,
                grid_size=grid_size,
                n_agents=n_agents,
                n_food=n_food,
                max_steps=max_steps,
                K=K,
                observe_agent_levels=False,
            )
            for i in range(n_envs)
        ]
        preprocess_gpl = lambda o, dev: preprocess_lbf(
            o, n_agents=n_agents, n_food=n_food, hidden_dim=100, device=dev, observe_agent_levels=False, from_ego_perspective=True
        )[0]
    else:
        n_agents = 3
        n_prey = 2
        action_dim = 5
        grid_size = 10
        max_steps = 50
        obs_dim = 8
        flat_obs_dim = 15  # 5 entities * 3 features
        envs = [
            make_benchmark_env(
                "wolfpack",
                sight=sight,
                seed=seed + i,
                grid_size=grid_size,
                n_wolves=n_agents,
                n_prey=n_prey,
                max_steps=max_steps,
            )
            for i in range(n_envs)
        ]
        preprocess_gpl = lambda o, dev: preprocess_wolfpack(
            o, n_wolves=n_agents, n_prey=n_prey, hidden_dim=100, device=dev, from_ego_perspective=True
        )[0]

    teammate_policy = get_teammate_policy(env_name, teammate_type, rng=rng)

    # Agent instantiation
    if algo == "transformer_q":
        agent = TransformerQAgent(
            action_dim=action_dim,
            feat_dim=3,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            lr=2.5e-4,
            gamma=0.99,
            t_update=4,
            polyak_tau=1.0e-3,
            device=device,
        )
    elif algo == "ippo_gru":
        rollout_len = 8 if smoke_test else 128
        agent = IPPOAgent(
            obs_dim=flat_obs_dim,
            action_dim=action_dim,
            n_envs=n_envs,
            rollout_length=rollout_len,
            encoder_hidden=128,
            gru_hidden=128,
            lr=3.0e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=2 if smoke_test else 4,
            device=device,
        )
    elif algo == "gpl":
        agent = GPLAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            type_dim=100,
            hidden_dim=100,
            n_gnn_layers=2,
            pairwise_rank=5,
            lr=2.5e-4,
            gamma=0.99,
            t_update=4,
            t_targ_update=1,
            polyak_tau=1.0e-3,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Training parameters
    target_steps = 200 if smoke_test else train_steps
    eps_init = 1.0
    eps_final = 0.05
    decay_steps = 200000

    def get_eps(step: int) -> float:
        frac = min(step / max(decay_steps, 1), 1.0)
        return eps_init + (eps_final - eps_init) * frac

    # Setup metrics CSV with comprehensive columns
    metrics_file = os.path.join(output_dir, "metrics.csv")
    csv_f = open(metrics_file, "w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow([
        "step", "episode", "return", "length", "loss", "success_rate",
        "runtime_sec", "visible_entities", "policy_loss", "value_loss",
        "entropy", "approx_kl", "clip_fraction", "explained_variance", "grad_norm"
    ])

    print(f"=== Starting Unified Training ===")
    print(f"Algo: {algo} | Env: {env_name} | Sight: {sight} ({sight_radius}) | Teammates: {teammate_type} | Seed: {seed}")
    print(f"Target steps: {target_steps} | N_envs: {n_envs} | Device: {device} | Output: {output_dir}")

    # Vectorized state tracking
    env_obs = [None] * n_envs
    env_done = [True] * n_envs
    env_ep_return = [0.0] * n_envs
    env_ep_len = [0] * n_envs
    env_hidden_gpl = [(None, None, None)] * n_envs

    global_step = 0
    completed_episodes = 0
    best_eval_return = -float("inf")
    last_eval_step = 0
    all_returns = []
    last_ppo_metrics = {}

    pbar = tqdm(total=target_steps, desc=f"{env_name}_{algo}_s{sight}", unit="step")

    while global_step < target_steps:
        # Check if IPPO rollout collection is needed
        if algo == "ippo_gru":
            # Collect one full rollout of length agent.rollout_length across n_envs
            for _ in range(agent.rollout_length):
                if global_step >= target_steps:
                    break

                # Handle resets
                for env_idx in range(n_envs):
                    if env_done[env_idx]:
                        if env_is_lbf:
                            ag_levels, fd_levels = sample_lbf_levels(n_agents, K, n_food, rng)
                            envs[env_idx].min_player_level = np.array(ag_levels)
                            envs[env_idx].max_player_level = np.array(ag_levels)
                            envs[env_idx].min_food_level = np.array(fd_levels)
                            envs[env_idx].max_food_level = np.array(fd_levels)
                        env_obs[env_idx] = unpack_reset(envs[env_idx].reset())
                        env_done[env_idx] = False
                        env_ep_return[env_idx] = 0.0
                        env_ep_len[env_idx] = 0

                # Extract flat observations for all envs
                batch_eb = extract_entities(env_obs, env_name, device=device)
                actions, log_probs, values, step_hiddens = agent.act(batch_eb.flat_obs)

                # Step environments
                rewards = np.zeros(n_envs, dtype=np.float32)
                dones = np.zeros(n_envs, dtype=bool)
                truncateds = np.zeros(n_envs, dtype=bool)

                for env_idx in range(n_envs):
                    ego_action = int(actions[env_idx])
                    joint_actions = [ego_action]
                    for tm_idx in range(1, n_agents):
                        tm_act = teammate_policy.select_action(tm_idx, env=envs[env_idx], obs=env_obs[env_idx])
                        joint_actions.append(tm_act)

                    next_obs, rew, done, term, trunc, _ = unpack_step_info(envs[env_idx].step(joint_actions))
                    rewards[env_idx] = rew
                    dones[env_idx] = done
                    truncateds[env_idx] = trunc
                    env_obs[env_idx] = next_obs
                    env_ep_return[env_idx] += rew
                    env_ep_len[env_idx] += 1
                    env_done[env_idx] = done
                    global_step += 1

                    if done:
                        completed_episodes += 1
                        all_returns.append(env_ep_return[env_idx])
                        vis_count = int(batch_eb.visible_mask[env_idx].sum().item())
                        csv_writer.writerow([
                            global_step,
                            completed_episodes,
                            env_ep_return[env_idx],
                            env_ep_len[env_idx],
                            last_ppo_metrics.get("actor_loss", 0.0),
                            1.0 if env_ep_return[env_idx] > 0 else 0.0,
                            f"{time.time() - start_time:.1f}",
                            vis_count,
                            last_ppo_metrics.get("actor_loss", 0.0),
                            last_ppo_metrics.get("critic_loss", 0.0),
                            last_ppo_metrics.get("entropy", 0.0),
                            last_ppo_metrics.get("approx_kl", 0.0),
                            last_ppo_metrics.get("clip_fraction", 0.0),
                            last_ppo_metrics.get("explained_variance", 0.0),
                            last_ppo_metrics.get("grad_norm", 0.0),
                        ])
                        try:
                            csv_f.flush()
                        except (BlockingIOError, OSError):
                            pass

                # Insert into rollout buffer with truncation distinction
                agent.observe(
                    obs=batch_eb.flat_obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    dones=dones,
                    values=values,
                    step_hiddens=step_hiddens,
                    truncated=truncateds,
                )
                pbar.update(n_envs)

            # Update PPO with termination vs truncation
            next_eb = extract_entities(env_obs, env_name, device=device)
            last_ppo_metrics = agent.update(next_eb.flat_obs, dones, truncateds)
            loss_val = last_ppo_metrics.get("actor_loss", 0.0)

        elif algo == "transformer_q":
            # Batched Transformer-Q stepping across parallel environments
            for env_idx in range(n_envs):
                if env_done[env_idx]:
                    if env_is_lbf:
                        ag_levels, fd_levels = sample_lbf_levels(n_agents, K, n_food, rng)
                        envs[env_idx].min_player_level = np.array(ag_levels)
                        envs[env_idx].max_player_level = np.array(ag_levels)
                        envs[env_idx].min_food_level = np.array(fd_levels)
                        envs[env_idx].max_food_level = np.array(fd_levels)
                    env_obs[env_idx] = unpack_reset(envs[env_idx].reset())
                    env_done[env_idx] = False
                    env_ep_return[env_idx] = 0.0
                    env_ep_len[env_idx] = 0

            eps = get_eps(global_step)
            batch_eb = extract_entities(env_obs, env_name, device=device)
            actions = agent.act_batch(batch_eb, epsilon=eps)

            rewards = np.zeros(n_envs, dtype=np.float32)
            dones = np.zeros(n_envs, dtype=bool)
            next_obs_list = [None] * n_envs

            for env_idx in range(n_envs):
                ego_action = int(actions[env_idx])
                joint_actions = [ego_action]
                for tm_idx in range(1, n_agents):
                    tm_act = teammate_policy.select_action(tm_idx, env=envs[env_idx], obs=env_obs[env_idx])
                    joint_actions.append(tm_act)

                next_obs, rew, done, term, trunc, _ = unpack_step_info(envs[env_idx].step(joint_actions))
                rewards[env_idx] = rew
                dones[env_idx] = done
                next_obs_list[env_idx] = next_obs
                env_obs[env_idx] = next_obs
                env_ep_return[env_idx] += rew
                env_ep_len[env_idx] += 1
                env_done[env_idx] = done
                global_step += 1

                if done:
                    completed_episodes += 1
                    all_returns.append(env_ep_return[env_idx])
                    vis_count = int(batch_eb.visible_mask[env_idx].sum().item())
                    csv_writer.writerow([
                        global_step, completed_episodes, env_ep_return[env_idx],
                        env_ep_len[env_idx], loss_val, 1.0 if env_ep_return[env_idx] > 0 else 0.0,
                        f"{time.time() - start_time:.1f}", vis_count,
                        loss_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    ])
                    try:
                        csv_f.flush()
                    except (BlockingIOError, OSError):
                        pass

            next_batch_eb = extract_entities(next_obs_list, env_name, device=device)
            step_metrics = agent.train_step_batch(batch_eb, actions, rewards, next_batch_eb, dones)
            if step_metrics:
                loss_val = step_metrics.get("q_loss", 0.0)
            pbar.update(n_envs)

        else:
            # GPL Q-learning
            for env_idx in range(n_envs):
                if global_step >= target_steps:
                    break

                if env_done[env_idx]:
                    if env_is_lbf:
                        ag_levels, fd_levels = sample_lbf_levels(n_agents, K, n_food, rng)
                        envs[env_idx].min_player_level = np.array(ag_levels)
                        envs[env_idx].max_player_level = np.array(ag_levels)
                        envs[env_idx].min_food_level = np.array(fd_levels)
                        envs[env_idx].max_food_level = np.array(fd_levels)
                    env_obs[env_idx] = unpack_reset(envs[env_idx].reset())
                    env_done[env_idx] = False
                    env_ep_return[env_idx] = 0.0
                    env_ep_len[env_idx] = 0
                    env_hidden_gpl[env_idx] = (None, None, None)

                obs = env_obs[env_idx]
                eps = get_eps(global_step)

                agent._hidden_q = env_hidden_gpl[env_idx][0]
                agent._hidden_agent = env_hidden_gpl[env_idx][1]
                agent._hidden_q_target = env_hidden_gpl[env_idx][2]
                B = preprocess_gpl(obs, device)
                B_np = B.cpu().numpy()
                ego_action = agent.act(B_np, learner_idx=0, epsilon=eps)
                vis_count = n_agents

                joint_actions = [ego_action]
                for tm_idx in range(1, n_agents):
                    tm_act = teammate_policy.select_action(tm_idx, env=envs[env_idx], obs=obs)
                    joint_actions.append(tm_act)

                next_obs, rew, done, term, trunc, _ = unpack_step_info(envs[env_idx].step(joint_actions))

                loss_val = 0.0
                B_next = preprocess_gpl(next_obs, device)
                step_metrics = agent.train_step_online(B_np, np.array(joint_actions), rew, B_next.cpu().numpy(), done, learner_idx=0)
                env_hidden_gpl[env_idx] = (agent._hidden_q, agent._hidden_agent, agent._hidden_q_target)
                if step_metrics:
                    loss_val = step_metrics.get("q_loss", 0.0)

                env_ep_return[env_idx] += rew
                env_ep_len[env_idx] += 1
                env_obs[env_idx] = next_obs
                env_done[env_idx] = done
                global_step += 1
                pbar.update(1)

                if done:
                    completed_episodes += 1
                    all_returns.append(env_ep_return[env_idx])
                    csv_writer.writerow([
                        global_step, completed_episodes, env_ep_return[env_idx],
                        env_ep_len[env_idx], loss_val, 1.0 if env_ep_return[env_idx] > 0 else 0.0,
                        f"{time.time() - start_time:.1f}", vis_count,
                        loss_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    ])
                    try:
                        csv_f.flush()
                    except (BlockingIOError, OSError):
                        pass

        # Periodic flush
        if completed_episodes % 50 == 0:
            try:
                csv_f.flush()
            except (BlockingIOError, OSError):
                pass

        # Periodic Evaluation
        if not smoke_test and (global_step - last_eval_step >= eval_interval or global_step >= target_steps):
            last_eval_step = global_step
            eval_ret, eval_succ, eval_len = run_evaluation(
                algo=algo,
                agent=agent,
                env_name=env_name,
                sight=sight,
                teammate_policy=teammate_policy,
                n_agents=n_agents,
                n_eval_episodes=n_eval_episodes,
                device=device,
                seed=seed + 1000,
            )
            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                agent.save(os.path.join(ckpt_dir, "model_best.pt"))
            agent.save(os.path.join(ckpt_dir, "model_latest.pt"))
            tqdm.write(f"[{algo} | Step {global_step:,}] Eval Return: {eval_ret:.3f} | Win Rate: {eval_succ:.2f} | Best: {best_eval_return:.3f}")

    pbar.close()
    csv_f.close()

    # Final evaluation (primary benchmark metric)
    final_eval_ret, final_eval_succ, final_eval_len = run_evaluation(
        algo=algo,
        agent=agent,
        env_name=env_name,
        sight=sight,
        teammate_policy=teammate_policy,
        n_agents=n_agents,
        n_eval_episodes=20 if not smoke_test else 5,
        device=device,
        seed=seed + 2000,
    )
    agent.save(os.path.join(ckpt_dir, "model_final.pt"))

    elapsed = time.time() - start_time
    summary_data = {
        "algorithm": algo,
        "environment": env_name,
        "seed": seed,
        "sight_radius": sight,
        "sight_arg": sight_radius,
        "train_steps": global_step,
        "completed_episodes": completed_episodes,
        "final_eval_return": round(final_eval_ret, 4),
        "final_eval_success_rate": round(final_eval_succ, 4),
        "best_eval_return": round(best_eval_return if best_eval_return != -float("inf") else final_eval_ret, 4),
        "runtime_seconds": round(elapsed, 2),
    }

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nTraining Complete in {elapsed:.1f}s!")
    print(f"Summary saved to {summary_file}:")
    print(json.dumps(summary_data, indent=2))
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Unified MARL Observability Benchmark")
    parser.add_argument("--algo", type=str, required=True, choices=["transformer_q", "ippo_gru", "gpl"])
    parser.add_argument("--env", type=str, required=True, choices=["lbf", "wolfpack"])
    parser.add_argument("--sight_radius", "--sight", dest="sight_radius", type=str, required=True,
                        help="Integer (e.g. 3, 4, 6, 8, 10) or string alias ('restricted', 'intermediate', 'full')")
    parser.add_argument("--teammate_type", type=str, default="greedy", choices=["greedy", "random"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=2000000)
    parser.add_argument("--n_envs", "--n-envs", dest="n_envs", type=int, default=16)
    parser.add_argument("--eval_interval", type=int, default=50000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        algo=args.algo,
        env_name=args.env,
        sight_radius=args.sight_radius,
        teammate_type=args.teammate_type,
        seed=args.seed,
        train_steps=args.train_steps,
        n_envs=args.n_envs,
        eval_interval=args.eval_interval,
        n_eval_episodes=args.n_eval_episodes,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        device=args.device,
    )


if __name__ == "__main__":
    main()
