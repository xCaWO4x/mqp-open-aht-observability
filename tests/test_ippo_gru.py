"""
Unit tests for Recurrent Independent PPO (IPPO-GRU).

Verifies:
1. Network output tensor shapes (step and sequence evaluation).
2. Invisible entity perturbation invariance (flat_obs zeroing guarantee).
3. Selective GRU hidden state reset on episode completion (done=True).
4. True termination vs. time-limit truncation in GAE advantage estimation.
5. PPO rollout update and detailed metric logging (explained variance, clip fraction, grad norm).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from agents.ippo_gru.ippo_gru_model import IPPORecurrentActorCritic
from agents.ippo_gru.ippo_gru_agent import IPPOAgent
from envs.entity_adapter import extract_entities_lbf, extract_entities_wolfpack, EntityBatch


def test_ippo_gru_model_shapes():
    """Verify Actor-Critic step and sequence shapes."""
    torch.manual_seed(42)
    obs_dim = 18
    action_dim = 6
    gru_hidden = 128
    B, T = 4, 16

    model = IPPORecurrentActorCritic(obs_dim=obs_dim, action_dim=action_dim, gru_hidden=gru_hidden)
    h0 = model.get_initial_hidden(batch_size=B)

    # 1. Single-step forward
    obs_step = torch.randn(B, obs_dim)
    action, log_prob, value, next_h = model.step(obs_step, h0)
    assert action.shape == (B,)
    assert log_prob.shape == (B,)
    assert value.shape == (B,)
    assert next_h.shape == (1, B, gru_hidden)

    # 2. Sequence evaluation
    obs_seq = torch.randn(B, T, obs_dim)
    actions = torch.randint(0, action_dim, (B, T))
    log_probs, entropy, values = model.evaluate_actions(obs_seq, h0, actions)
    assert log_probs.shape == (B, T)
    assert entropy.shape == (B, T)
    assert values.shape == (B, T)
    print("test_ippo_gru_model_shapes passed successfully.")


def test_invisible_entity_perturbation_invariance_ippo():
    """CRITICAL REQUIREMENT:
    Create a restricted-sight observation.
    Identify all invisible entities.
    Perturb ALL underlying features of invisible entities by extreme values:
    - coordinates
    - level
    - type/flags
    - any other entity attributes
    Reconstruct the IPPO observation.
    With identical GRU hidden state:
    actor logits before perturbation == actor logits after perturbation
    critic value before perturbation == critic value after perturbation
    Tolerance: < 1e-6 (strictly 0.0).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    obs_dim = 18  # 6 entities * 3 features
    action_dim = 6
    gru_hidden = 128

    model = IPPORecurrentActorCritic(obs_dim=obs_dim, action_dim=action_dim, gru_hidden=gru_hidden)
    model.eval()

    # Fixed initial hidden state
    h0 = model.get_initial_hidden(batch_size=1)

    # -------------------------------------------------------------
    # 1. LBF Restricted-Sight Observation
    # 3 foods (9 dims), 3 agents (9 dims with observe_agent_levels=True)
    # -------------------------------------------------------------
    # Food 0 at (2, 2, 1) -> visible
    # Food 1 at (-1, -1, 0) -> out of sight (invisible)
    # Food 2 at (-1, -1, 0) -> out of sight (invisible)
    # Self at (2, 3, 2) -> visible
    # Teammate 1 at (3, 3, 1) -> visible
    # Teammate 2 at (-1, -1, 0) -> out of sight (invisible)
    raw_obs_clean = np.array([
        2.0, 2.0, 1.0,   # food 0 (visible)
        -1.0, -1.0, 0.0, # food 1 (invisible)
        -1.0, -1.0, 0.0, # food 2 (invisible)
        2.0, 3.0, 2.0,   # self (visible)
        3.0, 3.0, 1.0,   # teammate 1 (visible)
        -1.0, -1.0, 0.0, # teammate 2 (invisible)
    ], dtype=np.float32)

    batch_clean = extract_entities_lbf(raw_obs_clean, n_agents=3, n_food=3, grid_size=8.0, observe_agent_levels=True)
    flat_clean = batch_clean.flat_obs

    # Evaluate clean observation
    with torch.no_grad():
        _, _, val_clean, _ = model.step(flat_clean, h0)
        # Compute exact actor logits
        h_enc_clean = model.encoder(flat_clean).unsqueeze(0)
        _, h_gru_clean = model.gru(h_enc_clean, h0)
        logits_clean = model.actor(h_gru_clean.squeeze(0))

    # Verify that invisible entities were correctly identified
    assert batch_clean.visible_mask[0, 0].item() is True   # self
    assert batch_clean.visible_mask[0, 1].item() is True   # tm 1
    assert batch_clean.visible_mask[0, 2].item() is False  # tm 2 is invisible
    assert batch_clean.visible_mask[0, 3].item() is True   # food 0
    assert batch_clean.visible_mask[0, 4].item() is False  # food 1 is invisible
    assert batch_clean.visible_mask[0, 5].item() is False  # food 2 is invisible

    # -------------------------------------------------------------
    # 2. Perturb ALL features of invisible entities by extreme values
    # Coordinates, levels, flags, etc.
    # -------------------------------------------------------------
    raw_obs_perturbed = raw_obs_clean.copy()
    # Perturb food 1 (indices 3, 4, 5 in raw_obs)
    raw_obs_perturbed[3] = -99999.0  # extreme sentinel coordinate y
    raw_obs_perturbed[4] = -88888.0  # extreme sentinel coordinate x
    raw_obs_perturbed[5] = 77777.0   # extreme food level (should be masked!)

    # Perturb food 2 (indices 6, 7, 8 in raw_obs)
    raw_obs_perturbed[6] = -12345.0  # extreme sentinel coordinate y
    raw_obs_perturbed[7] = -54321.0  # extreme sentinel coordinate x
    raw_obs_perturbed[8] = 99999.0   # extreme food level (should be masked!)

    # Perturb teammate 2 (indices 15, 16, 17 in raw_obs)
    raw_obs_perturbed[15] = -65432.0 # extreme sentinel coordinate y
    raw_obs_perturbed[16] = -23456.0 # extreme sentinel coordinate x
    raw_obs_perturbed[17] = 88888.0  # extreme teammate level (should be masked!)

    # Reconstruct IPPO observation
    batch_perturbed = extract_entities_lbf(raw_obs_perturbed, n_agents=3, n_food=3, grid_size=8.0, observe_agent_levels=True)
    flat_perturbed = batch_perturbed.flat_obs

    # Check flat_obs equality: all invisible entity features must be identically 0.0
    flat_diff = torch.max(torch.abs(flat_clean - flat_perturbed)).item()
    print(f"Max flat_obs difference after extreme invisible entity perturbation: {flat_diff}")
    assert flat_diff < 1e-6, f"flat_obs leaked invisible entity information! Max diff: {flat_diff}"

    # Evaluate perturbed observation with IDENTICAL hidden state
    with torch.no_grad():
        _, _, val_perturbed, _ = model.step(flat_perturbed, h0)
        h_enc_pert = model.encoder(flat_perturbed).unsqueeze(0)
        _, h_gru_pert = model.gru(h_enc_pert, h0)
        logits_perturbed = model.actor(h_gru_pert.squeeze(0))

    logits_diff = torch.max(torch.abs(logits_clean - logits_perturbed)).item()
    val_diff = torch.abs(val_clean - val_perturbed).item()

    print(f"Max actor logits difference: {logits_diff}")
    print(f"Critic value difference: {val_diff}")

    assert logits_diff < 1e-6, f"Actor logits changed by {logits_diff} after invisible entity perturbation!"
    assert val_diff < 1e-6, f"Critic value changed by {val_diff} after invisible entity perturbation!"

    # -------------------------------------------------------------
    # 3. Wolfpack Invariance Check
    # 2 prey (6 dims), 3 wolves (6 dims) = 12 dims
    # Wolf 2 invisible (-1, -1), Prey 1 invisible (-1, -1, 0)
    # -------------------------------------------------------------
    wolf_raw_clean = np.array([
        4.0, 5.0, 1.0,   # prey 0 (visible)
        -1.0, -1.0, 0.0, # prey 1 (invisible)
        3.0, 3.0,        # self wolf (visible)
        4.0, 3.0,        # wolf 1 (visible)
        -1.0, -1.0,      # wolf 2 (invisible)
    ], dtype=np.float32)

    wolf_batch_clean = extract_entities_wolfpack(wolf_raw_clean, n_wolves=3, n_prey=2, grid_size=10.0)

    wolf_raw_perturbed = wolf_raw_clean.copy()
    wolf_raw_perturbed[3] = -99999.0
    wolf_raw_perturbed[4] = -88888.0
    wolf_raw_perturbed[5] = 999.0   # prey active flag perturbed
    wolf_raw_perturbed[10] = -77777.0
    wolf_raw_perturbed[11] = -55555.0

    wolf_batch_perturbed = extract_entities_wolfpack(wolf_raw_perturbed, n_wolves=3, n_prey=2, grid_size=10.0)

    wolf_flat_diff = torch.max(torch.abs(wolf_batch_clean.flat_obs - wolf_batch_perturbed.flat_obs)).item()
    print(f"Wolfpack max flat_obs difference: {wolf_flat_diff}")
    assert wolf_flat_diff < 1e-6, f"Wolfpack flat_obs leaked invisible entity information! Diff: {wolf_flat_diff}"

    print("test_invisible_entity_perturbation_invariance_ippo passed successfully.")


def test_ippo_gru_hidden_reset_on_done():
    """Verify that when done=True occurs for specific envs, only those envs reset their GRU hidden state."""
    torch.manual_seed(42)
    n_envs = 4
    obs_dim = 18
    action_dim = 6
    gru_hidden = 128

    agent = IPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_envs=n_envs,
        rollout_length=8,
        gru_hidden=gru_hidden,
    )

    # Initial hidden is all zeros
    assert (agent.hidden == 0.0).all()

    # Step all environments
    obs = np.random.randn(n_envs, obs_dim).astype(np.float32)
    actions, log_probs, values, curr_hidden = agent.act(obs)

    # Now agent.hidden should be non-zero
    assert not (agent.hidden == 0.0).all()

    # Suppose env 1 and env 3 terminate (done=True), but env 0 and env 2 continue (done=False)
    dones = np.array([False, True, False, True], dtype=bool)
    rewards = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    agent.observe(
        obs=obs,
        actions=actions,
        log_probs=log_probs,
        rewards=rewards,
        dones=dones,
        values=values,
        step_hiddens=curr_hidden,
    )

    # Verify that env 1 and 3 are reset to 0.0, while env 0 and 2 retain their hidden states
    assert (agent.hidden[:, 1, :] == 0.0).all(), "Env 1 hidden state was not reset on done!"
    assert (agent.hidden[:, 3, :] == 0.0).all(), "Env 3 hidden state was not reset on done!"
    assert not (agent.hidden[:, 0, :] == 0.0).all(), "Env 0 hidden state was incorrectly zeroed!"
    assert not (agent.hidden[:, 2, :] == 0.0).all(), "Env 2 hidden state was incorrectly zeroed!"
    print("test_ippo_gru_hidden_reset_on_done passed successfully.")


def test_ippo_truncation_vs_termination_gae():
    """Verify that GAE correctly bootstraps on time-limit truncation, but zeroes out on true termination."""
    torch.manual_seed(42)
    n_envs = 2
    rollout_len = 4
    obs_dim = 18
    action_dim = 6
    device = torch.device("cpu")

    agent = IPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_envs=n_envs,
        rollout_length=rollout_len,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
    )

    # Fill buffer: Env 0 terminates at t=3 (terminated=True, truncated=False)
    # Env 1 truncates at t=3 (terminated=False, truncated=True)
    obs = torch.zeros((n_envs, obs_dim))
    act = torch.zeros(n_envs, dtype=torch.long)
    lp = torch.zeros(n_envs)
    val = torch.ones(n_envs) * 2.0  # V(s) = 2.0
    rew = torch.ones(n_envs) * 1.0  # r = 1.0
    h = torch.zeros((1, n_envs, 128))

    for t in range(rollout_len - 1):
        agent.buffer.insert(obs, act, lp, rew, torch.zeros(n_envs, dtype=torch.bool), val, h)

    # Step t=3: Env 0 done (term), Env 1 done (trunc)
    dones = torch.tensor([True, True], dtype=torch.bool)
    truncated = torch.tensor([False, True], dtype=torch.bool)
    agent.buffer.insert(obs, act, lp, rew, dones, val, h, truncated=truncated)

    next_value = torch.tensor([10.0, 10.0])  # V(s_{t+1}) = 10.0
    next_done = torch.tensor([True, True], dtype=torch.bool)
    next_trunc = torch.tensor([False, True], dtype=torch.bool)

    agent.buffer.compute_gae(next_value, next_done, next_trunc, gamma=0.99, gae_lambda=0.95)

    # For Env 0 (true termination): target value should be r = 1.0 (no bootstrapping from 10.0)
    # return at t=3 is delta_3 + V_3 = (1.0 - 2.0) + 2.0 = 1.0
    ret_env0 = agent.buffer.returns[3, 0].item()
    # For Env 1 (time-limit truncation): target value SHOULD bootstrap: r + gamma * 10.0 = 1.0 + 9.9 = 10.9
    ret_env1 = agent.buffer.returns[3, 1].item()

    print(f"GAE Return for True Termination (Env 0): {ret_env0:.2f} (expected 1.0)")
    print(f"GAE Return for Time-limit Truncation (Env 1): {ret_env1:.2f} (expected 10.9)")

    assert abs(ret_env0 - 1.0) < 1e-4, f"True termination return was {ret_env0}, expected 1.0"
    assert abs(ret_env1 - 10.9) < 1e-4, f"Truncation return was {ret_env1}, expected 10.9"
    print("test_ippo_truncation_vs_termination_gae passed successfully.")


def test_ippo_gru_update():
    """Verify that a complete rollout and PPO update runs cleanly and returns all requested metrics."""
    torch.manual_seed(42)
    n_envs = 2
    rollout_len = 8
    obs_dim = 18
    action_dim = 6

    agent = IPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_envs=n_envs,
        rollout_length=rollout_len,
        ppo_epochs=2,
    )

    for _ in range(rollout_len):
        obs = np.random.randn(n_envs, obs_dim).astype(np.float32)
        actions, log_probs, values, curr_hidden = agent.act(obs)
        rewards = np.array([0.5, 1.0], dtype=np.float32)
        dones = np.array([False, False], dtype=bool)

        agent.observe(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            values=values,
            step_hiddens=curr_hidden,
        )

    next_obs = np.random.randn(n_envs, obs_dim).astype(np.float32)
    metrics = agent.update(next_obs, dones)

    required_metrics = [
        "actor_loss", "critic_loss", "entropy", "approx_kl",
        "clip_fraction", "explained_variance", "grad_norm"
    ]
    for m in required_metrics:
        assert m in metrics, f"Metric {m} missing from update() return dict!"
        assert not np.isnan(metrics[m]), f"Metric {m} returned NaN!"
        print(f"Metric {m}: {metrics[m]:.4f}")

    print("test_ippo_gru_update passed successfully.")


if __name__ == "__main__":
    print("=== Running IPPO-GRU Unit Tests ===")
    test_ippo_gru_model_shapes()
    test_invisible_entity_perturbation_invariance_ippo()
    test_ippo_gru_hidden_reset_on_done()
    test_ippo_truncation_vs_termination_gae()
    test_ippo_gru_update()
    print("\nALL IPPO-GRU UNIT TESTS PASSED!")
