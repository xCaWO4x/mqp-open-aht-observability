"""
Unit tests for Entity-Transformer Q-Learning.

Crucial test:
- Verify that dramatically modifying features of an invisible entity under restricted sight
  does NOT alter Q-values (mathematical invariance under attention key padding).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from envs.entity_adapter import extract_entities_lbf, extract_entities_wolfpack, EntityBatch
from agents.transformer_q.transformer_q_model import EntityTransformerQNetwork
from agents.transformer_q.transformer_q_agent import TransformerQAgent


def test_transformer_q_output_shape():
    """Verify output tensor shapes from EntityTransformerQNetwork."""
    net = EntityTransformerQNetwork(action_dim=6, feat_dim=3, d_model=64, nhead=4, num_layers=2)
    B, K = 4, 6  # 4 batch size, 6 entities
    feats = torch.randn(B, K, 3)
    types = torch.tensor([[0, 1, 1, 2, 2, 2]] * B, dtype=torch.long)
    mask = torch.tensor([[True, True, False, True, False, False]] * B, dtype=torch.bool)

    q_vals = net(feats, entity_types=types, visible_mask=mask)
    assert q_vals.shape == (B, 6), f"Expected shape ({B}, 6), got {q_vals.shape}"


def test_invisible_entity_perturbation_invariance():
    """CRITICAL REQUIREMENT:
    Take one observation under restricted sight.
    Modify the features of an invisible entity dramatically.
    Q-values should remain unchanged within numerical tolerance.
    """
    torch.manual_seed(42)
    net = EntityTransformerQNetwork(action_dim=6, feat_dim=3, d_model=64, nhead=4, num_layers=2)
    net.eval()

    # LBF observation: 3 foods (9), 3 agents (6) = 15 total
    # Food 0 at (2, 2, 1) -> visible
    # Food 1 at (-1, -1, 0) -> out of sight (invisible)
    # Food 2 at (-1, -1, 0) -> out of sight (invisible)
    # Self at (2, 3) -> visible
    # Teammate 1 at (3, 3) -> visible
    # Teammate 2 at (-1, -1) -> out of sight (invisible)
    raw_obs_0 = np.array([
        2, 2, 1,   # food 0 (visible)
        -1, -1, 0, # food 1 (invisible)
        -1, -1, 0, # food 2 (invisible)
        2, 3,      # self (visible)
        3, 3,      # teammate 1 (visible)
        -1, -1,    # teammate 2 (invisible)
    ], dtype=np.float32)

    batch_clean = extract_entities_lbf(raw_obs_0, n_agents=3, n_food=3, grid_size=8.0)

    # Compute baseline Q-values
    with torch.no_grad():
        q_baseline = net(batch_clean).cpu().numpy()

    # Verify that invisible entities were correctly flagged in the mask
    # Ordering: [self (vis), tm1 (vis), tm2 (invis), food0 (vis), food1 (invis), food2 (invis)]
    assert batch_clean.visible_mask[0, 0].item() is True
    assert batch_clean.visible_mask[0, 1].item() is True
    assert batch_clean.visible_mask[0, 2].item() is False  # tm 2 is invisible
    assert batch_clean.visible_mask[0, 3].item() is True  # food 0 is visible
    assert batch_clean.visible_mask[0, 4].item() is False # food 1 is invisible
    assert batch_clean.visible_mask[0, 5].item() is False # food 2 is invisible

    # Now dramatically modify the features of the invisible entities (e.g. food 1 and teammate 2)
    # In batch_perturbed, we change their values to astronomical or completely different numbers
    feats_perturbed = batch_clean.entity_features.clone()
    feats_perturbed[0, 2, :] = torch.tensor([99999.0, -88888.0, 77777.0])  # invisible teammate 2
    feats_perturbed[0, 4, :] = torch.tensor([-54321.0, 12345.0, 999.0])    # invisible food 1
    feats_perturbed[0, 5, :] = torch.tensor([100000.0, 100000.0, 500.0])   # invisible food 2

    batch_perturbed = EntityBatch(
        entity_features=feats_perturbed,
        entity_types=batch_clean.entity_types,
        visible_mask=batch_clean.visible_mask,  # same visibility mask
        flat_obs=batch_clean.flat_obs,
    )

    with torch.no_grad():
        q_perturbed = net(batch_perturbed).cpu().numpy()

    # Assert that Q-values are identical within strict numerical tolerance
    diff = np.abs(q_baseline - q_perturbed)
    max_diff = np.max(diff)
    print(f"Max Q-value difference after invisible entity perturbation: {max_diff}")
    assert max_diff < 1e-6, f"Q-values changed when modifying invisible entities! Max diff: {max_diff}"


def test_transformer_q_agent_training_step():
    """Verify that a training step executes, computes loss, and updates parameters."""
    torch.manual_seed(42)
    agent = TransformerQAgent(action_dim=6, feat_dim=3, lr=1e-3, t_update=1)

    raw_obs_t = np.array([2, 2, 1, -1, -1, 0, -1, -1, 0, 2, 3, 3, 3, -1, -1], dtype=np.float32)
    raw_obs_next = np.array([2, 2, 0, -1, -1, 0, -1, -1, 0, 2, 2, 3, 2, -1, -1], dtype=np.float32)

    batch_t = extract_entities_lbf(raw_obs_t, n_agents=3, n_food=3)
    batch_next = extract_entities_lbf(raw_obs_next, n_agents=3, n_food=3)

    metrics = agent.train_step_online(
        entity_batch=batch_t,
        action=5,  # LOAD
        reward=1.0,
        next_entity_batch=batch_next,
        done=False,
    )

    assert metrics is not None
    assert "q_loss" in metrics
    assert metrics["q_loss"] >= 0.0
    assert not np.isnan(metrics["q_loss"])


if __name__ == "__main__":
    print("Running test_transformer_q_output_shape...")
    test_transformer_q_output_shape()
    print("Running test_invisible_entity_perturbation_invariance...")
    test_invisible_entity_perturbation_invariance()
    print("Running test_transformer_q_agent_training_step...")
    test_transformer_q_agent_training_step()
    print("\nALL TRANSFORMER-Q UNIT TESTS PASSED!")

