"""
Unit tests for PREPROCESS (Appendix C.1) and environment utilities.

Covers:
  - B_j = [x_j; u] construction
  - LSTM hidden state management for open agent sets
  - Edge cases: single agent, all agents replaced, agent reordering
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from envs.env_utils import preprocess, preprocess_lbf


# ======================================================================
# Fixtures
# ======================================================================

HIDDEN_DIM = 64


@pytest.fixture
def simple_obs():
    """Raw obs with 3 agents (4 features each) + 3 shared features = 15 total."""
    return np.arange(15, dtype=np.float32)


@pytest.fixture
def agent_slices_3():
    """Feature slices for 3 agents, 4 features each."""
    return {0: slice(0, 4), 1: slice(4, 8), 2: slice(8, 12)}


@pytest.fixture
def shared_slice():
    return slice(12, 15)


# ======================================================================
# B_j = [x_j ; u] construction
# ======================================================================

class TestPreprocessConstruction:

    def test_output_shape(self, simple_obs, agent_slices_3, shared_slice):
        B, hidden, agent_ids = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            hidden_dim=HIDDEN_DIM,
        )
        assert B.shape == (3, 7)  # 4 agent features + 3 shared
        assert hidden[0].shape == (3, HIDDEN_DIM)
        assert hidden[1].shape == (3, HIDDEN_DIM)
        assert agent_ids == [0, 1, 2]

    def test_bj_is_xj_concat_u(self, simple_obs, agent_slices_3, shared_slice):
        """B_j should be [x_j ; u] for each agent."""
        B, _, _ = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            hidden_dim=HIDDEN_DIM,
        )
        u = simple_obs[12:15]
        for j in range(3):
            x_j = simple_obs[j * 4:(j + 1) * 4]
            expected = np.concatenate([x_j, u])
            np.testing.assert_array_almost_equal(B[j].numpy(), expected)

    def test_shared_features_same_for_all(self, simple_obs, agent_slices_3, shared_slice):
        """The shared portion of B should be identical across all agents."""
        B, _, _ = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            hidden_dim=HIDDEN_DIM,
        )
        # Last 3 elements of each row should be the shared features
        for j in range(3):
            np.testing.assert_array_equal(
                B[j, 4:].numpy(), B[0, 4:].numpy()
            )

    def test_agent_features_differ(self, simple_obs, agent_slices_3, shared_slice):
        """Per-agent portions of B should differ (since obs values differ)."""
        B, _, _ = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            hidden_dim=HIDDEN_DIM,
        )
        assert not torch.equal(B[0, :4], B[1, :4])
        assert not torch.equal(B[1, :4], B[2, :4])

    def test_single_agent(self):
        """Should work with a single agent."""
        obs = np.array([1.0, 2.0, 3.0, 10.0, 20.0], dtype=np.float32)
        slices = {0: slice(0, 3)}
        B, hidden, ids = preprocess(
            obs, slices, slice(3, 5), hidden_dim=HIDDEN_DIM,
        )
        assert B.shape == (1, 5)
        assert ids == [0]
        expected = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(B[0].numpy(), expected)

    def test_curr_agent_ids_ordering(self, simple_obs, agent_slices_3, shared_slice):
        """If curr_agent_ids is given, B rows should follow that order."""
        B, _, ids = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            curr_agent_ids=[2, 0, 1],
            hidden_dim=HIDDEN_DIM,
        )
        assert ids == [2, 0, 1]
        # First row should be agent 2's features
        x_2 = simple_obs[8:12]
        u = simple_obs[12:15]
        expected = np.concatenate([x_2, u])
        np.testing.assert_array_almost_equal(B[0].numpy(), expected)


# ======================================================================
# Hidden state management for openness
# ======================================================================

class TestPreprocessHiddenStates:

    def test_no_prior_state_zeros(self, simple_obs, agent_slices_3, shared_slice):
        """Without prior hidden state, should return zeros."""
        _, hidden, _ = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            hidden_dim=HIDDEN_DIM,
        )
        assert (hidden[0] == 0).all()
        assert (hidden[1] == 0).all()

    def test_carry_forward_persistent_agents(self, simple_obs, agent_slices_3, shared_slice):
        """Hidden states should carry forward for agents present in both steps."""
        prev_h = torch.randn(3, HIDDEN_DIM)
        prev_c = torch.randn(3, HIDDEN_DIM)
        prev_ids = [0, 1, 2]

        _, hidden, _ = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[0, 1, 2],
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        assert torch.equal(hidden[0], prev_h)
        assert torch.equal(hidden[1], prev_c)

    def test_new_agent_gets_zeros(self, simple_obs, agent_slices_3, shared_slice):
        """A newly appearing agent should get zero-initialised hidden state."""
        prev_h = torch.randn(2, HIDDEN_DIM)
        prev_c = torch.randn(2, HIDDEN_DIM)
        prev_ids = [0, 1]

        _, hidden, ids = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[0, 1, 2],
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        # Agents 0 and 1 carried forward
        assert torch.equal(hidden[0][0], prev_h[0])
        assert torch.equal(hidden[0][1], prev_h[1])
        # Agent 2 is new → zeros
        assert (hidden[0][2] == 0).all()
        assert (hidden[1][2] == 0).all()

    def test_departed_agent_removed(self):
        """Departed agent's hidden state should not appear in output."""
        obs = np.arange(11, dtype=np.float32)  # 2 agents × 4 + 3 shared
        slices = {0: slice(0, 4), 1: slice(4, 8)}
        shared = slice(8, 11)

        prev_h = torch.randn(3, HIDDEN_DIM)
        prev_c = torch.randn(3, HIDDEN_DIM)
        prev_ids = [0, 1, 2]  # agent 2 was here before

        _, hidden, ids = preprocess(
            obs, slices, shared,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[0, 1],
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        assert ids == [0, 1]
        assert hidden[0].shape == (2, HIDDEN_DIM)
        # Agent 0 and 1 carried forward from their old positions
        assert torch.equal(hidden[0][0], prev_h[0])
        assert torch.equal(hidden[0][1], prev_h[1])

    def test_all_agents_replaced(self):
        """If all agents change, all hidden states should be zero."""
        obs = np.arange(11, dtype=np.float32)
        slices = {3: slice(0, 4), 4: slice(4, 8)}
        shared = slice(8, 11)

        prev_h = torch.randn(2, HIDDEN_DIM)
        prev_c = torch.randn(2, HIDDEN_DIM)
        prev_ids = [0, 1]  # completely different

        _, hidden, ids = preprocess(
            obs, slices, shared,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[3, 4],
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        assert ids == [3, 4]
        assert (hidden[0] == 0).all()
        assert (hidden[1] == 0).all()

    def test_agent_reordering(self):
        """Hidden states should follow agent IDs, not positions."""
        obs = np.arange(11, dtype=np.float32)
        slices = {1: slice(0, 4), 0: slice(4, 8)}
        shared = slice(8, 11)

        prev_h = torch.randn(2, HIDDEN_DIM)
        prev_c = torch.randn(2, HIDDEN_DIM)
        prev_ids = [0, 1]  # agent 0 was at index 0, agent 1 at index 1

        _, hidden, ids = preprocess(
            obs, slices, shared,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[1, 0],  # swapped order
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        assert ids == [1, 0]
        # Agent 1 (now at position 0) should have agent 1's old hidden state
        assert torch.equal(hidden[0][0], prev_h[1])
        # Agent 0 (now at position 1) should have agent 0's old hidden state
        assert torch.equal(hidden[0][1], prev_h[0])

    def test_mixed_new_and_persistent(self):
        """Some agents persist, some depart, some are new."""
        obs = np.arange(11, dtype=np.float32)
        slices = {0: slice(0, 4), 5: slice(4, 8)}
        shared = slice(8, 11)

        prev_h = torch.randn(3, HIDDEN_DIM)
        prev_c = torch.randn(3, HIDDEN_DIM)
        prev_ids = [0, 1, 2]
        # Agent 0 persists, agents 1 & 2 depart, agent 5 is new

        _, hidden, ids = preprocess(
            obs, slices, shared,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[0, 5],
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        assert ids == [0, 5]
        # Agent 0: carried forward
        assert torch.equal(hidden[0][0], prev_h[0])
        # Agent 5: new → zeros
        assert (hidden[0][1] == 0).all()
        assert (hidden[1][1] == 0).all()


# ======================================================================
# preprocess_lbf — LBF-specific observation parsing
# ======================================================================

class TestPreprocessLBF:
    """Tests for preprocess_lbf to verify correct food-first observation parsing."""

    def _make_lbf_obs(self, n_agents=3, n_food=3):
        """Create synthetic LBF per-agent observations.

        LBF obs format per agent i:
            [food_0(y,x,level), ..., food_m(y,x,level), self(y,x,level), other_0(y,x,level), ...]

        Food features come FIRST, then agent features (self first among agents).
        """
        # Distinct values so we can verify correct extraction
        food_positions = np.array([
            [1.0, 2.0, 3.0],   # food 0: y=1, x=2, level=3
            [4.0, 5.0, 6.0],   # food 1: y=4, x=5, level=6
            [7.0, 8.0, 9.0],   # food 2: y=7, x=8, level=9
        ], dtype=np.float32)[:n_food]

        agent_positions = np.array([
            [10.0, 11.0, 12.0],  # agent 0: y=10, x=11, level=12
            [20.0, 21.0, 22.0],  # agent 1: y=20, x=21, level=22
            [30.0, 31.0, 32.0],  # agent 2: y=30, x=31, level=32
        ], dtype=np.float32)[:n_agents]

        per_agent_obs = []
        for i in range(n_agents):
            # Food comes first
            food_part = food_positions.flatten()
            # Self comes next
            self_part = agent_positions[i]
            # Others follow
            others = [agent_positions[j] for j in range(n_agents) if j != i]
            others_part = np.concatenate(others) if others else np.array([], dtype=np.float32)
            obs_i = np.concatenate([food_part, self_part, others_part])
            per_agent_obs.append(obs_i)

        return per_agent_obs, agent_positions, food_positions

    def test_output_shape(self):
        """B should be (n_agents, 3 + 3*n_food)."""
        obs, _, _ = self._make_lbf_obs(n_agents=3, n_food=3)
        B, hidden, ids = preprocess_lbf(obs, n_agents=3, n_food=3, hidden_dim=HIDDEN_DIM)
        assert B.shape == (3, 3 + 3 * 3)  # (3, 12)
        assert hidden[0].shape == (3, HIDDEN_DIM)
        assert ids == [0, 1, 2]

    def test_agent_features_extracted_correctly(self):
        """Each agent's x_j should be their (y, x, level), NOT food features."""
        obs, agent_pos, food_pos = self._make_lbf_obs(n_agents=3, n_food=3)
        B, _, _ = preprocess_lbf(obs, n_agents=3, n_food=3, hidden_dim=HIDDEN_DIM)

        # B_j = [x_j; u] where x_j = agent j's (y, x, level), u = food features
        for j in range(3):
            x_j = B[j, :3].numpy()
            np.testing.assert_array_almost_equal(
                x_j, agent_pos[j],
                err_msg=f"Agent {j} features should be {agent_pos[j]}, got {x_j}"
            )

    def test_food_features_are_shared(self):
        """The shared portion u should be the food features, identical for all agents."""
        obs, _, food_pos = self._make_lbf_obs(n_agents=3, n_food=3)
        B, _, _ = preprocess_lbf(obs, n_agents=3, n_food=3, hidden_dim=HIDDEN_DIM)

        expected_food = food_pos.flatten()
        for j in range(3):
            u_j = B[j, 3:].numpy()
            np.testing.assert_array_almost_equal(
                u_j, expected_food,
                err_msg=f"Agent {j}'s shared features should be food, got {u_j}"
            )

    def test_food_not_confused_with_agents(self):
        """Regression: the first 3 features of B_j must NOT be food[0]."""
        obs, agent_pos, food_pos = self._make_lbf_obs(n_agents=3, n_food=3)
        B, _, _ = preprocess_lbf(obs, n_agents=3, n_food=3, hidden_dim=HIDDEN_DIM)

        # Agent 0's features should be [10, 11, 12], NOT food[0] = [1, 2, 3]
        x_0 = B[0, :3].numpy()
        assert not np.allclose(x_0, food_pos[0]), \
            f"BUG: Agent 0 features = food[0] ({food_pos[0]}), observation parsing is backwards!"
        np.testing.assert_array_almost_equal(x_0, agent_pos[0])

    def test_single_agent_single_food(self):
        """Edge case: 1 agent, 1 food."""
        obs, agent_pos, food_pos = self._make_lbf_obs(n_agents=1, n_food=1)
        B, _, ids = preprocess_lbf(obs, n_agents=1, n_food=1, hidden_dim=HIDDEN_DIM)
        assert B.shape == (1, 3 + 3)  # (1, 6)
        assert ids == [0]
        np.testing.assert_array_almost_equal(B[0, :3].numpy(), agent_pos[0])
        np.testing.assert_array_almost_equal(B[0, 3:].numpy(), food_pos[0])

    def test_flat_global_state_input(self):
        """When raw_obs is already a flat array, should work via else branch."""
        # Flat format: [agent_0(3), agent_1(3), food(3*2)]
        flat = np.array([10, 11, 12, 20, 21, 22, 1, 2, 3, 4, 5, 6], dtype=np.float32)
        B, _, ids = preprocess_lbf(flat, n_agents=2, n_food=2, hidden_dim=HIDDEN_DIM)
        assert B.shape == (2, 3 + 6)  # (2, 9)
        np.testing.assert_array_almost_equal(B[0, :3].numpy(), [10, 11, 12])
        np.testing.assert_array_almost_equal(B[1, :3].numpy(), [20, 21, 22])
        np.testing.assert_array_almost_equal(B[0, 3:].numpy(), [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_almost_equal(B[1, 3:].numpy(), [1, 2, 3, 4, 5, 6])
