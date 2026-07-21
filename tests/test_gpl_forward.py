"""
Formal tests for GPL sub-modules and top-level agent.

Covers:
  - TypeInferenceModel (§4.2, Eq. 7)
  - AgentModel (§4.4, Eqs. 11-13)
  - JointActionValueModel (§4.3, Eqs. 8-10)
  - GPLAgent (§4.1-4.6, Algorithms 2-5)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np

from agents.gpl.type_inference import TypeInferenceModel
from agents.gpl.agent_model import AgentModel
from agents.gpl.joint_action_value import JointActionValueModel
from agents.gpl.gpl_agent import GPLAgent


# ======================================================================
# Fixtures
# ======================================================================

OBS_DIM = 16
ACTION_DIM = 5
TYPE_DIM = 32
HIDDEN_DIM = 64
N_AGENTS = 3
PAIRWISE_RANK = 4
N_GNN_LAYERS = 2


@pytest.fixture
def type_net():
    return TypeInferenceModel(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM, type_dim=TYPE_DIM,
    )


@pytest.fixture
def agent_model():
    return AgentModel(
        type_dim=TYPE_DIM, lstm_hidden_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM,
        n_gnn_layers=N_GNN_LAYERS,
    )


@pytest.fixture
def q_network():
    return JointActionValueModel(
        type_dim=TYPE_DIM, action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM, pairwise_rank=PAIRWISE_RANK,
    )


@pytest.fixture
def gpl_agent():
    return GPLAgent(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        type_dim=TYPE_DIM, hidden_dim=HIDDEN_DIM,
        n_gnn_layers=N_GNN_LAYERS, pairwise_rank=PAIRWISE_RANK,
        t_update=2, t_targ_update=10,
    )


# ======================================================================
# TypeInferenceModel
# ======================================================================

class TestTypeInferenceModel:

    def test_output_shape(self, type_net):
        B_t = torch.randn(N_AGENTS, OBS_DIM)
        type_emb, (h, c) = type_net(B_t)
        assert type_emb.shape == (N_AGENTS, TYPE_DIM)
        assert h.shape == (N_AGENTS, HIDDEN_DIM)
        assert c.shape == (N_AGENTS, HIDDEN_DIM)

    def test_hidden_state_carryover(self, type_net):
        """Hidden state from step 1 should affect step 2's output."""
        B_t = torch.randn(N_AGENTS, OBS_DIM)
        _, hidden1 = type_net(B_t)
        emb_with_carry, _ = type_net(B_t, hidden1)
        emb_from_scratch, _ = type_net(B_t, None)
        # With carry-over vs fresh should differ
        assert not torch.allclose(emb_with_carry, emb_from_scratch, atol=1e-6)

    def test_none_hidden_defaults_to_zeros(self, type_net):
        """Passing hidden=None should be equivalent to passing zeros."""
        B_t = torch.randn(N_AGENTS, OBS_DIM)
        emb_none, _ = type_net(B_t, hidden=None)
        zero_h = type_net.reset_hidden(N_AGENTS)
        emb_zero, _ = type_net(B_t, hidden=zero_h)
        assert torch.allclose(emb_none, emb_zero, atol=1e-7)

    def test_forward_sequence_shape(self, type_net):
        T = 5
        B_seq = torch.randn(N_AGENTS, T, OBS_DIM)
        type_embs, (h, c) = type_net.forward_sequence(B_seq)
        assert type_embs.shape == (N_AGENTS, T, TYPE_DIM)
        assert h.shape == (N_AGENTS, HIDDEN_DIM)
        assert c.shape == (N_AGENTS, HIDDEN_DIM)

    def test_forward_sequence_matches_stepwise(self, type_net):
        """forward_sequence should produce identical results to step-by-step."""
        T = 4
        B_seq = torch.randn(N_AGENTS, T, OBS_DIM)
        embs_seq, hidden_seq = type_net.forward_sequence(B_seq)

        hidden = None
        embs_step = []
        for t in range(T):
            emb, hidden = type_net(B_seq[:, t, :], hidden)
            embs_step.append(emb)
        embs_step = torch.stack(embs_step, dim=1)

        assert torch.allclose(embs_seq, embs_step, atol=1e-6)
        assert torch.allclose(hidden_seq[0], hidden[0], atol=1e-6)

    def test_variable_agent_count(self, type_net):
        """Should handle different N between calls (openness)."""
        B_2 = torch.randn(2, OBS_DIM)
        B_5 = torch.randn(5, OBS_DIM)
        emb2, _ = type_net(B_2)
        emb5, _ = type_net(B_5)
        assert emb2.shape == (2, TYPE_DIM)
        assert emb5.shape == (5, TYPE_DIM)

    def test_reset_hidden_shape(self, type_net):
        h, c = type_net.reset_hidden(4, device="cpu")
        assert h.shape == (4, HIDDEN_DIM)
        assert c.shape == (4, HIDDEN_DIM)
        assert (h == 0).all()
        assert (c == 0).all()


# ======================================================================
# AgentModel
# ======================================================================

class TestAgentModel:

    def test_forward_shape(self, agent_model):
        theta = torch.randn(1, N_AGENTS, TYPE_DIM)
        cell = torch.randn(1, N_AGENTS, HIDDEN_DIM)
        logits = agent_model(theta, cell)
        assert logits.shape == (1, N_AGENTS, ACTION_DIM)

    def test_action_probs_sum_to_one(self, agent_model):
        theta = torch.randn(2, N_AGENTS, TYPE_DIM)
        cell = torch.randn(2, N_AGENTS, HIDDEN_DIM)
        probs = agent_model.action_probs(theta, cell)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_action_probs_nonnegative(self, agent_model):
        theta = torch.randn(1, N_AGENTS, TYPE_DIM)
        cell = torch.randn(1, N_AGENTS, HIDDEN_DIM)
        probs = agent_model.action_probs(theta, cell)
        assert (probs >= 0).all()

    def test_log_probs_consistent(self, agent_model):
        """log_probs should be log of action_probs."""
        theta = torch.randn(1, N_AGENTS, TYPE_DIM)
        cell = torch.randn(1, N_AGENTS, HIDDEN_DIM)
        probs = agent_model.action_probs(theta, cell)
        log_p = agent_model.log_probs(theta, cell)
        assert torch.allclose(log_p, torch.log(probs), atol=1e-5)

    def test_gnn_changes_output(self, agent_model):
        """With >1 agent, GNN message passing should change embeddings vs single agent."""
        theta_single = torch.randn(1, 1, TYPE_DIM)
        cell_single = torch.randn(1, 1, HIDDEN_DIM)
        logits_single = agent_model(theta_single, cell_single)

        # Same first agent but with a second agent present
        theta_pair = torch.cat([theta_single, torch.randn(1, 1, TYPE_DIM)], dim=1)
        cell_pair = torch.cat([cell_single, torch.randn(1, 1, HIDDEN_DIM)], dim=1)
        logits_pair = agent_model(theta_pair, cell_pair)

        # Agent 0's logits should differ when another agent is present
        assert not torch.allclose(logits_single[:, 0], logits_pair[:, 0], atol=1e-5)

    def test_batched_forward(self, agent_model):
        B = 4
        theta = torch.randn(B, N_AGENTS, TYPE_DIM)
        cell = torch.randn(B, N_AGENTS, HIDDEN_DIM)
        logits = agent_model(theta, cell)
        assert logits.shape == (B, N_AGENTS, ACTION_DIM)


# ======================================================================
# JointActionValueModel
# ======================================================================

class TestJointActionValueModel:

    def test_individual_q_shape(self, q_network):
        theta_all = torch.randn(1, N_AGENTS, TYPE_DIM)
        theta_i = torch.randn(1, TYPE_DIM)
        q_ind = q_network.individual_q(theta_all, theta_i)
        assert q_ind.shape == (1, N_AGENTS, ACTION_DIM)

    def test_pairwise_q_shape(self, q_network):
        theta_all = torch.randn(1, N_AGENTS, TYPE_DIM)
        theta_i = torch.randn(1, TYPE_DIM)
        factors = q_network.pairwise_q(theta_all, theta_i)
        assert factors.shape == (1, N_AGENTS, PAIRWISE_RANK, ACTION_DIM)

    def test_forward_returns_both(self, q_network):
        theta_all = torch.randn(2, N_AGENTS, TYPE_DIM)
        q_ind, pw_factors = q_network(theta_all, learner_idx=0)
        assert q_ind.shape == (2, N_AGENTS, ACTION_DIM)
        assert pw_factors.shape == (2, N_AGENTS, PAIRWISE_RANK, ACTION_DIM)

    def test_compute_joint_q_shape(self, q_network):
        theta_all = torch.randn(2, N_AGENTS, TYPE_DIM)
        q_ind, pw_factors = q_network(theta_all, learner_idx=0)
        actions = torch.randint(0, ACTION_DIM, (2, N_AGENTS))
        q_val = q_network.compute_joint_q(q_ind, pw_factors, actions)
        assert q_val.shape == (2,)

    def test_compute_joint_q_is_scalar_per_batch(self, q_network):
        """Joint Q should be a single scalar per batch element."""
        theta_all = torch.randn(1, N_AGENTS, TYPE_DIM)
        q_ind, pw_factors = q_network(theta_all, learner_idx=0)
        actions = torch.randint(0, ACTION_DIM, (1, N_AGENTS))
        q_val = q_network.compute_joint_q(q_ind, pw_factors, actions)
        assert q_val.dim() == 1 and q_val.shape[0] == 1

    def test_different_actions_different_q(self, q_network):
        """Different joint actions should generally give different Q-values."""
        torch.manual_seed(42)
        theta_all = torch.randn(1, N_AGENTS, TYPE_DIM)
        q_ind, pw_factors = q_network(theta_all, learner_idx=0)
        actions_a = torch.zeros(1, N_AGENTS, dtype=torch.long)
        actions_b = torch.ones(1, N_AGENTS, dtype=torch.long)
        q_a = q_network.compute_joint_q(q_ind, pw_factors, actions_a)
        q_b = q_network.compute_joint_q(q_ind, pw_factors, actions_b)
        assert not torch.allclose(q_a, q_b, atol=1e-6)

    def test_pairwise_identity_two_agents(self, q_network):
        """For 2 agents, the (Σf)²−Σf² identity should equal 2·f1^T·f2."""
        theta_all = torch.randn(1, 2, TYPE_DIM)
        q_ind, pw_factors = q_network(theta_all, learner_idx=0)
        actions = torch.randint(0, ACTION_DIM, (1, 2))

        # Get factors at chosen actions
        actions_exp = actions.unsqueeze(-1).unsqueeze(-1).expand(
            1, 2, PAIRWISE_RANK, 1
        )
        sel = pw_factors.gather(-1, actions_exp).squeeze(-1)  # (1, 2, K)
        f0, f1 = sel[:, 0, :], sel[:, 1, :]

        # Identity: (f0+f1)^2 - f0^2 - f1^2 = 2 * f0·f1
        via_identity = (f0 + f1).pow(2).sum() - f0.pow(2).sum() - f1.pow(2).sum()
        via_direct = 2 * (f0 * f1).sum()
        assert torch.allclose(via_identity, via_direct, atol=1e-5)


# ======================================================================
# GPLAgent — compute methods
# ======================================================================

class TestGPLAgentCompute:

    def test_compute_qv_shape(self, gpl_agent):
        B_t = torch.randn(N_AGENTS, OBS_DIM)
        q_bar, h_q, h_ag = gpl_agent.compute_qv(B_t, learner_idx=0)
        assert q_bar.shape == (ACTION_DIM,)
        # Hidden states should be (N, hidden_dim) tuples
        assert h_q[0].shape == (N_AGENTS, HIDDEN_DIM)
        assert h_ag[0].shape == (N_AGENTS, HIDDEN_DIM)

    def test_compute_qjoint_unbatched(self, gpl_agent):
        B_t = torch.randn(N_AGENTS, OBS_DIM)
        actions = torch.randint(0, ACTION_DIM, (N_AGENTS,))
        q_val, h_q = gpl_agent.compute_qjoint(B_t, actions, learner_idx=0)
        assert q_val.dim() == 0  # scalar
        assert h_q[0].shape == (N_AGENTS, HIDDEN_DIM)

    def test_compute_qjoint_batched(self, gpl_agent):
        B = 4
        B_t = torch.randn(B, N_AGENTS, OBS_DIM)
        actions = torch.randint(0, ACTION_DIM, (B, N_AGENTS))
        q_val, h_q = gpl_agent.compute_qjoint(B_t, actions, learner_idx=0)
        assert q_val.shape == (B,)
        assert h_q[0].shape == (B, N_AGENTS, HIDDEN_DIM)

    def test_compute_pteam_unbatched(self, gpl_agent):
        B_t = torch.randn(N_AGENTS, OBS_DIM)
        log_p, h_ag = gpl_agent.compute_pteam(B_t, learner_idx=0)
        assert log_p.shape == (N_AGENTS, ACTION_DIM)
        # Log probs should be negative
        assert (log_p <= 0).all()
        # Should be valid log probs (exp sums to 1)
        probs = log_p.exp()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(N_AGENTS), atol=1e-5)

    def test_compute_pteam_batched(self, gpl_agent):
        B = 3
        B_t = torch.randn(B, N_AGENTS, OBS_DIM)
        log_p, h_ag = gpl_agent.compute_pteam(B_t, learner_idx=0)
        assert log_p.shape == (B, N_AGENTS, ACTION_DIM)


# ======================================================================
# GPLAgent — act
# ======================================================================

class TestGPLAgentAct:

    def test_act_returns_valid_action(self, gpl_agent):
        B_t = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
        action = gpl_agent.act(B_t, learner_idx=0, epsilon=0.0)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < ACTION_DIM

    def test_act_greedy_deterministic(self, gpl_agent):
        """Greedy action should be deterministic for same input and state."""
        B_t = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
        gpl_agent.reset()
        a1 = gpl_agent.act(B_t, epsilon=0.0)
        gpl_agent.reset()
        a2 = gpl_agent.act(B_t, epsilon=0.0)
        assert a1 == a2

    def test_act_epsilon_one_is_random(self, gpl_agent):
        """With epsilon=1.0, actions should be random (not always the same)."""
        B_t = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
        actions = set()
        for _ in range(50):
            gpl_agent.reset()
            actions.add(gpl_agent.act(B_t, epsilon=1.0))
        # With 50 tries and 5 actions, should see at least 2 distinct
        assert len(actions) >= 2

    def test_act_advances_hidden(self, gpl_agent):
        """act() should update internal LSTM hidden states."""
        gpl_agent.reset()
        assert gpl_agent._hidden_q is None
        B_t = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
        gpl_agent.act(B_t)
        assert gpl_agent._hidden_q is not None
        assert gpl_agent._hidden_agent is not None


# ======================================================================
# GPLAgent — training
# ======================================================================

class TestGPLAgentTraining:

    def test_train_step_online_runs(self, gpl_agent):
        """train_step_online should run without error."""
        B_t = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
        actions = np.random.randint(0, ACTION_DIM, N_AGENTS)
        B_next = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
        metrics = gpl_agent.train_step_online(
            B_t, actions, reward=1.0, B_t_next=B_next, done=False,
        )
        # First step: t_update=2, so no gradient applied yet
        assert metrics is None

    def test_train_step_online_gradient_accumulation(self, gpl_agent):
        """Gradients should be applied every t_update steps (t_update=2)."""
        for i in range(2):
            B_t = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
            actions = np.random.randint(0, ACTION_DIM, N_AGENTS)
            B_next = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
            metrics = gpl_agent.train_step_online(
                B_t, actions, reward=1.0, B_t_next=B_next, done=False,
            )
        # Second step should trigger gradient application
        assert metrics is not None
        assert "q_loss" in metrics
        assert "agent_model_loss" in metrics

    def test_train_step_online_done_resets_hidden(self, gpl_agent):
        """done=True should reset hidden states."""
        B_t = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
        actions = np.random.randint(0, ACTION_DIM, N_AGENTS)
        B_next = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
        gpl_agent.train_step_online(
            B_t, actions, reward=0.0, B_t_next=B_next, done=True,
        )
        assert gpl_agent._hidden_q is None
        assert gpl_agent._hidden_agent is None
        assert gpl_agent._hidden_q_target is None

    def test_update_replay_buffer(self, gpl_agent):
        """Replay buffer update should return loss metrics."""
        B = 8
        batch = {
            "B_t": torch.randn(B, N_AGENTS, OBS_DIM),
            "actions": torch.randint(0, ACTION_DIM, (B, N_AGENTS)),
            "rewards": torch.randn(B),
            "B_t_next": torch.randn(B, N_AGENTS, OBS_DIM),
            "dones": torch.zeros(B),
            "learner_idx": 0,
            "teammate_indices": [1, 2],
        }
        metrics = gpl_agent.update(batch)
        assert "q_loss" in metrics
        assert "agent_model_loss" in metrics
        assert metrics["q_loss"] >= 0
        assert metrics["agent_model_loss"] >= 0

    def test_update_changes_weights(self, gpl_agent):
        """update() should modify network parameters."""
        B = 8
        batch = {
            "B_t": torch.randn(B, N_AGENTS, OBS_DIM),
            "actions": torch.randint(0, ACTION_DIM, (B, N_AGENTS)),
            "rewards": torch.randn(B),
            "B_t_next": torch.randn(B, N_AGENTS, OBS_DIM),
            "dones": torch.zeros(B),
            "learner_idx": 0,
            "teammate_indices": [1, 2],
        }
        params_before = {n: p.clone() for n, p in
                         gpl_agent.q_network.named_parameters()}
        gpl_agent.update(batch)
        changed = any(
            not torch.equal(params_before[n], p)
            for n, p in gpl_agent.q_network.named_parameters()
        )
        assert changed


# ======================================================================
# GPLAgent — persistence
# ======================================================================

class TestGPLAgentPersistence:

    def test_save_load_roundtrip(self, gpl_agent, tmp_path):
        """Save/load should restore identical state."""
        # Run a few steps to get non-default state
        B = 4
        batch = {
            "B_t": torch.randn(B, N_AGENTS, OBS_DIM),
            "actions": torch.randint(0, ACTION_DIM, (B, N_AGENTS)),
            "rewards": torch.randn(B),
            "B_t_next": torch.randn(B, N_AGENTS, OBS_DIM),
            "dones": torch.zeros(B),
            "learner_idx": 0,
            "teammate_indices": [1, 2],
        }
        gpl_agent.update(batch)

        path = str(tmp_path / "gpl_test.pt")
        gpl_agent.save(path)

        agent2 = GPLAgent(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            type_dim=TYPE_DIM, hidden_dim=HIDDEN_DIM,
            n_gnn_layers=N_GNN_LAYERS, pairwise_rank=PAIRWISE_RANK,
        )
        agent2.load(path)

        # Check weights match
        for p1, p2 in zip(gpl_agent.q_network.parameters(),
                          agent2.q_network.parameters()):
            assert torch.equal(p1, p2)

        # Check step count matches
        assert agent2._step_count == gpl_agent._step_count

    def test_reset_clears_hidden(self, gpl_agent):
        B_t = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
        gpl_agent.act(B_t)
        assert gpl_agent._hidden_q is not None
        gpl_agent.reset()
        assert gpl_agent._hidden_q is None
        assert gpl_agent._hidden_agent is None
        assert gpl_agent._hidden_q_target is None
