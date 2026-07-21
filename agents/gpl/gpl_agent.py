"""
Top-level GPL agent.

Implements the full GPL algorithm from Sections 4.1–4.6, Appendix A
Algorithm 5, and relevant parts of D.3 (Rahman et al. 2023,
arXiv:2210.05448).

Combines:
  - TypeInferenceModel (§4.2): LSTM mapping B_t → type vectors θ
  - AgentModel (§4.4):         RFM_ζ(θ', c') + MLP_η → teammate action probs
  - JointActionValueModel (§4.3): MLP_β(θ) + MLP_δ(θ) → CG Q-values

Algorithm 5 training loop (synchronous, per-timestep):
  Lines 10-11:  Carry hidden states, run QV to select action
  Line 13:      Execute action, observe r, s'
  Line 14:      QJOINT(s, a, h_Q) → Q(H_t, a_t)
  Line 15:      QV(s', target params, h_Q^targ, h'_q) → Q̄'(H', a^i)
  Line 16:      Compute TD target y
  Line 17:      PTEAM(s, a, h_q) → teammate probs for NLL loss
  Line 18:      Compute L_{β,δ} (Eq. 16) and L_{ζ,η} (Eq. 15)
  Line 19:      Accumulate gradients
  Lines 20-22:  Apply gradients every t_update steps
  Lines 24-25:  Copy to target network every t_targ_update steps
  Line 27:      Carry hidden states forward

Also supports replay-buffer-based training via update() for experience
replay variants (noted as straightforward in Appendix A).

PREPROCESS (Appendix C.1) is implemented in envs/env_utils.py.

# TODO: Partial observability via D.3 autoencoder (ρ_t = MLP_α(c_t))
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.gpl.type_inference import TypeInferenceModel
from agents.gpl.agent_model import AgentModel
from agents.gpl.joint_action_value import JointActionValueModel


class GPLAgent:
    """Graph-based Policy Learning agent.

    Parameters
    ----------
    obs_dim : int
        Flat observation dimension (preprocessed B_t per agent).
    action_dim : int
        Number of discrete actions.
    type_dim : int
        Latent type embedding size (output of LSTM projection).
    hidden_dim : int
        LSTM hidden/cell state size and MLP width.
    n_gnn_layers : int
        GNN message-passing rounds in agent model RFM.
    pairwise_rank : int
        Low-rank dimension K for pairwise Q factorisation.
    lr : float
        Learning rate for all sub-networks.
    gamma : float
        Discount factor.
    tau : float or None
        SPI temperature τ (Eq. 19).  None → Q-learning (Eq. 17).
    t_update : int
        Steps between gradient application (Alg. 5 line 20).
    t_targ_update : int
        Steps between target network updates (Alg. 5 line 24).
    device : str
        Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        type_dim: int = 32,
        hidden_dim: int = 128,
        n_gnn_layers: int = 2,
        pairwise_rank: int = 8,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = None,
        t_update: int = 1,
        t_targ_update: int = 200,
        polyak_tau: float = None,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.type_dim = type_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.t_update = t_update
        self.t_targ_update = t_targ_update
        self.polyak_tau = polyak_tau  # if set, use soft update instead of hard copy
        self.device = torch.device(device)
        self._step_count = 0

        # --- Two separate type inference networks (§4.2, Alg. 5 lines 2-3) ---
        # α_Q: feeds the joint action value model
        self.type_net_q = TypeInferenceModel(
            obs_dim, action_dim, hidden_dim, type_dim
        ).to(self.device)

        # α_q: feeds the agent model
        self.type_net_agent = TypeInferenceModel(
            obs_dim, action_dim, hidden_dim, type_dim
        ).to(self.device)

        # --- Agent model (§4.4): RFM_ζ(θ', c') + MLP_η ---
        self.agent_model = AgentModel(
            type_dim=type_dim,
            lstm_hidden_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            n_gnn_layers=n_gnn_layers,
        ).to(self.device)

        # --- Joint action value model (§4.3): MLP_β(θ) + MLP_δ(θ) ---
        self.q_network = JointActionValueModel(
            type_dim, action_dim, hidden_dim, pairwise_rank
        ).to(self.device)

        # --- Target networks (Alg. 5 line 4) ---
        self.q_network_target = JointActionValueModel(
            type_dim, action_dim, hidden_dim, pairwise_rank
        ).to(self.device)
        self.q_network_target.load_state_dict(self.q_network.state_dict())

        self.type_net_q_target = TypeInferenceModel(
            obs_dim, action_dim, hidden_dim, type_dim
        ).to(self.device)
        self.type_net_q_target.load_state_dict(self.type_net_q.state_dict())

        # --- Optimisers ---
        # Q-value path: α_Q + β + δ (Alg. 5 line 19: dα_Q, dβ, dδ)
        self.optimiser_q = torch.optim.Adam(
            list(self.type_net_q.parameters()) +
            list(self.q_network.parameters()),
            lr=lr,
        )
        # Agent model path: α_q + η + ζ (Alg. 5 line 19: dα_q, dη, dζ)
        self.optimiser_agent = torch.optim.Adam(
            list(self.type_net_agent.parameters()) +
            list(self.agent_model.parameters()),
            lr=lr,
        )

        # --- LSTM hidden states (Alg. 5 lines 5-6) ---
        # Online hidden states carried across timesteps within an episode
        self._hidden_q = None            # (θ_Q, c_Q)
        self._hidden_agent = None        # (θ_q, c_q)
        self._hidden_q_target = None     # (θ_Q^targ, c_Q^targ)

    # ------------------------------------------------------------------
    # Algorithm 2 — QV: Action Value Computation
    # ------------------------------------------------------------------

    def compute_qv(self, B_t, learner_idx=0, hidden_q=None, hidden_agent=None,
                   use_target_q=False):
        """Compute Q̄(H_t, a^i) for all ego actions (Algorithm 2).

        Parameters
        ----------
        B_t : Tensor, shape (N, obs_dim)
            Preprocessed state features for all N current agents.
        learner_idx : int
        hidden_q : optional (h, c) LSTM state for Q-path.
            If None, uses self._hidden_q.
        hidden_agent : optional (h, c) for agent-model-path.
            If None, uses self._hidden_agent.
        use_target_q : bool
            If True, use target Q-network and target type-net-Q (Alg. 5 line 15).

        Returns
        -------
        q_bar : Tensor, shape (action_dim,)
        hidden_q_out : updated Q-path LSTM state
        hidden_agent_out : updated agent-model-path LSTM state
        """
        h_q = hidden_q if hidden_q is not None else self._hidden_q
        h_ag = hidden_agent if hidden_agent is not None else self._hidden_agent

        # Select networks
        type_net_q = self.type_net_q_target if use_target_q else self.type_net_q
        q_net = self.q_network_target if use_target_q else self.q_network

        # Step 5: Q-path type inference
        theta_q, h_q_out = type_net_q(B_t, h_q)

        # Step 6: Agent-model-path type inference
        theta_ag, h_ag_out = self.type_net_agent(B_t, h_ag)

        # Steps 7-8: Agent model uses (θ'_q, c'_q)
        _, c_ag = h_ag_out
        theta_ag_b = theta_ag.unsqueeze(0)
        c_ag_b = c_ag.unsqueeze(0)
        q_probs = self.agent_model.action_probs(theta_ag_b, c_ag_b)

        # Steps 9-10: Joint action value uses only θ'_Q
        theta_q_b = theta_q.unsqueeze(0)
        q_ind, pw_factors = q_net(theta_q_b, learner_idx)

        # Steps 11-12: MARGINALIZE via Eq. 14
        q_bar = self._marginalize(q_ind, pw_factors, q_probs, learner_idx)

        return q_bar.squeeze(0), h_q_out, h_ag_out

    # ------------------------------------------------------------------
    # Algorithm 3 — QJOINT: Joint-Action Value Computation
    # ------------------------------------------------------------------

    def compute_qjoint(self, B_t, joint_actions, learner_idx=0, hidden_q=None):
        """Compute Q(s, a) for observed joint action (Algorithm 3).

        Parameters
        ----------
        B_t : Tensor, shape (N, obs_dim) or (B, N, obs_dim)
        joint_actions : LongTensor, shape (N,) or (B, N)
        learner_idx : int
        hidden_q : optional LSTM state.

        Returns
        -------
        q_value : Tensor
        hidden_q_out : updated LSTM state
        """
        batched = B_t.dim() == 3
        if not batched:
            B_t = B_t.unsqueeze(0)
            joint_actions = joint_actions.unsqueeze(0)

        B, N, _ = B_t.shape
        B_flat = B_t.reshape(B * N, self.obs_dim)

        if hidden_q is not None:
            h, c = hidden_q
            hidden_flat = (h.reshape(B * N, self.hidden_dim),
                           c.reshape(B * N, self.hidden_dim))
        else:
            hidden_flat = None

        theta_flat, (h_new, c_new) = self.type_net_q(B_flat, hidden_flat)
        theta_q = theta_flat.view(B, N, self.type_dim)
        hidden_q_out = (h_new.view(B, N, self.hidden_dim),
                        c_new.view(B, N, self.hidden_dim))

        q_ind, pw_factors = self.q_network(theta_q, learner_idx)
        q_value = self.q_network.compute_joint_q(q_ind, pw_factors, joint_actions)

        if not batched:
            q_value = q_value.squeeze(0)
            hidden_q_out = (hidden_q_out[0].squeeze(0),
                            hidden_q_out[1].squeeze(0))

        return q_value, hidden_q_out

    # ------------------------------------------------------------------
    # Algorithm 4 — PTEAM: Teammate Action Probability
    # ------------------------------------------------------------------

    def compute_pteam(self, B_t, learner_idx=0, hidden_agent=None):
        """Compute teammate action probabilities (Algorithm 4).

        Parameters
        ----------
        B_t : Tensor, shape (N, obs_dim) or (B, N, obs_dim)
        learner_idx : int
        hidden_agent : optional LSTM state.

        Returns
        -------
        log_probs : Tensor, shape (..., N, action_dim) — log action probs
        hidden_agent_out : updated LSTM state
        """
        batched = B_t.dim() == 3
        if not batched:
            B_t = B_t.unsqueeze(0)

        B, N, _ = B_t.shape
        B_flat = B_t.reshape(B * N, self.obs_dim)

        if hidden_agent is not None:
            h, c = hidden_agent
            hidden_flat = (h.reshape(B * N, self.hidden_dim),
                           c.reshape(B * N, self.hidden_dim))
        else:
            hidden_flat = None

        theta_flat, (h_new, c_new) = self.type_net_agent(B_flat, hidden_flat)
        theta_ag = theta_flat.view(B, N, self.type_dim)
        c_ag = c_new.view(B, N, self.hidden_dim)
        hidden_agent_out = (h_new.view(B, N, self.hidden_dim),
                            c_new.view(B, N, self.hidden_dim))

        log_p = self.agent_model.log_probs(theta_ag, c_ag)

        if not batched:
            log_p = log_p.squeeze(0)
            hidden_agent_out = (hidden_agent_out[0].squeeze(0),
                                hidden_agent_out[1].squeeze(0))

        return log_p, hidden_agent_out

    # ------------------------------------------------------------------
    # MARGINALIZE — Eq. 14
    # ------------------------------------------------------------------

    def _marginalize(self, q_ind, pw_factors, q_probs, learner_idx=0):
        """Compute Q̄(H_t, a^i) by marginalising over teammate actions (Eq. 14).

        Parameters
        ----------
        q_ind : Tensor, shape (B, N, A)
        pw_factors : Tensor, shape (B, N, K, A)
        q_probs : Tensor, shape (B, N, A)
        learner_idx : int

        Returns
        -------
        q_bar : Tensor, shape (B, A)
        """
        B, N, A = q_ind.shape

        teammate_mask = torch.ones(N, dtype=torch.bool, device=q_ind.device)
        teammate_mask[learner_idx] = False
        tm_idx = teammate_mask.nonzero(as_tuple=True)[0]

        # Term 1: Q^i_β(a^i | H_t)
        q_bar = q_ind[:, learner_idx, :]  # (B, A)

        if len(tm_idx) == 0:
            return q_bar

        q_tm = q_ind[:, tm_idx, :]                          # (B, N_tm, A)
        p_tm = q_probs[:, tm_idx, :]                         # (B, N_tm, A)
        factor_i = pw_factors[:, learner_idx, :, :]          # (B, K, A)
        factor_tm = pw_factors[:, tm_idx, :, :]              # (B, N_tm, K, A)

        # Term 2a: Σ_j Σ_{a^j} Q^j_β(a^j) q(a^j)
        singular_contrib = (q_tm * p_tm).sum(dim=(1, 2))     # (B,)
        q_bar = q_bar + singular_contrib.unsqueeze(1)

        # Term 2b: Σ_j Σ_{a^j} Q^{i,j}_δ(a^i, a^j) q(a^j)
        weighted_factor_tm = (factor_tm * p_tm.unsqueeze(2)).sum(dim=-1)  # (B, N_tm, K)
        sum_wf_tm = weighted_factor_tm.sum(dim=1)            # (B, K)
        ego_tm_pairwise = (factor_i * sum_wf_tm.unsqueeze(-1)).sum(dim=1)  # (B, A)
        q_bar = q_bar + ego_tm_pairwise

        # Term 3: Σ_{j,k; j≠k≠i} Q^{j,k} q(a^j) q(a^k)
        sq_total = (sum_wf_tm ** 2).sum(dim=1)               # (B,)
        sq_self = (weighted_factor_tm ** 2).sum(dim=(1, 2))   # (B,)
        q_bar = q_bar + (sq_total - sq_self).unsqueeze(1)

        return q_bar

    # ------------------------------------------------------------------
    # Action selection (§4.5, Alg. 5 lines 11-12)
    # ------------------------------------------------------------------

    def act(self, B_t, learner_idx=0, epsilon=0.0):
        """Select an action via ε-greedy over Q̄.

        Does NOT update internal hidden states — train_step_online is the sole
        updater of hidden states (Alg. 5 line 27).  This ensures QJOINT and
        QV in the training step both start from the same h_{t-1}, matching
        Algorithm 5 lines 11 and 14 which both receive h_Q (pre-step hidden).

        Parameters
        ----------
        B_t : np.ndarray, shape (N, obs_dim)
        learner_idx : int
        epsilon : float

        Returns
        -------
        action : int
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            B_t_tensor = torch.FloatTensor(B_t).to(self.device)
            q_bar, _, _ = self.compute_qv(B_t_tensor, learner_idx)
            return int(q_bar.argmax().item())

    # ------------------------------------------------------------------
    # Algorithm 5 — Online training step (synchronous)
    # ------------------------------------------------------------------

    def train_step_online(self, B_t, joint_actions, reward, B_t_next, done,
                          learner_idx=0, teammate_indices=None):
        """One step of Algorithm 5's synchronous training loop.

        Called once per environment timestep.  Accumulates gradients and
        applies them every t_update steps.

        Algorithm 5 lines 14-27:
          14. Q(H, a) ← QJOINT(s, a, h_Q)
          15. Q̄'(H', a^i) ← QV(s', target, h_Q^targ, h'_q)
          16. y ← r + γ max/SPI Q̄'
          17. q(a^{-i}) ← PTEAM(s, a, h_q)
          18. Compute L_{β,δ} and L_{ζ,η}
          19. Accumulate gradients
          20-22. Apply every t_update
          24-25. Update target every t_targ_update
          27. Carry hidden states forward

        Parameters
        ----------
        B_t : np.ndarray or Tensor, shape (N, obs_dim)
        joint_actions : array-like, shape (N,) — all agent actions
        reward : float — learner's reward
        B_t_next : np.ndarray or Tensor, shape (N', obs_dim)
        done : bool
        learner_idx : int
        teammate_indices : list[int] or None (auto-computed if None)

        Returns
        -------
        metrics : dict with "q_loss", "agent_model_loss" (or None if not updated)
        """
        B_t_t = torch.FloatTensor(np.asarray(B_t)).to(self.device)
        actions_t = torch.LongTensor(np.asarray(joint_actions)).to(self.device)
        B_next_t = torch.FloatTensor(np.asarray(B_t_next)).to(self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(float(done), device=self.device)

        N = B_t_t.shape[0]
        if teammate_indices is None:
            teammate_indices = [j for j in range(N) if j != learner_idx]

        # ---- Line 14: QJOINT(s, a, h_Q) ----
        q_current, h_q_new = self.compute_qjoint(
            B_t_t, actions_t, learner_idx, self._hidden_q
        )

        # ---- Line 15: QV(s', target params, h_Q^targ, h'_q) ----
        # First advance target hidden through current obs s_t (same as online),
        # then use the advanced hidden to compute Q' for s_{t+1}.
        with torch.no_grad():
            _, h_q_targ_at_t, h_ag_at_t = self.compute_qv(
                B_t_t, learner_idx,
                hidden_q=self._hidden_q_target,
                hidden_agent=self._hidden_agent,
                use_target_q=True,
            )
            q_bar_next, h_q_targ_new, _ = self.compute_qv(
                B_next_t, learner_idx,
                hidden_q=h_q_targ_at_t,
                hidden_agent=h_ag_at_t,
                use_target_q=True,
            )

        # ---- Line 16: Compute target y ----
        with torch.no_grad():
            if self.tau is None:
                # GPL-Q (Eq. 17)
                y = reward_t + self.gamma * (1 - done_t) * q_bar_next.max()
            else:
                # GPL-SPI (Eq. 18-19)
                p_spi = F.softmax(q_bar_next / self.tau, dim=-1)
                y = reward_t + self.gamma * (1 - done_t) * (p_spi * q_bar_next).sum()

        # ---- Line 18: Q-value loss (Eq. 16) ----
        q_loss = 0.5 * (q_current - y) ** 2

        # ---- Line 17: PTEAM(s, a, h_q) ----
        log_p, h_ag_new = self.compute_pteam(
            B_t_t, learner_idx, self._hidden_agent
        )

        # ---- Line 18: Agent model loss (Eq. 15) ----
        tm_actions = actions_t[teammate_indices]
        tm_log_p = log_p[teammate_indices, :]
        agent_loss = -tm_log_p[
            torch.arange(len(teammate_indices), device=self.device),
            tm_actions,
        ].sum()

        # ---- Line 19: Accumulate gradients ----
        # Scale losses for accumulation (will be applied every t_update)
        (q_loss / self.t_update).backward()
        (agent_loss / self.t_update).backward()

        self._step_count += 1
        metrics = None

        # ---- Lines 20-22: Apply gradients every t_update ----
        if self._step_count % self.t_update == 0:
            self.optimiser_q.step()
            self.optimiser_agent.step()
            self.optimiser_q.zero_grad()
            self.optimiser_agent.zero_grad()
            metrics = {
                "q_loss": q_loss.item(),
                "agent_model_loss": agent_loss.item(),
            }

        # ---- Lines 24-25: Target network update every t_targ_update ----
        if self._step_count % self.t_targ_update == 0:
            if self.polyak_tau is not None:
                # Soft update (Polyak averaging): φ' ← (1-α)φ' + αφ
                self._soft_update(self.q_network, self.q_network_target, self.polyak_tau)
                self._soft_update(self.type_net_q, self.type_net_q_target, self.polyak_tau)
            else:
                # Hard copy
                self.q_network_target.load_state_dict(self.q_network.state_dict())
                self.type_net_q_target.load_state_dict(self.type_net_q.state_dict())

        # ---- Line 27: Carry hidden states forward ----
        # h_q_new: online Q-path after seeing B_t (one step forward from h_{t-1})
        # h_ag_new: agent-model path after seeing B_t
        # h_q_targ_at_t: TARGET Q-path after seeing B_t (advance target by one obs)
        with torch.no_grad():
            self._hidden_q = (h_q_new[0].detach(), h_q_new[1].detach())
            self._hidden_agent = (h_ag_new[0].detach(), h_ag_new[1].detach())
            self._hidden_q_target = (h_q_targ_at_t[0].detach(), h_q_targ_at_t[1].detach())

        # Reset hidden states on episode end
        if done:
            self._hidden_q = None
            self._hidden_agent = None
            self._hidden_q_target = None

        return metrics

    # ------------------------------------------------------------------
    # Replay-buffer training (experience replay variant)
    # ------------------------------------------------------------------

    def update(self, batch):
        """One gradient step on a batch from a replay buffer.

        Replay-buffer variant of Algorithm 5.  Transitions are sampled
        independently so hidden states are not carried across.

        Parameters
        ----------
        batch : dict with keys:
            "B_t"              : Tensor (B, N, obs_dim)
            "actions"          : LongTensor (B, N)
            "rewards"          : Tensor (B,)
            "B_t_next"         : Tensor (B, N', obs_dim)
            "dones"            : Tensor (B,)
            "learner_idx"      : int
            "teammate_indices" : list[int]

        Returns
        -------
        metrics : dict with "agent_model_loss", "q_loss"
        """
        B_t = batch["B_t"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        B_t_next = batch["B_t_next"].to(self.device)
        dones = batch["dones"].to(self.device)
        learner_idx = batch["learner_idx"]
        tm_idx = batch["teammate_indices"]

        B, N, _ = B_t.shape

        # ---- Agent model loss (Eq. 15) ----
        log_p, _ = self.compute_pteam(B_t, learner_idx, hidden_agent=None)
        tm_actions = actions[:, tm_idx]
        tm_log_p = log_p[:, tm_idx, :]
        nll = F.nll_loss(
            tm_log_p.reshape(-1, self.action_dim),
            tm_actions.reshape(-1),
        )

        self.optimiser_agent.zero_grad()
        nll.backward()
        self.optimiser_agent.step()

        # ---- Joint action value loss (Eq. 16) ----
        q_current, _ = self.compute_qjoint(
            B_t, actions, learner_idx, hidden_q=None
        )

        with torch.no_grad():
            N_next = B_t_next.shape[1]

            # Target Q-path type inference
            B_next_flat = B_t_next.reshape(-1, self.obs_dim)
            theta_next_flat, (h_next, c_next) = self.type_net_q_target(
                B_next_flat, None
            )
            theta_q_next = theta_next_flat.view(B, N_next, self.type_dim)

            # Agent model probs for next state
            theta_ag_next_flat, (_, c_ag_next) = self.type_net_agent(
                B_next_flat, None
            )
            theta_ag_next = theta_ag_next_flat.view(B, N_next, self.type_dim)
            c_ag_next_b = c_ag_next.view(B, N_next, self.hidden_dim)
            q_probs_next = self.agent_model.action_probs(theta_ag_next, c_ag_next_b)

            # Target Q-values
            q_ind_next, pw_next = self.q_network_target(theta_q_next, learner_idx)
            q_bar_next = self._marginalize(
                q_ind_next, pw_next, q_probs_next, learner_idx
            )

            if self.tau is None:
                y = rewards + self.gamma * (1 - dones) * q_bar_next.max(dim=-1).values
            else:
                p_spi = F.softmax(q_bar_next / self.tau, dim=-1)
                y = rewards + self.gamma * (1 - dones) * (p_spi * q_bar_next).sum(dim=-1)

        td_loss = F.mse_loss(q_current, y)

        self.optimiser_q.zero_grad()
        td_loss.backward()
        self.optimiser_q.step()

        # Target network update
        self._step_count += 1
        if self._step_count % self.t_targ_update == 0:
            if self.polyak_tau is not None:
                self._soft_update(self.q_network, self.q_network_target, self.polyak_tau)
                self._soft_update(self.type_net_q, self.type_net_q_target, self.polyak_tau)
            else:
                self.q_network_target.load_state_dict(self.q_network.state_dict())
                self.type_net_q_target.load_state_dict(self.type_net_q.state_dict())

        return {
            "agent_model_loss": nll.item(),
            "q_loss": td_loss.item(),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def advance_hidden(self, B_t):
        """Advance LSTM hidden states by one timestep without training.

        Used during evaluation: act() no longer updates hidden states
        (train_step_online is the sole updater during training), so evaluation
        loops must call this after each step to keep the LSTM context current.

        Parameters
        ----------
        B_t : np.ndarray, shape (N, obs_dim)
        """
        with torch.no_grad():
            B_t_tensor = torch.FloatTensor(np.asarray(B_t)).to(self.device)
            _, self._hidden_q, self._hidden_agent = self.compute_qv(B_t_tensor)
            # Advance target hidden the same way using target type net
            _, h_q_targ, _ = self.compute_qv(
                B_t_tensor,
                hidden_q=self._hidden_q_target,
                hidden_agent=self._hidden_agent,
                use_target_q=True,
            )
            self._hidden_q_target = (h_q_targ[0].detach(), h_q_targ[1].detach())

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float):
        """Polyak averaging: φ' ← (1-τ)φ' + τφ."""
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.mul_(1.0 - tau).add_(src_p.data, alpha=tau)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save all model weights and optimiser state."""
        torch.save({
            "type_net_q": self.type_net_q.state_dict(),
            "type_net_agent": self.type_net_agent.state_dict(),
            "agent_model": self.agent_model.state_dict(),
            "q_network": self.q_network.state_dict(),
            "q_network_target": self.q_network_target.state_dict(),
            "type_net_q_target": self.type_net_q_target.state_dict(),
            "optimiser_q": self.optimiser_q.state_dict(),
            "optimiser_agent": self.optimiser_agent.state_dict(),
            "step_count": self._step_count,
        }, path)

    def load(self, path: str):
        """Load model weights and optimiser state."""
        try:
            ckpt = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.type_net_q.load_state_dict(ckpt["type_net_q"])
        self.type_net_agent.load_state_dict(ckpt["type_net_agent"])
        self.agent_model.load_state_dict(ckpt["agent_model"])
        self.q_network.load_state_dict(ckpt["q_network"])
        self.q_network_target.load_state_dict(ckpt["q_network_target"])
        self.type_net_q_target.load_state_dict(ckpt["type_net_q_target"])
        self.optimiser_q.load_state_dict(ckpt["optimiser_q"])
        self.optimiser_agent.load_state_dict(ckpt["optimiser_agent"])
        self._step_count = ckpt["step_count"]

    def reset(self):
        """Reset episode-level LSTM hidden states (Alg. 5 lines 5-6)."""
        self._hidden_q = None
        self._hidden_agent = None
        self._hidden_q_target = None
