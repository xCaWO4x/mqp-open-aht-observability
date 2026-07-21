"""
Coordination-graph-based joint action value model.

Implements GPL Section 4.3 (Rahman et al. 2023, arXiv:2210.05448).

The joint action value is factorised via a fully connected coordination
graph (CG) into singular and pairwise utility terms:

    Q_{π^i}(H_t, a_t) = Σ_j Q^j(a^j | H_t) + Σ_{j,k; j≠k} Q^{j,k}(a^j, a^k | H_t)   (Eq. 8)

Two MLPs compute these terms from type vectors:

    MLP_β (individual):   Q^j(a^j | H_t) = MLP_β(θ^j, θ^i)(a^j)                         (Eq. 9)
    MLP_δ (pairwise):     Q^{j,k}(a^j, a^k | H_t) = MLP_δ(θ^j, θ^i)^T MLP_δ(θ^k, θ^i) (Eq. 10)

Key design choices:
  - Both MLPs take (θ^j, θ^i) — agent j's type concatenated with learner's type.
  - MLP_β outputs |A|-dim vector (individual Q per action of agent j).
  - MLP_δ outputs K × |A| matrix (K ≪ |A|, low-rank factorisation of pairwise Q).
  - Same parameters shared across all agents / pairs.
  - Trained via TD loss (Eq. 16) with Q-learning or SPI targets (Eq. 17/18).
"""

import torch
import torch.nn as nn


class JointActionValueModel(nn.Module):
    """Coordination graph joint Q-value network with singular + pairwise terms.

    Parameters
    ----------
    type_dim : int
        Type embedding dimension (θ^j size).
    action_dim : int
        Number of discrete actions |A|.
    hidden_dim : int
        MLP hidden layer width.
    pairwise_rank : int
        Low-rank dimension K for MLP_δ output (K ≪ |A|).  Each agent's
        MLP_δ output is a (K, |A|) matrix; pairwise Q is their bilinear product.
    """

    def __init__(
        self,
        type_dim: int = 32,
        action_dim: int = 5,
        hidden_dim: int = 128,
        pairwise_rank: int = 8,
    ):
        super().__init__()
        self.type_dim = type_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.pairwise_rank = pairwise_rank

        # MLP_β: (θ^j, θ^i) → Q^j(a | H_t) of shape (|A|,)        (Eq. 9)
        # Paper Figure 7(b): FC(70)→ReLU→FC(60)→FC(|A|)
        self.mlp_beta = nn.Sequential(
            nn.Linear(type_dim * 2, 70),
            nn.ReLU(),
            nn.Linear(70, 60),
            nn.Linear(60, action_dim),
        )

        # MLP_δ: (θ^j, θ^i) → matrix of shape (K, |A|)             (Eq. 10)
        # Paper Figure 7(c): FC(70)→ReLU→FC(60)→FC(K×|A|)
        # Pairwise Q is computed as MLP_δ(j)^T MLP_δ(k) → (|A|, |A|).
        self.mlp_delta = nn.Sequential(
            nn.Linear(type_dim * 2, 70),
            nn.ReLU(),
            nn.Linear(70, 60),
            nn.Linear(60, pairwise_rank * action_dim),
        )

    def individual_q(self, theta_j, theta_i):
        """Compute individual utility Q^j(a^j | H_t) for each agent j (Eq. 9).

        Parameters
        ----------
        theta_j : Tensor, shape (B, N, type_dim)
            Type vectors for N agents.
        theta_i : Tensor, shape (B, type_dim)
            Learner's type vector.

        Returns
        -------
        q_individual : Tensor, shape (B, N, action_dim)
            Q^j(a^j | H_t) for each agent j and each of their actions.
        """
        B, N, _ = theta_j.shape
        # Expand learner type to match: (B, N, type_dim)
        theta_i_exp = theta_i.unsqueeze(1).expand(B, N, self.type_dim)
        # Concatenate (θ^j, θ^i)
        inp = torch.cat([theta_j, theta_i_exp], dim=-1)  # (B, N, 2*type_dim)
        return self.mlp_beta(inp)  # (B, N, action_dim)

    def pairwise_q(self, theta_j, theta_i):
        """Compute pairwise utility factors for all agent pairs (Eq. 10).

        Returns the low-rank factors rather than the full (|A| × |A|) matrix.

        Parameters
        ----------
        theta_j : Tensor, shape (B, N, type_dim)
            Type vectors for N agents.
        theta_i : Tensor, shape (B, type_dim)
            Learner's type vector.

        Returns
        -------
        factors : Tensor, shape (B, N, pairwise_rank, action_dim)
            MLP_δ output for each agent, reshaped.  Pairwise Q between agents
            j and k is: factors[j]^T @ factors[k] → (action_dim, action_dim).
        """
        B, N, _ = theta_j.shape
        theta_i_exp = theta_i.unsqueeze(1).expand(B, N, self.type_dim)
        inp = torch.cat([theta_j, theta_i_exp], dim=-1)  # (B, N, 2*type_dim)
        raw = self.mlp_delta(inp)  # (B, N, K * action_dim)
        return raw.view(B, N, self.pairwise_rank, self.action_dim)

    def forward(self, theta_all, learner_idx=0):
        """Compute the full joint action value Q(H_t, a_t) via Eq. 8.

        Parameters
        ----------
        theta_all : Tensor, shape (B, N, type_dim)
            Type vectors for ALL agents (learner + teammates).
        learner_idx : int
            Index of the learner within the N agents.

        Returns
        -------
        q_individual : Tensor, shape (B, N, action_dim)
            Per-agent individual Q^j(a^j | H_t).
        pairwise_factors : Tensor, shape (B, N, pairwise_rank, action_dim)
            Low-rank pairwise factors for computing Q^{j,k}.
        """
        B, N, _ = theta_all.shape
        theta_i = theta_all[:, learner_idx, :]  # (B, type_dim)

        q_ind = self.individual_q(theta_all, theta_i)        # (B, N, action_dim)
        pw_factors = self.pairwise_q(theta_all, theta_i)     # (B, N, K, action_dim)

        return q_ind, pw_factors

    def compute_joint_q(self, q_ind, pw_factors, joint_actions):
        """Evaluate Q(H_t, a_t) for a specific joint action (Eq. 8).

        Parameters
        ----------
        q_ind : Tensor, shape (B, N, action_dim)
        pw_factors : Tensor, shape (B, N, pairwise_rank, action_dim)
        joint_actions : LongTensor, shape (B, N)
            Action index for each agent.

        Returns
        -------
        q_value : Tensor, shape (B,)
        """
        B, N, _ = q_ind.shape

        # Singular terms: Σ_j Q^j(a^j)
        # Gather the Q-value for each agent's chosen action
        actions_exp = joint_actions.unsqueeze(-1)  # (B, N, 1)
        q_singular = q_ind.gather(-1, actions_exp).squeeze(-1).sum(dim=1)  # (B,)

        # Pairwise terms: Σ_{j≠k} Q^{j,k}(a^j, a^k) via low-rank (Eq. 10)
        # factors_j[a^j] dot factors_k[a^k]
        # Gather factors at chosen actions: (B, N, K)
        actions_exp_k = joint_actions.unsqueeze(-1).unsqueeze(-1).expand(
            B, N, self.pairwise_rank, 1
        )
        selected_factors = pw_factors.gather(-1, actions_exp_k).squeeze(-1)  # (B, N, K)

        # Q^{j,k} = factors_j^T @ factors_k (dot product over K dimension)
        # For all pairs j≠k, sum factor_j^T factor_k
        # = (Σ_j factor_j)^T (Σ_k factor_k) - Σ_j factor_j^T factor_j
        sum_factors = selected_factors.sum(dim=1)  # (B, K)
        q_pairwise_total = (sum_factors ** 2).sum(dim=1)  # (B,)
        q_self = (selected_factors ** 2).sum(dim=(1, 2))   # (B,)
        q_pairwise = q_pairwise_total - q_self              # (B,)

        return q_singular + q_pairwise
