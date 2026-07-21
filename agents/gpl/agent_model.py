"""
RFM-based agent model for teammate action prediction.

Implements GPL Section 4.4 / Appendix A Algorithm 4 (Rahman et al. 2023,
arXiv:2210.05448).

Architecture (Algorithm 4 — PTEAM):
  1. PREPROCESS(s, h_{t-1,q}) → B, θ_q, c_q
  2. θ'_q, c'_q ← LSTM_{α_q}(B, θ_q, c_q)
  3. ∀j, n̄_j ← (RFM_ζ(θ'_q, c'_q))_j          (Eq. 11)
  4. ∀j, q(·|s) ← Softmax(MLP_η(n̄_j))          (Eq. 12)
  5. π^{-i} ≈ Π_{j∈-i} q(a^j | s)               (Eq. 13)

Key detail from Appendix A: RFM_ζ receives BOTH the LSTM hidden state θ'_q
AND the cell state c'_q as node features (Algorithm 2 line 7, Algorithm 4
line 5).  This is in contrast to the joint action value model, which only
uses the hidden state θ.

Trained via negative log-likelihood of observed teammate actions (Eq. 15).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentModel(nn.Module):
    """GNN (RFM) + MLP that predicts per-agent action distributions.

    Per Appendix A, RFM_ζ takes (θ'_q, c'_q) — concatenation of LSTM
    hidden state and cell state — as node features.

    Parameters
    ----------
    type_dim : int
        Dimensionality of the type embedding θ (LSTM hidden state projection).
    lstm_hidden_dim : int
        LSTM hidden/cell state size.  The RFM node input is
        (θ, c) = type_dim + lstm_hidden_dim.  If type_dim == lstm_hidden_dim
        (no projection in TypeInferenceModel), node input is 2 * type_dim.
    hidden_dim : int
        Width of GNN message and MLP layers.
    action_dim : int
        Number of discrete actions per agent.
    n_gnn_layers : int
        Number of message-passing rounds in GNN_ζ.
    """

    def __init__(
        self,
        type_dim: int,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 128,
        action_dim: int = 5,
        n_gnn_layers: int = 2,
    ):
        super().__init__()
        self.type_dim = type_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.n_gnn_layers = n_gnn_layers

        # RFM node input: concatenation of θ (type_dim) and c (lstm_hidden_dim)
        node_input_dim = type_dim + lstm_hidden_dim

        # --- GNN_ζ (Eq. 11): message-passing on fully-connected agent graph ---
        # Paper Figure 7(d,e): edge FC(30)→ReLU→FC(70), node FC(30)→ReLU→FC(70)
        self.gnn_hidden = 70  # GNN internal width (paper)

        # Node encoder: (θ, c) → gnn_hidden
        self.node_encoder = nn.Linear(node_input_dim, self.gnn_hidden)

        # Message-passing layers.  Each round:
        #   message_{j→k} = MLP_msg(concat(h_j, h_k))
        #   h_k' = MLP_update(h_k + mean(messages to k))
        self.msg_fns = nn.ModuleList()
        self.update_fns = nn.ModuleList()
        for _ in range(n_gnn_layers):
            # Paper Figure 7(d): edge FC(30)→ReLU→FC(70)
            self.msg_fns.append(nn.Sequential(
                nn.Linear(self.gnn_hidden * 2, 30),
                nn.ReLU(),
                nn.Linear(30, self.gnn_hidden),
            ))
            # Paper Figure 7(e): node FC(30)→ReLU→FC(70)
            self.update_fns.append(nn.Sequential(
                nn.Linear(self.gnn_hidden, 30),
                nn.ReLU(),
                nn.Linear(30, self.gnn_hidden),
            ))

        # --- MLP_η (Eq. 12): per-agent embedding → action logits ---
        # Paper Figure 7(f): FC(20)→ReLU→FC(|A|)
        self.action_head = nn.Sequential(
            nn.Linear(self.gnn_hidden, 20),
            nn.ReLU(),
            nn.Linear(20, action_dim),
        )

    def forward(self, theta, cell_state):
        """Compute action logits for each agent via RFM_ζ(θ', c') + MLP_η.

        Corresponds to Algorithm 4 lines 5-6 / Algorithm 2 lines 7-8.

        Parameters
        ----------
        theta : Tensor, shape (B, N, type_dim)
            Type vectors θ'_q (LSTM hidden state projections) for all N agents.
        cell_state : Tensor, shape (B, N, lstm_hidden_dim)
            LSTM cell states c'_q for all N agents.

        Returns
        -------
        action_logits : Tensor, shape (B, N, action_dim)
            Unnormalised log-probabilities per agent per action.
        """
        B, N, _ = theta.shape

        # Concatenate (θ, c) as node features — Appendix A detail
        node_features = torch.cat([theta, cell_state], dim=-1)  # (B, N, type_dim + lstm_hidden)

        # Encode node features
        h = torch.relu(self.node_encoder(node_features))  # (B, N, hidden)

        # Message passing (Eq. 11): RFM_ζ
        for msg_fn, update_fn in zip(self.msg_fns, self.update_fns):
            # Pairwise messages on fully-connected graph
            h_i = h.unsqueeze(2).expand(B, N, N, self.gnn_hidden)
            h_j = h.unsqueeze(1).expand(B, N, N, self.gnn_hidden)

            # Message from j to i: msg_fn(concat(h_j, h_i))
            messages = msg_fn(torch.cat([h_j, h_i], dim=-1))  # (B, N, N, hidden)

            # Mask self-messages
            mask = ~torch.eye(N, dtype=torch.bool, device=h.device)
            mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
            messages = messages * mask

            # Aggregate: mean of incoming messages (excluding self)
            n_neighbors = max(N - 1, 1)
            agg = messages.sum(dim=2) / n_neighbors  # (B, N, hidden)

            # Update node embeddings
            h = update_fn(h + agg)  # (B, N, hidden)

        # n̄_j = h after message passing (Eq. 11)
        # Action logits (Eq. 12): q(a^j | s) = Softmax(MLP_η(n̄_j))
        logits = self.action_head(h)  # (B, N, action_dim)
        return logits

    def action_probs(self, theta, cell_state):
        """Softmax action distribution q(a^j | s) per agent (Eq. 12).

        Parameters
        ----------
        theta : Tensor, shape (B, N, type_dim)
        cell_state : Tensor, shape (B, N, lstm_hidden_dim)

        Returns
        -------
        probs : Tensor, shape (B, N, action_dim)
        """
        return F.softmax(self.forward(theta, cell_state), dim=-1)

    def log_probs(self, theta, cell_state):
        """Log softmax action distribution per agent.

        Used for computing the agent model loss (Eq. 15).

        Parameters
        ----------
        theta : Tensor, shape (B, N, type_dim)
        cell_state : Tensor, shape (B, N, lstm_hidden_dim)

        Returns
        -------
        log_probs : Tensor, shape (B, N, action_dim)
        """
        return F.log_softmax(self.forward(theta, cell_state), dim=-1)
