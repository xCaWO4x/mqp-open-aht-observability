"""
LSTM-based type inference module.

Implements GPL Section 4.2 (Rahman et al. 2023, arXiv:2210.05448).

The type inference model represents teammate types as continuous vectors
computed by an LSTM.  The LSTM hidden state θ_t serves as the type vector:

    c_t, θ_t = LSTM_α(B_t, c_{t-1}, θ_{t-1})          (Eq. 7)

Key design choices from the paper:
  - Types are continuous vectors, NOT discrete labels — enables generalisation
    to previously unseen teammates.
  - Trained without ground-truth types; gradients flow back from the agent
    model loss (Eq. 15) and joint action value loss (Eq. 16).
  - In practice, TWO separate LSTM copies are used: one feeds the Joint Action
    Value model, one feeds the Agent Model, to prevent loss interference.
  - Between timesteps, type vectors of departed agents are removed and new
    vectors are added for newly-arrived agents (openness handling).

B_t construction and hidden-state management (Appendix C.1) are handled
by preprocess_lbf() in envs/env_utils.py.
"""

import torch
import torch.nn as nn


class TypeInferenceModel(nn.Module):
    """LSTM that maps state feature sequences to per-agent type embeddings.

    The LSTM hidden state IS the type vector θ (Eq. 7).  The cell state c
    provides long-term memory.  A projection layer maps the raw LSTM hidden
    state to the desired type_dim if it differs from hidden_dim.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the preprocessed input batch B_t per agent.
    action_dim : int
        Number of discrete actions (not used in the basic version, but
        reserved for input variants that concatenate one-hot actions).
    hidden_dim : int
        LSTM hidden / cell state size.
    type_dim : int
        Output type embedding dimensionality.  If equal to hidden_dim, the
        projection layer is an identity.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        type_dim: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.type_dim = type_dim

        # Input projection: B_t → LSTM input
        # Paper Figure 7(a): FC(100)→ReLU→FC(100)→LSTM(100)→ReLU
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Core LSTM (Eq. 7): processes one timestep at a time to allow
        # incremental updates and agent set changes between steps.
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        # Project LSTM hidden state → type embedding θ
        if type_dim == hidden_dim:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(hidden_dim, type_dim)

    def forward(self, B_t, hidden=None):
        """Process one timestep of state features for a batch of agents.

        Implements Eq. 7: c_t, θ_t = LSTM_α(B_t, c_{t-1}, θ_{t-1})

        Parameters
        ----------
        B_t : Tensor, shape (N, obs_dim)
            Preprocessed state features for N agents at current timestep.
            N can vary between calls (openness).
        hidden : tuple of (h, c) each shape (N, hidden_dim), or None
            Previous LSTM state.  None initialises to zeros.
            When the agent set changes, the caller must handle adding/removing
            rows to match the new N.

        Returns
        -------
        type_emb : Tensor, shape (N, type_dim)
            Type vectors θ_t for each agent.
        hidden : tuple of (h, c) each shape (N, hidden_dim)
            Updated LSTM state for carry-over to next timestep.
        """
        N = B_t.shape[0]
        device = B_t.device

        # Initialise hidden state if not provided
        if hidden is None:
            h = torch.zeros(N, self.hidden_dim, device=device)
            c = torch.zeros(N, self.hidden_dim, device=device)
        else:
            h, c = hidden

        # Input projection: FC→ReLU→FC
        x = self.input_proj(B_t)  # (N, hidden_dim)

        # LSTM update (Eq. 7)
        h_new, c_new = self.lstm(x, (h, c))

        # ReLU after LSTM (paper Figure 7(a))
        h_activated = torch.relu(h_new)

        # Project hidden state to type embedding
        type_emb = self.output_proj(h_activated)  # (N, type_dim)

        return type_emb, (h_new, c_new)

    def forward_sequence(self, B_seq, hidden=None):
        """Process a full sequence of timesteps (convenience method).

        Parameters
        ----------
        B_seq : Tensor, shape (N, T, obs_dim)
            State features for N agents over T timesteps.
        hidden : optional initial LSTM state.

        Returns
        -------
        type_embs : Tensor, shape (N, T, type_dim)
            Type vectors at each timestep.
        hidden : final LSTM state.
        """
        N, T, _ = B_seq.shape
        all_embs = []
        for t in range(T):
            emb, hidden = self.forward(B_seq[:, t, :], hidden)
            all_embs.append(emb)
        return torch.stack(all_embs, dim=1), hidden

    def reset_hidden(self, n_agents, device="cpu"):
        """Create fresh zero-initialised hidden state for n_agents.

        Parameters
        ----------
        n_agents : int
        device : str or torch.device

        Returns
        -------
        hidden : tuple of (h, c) each shape (n_agents, hidden_dim)
        """
        h = torch.zeros(n_agents, self.hidden_dim, device=device)
        c = torch.zeros(n_agents, self.hidden_dim, device=device)
        return (h, c)
