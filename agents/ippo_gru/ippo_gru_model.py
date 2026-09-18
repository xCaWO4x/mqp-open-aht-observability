"""
Recurrent Actor-Critic Model for Independent PPO (IPPO-GRU).

Architecture:
    Observation -> MLP Encoder (128) -> GRU (128) -> Actor Head & Critic Head.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical


class IPPORecurrentActorCritic(nn.Module):
    """Recurrent Actor-Critic network with GRU core.

    Parameters
    ----------
    obs_dim : int
        Dimension of the local ego observation.
    action_dim : int
        Number of discrete actions.
    encoder_hidden : int, default 128
        Hidden dimension of MLP observation encoder.
    gru_hidden : int, default 128
        Hidden state dimension of GRU.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        encoder_hidden: int = 128,
        gru_hidden: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gru_hidden = gru_hidden

        # Observation MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, gru_hidden),
            nn.ReLU(),
        )

        # Recurrent GRU core
        self.gru = nn.GRU(gru_hidden, gru_hidden, batch_first=True)

        # Actor head: predicts action logits
        self.actor = nn.Linear(gru_hidden, action_dim)

        # Decentralized Critic head: predicts scalar state value V(o)
        self.critic = nn.Linear(gru_hidden, 1)

    def get_initial_hidden(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Create zero-initialized GRU hidden state of shape (1, batch_size, gru_hidden)."""
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-timestep forward pass for rollout collection.

        Parameters
        ----------
        obs : Tensor of shape (B, obs_dim)
        hidden : Tensor of shape (1, B, gru_hidden)
        action_mask : Optional BoolTensor of shape (B, action_dim)

        Returns
        -------
        action : LongTensor of shape (B,)
        log_prob : Tensor of shape (B,)
        value : Tensor of shape (B,)
        next_hidden : Tensor of shape (1, B, gru_hidden)
        """
        B = obs.shape[0]
        # Encode observation
        enc = self.encoder(obs).unsqueeze(1)  # (B, 1, gru_hidden)

        # GRU step
        gru_out, next_hidden = self.gru(enc, hidden)  # (B, 1, gru_hidden), (1, B, gru_hidden)
        h = gru_out.squeeze(1)  # (B, gru_hidden)

        # Policy distribution
        logits = self.actor(h)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        dist = Categorical(logits=logits)

        # Sample action and value
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(h).squeeze(-1)

        return action, log_prob, value, next_hidden

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        hidden_init: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence evaluation for PPO update.

        Parameters
        ----------
        obs_seq : Tensor of shape (B, T, obs_dim)
        hidden_init : Tensor of shape (1, B, gru_hidden)
        actions : LongTensor of shape (B, T)
        action_masks : Optional BoolTensor of shape (B, T, action_dim)

        Returns
        -------
        log_probs : Tensor of shape (B, T)
        entropy : Tensor of shape (B, T)
        values : Tensor of shape (B, T)
        """
        B, T, _ = obs_seq.shape
        # Encode all observations in the sequence
        obs_flat = obs_seq.reshape(B * T, -1)
        enc_flat = self.encoder(obs_flat)
        enc_seq = enc_flat.reshape(B, T, self.gru_hidden)

        # Pass sequence through GRU
        gru_out, _ = self.gru(enc_seq, hidden_init)  # (B, T, gru_hidden)
        h_flat = gru_out.reshape(B * T, self.gru_hidden)

        # Compute logits and distributions
        logits = self.actor(h_flat)
        if action_masks is not None:
            mask_flat = action_masks.reshape(B * T, -1)
            logits = logits.masked_fill(~mask_flat, -1e9)
        dist = Categorical(logits=logits)

        # Evaluate actions
        actions_flat = actions.reshape(B * T)
        log_probs = dist.log_prob(actions_flat).reshape(B, T)
        entropy = dist.entropy().reshape(B, T)
        values = self.critic(h_flat).squeeze(-1).reshape(B, T)

        return log_probs, entropy, values
