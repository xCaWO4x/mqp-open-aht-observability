"""
Entity-Transformer Q-Learning Agent.

Matches GPL's learning rate, target network updates, optimizer, and discount factor
while replacing relational GNN message passing with multi-head self-attention.
"""

import os
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.transformer_q.transformer_q_model import EntityTransformerQNetwork
from envs.entity_adapter import EntityBatch


class TransformerQAgent:
    """Q-learning agent with Entity-Transformer representation.

    Parameters
    ----------
    action_dim : int
        Number of discrete actions.
    feat_dim : int, default 3
        Dimensionality of raw entity features.
    d_model : int, default 64
        Transformer embedding dimension.
    nhead : int, default 4
        Number of self-attention heads.
    num_layers : int, default 2
        Number of Transformer encoder layers.
    dim_feedforward : int, default 128
        Dimension of feedforward layer.
    lr : float, default 2.5e-4
        Adam learning rate (matches GPL).
    gamma : float, default 0.99
        Discount factor (matches GPL).
    t_update : int, default 4
        Gradient accumulation step frequency (matches GPL).
    t_targ_update : int, default 1
        Target update frequency.
    polyak_tau : float, default 1.0e-3
        Polyak soft target update rate (matches GPL).
    device : str, default "cpu"
        PyTorch device.
    """

    def __init__(
        self,
        action_dim: int,
        feat_dim: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        t_update: int = 4,
        t_targ_update: int = 1,
        polyak_tau: float = 1.0e-3,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.t_update = t_update
        self.t_targ_update = t_targ_update
        self.polyak_tau = polyak_tau
        self.device = torch.device(device)
        self._step_count = 0

        # Online and Target Q-networks
        self.q_network = EntityTransformerQNetwork(
            action_dim=action_dim,
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)

        self.q_network_target = EntityTransformerQNetwork(
            action_dim=action_dim,
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)
        self.q_network_target.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    def act(
        self,
        entity_batch: EntityBatch,
        epsilon: float = 0.0,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Select discrete action using epsilon-greedy exploration.

        Parameters
        ----------
        entity_batch : EntityBatch
            Preprocessed entity tokens and visibility masks.
        epsilon : float
            Exploration probability.
        action_mask : Optional ndarray of bool
            True for valid actions, False for invalid actions.

        Returns
        -------
        action : int
        """
        if np.random.rand() < epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                return int(np.random.choice(valid_actions))
            return int(np.random.randint(0, self.action_dim))

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(entity_batch)[0]  # Shape: (action_dim,)
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, device=self.device, dtype=torch.bool)
                q_values[~mask_t] = -float("inf")
            action = int(q_values.argmax().item())
        return action

    def act_batch(
        self,
        entity_batch: EntityBatch,
        epsilon: float = 0.0,
        action_masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batched action selection across parallel environments."""
        B = entity_batch.entity_features.shape[0]
        actions = np.zeros(B, dtype=np.int64)
        rand_mask = np.random.rand(B) < epsilon

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(entity_batch)  # Shape: (B, action_dim)
            if action_masks is not None:
                mask_t = torch.tensor(action_masks, device=self.device, dtype=torch.bool)
                q_values[~mask_t] = -float("inf")
            greedy_actions = q_values.argmax(dim=-1).cpu().numpy()

        for b in range(B):
            if rand_mask[b]:
                actions[b] = np.random.randint(0, self.action_dim)
            else:
                actions[b] = greedy_actions[b]
        return actions

    def train_step_batch(
        self,
        entity_batch: EntityBatch,
        actions: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
        next_entity_batch: EntityBatch,
        dones: Union[np.ndarray, torch.Tensor],
    ) -> Optional[Dict[str, float]]:
        """Batched transition training step across parallel environments."""
        self.q_network.train()
        B = entity_batch.entity_features.shape[0]

        if not isinstance(actions, torch.Tensor):
            actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        else:
            actions_t = actions.to(self.device)

        if not isinstance(rewards, torch.Tensor):
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        else:
            rewards_t = rewards.to(self.device)

        if not isinstance(dones, torch.Tensor):
            dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        else:
            dones_t = dones.to(dtype=torch.float32, device=self.device)

        # 1. Current Q-values: (B, action_dim)
        q_pred = self.q_network(entity_batch)
        q_val = q_pred.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # 2. Target Q-values: y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.q_network_target(next_entity_batch)  # (B, action_dim)
            max_q_next = q_next.max(dim=1)[0]
            y_tensor = rewards_t + (1.0 - dones_t) * (self.gamma * max_q_next)

        # 3. TD Loss
        loss = 0.5 * ((q_val - y_tensor) ** 2).mean()

        # 4. Backward & update
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._step_count += B

        # 5. Target update
        if self.polyak_tau is not None:
            for p, p_targ in zip(self.q_network.parameters(), self.q_network_target.parameters()):
                p_targ.data.mul_(1.0 - self.polyak_tau).add_(p.data, alpha=self.polyak_tau)
        else:
            self.q_network_target.load_state_dict(self.q_network.state_dict())

        return {"q_loss": loss.item()}

    def train_step_online(
        self,
        entity_batch: EntityBatch,
        action: Union[int, np.ndarray, torch.Tensor],
        reward: float,
        next_entity_batch: EntityBatch,
        done: bool,
    ) -> Optional[Dict[str, float]]:
        """Single online transition training step (matches GPL Alg. 5 mechanics).

        Accumulates gradients over t_update transitions and applies Polyak
        soft updates to the target network.
        """
        self.q_network.train()

        # Handle scalar / array action
        if isinstance(action, (int, np.integer)):
            a_idx = int(action)
        elif isinstance(action, (np.ndarray, list)):
            a_idx = int(action[0])
        elif isinstance(action, torch.Tensor):
            a_idx = int(action.item())
        else:
            a_idx = int(action)

        # 1. Current Q(s, a)
        q_pred = self.q_network(entity_batch)  # Shape: (1, action_dim)
        q_val = q_pred[0, a_idx]

        # 2. Target Q-value: y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.q_network_target(next_entity_batch)  # Shape: (1, action_dim)
            max_q_next = q_next.max(dim=1)[0].item()
            y = reward + (0.0 if done else (self.gamma * max_q_next))
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        # 3. TD Loss
        loss = 0.5 * (q_val - y_tensor) ** 2

        # 4. Gradient accumulation
        (loss / self.t_update).backward()
        self._step_count += 1
        metrics = None

        # 5. Optimizer step
        if self._step_count % self.t_update == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            metrics = {"q_loss": loss.item()}

        # 6. Target network update
        if self._step_count % self.t_targ_update == 0:
            if self.polyak_tau is not None:
                for p, p_targ in zip(self.q_network.parameters(), self.q_network_target.parameters()):
                    p_targ.data.mul_(1.0 - self.polyak_tau).add_(p.data, alpha=self.polyak_tau)
            else:
                self.q_network_target.load_state_dict(self.q_network.state_dict())

        return metrics

    def reset(self):
        """Reset episode state (stateless for feedforward Transformer-Q)."""
        pass

    def save(self, path: str):
        """Save model and optimizer state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "q_network_target": self.q_network_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
        }, path)

    def load(self, path: str):
        """Load model and optimizer state."""
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.q_network_target.load_state_dict(ckpt["q_network_target"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step_count = ckpt.get("step_count", 0)
