"""
Recurrent Independent PPO (IPPO-GRU) Agent.

Maintains per-environment GRU hidden states across parallel rollouts,
computes Generalized Advantage Estimation (GAE) with proper distinction
between true termination and time-limit truncation, and performs recurrent
mini-batch PPO updates on contiguous sequence chunks.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from agents.ippo_gru.ippo_gru_model import IPPORecurrentActorCritic


class RecurrentRolloutBuffer:
    """Rollout buffer for vectorized environments with recurrent state tracking."""

    def __init__(
        self,
        rollout_length: int,
        n_envs: int,
        obs_dim: int,
        gru_hidden: int,
        device: torch.device,
    ):
        self.rollout_length = rollout_length
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.gru_hidden = gru_hidden
        self.device = device
        self.reset()

    def reset(self):
        self.obs = torch.zeros((self.rollout_length, self.n_envs, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.rollout_length, self.n_envs), dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((self.rollout_length, self.n_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.rollout_length, self.n_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.rollout_length, self.n_envs), dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros((self.rollout_length, self.n_envs), dtype=torch.bool, device=self.device)
        self.values = torch.zeros((self.rollout_length, self.n_envs), dtype=torch.float32, device=self.device)
        # Store the initial hidden state at the start of each rollout step
        self.hiddens = torch.zeros((self.rollout_length, 1, self.n_envs, self.gru_hidden), dtype=torch.float32, device=self.device)

        self.advantages = torch.zeros((self.rollout_length, self.n_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((self.rollout_length, self.n_envs), dtype=torch.float32, device=self.device)
        self.step_idx = 0

    def insert(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        hidden: torch.Tensor,
        truncated: Optional[torch.Tensor] = None,
    ):
        t = self.step_idx
        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = done
        self.truncated[t] = truncated if truncated is not None else torch.zeros_like(done)
        self.values[t] = value
        self.hiddens[t] = hidden.detach()
        self.step_idx += 1

    def compute_gae(
        self,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        next_truncated: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute Generalized Advantage Estimation (GAE) backwards in time.

        Correctly distinguishes true termination from time-limit truncation:
        - True termination (terminated=True): next_non_terminal=0, V_next=0 (no bootstrapping).
        - Time-limit truncation (truncated=True): bootstraps V(s_{t+1}) into return,
          but does NOT propagate GAE backwards across the episode boundary.
        - Normal transition (done=False): bootstraps V(s_{t+1}) and propagates GAE.
        """
        last_gae = 0.0
        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                is_done = next_done
                is_trunc = next_truncated if next_truncated is not None else torch.zeros_like(next_done)
                next_val = next_value
            else:
                is_done = self.dones[t + 1]
                is_trunc = self.truncated[t + 1]
                next_val = self.values[t + 1]

            is_true_terminal = is_done & (~is_trunc)
            next_non_terminal = (~is_true_terminal).float()
            non_reset_flag = (~is_done).float()

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_reset_flag * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]


class IPPOAgent:
    """Recurrent Independent PPO Agent."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_envs: int = 16,
        rollout_length: int = 128,
        encoder_hidden: int = 128,
        gru_hidden: int = 128,
        lr: float = 3.0e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.device = torch.device(device)

        # Actor-Critic network
        self.ac = IPPORecurrentActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            encoder_hidden=encoder_hidden,
            gru_hidden=gru_hidden,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

        # Rollout buffer
        self.buffer = RecurrentRolloutBuffer(
            rollout_length=rollout_length,
            n_envs=n_envs,
            obs_dim=obs_dim,
            gru_hidden=gru_hidden,
            device=self.device,
        )

        # Current per-env GRU hidden states: shape (1, n_envs, gru_hidden)
        self.hidden = self.ac.get_initial_hidden(batch_size=n_envs, device=self.device)

    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions for all parallel environments."""
        self.ac.eval()
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            else:
                obs_t = obs.to(self.device)

            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)

            mask_t = None
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
                if mask_t.ndim == 1:
                    mask_t = mask_t.unsqueeze(0)

            curr_hidden = self.hidden.clone()
            action, log_prob, value, next_hidden = self.ac.step(obs_t, self.hidden, mask_t)
            self.hidden = next_hidden

        return action.cpu().numpy(), log_prob, value, curr_hidden

    def observe(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        log_probs: torch.Tensor,
        rewards: Union[np.ndarray, torch.Tensor, float],
        dones: Union[np.ndarray, torch.Tensor, bool],
        values: torch.Tensor,
        step_hiddens: torch.Tensor,
        truncated: Optional[Union[np.ndarray, torch.Tensor, bool]] = None,
    ):
        """Insert step transition into rollout buffer and reset dones."""
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.to(self.device)

        if not isinstance(actions, torch.Tensor):
            act_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        else:
            act_t = actions.to(self.device)

        if not isinstance(rewards, torch.Tensor):
            rew_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        else:
            rew_t = rewards.to(self.device)

        if not isinstance(dones, torch.Tensor):
            don_t = torch.tensor(dones, dtype=torch.bool, device=self.device)
        else:
            don_t = dones.to(self.device)

        trunc_t = None
        if truncated is not None:
            if not isinstance(truncated, torch.Tensor):
                trunc_t = torch.tensor(truncated, dtype=torch.bool, device=self.device)
            else:
                trunc_t = truncated.to(self.device)

        self.buffer.insert(
            obs=obs_t,
            action=act_t,
            log_prob=log_probs.to(self.device),
            reward=rew_t,
            done=don_t,
            value=values.to(self.device),
            hidden=step_hiddens.to(self.device),
            truncated=trunc_t,
        )

        # Reset GRU hidden state for completed environments (done = True)
        if don_t.any():
            self.hidden[:, don_t, :] = 0.0

    def update(
        self,
        next_obs: Union[np.ndarray, torch.Tensor],
        next_dones: Union[np.ndarray, torch.Tensor],
        next_truncated: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Run PPO optimization on collected rollout buffer with detailed metric logging."""
        self.ac.eval()
        with torch.no_grad():
            if not isinstance(next_obs, torch.Tensor):
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            else:
                next_obs_t = next_obs.to(self.device)

            if not isinstance(next_dones, torch.Tensor):
                next_don_t = torch.tensor(next_dones, dtype=torch.bool, device=self.device)
            else:
                next_don_t = next_dones.to(self.device)

            next_trunc_t = None
            if next_truncated is not None:
                if not isinstance(next_truncated, torch.Tensor):
                    next_trunc_t = torch.tensor(next_truncated, dtype=torch.bool, device=self.device)
                else:
                    next_trunc_t = next_truncated.to(self.device)

            _, _, next_values, _ = self.ac.step(next_obs_t, self.hidden)

        # 1. Compute GAE with termination vs. truncation awareness
        self.buffer.compute_gae(next_values, next_don_t, next_trunc_t, self.gamma, self.gae_lambda)

        # 2. Reshape into sequences per environment: (N_envs, T, ...)
        obs_seq = self.buffer.obs.permute(1, 0, 2)            # (N, T, obs_dim)
        actions_seq = self.buffer.actions.permute(1, 0)        # (N, T)
        old_log_probs = self.buffer.log_probs.permute(1, 0)    # (N, T)
        returns_seq = self.buffer.returns.permute(1, 0)        # (N, T)
        advantages_seq = self.buffer.advantages.permute(1, 0)  # (N, T)

        # Initial hidden states for each sequence at timestep 0: shape (1, N, gru_hidden)
        initial_hiddens = self.buffer.hiddens[0]               # (1, N, gru_hidden)

        # Advantage normalization
        adv_mean = advantages_seq.mean()
        adv_std = advantages_seq.std() + 1e-8
        norm_advantages = (advantages_seq - adv_mean) / adv_std

        # 3. PPO Optimization Epochs
        self.ac.train()
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        total_grad_norm = 0.0
        approx_kl = 0.0
        last_new_values = None

        for _ in range(self.ppo_epochs):
            # Recurrent evaluation of the entire contiguous sequence
            new_log_probs, entropy, new_values = self.ac.evaluate_actions(
                obs_seq=obs_seq,
                hidden_init=initial_hiddens,
                actions=actions_seq,
            )
            last_new_values = new_values

            # Policy loss
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()

            surr1 = ratio * norm_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * norm_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = 0.5 * ((new_values - returns_seq) ** 2).mean()

            # Total loss
            entropy_loss = -entropy.mean()
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm).item()
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
            total_clip_frac += clip_frac
            total_grad_norm += grad_norm

        # Explained variance: 1 - Var(returns - values) / Var(returns)
        with torch.no_grad():
            y_true = returns_seq.reshape(-1)
            y_pred = last_new_values.reshape(-1)
            var_y = torch.var(y_true)
            explained_var = (1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)).item() if var_y > 1e-8 else 0.0

        # Reset buffer for next rollout
        self.buffer.reset()

        return {
            "actor_loss": total_actor_loss / self.ppo_epochs,
            "critic_loss": total_critic_loss / self.ppo_epochs,
            "entropy": total_entropy / self.ppo_epochs,
            "approx_kl": approx_kl,
            "clip_fraction": total_clip_frac / self.ppo_epochs,
            "explained_variance": explained_var,
            "grad_norm": total_grad_norm / self.ppo_epochs,
        }

    def reset(self):
        """Reset all per-env hidden states."""
        self.hidden = self.ac.get_initial_hidden(batch_size=self.n_envs, device=self.device)

    def save(self, path: str):
        """Save model and optimizer state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "ac": self.ac.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model and optimizer state."""
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
