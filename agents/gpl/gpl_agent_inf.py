"""
GPL agent with an auxiliary teammate-level inference head.

The auxiliary head predicts teammate LBF levels from the type-inference
embedding during centralized training. The level labels are privileged
training targets only; they are not fed to the execution policy.
"""

import torch

from agents.gpl.gpl_agent import GPLAgent
from agents.gpl.auxiliary_head import AuxiliaryLevelHead


class GPLAgentInf(GPLAgent):
    """GPL agent with an auxiliary level-prediction objective."""

    def __init__(
        self,
        obs_dim: int,
        aux_n_classes: int = 3,
        aux_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(obs_dim=obs_dim, **kwargs)

        type_dim = kwargs.get("type_dim", 32)
        self.aux_head = AuxiliaryLevelHead(
            type_dim=type_dim,
            n_classes=aux_n_classes,
        ).to(self.device)
        self.aux_weight = aux_weight

        self.optimiser_agent.add_param_group(
            {"params": self.aux_head.parameters()}
        )

    def train_step_online_inf(
        self,
        B_t: torch.Tensor,
        joint_actions,
        reward: float,
        B_t_next: torch.Tensor,
        done: bool,
        agent_levels: list,
        learner_idx: int = 0,
        teammate_indices=None,
    ):
        """Run a GPL training step plus the auxiliary level loss.

        `agent_levels` are privileged labels used only for the auxiliary loss.
        The policy receives the same preprocessed observation as the baseline.
        """
        # Snapshot the pre-update hidden state so the aux forward sees the
        # same temporal context as the policy. Passing None here would reduce
        # the task to a single-frame prediction and pin CE near uniform.
        hidden_agent_prev = self._hidden_agent

        metrics = super().train_step_online(
            B_t,
            joint_actions,
            reward,
            B_t_next,
            done,
            learner_idx=learner_idx,
            teammate_indices=teammate_indices,
        )

        B_t_tensor = torch.FloatTensor(B_t).to(self.device)
        type_emb, _ = self.type_net_agent(B_t_tensor, hidden_agent_prev)
        levels_tensor = torch.LongTensor(agent_levels).to(self.device)

        n_agents = len(agent_levels)
        if teammate_indices is None:
            teammate_indices = [j for j in range(n_agents) if j != learner_idx]

        if teammate_indices and self.aux_weight > 0:
            tm_emb = type_emb[teammate_indices]
            tm_levels = levels_tensor[teammate_indices]
            aux_loss = self.aux_head.loss(tm_emb, tm_levels) * self.aux_weight
            (aux_loss / self.t_update).backward()

            if metrics is not None:
                metrics["aux_loss"] = aux_loss.item()

        return metrics

    def act_inf(self, B_t, learner_idx=0, epsilon=0.0):
        """Action selection for the auxiliary-head variant."""
        return self.act(B_t, learner_idx=learner_idx, epsilon=epsilon)

    def advance_hidden_inf(self, B_t):
        """Advance hidden states for the auxiliary-head variant."""
        self.advance_hidden(B_t)


    def save(self, path: str):
        """Save all model weights including the auxiliary head."""
        state = {
            "type_net_q": self.type_net_q.state_dict(),
            "type_net_agent": self.type_net_agent.state_dict(),
            "agent_model": self.agent_model.state_dict(),
            "q_network": self.q_network.state_dict(),
            "q_network_target": self.q_network_target.state_dict(),
            "type_net_q_target": self.type_net_q_target.state_dict(),
            "optimiser_q": self.optimiser_q.state_dict(),
            "optimiser_agent": self.optimiser_agent.state_dict(),
            "step_count": self._step_count,
            "aux_head": self.aux_head.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load model weights including the auxiliary head when present."""
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
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
        if "aux_head" in ckpt:
            self.aux_head.load_state_dict(ckpt["aux_head"])
