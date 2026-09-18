"""
Recurrent Independent PPO (IPPO-GRU) baseline package.
"""

from agents.ippo_gru.ippo_gru_model import IPPORecurrentActorCritic
from agents.ippo_gru.ippo_gru_agent import IPPOAgent

__all__ = ["IPPORecurrentActorCritic", "IPPOAgent"]
