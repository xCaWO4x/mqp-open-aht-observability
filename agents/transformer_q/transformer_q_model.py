"""
Entity-Transformer Q-Network Architecture.

Represents visible entities as tokens with learned entity-type embeddings,
prepends a learned [AGENT] token, applies multi-head self-attention with
strict key padding masking for out-of-sight entities, and predicts Q-values
from the transformed [AGENT] token embedding.
"""

import torch
import torch.nn as nn
from typing import Optional, Union

from envs.entity_adapter import EntityBatch


class EntityTransformerQNetwork(nn.Module):
    """Entity-Transformer Q-Network with key padding mask for invisible entities.

    Parameters
    ----------
    feat_dim : int, default 3
        Dimensionality of raw entity features (e.g. y, x, val).
    action_dim : int
        Number of discrete actions.
    d_model : int, default 64
        Transformer embedding dimension.
    nhead : int, default 4
        Number of self-attention heads.
    num_layers : int, default 2
        Number of Transformer encoder layers.
    dim_feedforward : int, default 128
        Dimension of feedforward network inside Transformer layer.
    dropout : float, default 0.0
        Dropout probability.
    num_entity_types : int, default 3
        Number of discrete entity types (Self=0, Teammate=1, Object=2).
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
        num_entity_types: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.d_model = d_model

        # Entity feature projection + Entity type embedding
        self.entity_proj = nn.Linear(feat_dim, d_model)
        self.entity_type_embed = nn.Embedding(num_entity_types, d_model)

        # Learned [AGENT] / [CLS] token prepended to the sequence
        self.agent_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Multi-layer Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Q-value projection head from [AGENT] token representation
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )

    def forward(
        self,
        batch: Union[EntityBatch, torch.Tensor],
        entity_types: Optional[torch.Tensor] = None,
        visible_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Q-values for all discrete actions.

        Parameters
        ----------
        batch : EntityBatch or Tensor of shape (B, K, feat_dim)
        entity_types : Optional LongTensor of shape (B, K)
        visible_mask : Optional BoolTensor of shape (B, K), True = visible, False = hidden

        Returns
        -------
        q_values : Tensor of shape (B, action_dim)
        """
        if isinstance(batch, EntityBatch):
            entity_features = batch.entity_features
            entity_types = batch.entity_types
            visible_mask = batch.visible_mask
        else:
            entity_features = batch
            assert entity_types is not None and visible_mask is not None

        B, K, _ = entity_features.shape

        # 1. Project entity features + add entity type embedding
        entity_tokens = self.entity_proj(entity_features) + self.entity_type_embed(entity_types)

        # 2. Prepend learned [AGENT] token
        agent_tok = self.agent_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([agent_tok, entity_tokens], dim=1)  # (B, 1 + K, d_model)

        # 3. Construct attention padding mask: True means position is IGNORED
        # [AGENT] token at index 0 is ALWAYS False (never ignored)
        # Entity tokens at 1..K are ignored if visible_mask is False
        src_key_padding_mask = torch.zeros((B, 1 + K), dtype=torch.bool, device=entity_features.device)
        src_key_padding_mask[:, 1:] = ~visible_mask

        # 4. Self-attention encoding
        encoded = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)

        # 5. Extract transformed [AGENT] token embedding at index 0
        agent_rep = encoded[:, 0, :]  # (B, d_model)

        # 6. Predict Q-values
        q_values = self.q_head(agent_rep)  # (B, action_dim)

        return q_values
