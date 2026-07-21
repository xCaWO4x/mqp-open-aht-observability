"""
Auxiliary inference head for teammate level prediction.

Predicts teammate level class (1, 2, or 3) from the type inference LSTM's
hidden state / type embedding. Trained with cross-entropy against privileged
labels during centralized training, but hidden from the execution policy.

This forces the type embeddings to encode meaningful teammate structure
even when observe_agent_levels=False, recovering the information the
hidden-level observation removes.

Used by the Q3-inf-aux observability experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryLevelHead(nn.Module):
    """Small MLP predicting teammate level class from type embedding.

    Architecture: type_dim → 64 → ReLU → n_classes (cross-entropy).

    Parameters
    ----------
    type_dim : int
        Dimension of the type embedding (output of TypeInferenceModel).
    n_classes : int
        Number of level classes (default 3 for LBF levels {1, 2, 3}).
    hidden_dim : int
        Hidden layer width (default 64).
    """

    def __init__(self, type_dim: int, n_classes: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.n_classes = n_classes
        self.head = nn.Sequential(
            nn.Linear(type_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, type_emb: torch.Tensor) -> torch.Tensor:
        """Predict level logits from type embeddings.

        Parameters
        ----------
        type_emb : Tensor, shape (..., type_dim)

        Returns
        -------
        logits : Tensor, shape (..., n_classes)
        """
        return self.head(type_emb)

    def loss(self, type_emb: torch.Tensor, true_levels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss against privileged level labels.

        Parameters
        ----------
        type_emb : Tensor, shape (N, type_dim)
            Type embeddings for N agents.
        true_levels : LongTensor, shape (N,)
            Ground-truth LBF levels (1-indexed). Converted to 0-indexed internally.

        Returns
        -------
        loss : scalar Tensor
        """
        logits = self.forward(type_emb)  # (N, n_classes)
        # LBF levels are 1-indexed; convert to 0-indexed for cross-entropy
        targets = (true_levels - 1).clamp(0, self.n_classes - 1)
        return F.cross_entropy(logits, targets)

    def predict(self, type_emb: torch.Tensor) -> torch.Tensor:
        """Predicted level class (1-indexed, matching LBF convention).

        Parameters
        ----------
        type_emb : Tensor, shape (..., type_dim)

        Returns
        -------
        predicted_levels : LongTensor, shape (...)
        """
        logits = self.forward(type_emb)
        return logits.argmax(dim=-1) + 1  # back to 1-indexed
