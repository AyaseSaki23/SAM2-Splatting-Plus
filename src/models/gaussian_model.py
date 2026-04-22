"""Semantic-enhanced Gaussian model core for Stage 2.

This module adds per-Gaussian semantic features and a lightweight semantic head.
It is renderer-agnostic and can be integrated into existing 3DGS/Splatfacto loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GaussianRenderOutput:
    """Container for renderer outputs used by the training step."""

    rgb: torch.Tensor
    alpha: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    gaussian_ids: Optional[torch.Tensor] = None


class GaussianModel(nn.Module):
    """Gaussian model with semantic feature injection.

    Expected integration:
    - Keep your existing RGB Gaussian rendering path.
    - Use `semantic_features` per Gaussian and aggregate with rasterization weights.
    - Map semantic embedding -> class logits by `semantic_head`.
    """

    def __init__(
        self,
        num_gaussians: int,
        semantic_dim: int = 16,
        num_classes: int = 4,
        semantic_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if num_gaussians <= 0:
            raise ValueError("num_gaussians must be > 0")
        if semantic_dim <= 0:
            raise ValueError("semantic_dim must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be >= 2")

        self.num_gaussians = int(num_gaussians)
        self.semantic_dim = int(semantic_dim)
        self.num_classes = int(num_classes)

        # Per-Gaussian learnable semantic embedding s_i in R^D.
        semantic = torch.randn(num_gaussians, semantic_dim) * float(semantic_init_std)
        self.semantic_features = nn.Parameter(semantic)

        # Lightweight semantic classifier: D -> C.
        self.semantic_head = nn.Linear(semantic_dim, num_classes)

    def initialize_semantic_features_from_ids(
        self,
        gaussian_object_ids: torch.Tensor,
        class_id_to_embed: Optional[torch.Tensor] = None,
        random_std: float = 0.01,
    ) -> None:
        """Optional semantic initialization with object/class ids.

        Args:
            gaussian_object_ids: [N] int tensor, each gaussian's object/class id.
            class_id_to_embed: Optional [K, D] embedding table. If None, random init.
            random_std: std used when class id has no table row.
        """
        if gaussian_object_ids.ndim != 1 or gaussian_object_ids.shape[0] != self.num_gaussians:
            raise ValueError("gaussian_object_ids must have shape [num_gaussians]")

        with torch.no_grad():
            if class_id_to_embed is None:
                self.semantic_features.copy_(
                    torch.randn_like(self.semantic_features) * float(random_std)
                )
                return

            if class_id_to_embed.ndim != 2 or class_id_to_embed.shape[1] != self.semantic_dim:
                raise ValueError("class_id_to_embed must have shape [K, semantic_dim]")

            embeds = torch.randn_like(self.semantic_features) * float(random_std)
            valid = (gaussian_object_ids >= 0) & (gaussian_object_ids < class_id_to_embed.shape[0])
            if valid.any():
                embeds[valid] = class_id_to_embed[gaussian_object_ids[valid].long()]
            self.semantic_features.copy_(embeds)

    def aggregate_semantic_embeddings(
        self,
        weights: torch.Tensor,
        gaussian_ids: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Aggregate Gaussian semantic features to per-pixel semantic embedding.

        Args:
            weights: [M] alpha/raster weights for splats contributing to pixels.
            gaussian_ids: [M] gaussian index per contribution.
            image_shape: (H, W)

        Returns:
            sem_embed: [H, W, D]

        Notes:
            This function assumes contributions are flattened in row-major pixel order.
            If your rasterizer uses packed COO-style output, map into pixels before calling
            or adapt this function to your index format.
        """
        if weights.ndim != 1 or gaussian_ids.ndim != 1:
            raise ValueError("weights and gaussian_ids must be 1D tensors")
        if weights.shape[0] != gaussian_ids.shape[0]:
            raise ValueError("weights and gaussian_ids must have same length")

        h, w = image_shape
        if h <= 0 or w <= 0:
            raise ValueError("image_shape must be positive")

        if gaussian_ids.min() < 0 or gaussian_ids.max() >= self.num_gaussians:
            raise ValueError("gaussian_ids out of range")

        contrib = self.semantic_features[gaussian_ids.long()] * weights[:, None]

        # Fallback aggregation: assume each pixel has variable contributions appended,
        # and caller already pre-groups by pixel in flattened order.
        # For minimal baseline, if M == H*W we treat one contrib per pixel.
        if contrib.shape[0] == h * w:
            return contrib.view(h, w, self.semantic_dim)

        raise ValueError(
            "Unsupported contribution layout for aggregate_semantic_embeddings. "
            "Provide per-pixel-aligned weights/ids or customize this method for packed splats."
        )

    def semantic_logits_from_embedding(self, semantic_embedding: torch.Tensor) -> torch.Tensor:
        """Map per-pixel semantic embedding [H,W,D]/[B,H,W,D] to logits [..,C]."""
        if semantic_embedding.ndim == 3:
            h, w, d = semantic_embedding.shape
            if d != self.semantic_dim:
                raise ValueError("semantic embedding dim mismatch")
            flat = semantic_embedding.view(-1, d)
            logits = self.semantic_head(flat)
            return logits.view(h, w, self.num_classes)

        if semantic_embedding.ndim == 4:
            b, h, w, d = semantic_embedding.shape
            if d != self.semantic_dim:
                raise ValueError("semantic embedding dim mismatch")
            flat = semantic_embedding.view(-1, d)
            logits = self.semantic_head(flat)
            return logits.view(b, h, w, self.num_classes)

        raise ValueError("semantic_embedding must be [H,W,D] or [B,H,W,D]")

    def forward_semantic(
        self,
        weights: torch.Tensor,
        gaussian_ids: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        sem_embed = self.aggregate_semantic_embeddings(
            weights=weights,
            gaussian_ids=gaussian_ids,
            image_shape=image_shape,
        )
        sem_logits = self.semantic_logits_from_embedding(sem_embed)
        return {
            "semantic_embedding": sem_embed,
            "semantic_logits": sem_logits,
        }
