"""Loss definitions for semantic-enhanced 3DGS training.

Stage 2 target formula:
    L_total = L_rgb + lambda_sem * L_ce
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    rgb: torch.Tensor
    semantic: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "loss_total": self.total,
            "loss_rgb": self.rgb,
            "loss_sem": self.semantic,
        }


class JointSemanticLoss(nn.Module):
    """Joint RGB + semantic CE loss for Geo-Semantic Splatting."""

    def __init__(
        self,
        lambda_sem: float = 0.1,
        rgb_loss_type: str = "l1",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if rgb_loss_type not in {"l1", "mse"}:
            raise ValueError("rgb_loss_type must be 'l1' or 'mse'.")
        self.lambda_sem = float(lambda_sem)
        self.rgb_loss_type = rgb_loss_type
        self.ignore_index = int(ignore_index)

    def _rgb_loss(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred_rgb.shape != gt_rgb.shape:
            raise ValueError(
                f"RGB shape mismatch: pred={tuple(pred_rgb.shape)}, "
                f"gt={tuple(gt_rgb.shape)}"
            )
        if self.rgb_loss_type == "l1":
            diff = torch.abs(pred_rgb - gt_rgb)
        else:
            diff = (pred_rgb - gt_rgb) ** 2

        if valid_mask is None:
            return diff.mean()

        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(-1)
        if valid_mask.shape[:-1] != diff.shape[:-1]:
            raise ValueError("valid_mask spatial shape must match RGB shape.")

        weighted = diff * valid_mask.float()
        denom = valid_mask.float().sum() * diff.shape[-1]
        denom = torch.clamp(denom, min=1.0)
        return weighted.sum() / denom

    def _semantic_ce_loss(
        self,
        pred_sem_logits: torch.Tensor,
        gt_sem: torch.Tensor,
    ) -> torch.Tensor:
        # Supports [B, C, H, W] + [B, H, W] and [N, C] + [N].
        if pred_sem_logits.ndim == 4 and gt_sem.ndim == 3:
            return F.cross_entropy(
                pred_sem_logits,
                gt_sem.long(),
                ignore_index=self.ignore_index,
            )
        if pred_sem_logits.ndim == 2 and gt_sem.ndim == 1:
            return F.cross_entropy(
                pred_sem_logits,
                gt_sem.long(),
                ignore_index=self.ignore_index,
            )
        raise ValueError(
            "Unsupported semantic tensor shapes. "
            "Expected [B,C,H,W]+[B,H,W] or [N,C]+[N]."
        )

    def forward(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        pred_sem_logits: torch.Tensor,
        gt_sem: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        rgb = self._rgb_loss(pred_rgb, gt_rgb, valid_mask=valid_mask)
        sem = self._semantic_ce_loss(pred_sem_logits, gt_sem)
        total = rgb + self.lambda_sem * sem
        return LossOutput(total=total, rgb=rgb, semantic=sem)


def compute_joint_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    pred_sem_logits: torch.Tensor,
    gt_sem: torch.Tensor,
    lambda_sem: float = 0.1,
    valid_mask: Optional[torch.Tensor] = None,
    rgb_loss_type: str = "l1",
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """Functional helper for training loops."""
    criterion = JointSemanticLoss(
        lambda_sem=lambda_sem,
        rgb_loss_type=rgb_loss_type,
        ignore_index=ignore_index,
    )
    output = criterion(
        pred_rgb=pred_rgb,
        gt_rgb=gt_rgb,
        pred_sem_logits=pred_sem_logits,
        gt_sem=gt_sem,
        valid_mask=valid_mask,
    )
    return output.as_dict()
