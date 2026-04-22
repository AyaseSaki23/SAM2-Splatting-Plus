"""Stage-2 training glue for Geo-Semantic injection.

This module provides a minimal train step that combines:
1) RGB reconstruction loss (L1/MSE, and optional SSIM hook)
2) Semantic cross-entropy loss from rendered semantic logits vs SAM2 id masks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .loss_functions import JointSemanticLoss


@dataclass
class SemanticTrainConfig:
    lambda_sem: float = 0.1
    rgb_loss_type: str = "l1"
    ignore_index: int = -100
    use_ssim: bool = False
    ssim_lambda: float = 0.2


def _to_nchw_logits(logits_hwc: torch.Tensor) -> torch.Tensor:
    """Convert [H,W,C] or [B,H,W,C] logits to [B,C,H,W]."""
    if logits_hwc.ndim == 3:
        h, w, c = logits_hwc.shape
        return logits_hwc.view(1, h, w, c).permute(0, 3, 1, 2).contiguous()
    if logits_hwc.ndim == 4:
        return logits_hwc.permute(0, 3, 1, 2).contiguous()
    raise ValueError("semantic logits must be [H,W,C] or [B,H,W,C]")


def _to_bhw_mask(mask_hw: torch.Tensor) -> torch.Tensor:
    """Convert [H,W] or [B,H,W] semantic ids to [B,H,W]."""
    if mask_hw.ndim == 2:
        return mask_hw.unsqueeze(0).long()
    if mask_hw.ndim == 3:
        return mask_hw.long()
    raise ValueError("semantic mask must be [H,W] or [B,H,W]")


def train_step_joint(
    *,
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    pred_semantic_logits_hwc: torch.Tensor,
    gt_semantic_mask: torch.Tensor,
    cfg: SemanticTrainConfig,
    valid_mask: Optional[torch.Tensor] = None,
    ssim_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute Stage-2 joint loss.

    Args:
        pred_rgb: [H,W,3] or [B,H,W,3] in [0,1]
        gt_rgb: same shape as pred_rgb
        pred_semantic_logits_hwc: [H,W,C] or [B,H,W,C]
        gt_semantic_mask: [H,W] or [B,H,W] class ids
        cfg: semantic/rgb loss config
        valid_mask: optional pixel mask aligned with pred_rgb
        ssim_fn: optional SSIM callable (returns larger-is-better similarity)
    """
    criterion = JointSemanticLoss(
        lambda_sem=cfg.lambda_sem,
        rgb_loss_type=cfg.rgb_loss_type,
        ignore_index=cfg.ignore_index,
    )

    sem_logits_nchw = _to_nchw_logits(pred_semantic_logits_hwc)
    gt_sem_bhw = _to_bhw_mask(gt_semantic_mask)

    # Ensure RGB is 4D for optional SSIM branch.
    if pred_rgb.ndim == 3:
        pred_rgb_bhwc = pred_rgb.unsqueeze(0)
        gt_rgb_bhwc = gt_rgb.unsqueeze(0)
    else:
        pred_rgb_bhwc = pred_rgb
        gt_rgb_bhwc = gt_rgb

    # Base joint loss: L_rgb + lambda * L_ce
    base = criterion(
        pred_rgb=pred_rgb_bhwc,
        gt_rgb=gt_rgb_bhwc,
        pred_sem_logits=sem_logits_nchw,
        gt_sem=gt_sem_bhw,
        valid_mask=valid_mask,
    )

    total = base.total
    extra = {}

    if cfg.use_ssim:
        if ssim_fn is None:
            # Fallback pseudo-SSIM penalty if function is not wired yet.
            # This keeps pipeline runnable but should be replaced by real SSIM.
            pred_nchw = pred_rgb_bhwc.permute(0, 3, 1, 2).contiguous()
            gt_nchw = gt_rgb_bhwc.permute(0, 3, 1, 2).contiguous()
            mse = F.mse_loss(pred_nchw, gt_nchw)
            ssim_penalty = torch.clamp(mse, min=0.0)
        else:
            pred_nchw = pred_rgb_bhwc.permute(0, 3, 1, 2).contiguous()
            gt_nchw = gt_rgb_bhwc.permute(0, 3, 1, 2).contiguous()
            ssim_val = ssim_fn(pred_nchw, gt_nchw)
            ssim_penalty = 1.0 - ssim_val

        total = total + cfg.ssim_lambda * ssim_penalty
        extra["loss_ssim"] = ssim_penalty

    out: Dict[str, torch.Tensor] = {
        "loss_total": total,
        "loss_rgb": base.rgb,
        "loss_sem": base.semantic,
        "pred_semantic_logits_nchw": sem_logits_nchw,
    }
    out.update(extra)
    return out

