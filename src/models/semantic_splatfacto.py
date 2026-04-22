"""Semantic-augmented Splatfacto model for Stage-2 Geo-Semantic refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from gsplat.rendering import rasterization
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat


@dataclass
class SemanticSplatfactoModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SemanticSplatfactoModel)
    semantic_dim: int = 16
    num_classes: int = 4
    lambda_sem: float = 0.1
    lambda_mv_consistency: float = 0.02
    lambda_geo_alignment: float = 0.05
    lambda_fg_aux: float = 0.1
    lambda_class_dist: float = 0.05
    fg_boost_factor: float = 5.0
    fg_dice_weight: float = 0.1
    semantic_warmup_steps: int = 1000
    semantic_reg_start_delay: int = 500
    semantic_reg_warmup_steps: int = 1000
    semantic_warmup_start_step: int = 0
    two_stage_semantic_training: bool = True
    semantic_stage1_steps: int = 1200
    semantic_stage2_ramp_steps: int = 1200
    semantic_train_start_step: int = 0
    stage2_binary_keep_weight: float = 0.25
    semantic_head_bg_bias_init: float = 0.0
    semantic_head_fg_bias_init: float = 0.0
    geo_alignment_beta: float = 20.0
    focal_gamma: float = 2.0
    semantic_ignore_index: int = -100
    semantic_loss_every: int = 1
    freeze_geometry: bool = True
    use_rgb_loss: bool = True
    class_weights: Optional[Tuple[float, ...]] = None
    bg_class_id: int = 0
    enable_object_culling: bool = True
    semantic_stats_every: int = 200
    semantic_stats_print: bool = True
    semantic_stats_file: Optional[str] = None


class SemanticSplatfactoModel(SplatfactoModel):
    """Splatfacto with per-Gaussian semantic features and semantic CE supervision."""

    config: SemanticSplatfactoModelConfig
    _semantic_log_file_path: Optional[Path] = None

    @staticmethod
    def _rand_semantic_features(num_points: int, semantic_dim: int, device: torch.device) -> torch.Tensor:
        # Small random init avoids the "all-zero -> only bias learns" deadlock.
        return 1e-3 * torch.randn((num_points, semantic_dim), device=device, dtype=torch.float32)

    def populate_modules(self):
        super().populate_modules()
        n = self.gauss_params["means"].shape[0]
        self.semantic_features = torch.nn.Parameter(
            self._rand_semantic_features(n, self.config.semantic_dim, self.gauss_params["means"].device)
        )
        self.semantic_head = torch.nn.Linear(self.config.semantic_dim, self.config.num_classes)
        with torch.no_grad():
            self._init_semantic_head_bias(self.semantic_head.bias)
        self._semantic_log_file_path = None

    def _init_semantic_head_bias(self, bias: torch.Tensor) -> None:
        bias.zero_()
        if 0 <= self.config.bg_class_id < bias.numel():
            bias[self.config.bg_class_id] = float(self.config.semantic_head_bg_bias_init)
            fg_mask = torch.ones_like(bias, dtype=torch.bool)
            fg_mask[self.config.bg_class_id] = False
            bias[fg_mask] = float(self.config.semantic_head_fg_bias_init)

    def load_state_dict(self, state_dict, **kwargs):  # type: ignore[override]
        bootstrap_semantic = False
        if "semantic_features" in state_dict and "semantic_head.weight" in state_dict:
            sf = state_dict["semantic_features"]
            sw = state_dict["semantic_head.weight"]
            # If both are entirely zero in a training load, we bootstrap to break deadlock.
            bootstrap_semantic = torch.count_nonzero(sf) == 0 and torch.count_nonzero(sw) == 0

        n_ckpt = state_dict["gauss_params.means"].shape[0]
        if self.semantic_features.shape[0] != n_ckpt:
            self.semantic_features = nn.Parameter(
                self._rand_semantic_features(n_ckpt, self.config.semantic_dim, self.semantic_features.device)
            )

        if "semantic_features" not in state_dict:
            state_dict["semantic_features"] = self._rand_semantic_features(
                n_ckpt,
                self.config.semantic_dim,
                state_dict["gauss_params.means"].device,
            )
        if "semantic_head.weight" not in state_dict:
            device = state_dict["gauss_params.means"].device
            weight = torch.empty((self.config.num_classes, self.config.semantic_dim), device=device)
            nn.init.xavier_uniform_(weight)
            state_dict["semantic_head.weight"] = weight
            state_dict["semantic_head.bias"] = torch.zeros((self.config.num_classes,), device=device)
        else:
            # If class count changed, ignore old semantic head weights and re-init.
            if state_dict["semantic_head.weight"].shape != self.semantic_head.weight.shape:
                del state_dict["semantic_head.weight"]
                if "semantic_head.bias" in state_dict:
                    del state_dict["semantic_head.bias"]
            if "semantic_head.weight" not in state_dict:
                device = state_dict["gauss_params.means"].device
                weight = torch.empty((self.config.num_classes, self.config.semantic_dim), device=device)
                nn.init.xavier_uniform_(weight)
                state_dict["semantic_head.weight"] = weight
                state_dict["semantic_head.bias"] = torch.zeros((self.config.num_classes,), device=device)
        super().load_state_dict(state_dict, **kwargs)

        if bootstrap_semantic:
            with torch.no_grad():
                self.semantic_features.copy_(
                    self._rand_semantic_features(
                        self.semantic_features.shape[0],
                        self.config.semantic_dim,
                        self.semantic_features.device,
                    )
                )
                nn.init.xavier_uniform_(self.semantic_head.weight)
                self._init_semantic_head_bias(self.semantic_head.bias)
        elif torch.count_nonzero(self.semantic_head.bias) == 0:
            with torch.no_grad():
                self._init_semantic_head_bias(self.semantic_head.bias)

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_gaussian_param_groups()
        groups["semantic_features"] = [self.semantic_features]
        groups["semantic_head"] = list(self.semantic_head.parameters())
        return groups

    def _get_class_weights(self, device: torch.device) -> Optional[torch.Tensor]:
        class_weights = self.config.class_weights
        if class_weights is None:
            return None
        if len(class_weights) != self.config.num_classes:
            return None
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        if not torch.isfinite(weight_tensor).all():
            return None
        weight_tensor = torch.clamp(weight_tensor, min=1e-6)
        return weight_tensor / weight_tensor.mean()

    def _get_fg_class_weights(self, device: torch.device) -> Optional[torch.Tensor]:
        class_weights = self._get_class_weights(device)
        if class_weights is None or class_weights.numel() <= 1:
            return None
        fg_weights = class_weights[1:].clone()
        if not torch.isfinite(fg_weights).all():
            return None
        fg_weights = torch.clamp(fg_weights, min=1e-6)
        return fg_weights / fg_weights.mean()

    def _semantic_rel_step(self) -> int:
        return max(0, int(self.step) - int(self.config.semantic_train_start_step))

    def _two_stage_mix(self) -> Tuple[float, float]:
        if not self.config.two_stage_semantic_training:
            return 0.0, 1.0
        rel_step = self._semantic_rel_step()
        stage1_steps = max(0, int(self.config.semantic_stage1_steps))
        stage2_ramp = max(1, int(self.config.semantic_stage2_ramp_steps))
        if rel_step < stage1_steps:
            stage1_alpha = 1.0 - float(rel_step) / float(max(stage1_steps, 1))
            return stage1_alpha, 0.0
        stage1_alpha = 0.0
        stage2_alpha = min(1.0, float(rel_step - stage1_steps) / float(stage2_ramp))
        return stage1_alpha, stage2_alpha

    def _log_semantic_stats(
        self,
        logits_nchw: torch.Tensor,
        target_bhw: torch.Tensor,
        metrics_dict: Optional[Dict] = None,
    ) -> None:
        every = max(1, int(self.config.semantic_stats_every))
        if self.step % every != 0:
            return

        with torch.no_grad():
            pred_bhw = torch.argmax(logits_nchw, dim=1)
            valid = target_bhw != self.config.semantic_ignore_index
            pred_valid = pred_bhw[valid]
            tgt_valid = target_bhw[valid]
            conf_valid = F.softmax(logits_nchw, dim=1).max(dim=1).values[valid]

            pred_frac = [0.0 for _ in range(self.config.num_classes)]
            tgt_frac = [0.0 for _ in range(self.config.num_classes)]
            if pred_valid.numel() > 0:
                pred_hist = torch.bincount(pred_valid.view(-1), minlength=self.config.num_classes).to(torch.float32)
                pred_hist = pred_hist / pred_hist.sum().clamp_min(1.0)
                pred_frac = [float(x) for x in pred_hist.detach().cpu()]
            if tgt_valid.numel() > 0:
                tgt_hist = torch.bincount(tgt_valid.view(-1), minlength=self.config.num_classes).to(torch.float32)
                tgt_hist = tgt_hist / tgt_hist.sum().clamp_min(1.0)
                tgt_frac = [float(x) for x in tgt_hist.detach().cpu()]

            conf_mean = float(conf_valid.mean().detach().cpu()) if conf_valid.numel() > 0 else 0.0

            if metrics_dict is not None:
                metrics_dict["semantic_conf_mean"] = torch.tensor(conf_mean, device=self.device)
                for i in range(self.config.num_classes):
                    metrics_dict[f"semantic_pred_frac_c{i}"] = torch.tensor(pred_frac[i], device=self.device)
                    metrics_dict[f"semantic_tgt_frac_c{i}"] = torch.tensor(tgt_frac[i], device=self.device)

            if self.config.semantic_stats_print:
                pred_str = ", ".join([f"c{i}:{pred_frac[i]:.3f}" for i in range(self.config.num_classes)])
                tgt_str = ", ".join([f"c{i}:{tgt_frac[i]:.3f}" for i in range(self.config.num_classes)])
                line = f"[SemanticStats step={self.step}] conf={conf_mean:.3f} | pred=({pred_str}) | target=({tgt_str})"
                # Use plain print to avoid being swallowed by rich progress re-draw.
                print(line, flush=True)
                if self.config.semantic_stats_file:
                    if self._semantic_log_file_path is None:
                        out_dir = Path(self.config.semantic_stats_file)
                        out_dir.parent.mkdir(parents=True, exist_ok=True)
                        self._semantic_log_file_path = out_dir
                    with self._semantic_log_file_path.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")

    def get_outputs(self, camera):
        outputs = super().get_outputs(camera)

        if self.training and (self.step % max(1, self.config.semantic_loss_every) != 0):
            return outputs

        if self.training:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return outputs
        else:
            crop_ids = None

        if crop_ids is not None:
            means_crop = self.means[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            opacities_crop = self.opacities[crop_ids]
            sem_feat_crop = self.semantic_features[crop_ids]
        else:
            means_crop = self.means
            scales_crop = self.scales
            quats_crop = self.quats
            opacities_crop = self.opacities
            sem_feat_crop = self.semantic_features

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        w = int(camera.width.item())
        h = int(camera.height.item())
        camera.rescale_output_resolution(camera_scale_fac)

        sem_render, sem_alpha, _ = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=sem_feat_crop,
            viewmats=viewmat,
            Ks=K,
            width=w,
            height=h,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=None,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
        )
        sem_embed = sem_render.squeeze(0)
        sem_alpha = sem_alpha.squeeze(0)
        logits = self.semantic_head(sem_embed)
        if self.training:
            outputs["semantic_logits"] = logits
        else:
            # Viewer-friendly output: force a 1-channel semantic id map so it appears
            # in Output type and avoids PCA colormap on high-dim logits.
            probs = F.softmax(logits, dim=-1)
            sem_ids = torch.argmax(probs, dim=-1, keepdim=True).to(torch.float32)
            denom = float(max(self.config.num_classes - 1, 1))
            outputs["semantic"] = sem_ids / denom
            outputs["semantic_confidence"] = probs.max(dim=-1, keepdim=True).values
            if self.config.enable_object_culling and "rgb" in outputs:
                fg_mask = (sem_ids != float(self.config.bg_class_id)).to(torch.float32)
                outputs["semantic_fg_mask"] = fg_mask
                outputs["rgb_object_culled"] = outputs["rgb"] * fg_mask
                if "depth" in outputs:
                    outputs["depth_object_culled"] = outputs["depth"] * fg_mask
        outputs["semantic_accumulation"] = sem_alpha
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict=metrics_dict)
        stage1_alpha, stage2_alpha = self._two_stage_mix()

        if self.config.freeze_geometry:
            loss_dict["main_loss"] = loss_dict["main_loss"] * 0.0
            if "scale_reg" in loss_dict:
                loss_dict["scale_reg"] = loss_dict["scale_reg"] * 0.0

        if self.config.use_rgb_loss is False:
            loss_dict["main_loss"] = loss_dict["main_loss"] * 0.0

        if "semantic_logits" not in outputs or "semantic_mask" not in batch:
            loss_dict["semantic_loss"] = torch.zeros((), device=self.device)
            return loss_dict

        semantic_mask = batch["semantic_mask"].to(self.device).long()
        if semantic_mask.ndim == 3 and semantic_mask.shape[-1] == 1:
            semantic_mask = semantic_mask[..., 0]

        semantic_mask_ds = self._downscale_if_required(semantic_mask.to(torch.float32).unsqueeze(-1))
        semantic_mask_ds = semantic_mask_ds.squeeze(-1).round().long()

        logits = outputs["semantic_logits"]
        logits_nchw = logits.permute(2, 0, 1).unsqueeze(0)
        target_bhw = semantic_mask_ds.unsqueeze(0)

        # Final guard: force target spatial size to match CE input spatial size.
        in_h, in_w = logits_nchw.shape[-2], logits_nchw.shape[-1]
        tgt_h, tgt_w = target_bhw.shape[-2], target_bhw.shape[-1]
        if (in_h, in_w) != (tgt_h, tgt_w):
            target_bhw = F.interpolate(
                target_bhw.unsqueeze(1).to(torch.float32),
                size=(in_h, in_w),
                mode="nearest",
            ).squeeze(1).long()

        probs = F.softmax(logits_nchw, dim=1)
        valid_mask = target_bhw != self.config.semantic_ignore_index
        fg_mask = valid_mask & (target_bhw != self.config.bg_class_id)
        bg_id = int(self.config.bg_class_id)
        fg_prob = 1.0 - probs[:, bg_id, :, :]
        binary_fg_target = fg_mask.to(torch.float32)
        binary_bg_target = (~fg_mask & valid_mask).to(torch.float32)

        binary_loss = torch.zeros((), device=self.device)
        if valid_mask.any():
            binary_loss = F.binary_cross_entropy(fg_prob[valid_mask], binary_fg_target[valid_mask])
        fg_dice_loss = torch.zeros((), device=self.device)
        if valid_mask.any():
            valid_fg_prob = fg_prob[valid_mask]
            valid_fg_target = binary_fg_target[valid_mask]
            inter = (valid_fg_prob * valid_fg_target).sum()
            denom = valid_fg_prob.sum() + valid_fg_target.sum()
            fg_dice_loss = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)

        if valid_mask.any() and fg_mask.any():
            fg_target = target_bhw.clone()
            fg_target[fg_target == bg_id] = self.config.semantic_ignore_index
            fg_target = fg_target - 1
            fg_target[fg_target < 0] = self.config.semantic_ignore_index
            fg_logits = logits_nchw[:, 1:, :, :]
            fg_weights = self._get_fg_class_weights(logits_nchw.device)
            fine_loss = F.cross_entropy(
                fg_logits,
                fg_target,
                ignore_index=self.config.semantic_ignore_index,
                weight=fg_weights,
            )
        else:
            fine_loss = torch.zeros((), device=self.device)

        bg_keep_loss = torch.zeros((), device=self.device)
        if valid_mask.any():
            bg_keep_loss = F.binary_cross_entropy(probs[:, bg_id, :, :][valid_mask], binary_bg_target[valid_mask])

        stage1_loss = binary_loss + self.config.fg_dice_weight * fg_dice_loss + 0.5 * bg_keep_loss
        stage2_loss = fine_loss + self.config.stage2_binary_keep_weight * bg_keep_loss

        loss_dict["semantic_binary_loss"] = (1.0 - stage2_alpha) * self.config.lambda_sem * stage1_loss
        loss_dict["semantic_fine_loss"] = stage2_alpha * self.config.lambda_sem * stage2_loss

        self._log_semantic_stats(logits_nchw, target_bhw, metrics_dict=metrics_dict)
        return loss_dict
