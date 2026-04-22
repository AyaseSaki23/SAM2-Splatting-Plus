"""Stage-2 Geo-Semantic training entrypoint.

This script reuses a finished Splatfacto checkpoint and performs semantic refinement
with SAM2 id masks, without rerunning Stage-1 RGB reconstruction from scratch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.scripts.train import main as ns_train_main

from src.models.semantic_datamanager import SemanticDatamanagerConfig
from src.models.semantic_splatfacto import SemanticSplatfactoModel, SemanticSplatfactoModelConfig


def estimate_class_weights(
    masks_dir: Path,
    num_classes: int,
    semantic_ignore_index: int,
    ignore_bg_class: bool = False,
    bg_class_id: int = 0,
    mask_ext: str = ".png",
    max_files: int = 400,
) -> tuple[float, ...]:
    files = sorted(masks_dir.glob(f"*{mask_ext}"))
    if not files:
        return tuple(1.0 for _ in range(num_classes))

    hist = np.zeros(num_classes, dtype=np.float64)
    for p in files[:max_files]:
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[:, :, 0]
        m = np.asarray(m, dtype=np.int64)
        if ignore_bg_class:
            m[m == bg_class_id] = semantic_ignore_index
        valid = m != semantic_ignore_index
        if not np.any(valid):
            continue
        m = m[valid]
        m = m[(m >= 0) & (m < num_classes)]
        if m.size == 0:
            continue
        u, c = np.unique(m, return_counts=True)
        hist[u] += c

    if hist.sum() <= 0:
        return tuple(1.0 for _ in range(num_classes))

    inv = 1.0 / np.maximum(hist, 1.0)
    inv = inv / inv.mean()
    inv = np.clip(inv, 0.1, 10.0)
    return tuple(float(x) for x in inv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-2 semantic refinement for Geo-Semantic Splatting.")
    parser.add_argument("--data", type=Path, required=True, help="Nerfstudio data root (contains transforms.json).")
    parser.add_argument("--load-checkpoint", type=Path, required=True, help="Base splatfacto ckpt path.")
    parser.add_argument(
        "--semantic-masks-dir",
        type=Path,
        default=None,
        help="Directory of SAM2 id masks (default: <data>/sam2_out/id_masks).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output root directory.",
    )
    parser.add_argument("--experiment-name", type=str, default="scene_01")
    parser.add_argument("--timestamp", type=str, default="{timestamp}")
    parser.add_argument(
        "--max-num-iterations",
        type=int,
        default=5000,
        help=(
            "Absolute global stop step (inclusive). Example: load step 29999 and set 50000 -> "
            "this run trains to step 50000 then stops."
        ),
    )
    parser.add_argument("--lambda-sem", type=float, default=0.1)
    parser.add_argument("--semantic-dim", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--semantic-lr", type=float, default=1e-3)
    parser.add_argument("--semantic-head-lr", type=float, default=1e-3)
    parser.add_argument("--lambda-mv-consistency", type=float, default=0.02)
    parser.add_argument("--lambda-geo-alignment", type=float, default=0.05)
    parser.add_argument("--lambda-fg-aux", type=float, default=0.1)
    parser.add_argument("--lambda-class-dist", type=float, default=0.05)
    parser.add_argument("--fg-boost-factor", type=float, default=5.0)
    parser.add_argument("--fg-dice-weight", type=float, default=0.1)
    parser.add_argument("--semantic-warmup-steps", type=int, default=1000)
    parser.add_argument("--semantic-reg-start-delay", type=int, default=500)
    parser.add_argument("--semantic-reg-warmup-steps", type=int, default=1000)
    parser.add_argument("--semantic-warmup-start-step", type=int, default=0)
    parser.add_argument("--disable-two-stage-semantic-training", action="store_true")
    parser.add_argument("--semantic-stage1-steps", type=int, default=1200)
    parser.add_argument("--semantic-stage2-ramp-steps", type=int, default=1200)
    parser.add_argument("--semantic-head-bg-bias-init", type=float, default=-2.0)
    parser.add_argument("--semantic-head-fg-bias-init", type=float, default=0.25)
    parser.add_argument("--geo-alignment-beta", type=float, default=20.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--bg-class-id", type=int, default=0)
    parser.add_argument("--disable-object-culling", action="store_true")
    parser.add_argument("--auto-class-weights", action="store_true")
    parser.add_argument("--ignore-bg-class", action="store_true")
    parser.add_argument("--semantic-stats-every", type=int, default=200)
    parser.add_argument("--no-semantic-stats-print", action="store_true")
    parser.add_argument("--semantic-stats-file", type=Path, default=None)
    parser.add_argument("--freeze-geometry", action="store_true", help="Freeze RGB geometry/color params.")
    parser.add_argument("--strict-semantic-masks", action="store_true")
    parser.add_argument("--semantic-ignore-index", type=int, default=-100)
    parser.add_argument("--vis", type=str, default="viewer", choices=["viewer", "tensorboard", "wandb", "comet"])
    parser.add_argument(
        "--quit-on-train-completion",
        action="store_true",
        help="Exit process immediately after training finishes (recommended when using viewer).",
    )
    parser.add_argument(
        "--keep-viewer-alive",
        action="store_true",
        help="If set, keep viewer process alive after training completion.",
    )
    parser.add_argument("--max-jobs", type=int, default=1, help="Set MAX_JOBS for CUDA JIT stability on Windows.")
    return parser


def get_checkpoint_step(ckpt_path: Path) -> int:
    loaded_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "step" not in loaded_state:
        raise KeyError(f"'step' not found in checkpoint: {ckpt_path}")
    return int(loaded_state["step"])


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("MAX_JOBS", str(args.max_jobs))

    if not args.load_checkpoint.exists():
        raise FileNotFoundError(f"--load-checkpoint not found: {args.load_checkpoint}")
    start_ckpt_step = get_checkpoint_step(args.load_checkpoint)
    # Trainer starts from (loaded_step + 1), stop step is inclusive.
    train_iterations = args.max_num_iterations - start_ckpt_step
    if train_iterations <= 0:
        raise ValueError(
            f"--max-num-iterations must be greater than loaded checkpoint step {start_ckpt_step}. "
            f"Got {args.max_num_iterations}."
        )

    cfg = method_configs["splatfacto"]
    cfg.method_name = "semantic-splatfacto"
    cfg.data = args.data
    cfg.output_dir = args.output_dir.resolve()
    cfg.experiment_name = args.experiment_name
    cfg.timestamp = args.timestamp
    cfg.max_num_iterations = train_iterations
    cfg.load_checkpoint = args.load_checkpoint
    cfg.vis = args.vis
    # Default to exiting after completion to avoid appearing 'stuck' in viewer loop.
    cfg.viewer.quit_on_train_completion = True
    if args.keep_viewer_alive:
        cfg.viewer.quit_on_train_completion = False
    elif args.quit_on_train_completion:
        cfg.viewer.quit_on_train_completion = True
    cfg.steps_per_eval_batch = 0
    cfg.steps_per_eval_image = max(100, min(1000, train_iterations))
    cfg.steps_per_save = max(500, min(2000, train_iterations))
    cfg.load_scheduler = False

    sem_dm = SemanticDatamanagerConfig()
    sem_dm.data = args.data
    sem_dm.dataparser = cfg.pipeline.datamanager.dataparser
    sem_dm.cache_images = "gpu"
    sem_dm.cache_images_type = "uint8"
    sem_dm.semantic_masks_dir = args.semantic_masks_dir or (args.data / "sam2_out" / "id_masks")
    sem_dm.semantic_mask_ext = ".png"
    sem_dm.strict_semantic_masks = args.strict_semantic_masks
    sem_dm.semantic_ignore_index = args.semantic_ignore_index
    sem_dm.ignore_bg_class = args.ignore_bg_class
    sem_dm.bg_class_id = args.bg_class_id
    cfg.pipeline.datamanager = sem_dm
    semantic_masks_dir = sem_dm.semantic_masks_dir
    if semantic_masks_dir is None or (not Path(semantic_masks_dir).exists()):
        raise FileNotFoundError(f"Semantic masks directory not found: {semantic_masks_dir}")

    sem_model = SemanticSplatfactoModelConfig()
    base_model = cfg.pipeline.model
    for key in vars(base_model):
        if key == "_target":
            continue
        setattr(sem_model, key, getattr(base_model, key))
    sem_model._target = SemanticSplatfactoModel
    sem_model.semantic_dim = args.semantic_dim
    sem_model.num_classes = args.num_classes
    sem_model.lambda_sem = args.lambda_sem
    sem_model.lambda_mv_consistency = args.lambda_mv_consistency
    sem_model.lambda_geo_alignment = args.lambda_geo_alignment
    sem_model.lambda_fg_aux = args.lambda_fg_aux
    sem_model.lambda_class_dist = args.lambda_class_dist
    sem_model.fg_boost_factor = args.fg_boost_factor
    sem_model.fg_dice_weight = args.fg_dice_weight
    sem_model.semantic_warmup_steps = args.semantic_warmup_steps
    sem_model.semantic_reg_start_delay = args.semantic_reg_start_delay
    sem_model.semantic_reg_warmup_steps = args.semantic_reg_warmup_steps
    sem_model.semantic_warmup_start_step = args.semantic_warmup_start_step
    sem_model.two_stage_semantic_training = not args.disable_two_stage_semantic_training
    sem_model.semantic_stage1_steps = args.semantic_stage1_steps
    sem_model.semantic_stage2_ramp_steps = args.semantic_stage2_ramp_steps
    sem_model.semantic_head_bg_bias_init = args.semantic_head_bg_bias_init
    sem_model.semantic_head_fg_bias_init = args.semantic_head_fg_bias_init
    sem_model.geo_alignment_beta = args.geo_alignment_beta
    sem_model.focal_gamma = args.focal_gamma
    sem_model.semantic_ignore_index = args.semantic_ignore_index
    sem_model.bg_class_id = args.bg_class_id
    sem_model.enable_object_culling = not args.disable_object_culling
    sem_model.semantic_stats_every = args.semantic_stats_every
    sem_model.semantic_stats_print = not args.no_semantic_stats_print
    if args.semantic_stats_file is not None:
        sem_model.semantic_stats_file = str(args.semantic_stats_file)
    else:
        sem_model.semantic_stats_file = str(
            args.output_dir.resolve() / args.experiment_name / "semantic-splatfacto" / "semantic_stats_latest.log"
        )
    sem_model.freeze_geometry = args.freeze_geometry
    if args.auto_class_weights:
        sem_model.class_weights = estimate_class_weights(
            masks_dir=semantic_masks_dir,
            num_classes=args.num_classes,
            semantic_ignore_index=args.semantic_ignore_index,
            ignore_bg_class=args.ignore_bg_class,
            bg_class_id=args.bg_class_id,
            mask_ext=sem_dm.semantic_mask_ext,
        )
        print(f"[Stage2] auto class weights: {sem_model.class_weights}")
    else:
        sem_model.class_weights = None
    cfg.pipeline.model = sem_model

    cfg.optimizers["semantic_features"] = {
        "optimizer": AdamOptimizerConfig(lr=args.semantic_lr, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=max(args.semantic_lr * 0.1, 1e-6),
            max_steps=train_iterations,
        ),
    }
    cfg.optimizers["semantic_head"] = {
        "optimizer": AdamOptimizerConfig(lr=args.semantic_head_lr, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=max(args.semantic_head_lr * 0.1, 1e-6),
            max_steps=train_iterations,
        ),
    }

    # Stage 1: only binary fg/bg learning. Stage 2: full multi-class refinement.
    if sem_model.two_stage_semantic_training:
        for group, base_lr in [("semantic_features", args.semantic_lr), ("semantic_head", args.semantic_head_lr)]:
            if group not in cfg.optimizers:
                continue
            cfg.optimizers[group]["optimizer"].lr = base_lr * 0.5
            scheduler = cfg.optimizers[group].get("scheduler")
            if scheduler is not None:
                scheduler.lr_final = max(base_lr * 0.05, 1e-6)

    print(
        "[Stage2] loaded_step=",
        start_ckpt_step,
        " target_stop_step=",
        args.max_num_iterations,
        " run_iterations=",
        train_iterations,
        " two_stage=",
        sem_model.two_stage_semantic_training,
    )

    if args.freeze_geometry:
        for group in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
            cfg.optimizers[group]["optimizer"].lr = 0.0
            cfg.optimizers[group]["scheduler"] = None

    ns_train_main(cfg)


if __name__ == "__main__":
    main()
