"""Stage 1 preprocessing: SAM2 video inference for consistent object IDs.

The script expects extracted frames (e.g. frame_000000.jpg) and writes:
1) per-frame ID mask with the same file stem (e.g. frame_000000.png)
2) metadata JSON containing object/frame mapping
3) optional per-object binary masks and overlays
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class PromptAnnotation:
    obj_id: int
    frame_idx: int
    points: Optional[np.ndarray]
    labels: Optional[np.ndarray]
    box: Optional[np.ndarray]
    label_name: Optional[str] = None


@dataclass
class InferenceSummary:
    images_dir: str
    output_dir: str
    total_input_frames: int
    propagated_frames: int
    object_ids: List[int]
    threshold: float
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM2 video propagation to generate per-frame object ID masks."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Input frames dir.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root dir.")
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        required=True,
        help="Path to SAM2 checkpoint file.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="SAM2 model config name/path used by build_sam2_video_predictor.",
    )
    parser.add_argument(
        "--prompts-json",
        type=Path,
        required=True,
        help=(
            "Prompt definition JSON. Expected key: annotations. "
            "Each annotation should include obj_id, frame_idx and points and/or box."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Logit threshold used to binarize SAM2 mask logits.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device, e.g. cuda or cpu. Default: auto detect.",
    )
    parser.add_argument(
        "--save-object-masks",
        action="store_true",
        help="Also save per-object binary masks in output_dir/objects/obj_xxxxx/.",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save colored overlay previews in output_dir/overlays/.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing to non-empty output subdirectories.",
    )
    return parser.parse_args()


def _ensure_sam2_import():
    try:
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Failed to import SAM2. Install it first, then retry.\n"
            "Expected import: from sam2.build_sam import build_sam2_video_predictor"
        ) from exc
    return build_sam2_video_predictor


def _resolve_model_config_name(raw_config: str) -> str:
    """Normalize user input to SAM2 Hydra config name.

    Examples:
    - sam2_hiera_l -> configs/sam2/sam2_hiera_l.yaml
    - sam2_hiera_l.yaml -> configs/sam2/sam2_hiera_l.yaml
    - sam2.1_hiera_l -> configs/sam2.1/sam2.1_hiera_l.yaml
    - configs/sam2/sam2_hiera_l.yaml -> unchanged
    """
    cfg = str(raw_config).strip().replace("\\", "/")
    if not cfg:
        raise ValueError("--model-config cannot be empty.")

    if cfg.startswith("configs/"):
        return cfg

    name = cfg
    if name.endswith(".yaml"):
        name = name[:-5]

    # sam2.1 variants
    if name.startswith("sam2.1_"):
        return f"configs/sam2.1/{name}.yaml"

    # default sam2 variants
    if name.startswith("sam2_hiera_"):
        return f"configs/sam2/{name}.yaml"

    # Fallback: keep as-is (maybe user passes custom hydra config name).
    return cfg


def _list_images(images_dir: Path, max_frames: Optional[int]) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    images.sort(key=lambda p: p.name)
    if not images:
        raise ValueError(f"No images found in {images_dir}")
    if max_frames is not None:
        images = images[:max_frames]
    return images


def _safe_prepare_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output dir is not empty: {path}. Use --overwrite to continue.")


def _parse_points(raw_points: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if raw_points is None:
        return None, None
    if not isinstance(raw_points, list) or len(raw_points) == 0:
        raise ValueError("points must be a non-empty list when provided.")

    coords: List[List[float]] = []
    labels: List[int] = []
    for item in raw_points:
        if not isinstance(item, (list, tuple)):
            raise ValueError("Each point must be [x, y, label] or [x, y].")
        if len(item) == 3:
            x, y, lbl = item
        elif len(item) == 2:
            x, y = item
            lbl = 1
        else:
            raise ValueError("Point format must be [x, y, label] or [x, y].")
        coords.append([float(x), float(y)])
        labels.append(int(lbl))

    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _parse_box(raw_box: Any) -> Optional[np.ndarray]:
    if raw_box is None:
        return None
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise ValueError("box must be [x1, y1, x2, y2].")
    return np.asarray([float(v) for v in raw_box], dtype=np.float32)


def _load_prompt_annotations(prompts_json: Path) -> List[PromptAnnotation]:
    if not prompts_json.exists():
        raise FileNotFoundError(f"prompts json not found: {prompts_json}")
    payload = json.loads(prompts_json.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and "annotations" in payload:
        records = payload["annotations"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("prompts json must be a list or dict with key 'annotations'.")

    annotations: List[PromptAnnotation] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"annotations[{idx}] must be an object.")

        try:
            obj_id = int(record["obj_id"])
            frame_idx = int(record["frame_idx"])
        except KeyError as exc:
            raise ValueError(
                f"annotations[{idx}] missing required key: {exc.args[0]}"
            ) from exc

        points, labels = _parse_points(record.get("points"))
        box = _parse_box(record.get("box"))
        if points is None and box is None:
            raise ValueError(
                f"annotations[{idx}] must provide points and/or box for obj_id={obj_id}."
            )

        annotations.append(
            PromptAnnotation(
                obj_id=obj_id,
                frame_idx=frame_idx,
                points=points,
                labels=labels,
                box=box,
                label_name=record.get("label_name"),
            )
        )

    if not annotations:
        raise ValueError("No annotations found in prompts json.")
    return annotations


def _color_from_id(obj_id: int) -> Tuple[int, int, int]:
    # Stable pseudo-random RGB color from object id.
    rng = np.random.default_rng(seed=obj_id * 9973 + 17)
    color = rng.integers(30, 255, size=(3,), dtype=np.int32).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def _apply_prompt_with_compat(
    predictor: Any,
    inference_state: Any,
    ann: PromptAnnotation,
) -> Tuple[Any, Iterable[int], Any]:
    if not hasattr(predictor, "add_new_points_or_box"):
        raise AttributeError(
            "SAM2 predictor is missing add_new_points_or_box. "
            "Please check your SAM2 version."
        )

    kwargs: Dict[str, Any] = {
        "inference_state": inference_state,
        "frame_idx": ann.frame_idx,
        "obj_id": ann.obj_id,
    }
    if ann.points is not None:
        kwargs["points"] = ann.points
        kwargs["labels"] = ann.labels
    if ann.box is not None:
        kwargs["box"] = ann.box

    return predictor.add_new_points_or_box(**kwargs)


def _compose_id_mask(
    out_obj_ids: Iterable[int],
    out_mask_logits: torch.Tensor,
    threshold: float,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    masks_by_obj: Dict[int, np.ndarray] = {}
    out_obj_ids = list(int(v) for v in out_obj_ids)
    logits = out_mask_logits
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().float().cpu()
    else:
        logits = torch.as_tensor(logits).detach().float().cpu()

    if logits.ndim == 4:
        # [N, 1, H, W]
        logits = logits[:, 0]
    if logits.ndim != 3:
        raise ValueError(f"Unexpected mask logits shape: {tuple(logits.shape)}")

    h, w = int(logits.shape[-2]), int(logits.shape[-1])
    id_mask = np.zeros((h, w), dtype=np.uint16)

    for idx, obj_id in enumerate(out_obj_ids):
        binary = (logits[idx].numpy() > threshold).astype(np.uint8)
        masks_by_obj[obj_id] = binary
        id_mask[binary > 0] = np.uint16(obj_id)

    return id_mask, masks_by_obj


def _save_overlay(image_path: Path, id_mask: np.ndarray, overlay_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image for overlay: {image_path}")

    overlay = image.copy()
    unique_ids = [int(v) for v in np.unique(id_mask) if int(v) > 0]
    for obj_id in unique_ids:
        region = id_mask == obj_id
        color = _color_from_id(obj_id)
        overlay[region] = (
            0.45 * overlay[region] + 0.55 * np.array(color, dtype=np.float32)
        ).astype(np.uint8)

    blend = cv2.addWeighted(image, 0.5, overlay, 0.5, 0.0)
    ok = cv2.imwrite(str(overlay_path), blend)
    if not ok:
        raise RuntimeError(f"Failed to save overlay: {overlay_path}")


def run_inference(args: argparse.Namespace) -> InferenceSummary:
    image_paths = _list_images(args.images_dir, args.max_frames)
    annotations = _load_prompt_annotations(args.prompts_json)

    id_masks_dir = args.output_dir / "id_masks"
    meta_dir = args.output_dir / "meta"
    _safe_prepare_dir(id_masks_dir, args.overwrite)
    _safe_prepare_dir(meta_dir, args.overwrite)

    objects_dir = args.output_dir / "objects"
    overlays_dir = args.output_dir / "overlays"
    if args.save_object_masks:
        _safe_prepare_dir(objects_dir, args.overwrite)
    if args.save_overlays:
        _safe_prepare_dir(overlays_dir, args.overwrite)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    build_sam2_video_predictor = _ensure_sam2_import()
    resolved_model_config = _resolve_model_config_name(args.model_config)
    predictor = build_sam2_video_predictor(
        resolved_model_config,
        str(args.sam2_checkpoint),
        device=device,
    )
    inference_state = predictor.init_state(video_path=str(args.images_dir))
    if hasattr(predictor, "reset_state"):
        predictor.reset_state(inference_state)

    for ann in annotations:
        if ann.frame_idx < 0 or ann.frame_idx >= len(image_paths):
            raise IndexError(
                f"Annotation frame_idx out of range: {ann.frame_idx}, "
                f"valid=[0, {len(image_paths)-1}]"
            )
        _apply_prompt_with_compat(predictor, inference_state, ann)

    frame_records: List[Dict[str, Any]] = []
    all_obj_ids: set[int] = set()
    propagated_frames = 0

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        if out_frame_idx < 0 or out_frame_idx >= len(image_paths):
            continue

        image_path = image_paths[out_frame_idx]
        id_mask, masks_by_obj = _compose_id_mask(
            out_obj_ids=out_obj_ids,
            out_mask_logits=out_mask_logits,
            threshold=args.threshold,
        )
        all_obj_ids.update(masks_by_obj.keys())
        propagated_frames += 1

        mask_out_path = id_masks_dir / f"{image_path.stem}.png"
        ok = cv2.imwrite(str(mask_out_path), id_mask)
        if not ok:
            raise RuntimeError(f"Failed to write id mask: {mask_out_path}")

        if args.save_object_masks:
            for obj_id, binary in masks_by_obj.items():
                obj_dir = objects_dir / f"obj_{obj_id:05d}"
                obj_dir.mkdir(parents=True, exist_ok=True)
                obj_mask_path = obj_dir / f"{image_path.stem}.png"
                ok = cv2.imwrite(str(obj_mask_path), binary * 255)
                if not ok:
                    raise RuntimeError(f"Failed to write object mask: {obj_mask_path}")

        if args.save_overlays:
            overlay_out_path = overlays_dir / f"{image_path.stem}.jpg"
            _save_overlay(image_path, id_mask, overlay_out_path)

        frame_records.append(
            {
                "frame_idx": int(out_frame_idx),
                "image_name": image_path.name,
                "id_mask_name": mask_out_path.name,
                "object_ids": [int(v) for v in sorted(masks_by_obj.keys())],
            }
        )

    summary = InferenceSummary(
        images_dir=str(args.images_dir),
        output_dir=str(args.output_dir),
        total_input_frames=len(image_paths),
        propagated_frames=propagated_frames,
        object_ids=sorted(int(v) for v in all_obj_ids),
        threshold=float(args.threshold),
        device=str(device),
    )

    prompts_export = []
    for ann in annotations:
        prompts_export.append(
            {
                "obj_id": ann.obj_id,
                "frame_idx": ann.frame_idx,
                "label_name": ann.label_name,
                "points": ann.points.tolist() if ann.points is not None else None,
                "labels": ann.labels.tolist() if ann.labels is not None else None,
                "box": ann.box.tolist() if ann.box is not None else None,
            }
        )

    metadata = {
        "summary": asdict(summary),
        "prompts": prompts_export,
        "frames": sorted(frame_records, key=lambda x: x["frame_idx"]),
    }
    (meta_dir / "sam2_inference_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_inference(args)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
