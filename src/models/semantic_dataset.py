"""Semantic supervision data utilities for Stage 2 training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


def load_semantic_classes(path: Path) -> Dict[int, str]:
    """Load class id -> name mapping from json.

    Expected format:
    {
      "classes": {
        "0": "background",
        "1": "cup",
        ...
      }
    }
    """
    if not path.exists():
        raise FileNotFoundError(f"semantic classes file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("semantic classes json must include non-empty dict key 'classes'")
    out: Dict[int, str] = {}
    for k, v in classes.items():
        out[int(k)] = str(v)
    return dict(sorted(out.items(), key=lambda x: x[0]))


def load_semantic_mask(mask_path: Path, device: Optional[torch.device] = None) -> torch.Tensor:
    """Load semantic id mask from png into LongTensor [H, W]."""
    if not mask_path.exists():
        raise FileNotFoundError(f"mask not found: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {mask_path}")
    if mask.ndim == 3:
        # In case saved as RGB, use first channel.
        mask = mask[:, :, 0]
    if mask.dtype not in (np.uint8, np.uint16, np.int32):
        mask = mask.astype(np.int32)
    t = torch.from_numpy(mask.astype(np.int64))
    if device is not None:
        t = t.to(device)
    return t


def load_rgb_image(image_path: Path, device: Optional[torch.device] = None) -> torch.Tensor:
    """Load image as float tensor [H, W, 3] in [0, 1]."""
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img)
    if device is not None:
        t = t.to(device)
    return t


def resolve_pair_paths(
    images_dir: Path,
    semantic_masks_dir: Path,
    image_name: str,
    mask_ext: str = ".png",
) -> Tuple[Path, Path]:
    """Resolve aligned image/mask pair by same stem."""
    image_path = images_dir / image_name
    stem = Path(image_name).stem
    mask_path = semantic_masks_dir / f"{stem}{mask_ext}"
    return image_path, mask_path

