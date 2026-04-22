"""Semantic-aware datamanager/dataset for Stage-2 training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Type, Union

import cv2
import numpy as np
import torch

from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset


class SemanticInputDataset(InputDataset):
    """Input dataset that additionally returns a per-pixel semantic id map."""

    exclude_batch_keys_from_device = ["image", "mask", "semantic_mask"]

    def __init__(
        self,
        dataparser_outputs,
        scale_factor: float = 1.0,
        *,
        semantic_masks_dir: Union[Path, str],
        semantic_mask_ext: str = ".png",
        strict_semantic_masks: bool = False,
        semantic_ignore_index: int = -100,
        ignore_bg_class: bool = False,
        bg_class_id: int = 0,
    ):
        super().__init__(dataparser_outputs=dataparser_outputs, scale_factor=scale_factor)
        self.semantic_masks_dir = Path(semantic_masks_dir)
        self.semantic_mask_ext = semantic_mask_ext
        self.strict_semantic_masks = strict_semantic_masks
        self.semantic_ignore_index = int(semantic_ignore_index)
        self.ignore_bg_class = bool(ignore_bg_class)
        self.bg_class_id = int(bg_class_id)

    def get_metadata(self, data: Dict) -> Dict:
        image_idx = int(data["image_idx"])
        image_path = self._dataparser_outputs.image_filenames[image_idx]
        mask_path = self.semantic_masks_dir / f"{image_path.stem}{self.semantic_mask_ext}"
        h, w = data["image"].shape[:2]

        if not mask_path.exists():
            if self.strict_semantic_masks:
                raise FileNotFoundError(f"Semantic mask not found for {image_path.name}: {mask_path}")
            semantic_mask = torch.full((h, w), self.semantic_ignore_index, dtype=torch.long)
            return {"semantic_mask": semantic_mask}

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read semantic mask: {mask_path}")

        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_np = np.asarray(mask, dtype=np.int64)
        if self.ignore_bg_class:
            mask_np[mask_np == self.bg_class_id] = self.semantic_ignore_index
        return {"semantic_mask": torch.from_numpy(mask_np)}


@dataclass
class SemanticDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: SemanticDatamanager)
    semantic_masks_dir: Optional[Path] = None
    semantic_mask_ext: str = ".png"
    strict_semantic_masks: bool = False
    semantic_ignore_index: int = -100
    ignore_bg_class: bool = False
    bg_class_id: int = 0


class SemanticDatamanager(FullImageDatamanager[SemanticInputDataset]):
    """Full-image datamanager that injects semantic id masks into each training batch."""

    config: SemanticDatamanagerConfig

    def __init__(
        self,
        config: SemanticDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self._semantic_masks_dir = Path(config.semantic_masks_dir) if config.semantic_masks_dir is not None else None
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )

    def _resolve_semantic_masks_dir(self) -> Path:
        if self._semantic_masks_dir is not None:
            return self._semantic_masks_dir
        return Path(self.config.data) / "sam2_out" / "id_masks"

    def create_train_dataset(self) -> SemanticInputDataset:
        return SemanticInputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            semantic_masks_dir=self._resolve_semantic_masks_dir(),
            semantic_mask_ext=self.config.semantic_mask_ext,
            strict_semantic_masks=self.config.strict_semantic_masks,
            semantic_ignore_index=self.config.semantic_ignore_index,
            ignore_bg_class=self.config.ignore_bg_class,
            bg_class_id=self.config.bg_class_id,
        )

    def create_eval_dataset(self) -> SemanticInputDataset:
        return SemanticInputDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            semantic_masks_dir=self._resolve_semantic_masks_dir(),
            semantic_mask_ext=self.config.semantic_mask_ext,
            strict_semantic_masks=self.config.strict_semantic_masks,
            semantic_ignore_index=self.config.semantic_ignore_index,
            ignore_bg_class=self.config.ignore_bg_class,
            bg_class_id=self.config.bg_class_id,
        )
