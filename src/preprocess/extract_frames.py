"""Stage 1 preprocessing: extract high-quality frames from a video.

This script standardizes frame naming to keep downstream image/mask
alignment deterministic (e.g., frame_000001.jpg).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2


@dataclass
class ExtractionSummary:
    input_video: str
    output_dir: str
    source_fps: float
    source_frame_count: int
    source_duration_sec: float
    requested_fps: Optional[float]
    saved_frames: int
    start_sec: float
    end_sec: Optional[float]
    resize_long_edge: Optional[int]
    image_ext: str


def _validate_args(args: argparse.Namespace) -> None:
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.start_sec < 0:
        raise ValueError("--start-sec must be >= 0")
    if args.end_sec is not None and args.end_sec <= args.start_sec:
        raise ValueError("--end-sec must be larger than --start-sec")
    if args.resize_long_edge is not None and args.resize_long_edge <= 0:
        raise ValueError("--resize-long-edge must be > 0")
    if args.jpg_quality < 1 or args.jpg_quality > 100:
        raise ValueError("--jpg-quality must be in [1, 100]")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")


def _ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    has_files = any(output_dir.iterdir())
    if has_files and not overwrite:
        raise FileExistsError(
            f"Output dir is not empty: {output_dir}. "
            "Use --overwrite to continue."
        )


def _resize_keep_aspect(image, long_edge: int):
    h, w = image.shape[:2]
    current_long_edge = max(h, w)
    if current_long_edge <= long_edge:
        return image
    scale = long_edge / float(current_long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _should_save_frame(
    frame_idx: int,
    timestamp_sec: float,
    source_fps: float,
    stride: int,
    target_fps: Optional[float],
    next_sample_time: float,
) -> tuple[bool, float]:
    if frame_idx % stride != 0:
        return False, next_sample_time

    if target_fps is None:
        return True, next_sample_time

    # Time-based sampling is more stable than fixed frame steps for variable FPS.
    if timestamp_sec + (0.5 / max(source_fps, 1e-6)) >= next_sample_time:
        return True, next_sample_time + (1.0 / target_fps)
    return False, next_sample_time


def extract_frames(args: argparse.Namespace) -> ExtractionSummary:
    _validate_args(args)
    _ensure_output_dir(args.output_dir, args.overwrite)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = 30.0
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_duration_sec = (
        source_frame_count / source_fps if source_frame_count > 0 else 0.0
    )

    end_sec = args.end_sec if args.end_sec is not None else source_duration_sec
    if end_sec <= args.start_sec:
        cap.release()
        raise ValueError("Effective end time must be greater than start time.")

    start_frame = max(0, int(args.start_sec * source_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved = 0
    next_sample_time = args.start_sec
    metadata = []

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality]
    file_digits = max(args.zero_padding, 1)
    ext = args.image_ext.lower().lstrip(".")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_sec = frame_idx / source_fps
        if timestamp_sec >= end_sec:
            break

        should_save, next_sample_time = _should_save_frame(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            source_fps=source_fps,
            stride=args.stride,
            target_fps=args.fps,
            next_sample_time=next_sample_time,
        )

        if should_save:
            if args.resize_long_edge is not None:
                frame = _resize_keep_aspect(frame, args.resize_long_edge)

            out_name = f"{args.prefix}_{saved:0{file_digits}d}.{ext}"
            out_path = args.output_dir / out_name
            write_ok = cv2.imwrite(str(out_path), frame, encode_params)
            if not write_ok:
                cap.release()
                raise RuntimeError(f"Failed to write frame: {out_path}")

            metadata.append(
                {
                    "saved_index": saved,
                    "source_frame_index": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 6),
                    "file_name": out_name,
                }
            )
            saved += 1

            if args.max_frames is not None and saved >= args.max_frames:
                break

        frame_idx += 1

    cap.release()

    summary = ExtractionSummary(
        input_video=str(args.video),
        output_dir=str(args.output_dir),
        source_fps=source_fps,
        source_frame_count=source_frame_count,
        source_duration_sec=source_duration_sec,
        requested_fps=args.fps,
        saved_frames=saved,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        resize_long_edge=args.resize_long_edge,
        image_ext=ext,
    )

    if args.write_metadata:
        with (args.output_dir / "frames_meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": asdict(summary),
                    "frames": metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frames for Geo-Semantic Splatting Stage 1."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save extracted images.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target sampling FPS. If not set, keep source frame rate.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Save every N-th source frame before FPS filtering.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start extraction at this timestamp (seconds).",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Stop extraction at this timestamp (seconds).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to save.",
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=None,
        help="Resize so long edge equals this value while keeping aspect ratio.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality in [1, 100].",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Output file name prefix.",
    )
    parser.add_argument(
        "--zero-padding",
        type=int,
        default=6,
        help="Zero-padding width for frame index.",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="jpg",
        help="Image extension (recommended: jpg).",
    )
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Write frames_meta.json with index/timestamp mapping.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing to a non-empty output directory.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = extract_frames(args)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
