"""Interactive prompt annotation tool for SAM2 video inference.

Exports JSON format expected by src/preprocess/sam2_inference.py:
{
  "annotations": [
    {
      "obj_id": 1,
      "label_name": "chair",
      "frame_idx": 0,
      "points": [[x, y, label], ...],   # optional
      "box": [x1, y1, x2, y2]            # optional
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class PromptRecord:
    obj_id: int
    frame_idx: int
    label_name: Optional[str]
    points: List[List[float]]
    box: Optional[List[float]]

    def to_json(self) -> Dict:
        return {
            "obj_id": int(self.obj_id),
            "label_name": self.label_name,
            "frame_idx": int(self.frame_idx),
            "points": self.points if self.points else None,
            "box": self.box,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive tool to create prompts_json for SAM2 inference."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Input frames directory.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output prompts JSON path.")
    parser.add_argument(
        "--load-json",
        type=Path,
        default=None,
        help="Optional existing prompts JSON to continue editing.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Initial frame index.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="SAM2 Prompt Annotator",
        help="OpenCV window title.",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort(key=lambda p: p.name)
    if not images:
        raise ValueError(f"No images found in: {images_dir}")
    return images


def color_from_id(obj_id: int) -> Tuple[int, int, int]:
    # BGR for OpenCV.
    base = (obj_id * 11939 + 17) % 255
    return int((base + 40) % 255), int((base * 2 + 80) % 255), int((base * 3 + 120) % 255)


class PromptAnnotator:
    def __init__(
        self,
        image_paths: List[Path],
        output_json: Path,
        window_name: str,
        start_frame: int = 0,
    ) -> None:
        self.image_paths = image_paths
        self.output_json = output_json
        self.window_name = window_name
        self.frame_idx = max(0, min(start_frame, len(image_paths) - 1))

        self.mode = "point"  # point | box
        self.current_obj_id = 1
        self.current_label_name: Optional[str] = "object_1"
        self.records: Dict[Tuple[int, int], PromptRecord] = {}

        self.dragging = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.temp_box: Optional[List[float]] = None

    def key(self) -> Tuple[int, int]:
        return self.current_obj_id, self.frame_idx

    def get_or_create_current(self) -> PromptRecord:
        k = self.key()
        if k not in self.records:
            self.records[k] = PromptRecord(
                obj_id=self.current_obj_id,
                frame_idx=self.frame_idx,
                label_name=self.current_label_name,
                points=[],
                box=None,
            )
        rec = self.records[k]
        if self.current_label_name and (not rec.label_name):
            rec.label_name = self.current_label_name
        return rec

    def sync_current_from_records(self) -> None:
        rec = self.records.get(self.key())
        if rec is not None:
            self.current_label_name = rec.label_name

    def load_existing(self, json_path: Path) -> None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("annotations", [])
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError("Invalid prompts json format.")

        for item in items:
            if not isinstance(item, dict):
                continue
            obj_id = int(item["obj_id"])
            frame_idx = int(item["frame_idx"])
            if frame_idx < 0 or frame_idx >= len(self.image_paths):
                continue
            points = item.get("points") or []
            points = [[float(p[0]), float(p[1]), int(p[2])] for p in points]
            box = item.get("box")
            if box is not None:
                box = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            rec = PromptRecord(
                obj_id=obj_id,
                frame_idx=frame_idx,
                label_name=item.get("label_name"),
                points=points,
                box=box,
            )
            self.records[(obj_id, frame_idx)] = rec

    def save_json(self) -> None:
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        annotations = [r.to_json() for r in sorted(self.records.values(), key=lambda x: (x.frame_idx, x.obj_id))]
        payload = {"annotations": annotations}
        self.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[saved] {self.output_json} ({len(annotations)} annotations)")

    def delete_current_record(self) -> None:
        self.records.pop(self.key(), None)

    def clear_current_content(self) -> None:
        rec = self.get_or_create_current()
        rec.points = []
        rec.box = None
        self.temp_box = None

    def add_point(self, x: int, y: int, label: int) -> None:
        rec = self.get_or_create_current()
        rec.points.append([float(x), float(y), int(label)])

    def set_box(self, x1: int, y1: int, x2: int, y2: int) -> None:
        rec = self.get_or_create_current()
        xx1, xx2 = sorted([int(x1), int(x2)])
        yy1, yy2 = sorted([int(y1), int(y2)])
        rec.box = [float(xx1), float(yy1), float(xx2), float(yy2)]
        self.temp_box = None

    def undo_point(self) -> None:
        rec = self.get_or_create_current()
        if rec.points:
            rec.points.pop()

    def on_mouse(self, event, x, y, flags, param) -> None:
        if self.mode == "point":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.add_point(x, y, 1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.add_point(x, y, 0)
            return

        # box mode
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.temp_box = [float(x), float(y), float(x), float(y)]
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.drag_start is not None:
            sx, sy = self.drag_start
            self.temp_box = [float(sx), float(sy), float(x), float(y)]
        elif event == cv2.EVENT_LBUTTONUP and self.dragging and self.drag_start is not None:
            sx, sy = self.drag_start
            self.set_box(sx, sy, x, y)
            self.dragging = False
            self.drag_start = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            rec = self.get_or_create_current()
            rec.box = None
            self.temp_box = None

    def draw(self) -> None:
        frame_path = self.image_paths[self.frame_idx]
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {frame_path}")

        # Draw all annotations on this frame.
        for (obj_id, fidx), rec in self.records.items():
            if fidx != self.frame_idx:
                continue
            color = color_from_id(obj_id)
            thickness = 2 if obj_id == self.current_obj_id else 1

            if rec.box is not None:
                x1, y1, x2, y2 = [int(v) for v in rec.box]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            for px, py, plabel in rec.points:
                center = (int(px), int(py))
                if int(plabel) == 1:
                    cv2.circle(image, center, 4, (0, 255, 0), -1)
                else:
                    cv2.circle(image, center, 4, (0, 0, 255), -1)
                if obj_id == self.current_obj_id:
                    cv2.circle(image, center, 7, color, 1)

            txt = f"id={obj_id} {rec.label_name or ''}".strip()
            cv2.putText(
                image,
                txt,
                (8, 24 + obj_id * 18 % 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if self.temp_box is not None:
            x1, y1, x2, y2 = [int(v) for v in self.temp_box]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1)

        rec = self.records.get(self.key())
        current_points = len(rec.points) if rec else 0
        has_box = "Y" if (rec and rec.box is not None) else "N"
        info_lines = [
            f"frame {self.frame_idx + 1}/{len(self.image_paths)} : {frame_path.name}",
            f"obj_id={self.current_obj_id} label={self.current_label_name or 'None'} mode={self.mode}",
            f"current points={current_points} box={has_box} total annotations={len(self.records)}",
            "keys: [/] prev/next frame, o set object, p point, b box, z undo point",
            "keys: x clear current, r remove current record, s save, q save+quit, ESC quit",
            "mouse(point): L positive, R negative | mouse(box): drag L draw box, R clear box",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(
                image,
                line,
                (10, image.shape[0] - 120 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(self.window_name, image)

    def prompt_set_object(self) -> None:
        raw_obj = input(f"obj_id [{self.current_obj_id}]: ").strip()
        if raw_obj:
            self.current_obj_id = int(raw_obj)
        raw_name = input(f"label_name [{self.current_label_name or ''}]: ").strip()
        if raw_name:
            self.current_label_name = raw_name
        self.sync_current_from_records()

    def run(self) -> None:
        print("== SAM2 Prompt Annotator ==")
        print("Press 'h' for terminal help.")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while True:
            self.draw()
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue
            if key == ord("h"):
                print(
                    "[help]\n"
                    "  [ / ] : prev / next frame\n"
                    "  o     : set current obj_id and label_name\n"
                    "  p/b   : point mode / box mode\n"
                    "  z     : undo last point\n"
                    "  x     : clear current points+box\n"
                    "  r     : remove current record(obj_id+frame_idx)\n"
                    "  s     : save json\n"
                    "  q     : save and quit\n"
                    "  ESC   : quit without save"
                )
            elif key == ord("["):
                self.frame_idx = max(0, self.frame_idx - 1)
                self.temp_box = None
                self.sync_current_from_records()
            elif key == ord("]"):
                self.frame_idx = min(len(self.image_paths) - 1, self.frame_idx + 1)
                self.temp_box = None
                self.sync_current_from_records()
            elif key == ord("o"):
                self.prompt_set_object()
            elif key == ord("p"):
                self.mode = "point"
            elif key == ord("b"):
                self.mode = "box"
            elif key == ord("z"):
                self.undo_point()
            elif key == ord("x"):
                self.clear_current_content()
            elif key == ord("r"):
                self.delete_current_record()
            elif key == ord("s"):
                self.save_json()
            elif key == ord("q"):
                self.save_json()
                break
            elif key == 27:  # ESC
                break

        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    images = list_images(args.images_dir)
    annotator = PromptAnnotator(
        image_paths=images,
        output_json=args.output_json,
        window_name=args.window_name,
        start_frame=args.start_frame,
    )
    if args.load_json is not None and args.load_json.exists():
        annotator.load_existing(args.load_json)
        print(f"[loaded] {args.load_json}")
    annotator.run()


if __name__ == "__main__":
    main()

