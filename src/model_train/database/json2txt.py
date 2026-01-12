#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert LabelMe-style JSON annotations to YOLO-OBB txt labels.

Input directory contains images (*.jpg/*.png/...) and matching LabelMe JSON files (*.json) mixed together.
Output directory will have:

  output/
    images/  # copied/moved images
    labels/  # YOLO-OBB txt labels: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)

Supported LabelMe shapes:
  - polygon: points: [[x, y], ...]
  - rectangle: points: [[x1, y1], [x2, y2]] (top-left/bottom-right or any 2 corners)

If a polygon has more than 4 points, we fit a minimum-area rotated rectangle (cv2.minAreaRect) when OpenCV is
available; otherwise we error out (because YOLO-OBB needs 4 corners).
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class LabelmeObject:
    cls: int
    points_xy: List[Tuple[float, float]]  # image pixel coords


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _order_points_clockwise(pts: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Order points clockwise around centroid to avoid self-intersections from arbitrary order."""
    if len(pts) != 4:
        raise ValueError(f"Expected 4 points, got {len(pts)}")
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0

    def angle(p: Tuple[float, float]) -> float:
        return math.atan2(p[1] - cy, p[0] - cx)

    ordered = sorted(pts, key=angle)  # CCW order
    # Convert to clockwise by reversing (keeping first point stable)
    return [ordered[0], ordered[3], ordered[2], ordered[1]]


def _rect_from_two_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Axis-aligned rectangle corners from two points."""
    x1, y1 = p1
    x2, y2 = p2
    xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def _min_area_rect(pts: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Fit a minimum-area rotated rectangle to N>=3 points. Requires OpenCV."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Polygon has != 4 points; need OpenCV to fit minAreaRect. "
            "Install with: pip install opencv-python"
        ) from e

    arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(arr)
    box = cv2.boxPoints(rect)  # (4, 2)
    return [(float(x), float(y)) for x, y in box]


def _parse_labelme_json(
    json_path: Path, *, class_map: Dict[str, int], strict: bool = False
) -> Tuple[Path, int, int, List[LabelmeObject]]:
    """
    Returns: (image_path_relative, width, height, objects)
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img_w = int(data.get("imageWidth") or 0)
    img_h = int(data.get("imageHeight") or 0)
    image_path = Path(data.get("imagePath") or (json_path.stem + ".jpg"))

    objects: List[LabelmeObject] = []
    for shape in data.get("shapes", []) or []:
        label = str(shape.get("label", "0"))
        if label.isdigit():
            cls_id = int(label)
        else:
            if label not in class_map:
                if strict:
                    raise ValueError(f"Unknown class label '{label}' in {json_path}")
                class_map[label] = len(class_map)
            cls_id = class_map[label]

        shape_type = str(shape.get("shape_type") or "polygon").lower()
        points = shape.get("points") or []
        pts_xy = [(float(p[0]), float(p[1])) for p in points]

        if shape_type == "rectangle":
            if len(pts_xy) != 2:
                if strict:
                    raise ValueError(f"rectangle expects 2 points, got {len(pts_xy)} in {json_path}")
                continue
            rect_pts = _rect_from_two_points(pts_xy[0], pts_xy[1])
            objects.append(LabelmeObject(cls=cls_id, points_xy=rect_pts))
        else:
            # polygon (or unknown -> treat as polygon)
            if len(pts_xy) < 3:
                if strict:
                    raise ValueError(f"polygon expects >=3 points, got {len(pts_xy)} in {json_path}")
                continue
            if len(pts_xy) != 4:
                rect_pts = _min_area_rect(pts_xy)
            else:
                rect_pts = pts_xy
            objects.append(LabelmeObject(cls=cls_id, points_xy=list(rect_pts)))

    return image_path, img_w, img_h, objects


def _find_image_for_json(input_dir: Path, image_path_from_json: Path, json_stem: str) -> Optional[Path]:
    # 1) Prefer imagePath from json if it exists relative to input_dir
    cand = input_dir / image_path_from_json
    if cand.exists():
        return cand
    # 2) Fallback to same stem with common extensions
    for ext in IMG_EXTS:
        cand2 = input_dir / f"{json_stem}{ext}"
        if cand2.exists():
            return cand2
    return None


def _write_yolo_obb_txt(
    out_txt: Path, *, objects: Sequence[LabelmeObject], img_w: int, img_h: int
) -> None:
    lines: List[str] = []
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: w={img_w}, h={img_h}. JSON missing imageWidth/imageHeight?")

    for obj in objects:
        pts = obj.points_xy
        if len(pts) != 4:
            raise ValueError(f"Expected 4 points after conversion, got {len(pts)}")
        pts = _order_points_clockwise(pts)
        norm = []
        for (x, y) in pts:
            norm.append(_clamp01(x / img_w))
            norm.append(_clamp01(y / img_h))
        lines.append(f"{obj.cls} " + " ".join(f"{v:.6f}" for v in norm))

    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _iter_images(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def convert_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    move: bool = False,
    include_unlabeled_images: bool = True,
    strict: bool = False,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    class_map: Dict[str, int] = {}
    json_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])

    # Convert labels + copy/move images referenced by json
    seen_images: set[Path] = set()
    converted = 0
    skipped = 0
    for jp in json_files:
        try:
            img_rel, img_w, img_h, objects = _parse_labelme_json(jp, class_map=class_map, strict=strict)
            img_path = _find_image_for_json(input_dir, img_rel, jp.stem)
            if img_path is None:
                skipped += 1
                continue

            seen_images.add(img_path.resolve())
            # copy/move image
            dst_img = out_images / img_path.name
            if move:
                shutil.move(str(img_path), str(dst_img))
            else:
                shutil.copy2(str(img_path), str(dst_img))

            # write txt
            out_txt = out_labels / f"{img_path.stem}.txt"
            _write_yolo_obb_txt(out_txt, objects=objects, img_w=img_w, img_h=img_h)
            converted += 1

            # optionally move json out (to keep input clean)
            if move:
                jp.unlink(missing_ok=True)
        except Exception:
            if strict:
                raise
            skipped += 1

    # Optionally also copy/move images without labels
    if include_unlabeled_images:
        for img in _iter_images(input_dir):
            img_r = img.resolve()
            if img_r in seen_images:
                continue
            dst_img = out_images / img.name
            if move:
                shutil.move(str(img), str(dst_img))
            else:
                shutil.copy2(str(img), str(dst_img))
            # create empty label file
            (out_labels / f"{img.stem}.txt").write_text("", encoding="utf-8")

    # Write classes mapping if we saw non-numeric labels
    if class_map:
        # If user labels were numeric strings, class_map would remain empty; this is only for string labels.
        classes_path = output_dir / "classes.txt"
        inv = sorted(class_map.items(), key=lambda kv: kv[1])
        classes_path.write_text("\n".join([name for name, _ in inv]) + "\n", encoding="utf-8")

    print(f"[OK] input:  {input_dir}")
    print(f"[OK] output: {output_dir}")
    print(f"[OK] converted json: {converted}, skipped json: {skipped}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="把 '照片+LabelMe JSON' 混合目录转换为 YOLO-OBB (xyxyxyxy) 的 images/labels 结构。"
    )
    parser.add_argument(
        "input_dir",
        help="输入目录（照片和 *.json 混放的文件夹）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认: <input_dir>_yolo_obb）",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="移动文件而不是复制（会清空输入目录里的图片/json）",
    )
    parser.add_argument(
        "--no-unlabeled",
        action="store_true",
        help="不复制没有对应 JSON 的图片（默认会复制并生成空 txt）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：遇到异常立即报错停止；否则跳过有问题的文件",
    )
    args = parser.parse_args(argv)

    in_dir = Path(args.input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else Path(str(in_dir) + "_obb")
    convert_folder(
        in_dir,
        out_dir,
        move=args.move,
        include_unlabeled_images=not args.no_unlabeled,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()

