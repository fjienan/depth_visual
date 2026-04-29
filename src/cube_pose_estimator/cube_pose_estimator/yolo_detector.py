"""Two-stage YOLO detector used by cube_pose_node.

Stage-1: OBB model to localize target area.
Stage-2: Pose model to regress 4 corners on the warped patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from cube_pose_estimator.geometry_utils import GeometryUtils


@dataclass(frozen=True)
class CascadeDetectorConfig:
    """Runtime config for the two-stage cascade."""

    obb_model_path: str
    pose_model_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    device: str = ""
    pad_ratio: float = 1.2
    warp_size: Tuple[int, int] = (256, 256)
    imgsz_obb: int = 0
    imgsz_pose: int = 0


class YoloCascadeDetector:
    """YOLO OBB + YOLO Pose cascade detector."""

    def __init__(self, cfg: CascadeDetectorConfig):
        self._cfg = cfg
        self._obb_model = None
        self._pose_model = None
        self._load_models()

    def _load_models(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "ultralytics is required for YoloCascadeDetector. Install it with: pip install ultralytics"
            ) from exc

        obb_path = Path(self._cfg.obb_model_path)
        pose_path = Path(self._cfg.pose_model_path)
        if not obb_path.exists():
            raise FileNotFoundError(f"OBB model not found: {obb_path}")
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose model not found: {pose_path}")

        self._obb_model = YOLO(str(obb_path))
        self._pose_model = YOLO(str(pose_path))
        if self._cfg.device:
            self._obb_model.to(self._cfg.device)
            self._pose_model.to(self._cfg.device)

    def _build_predict_kwargs(self, imgsz: int) -> dict:
        kw = {
            "conf": float(self._cfg.conf_threshold),
            "iou": float(self._cfg.iou_threshold),
            "verbose": False,
        }
        if self._cfg.device:
            kw["device"] = self._cfg.device
        if int(imgsz) > 0:
            kw["imgsz"] = int(imgsz)
        return kw

    @staticmethod
    def _select_best_det_index(det_obj) -> int:
        if det_obj is None or len(det_obj) == 0:
            return -1
        try:
            conf = det_obj.conf
            if conf is not None and len(conf) > 0:
                return int(np.argmax(conf.detach().cpu().numpy()))
        except Exception:
            pass
        return 0

    def detect_face_corners(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return corners in image space ordered as [TL, TR, BR, BL], shape (4,2)."""
        if image_bgr is None or image_bgr.ndim != 3:
            return None

        try:
            # Stage 1: OBB
            obb_results = self._obb_model.predict(
                image_bgr, **self._build_predict_kwargs(self._cfg.imgsz_obb)
            )
            if len(obb_results) == 0 or obb_results[0].obb is None or len(obb_results[0].obb) == 0:
                return None

            obb_obj = obb_results[0].obb
            obb_idx = self._select_best_det_index(obb_obj)
            if obb_idx < 0:
                return None
            obb_xywhr = obb_obj.xywhr[obb_idx].detach().cpu().numpy().astype(np.float32)

            obb_points = GeometryUtils.get_dilated_box_points(
                obb_xywhr, pad_ratio=float(self._cfg.pad_ratio)
            )
            ordered_obb_points = GeometryUtils.order_points(obb_points)
            warp_res = GeometryUtils.warp_image(
                image_bgr, ordered_obb_points, tuple(map(int, self._cfg.warp_size))
            )

            # Stage 2: keypoints
            pose_results = self._pose_model.predict(
                warp_res.warped_image, **self._build_predict_kwargs(self._cfg.imgsz_pose)
            )
            if (
                len(pose_results) == 0
                or pose_results[0].keypoints is None
                or len(pose_results[0].keypoints) == 0
            ):
                return None

            kpts_obj = pose_results[0].keypoints
            kpt_idx = self._select_best_det_index(getattr(pose_results[0], "boxes", None))
            if kpt_idx < 0 or kpt_idx >= len(kpts_obj):
                kpt_idx = 0

            kpts_local = kpts_obj.xy[kpt_idx].detach().cpu().numpy().astype(np.float32)
            if kpts_local.shape[0] != 4:
                return None

            order_idx = GeometryUtils.order_points_indices(kpts_local[:, :2])
            kpts_local = kpts_local[order_idx]
            kpts_img = GeometryUtils.map_points_back(kpts_local, warp_res.transform_matrix)
            if kpts_img.shape != (4, 2):
                return None
            return kpts_img.astype(np.float32)
        except Exception:
            # Keep node runtime stable; caller treats None as detection failure.
            return None

