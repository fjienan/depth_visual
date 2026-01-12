"""ROS 2 node: detect a cube face corners and estimate cube center pose via PnP."""

from __future__ import annotations

from collections import deque
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PolygonStamped, PoseStamped, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from cube_pose_estimator.pnp_solver import CameraModel, PnPConfig, SquareFacePnPSolver
from cube_pose_estimator.yolo_detector import CascadeDetectorConfig, YoloCascadeDetector


class CubePoseEstimatorNode(Node):
    """Cube pose estimation node."""

    def __init__(self) -> None:
        super().__init__("cube_pose_estimator")

        self._bridge = CvBridge()
        self._tf_broadcaster = TransformBroadcaster(self)
        self._cap: Optional[cv2.VideoCapture] = None
        self._timer = None

        self._declare_parameters()
        self._load_parameters()

        self._detector = YoloCascadeDetector(self._det_cfg)
        self._pnp = SquareFacePnPSolver(self._camera, self._cube_size_mm, self._pnp_cfg)

        self._sub = None
        self._pub_pose = self.create_publisher(PoseStamped, "/cube_pose/pose", 10)
        self._pub_corners = self.create_publisher(PolygonStamped, "/cube_pose/face_corners_3d", 10)
        self._pub_vis = self.create_publisher(Image, "/cube_pose/visualization", 10)
        self._pub_markers = self.create_publisher(MarkerArray, self._markers_topic, 10)

        self._last_ts = time.perf_counter()
        self._fps = 0.0
        self._last_pnp_fail_log_ts = 0.0
        self._last_yolo_log_ts = 0.0
        self._last_pnp_ok_log_ts = 0.0
        self._corners_window = deque(maxlen=max(1, int(self._pnp_cfg.ba_window_size)))

        if self._input_mode == "topic":
            self._sub = self.create_subscription(Image, self._image_topic, self._on_image, 10)
        elif self._input_mode == "usb":
            self._start_usb_capture()
        else:
            raise ValueError(f"Unsupported input.mode: {self._input_mode}")

        self.get_logger().info("cube_pose_estimator node started.")

    def _declare_parameters(self) -> None:
        self.declare_parameter("input.mode", "topic")
        self.declare_parameter("image_topic", "/camera/color/image_raw")

        self.declare_parameter("usb.camera_id", 0)
        self.declare_parameter("usb.width", 1280)
        self.declare_parameter("usb.height", 720)
        self.declare_parameter("usb.fps", 30.0)
        self.declare_parameter("usb.backend", "any")
        # USB camera controls (best-effort; depends on driver/camera)
        self.declare_parameter("usb.auto_exposure", True)
        # OpenCV exposure units are backend-dependent (often log-scale or ms). We'll just pass through.
        self.declare_parameter("usb.exposure", -1.0)  # set only when auto_exposure=false and exposure>=0
        self.declare_parameter("usb.gain", -1.0)  # set only when gain>=0

        self.declare_parameter("camera.intrinsics.fx", 615.0)
        self.declare_parameter("camera.intrinsics.fy", 615.0)
        self.declare_parameter("camera.intrinsics.cx", 320.0)
        self.declare_parameter("camera.intrinsics.cy", 240.0)
        self.declare_parameter("camera.distortion", [0.0, 0.0, 0.0, 0.0, 0.0])

        self.declare_parameter("target.cube_size_mm", 100.0)

        self.declare_parameter("yolo.obb_model_path", "")
        self.declare_parameter("yolo.pose_model_path", "")
        self.declare_parameter("yolo.conf_threshold", 0.25)
        self.declare_parameter("yolo.iou_threshold", 0.7)
        self.declare_parameter("yolo.device", "cuda")
        self.declare_parameter("yolo.pad_ratio", 1.2)
        self.declare_parameter("yolo.warp_size", [256, 256])
        self.declare_parameter("yolo.imgsz_obb", 0)
        self.declare_parameter("yolo.imgsz_pose", 0)

        self.declare_parameter("pnp.method", "IPPE_SQUARE")
        self.declare_parameter("pnp.reproj_error_max_px", 8.0)
        # Optional guard: if face is too small in pixels, PnP becomes ill-conditioned. 0 disables.
        self.declare_parameter("pnp.min_face_size_px", 0.0)
        # Optional pose refinement (single-frame BA / reprojection LM)
        self.declare_parameter("pnp.refine", "NONE")  # NONE | LM
        self.declare_parameter("pnp.refine_iterations", 20)
        self.declare_parameter("pnp.refine_eps", 1e-6)
        # Multi-frame BA window size (static pose assumption). <=1 disables.
        self.declare_parameter("pnp.ba_window_size", 1)

        self.declare_parameter("frame_ids.camera_frame", "camera_color_optical_frame")
        self.declare_parameter("frame_ids.cube_center_frame", "cube_center")

        self.declare_parameter("publish.visualization", True)
        self.declare_parameter("publish.pose", True)
        self.declare_parameter("publish.face_corners_3d", True)
        self.declare_parameter("publish.tf", True)
        self.declare_parameter("publish.markers", True)

        self.declare_parameter("visualization.axes_length_mm", 50.0)
        self.declare_parameter("visualization.show_fps", True)

        self.declare_parameter("markers.topic", "/cube_pose/markers")
        self.declare_parameter("markers.cube_alpha", 0.35)
        self.declare_parameter("markers.cube_color", [0.1, 0.6, 1.0])
        self.declare_parameter("markers.center_color", [1.0, 0.2, 0.2])

        # Debug / observability
        self.declare_parameter("debug.log_pnp_failures", True)
        self.declare_parameter("debug.pnp_failure_log_interval_sec", 1.0)
        self.declare_parameter("debug.log_pnp_residuals", True)
        self.declare_parameter("debug.log_yolo_points", False)
        self.declare_parameter("debug.yolo_log_interval_sec", 0.5)
        self.declare_parameter("debug.yolo_log_as_int", True)
        self.declare_parameter("debug.log_pnp_success", False)
        self.declare_parameter("debug.pnp_success_log_interval_sec", 0.5)

    def _load_parameters(self) -> None:
        self._input_mode = str(self.get_parameter("input.mode").value).strip().lower()
        self._image_topic = self.get_parameter("image_topic").get_parameter_value().string_value

        self._usb_camera_id = int(self.get_parameter("usb.camera_id").value)
        self._usb_width = int(self.get_parameter("usb.width").value)
        self._usb_height = int(self.get_parameter("usb.height").value)
        self._usb_fps = float(self.get_parameter("usb.fps").value)
        self._usb_backend = str(self.get_parameter("usb.backend").value).strip().lower()
        self._usb_auto_exposure = bool(self.get_parameter("usb.auto_exposure").value)
        self._usb_exposure = float(self.get_parameter("usb.exposure").value)
        self._usb_gain = float(self.get_parameter("usb.gain").value)

        fx = self.get_parameter("camera.intrinsics.fx").value
        fy = self.get_parameter("camera.intrinsics.fy").value
        cx = self.get_parameter("camera.intrinsics.cx").value
        cy = self.get_parameter("camera.intrinsics.cy").value
        dist = self.get_parameter("camera.distortion").value

        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        d = np.array(dist, dtype=np.float64).reshape(1, -1)
        self._camera = CameraModel(camera_matrix=k, dist_coeffs=d)

        self._cube_size_mm = float(self.get_parameter("target.cube_size_mm").value)

        obb_model = self.get_parameter("yolo.obb_model_path").value
        pose_model = self.get_parameter("yolo.pose_model_path").value
        conf = float(self.get_parameter("yolo.conf_threshold").value)
        iou = float(self.get_parameter("yolo.iou_threshold").value)
        device = self.get_parameter("yolo.device").value
        pad_ratio = float(self.get_parameter("yolo.pad_ratio").value)
        warp_size_list = self.get_parameter("yolo.warp_size").value
        warp_size = (int(warp_size_list[0]), int(warp_size_list[1]))
        imgsz_obb = int(self.get_parameter("yolo.imgsz_obb").value)
        imgsz_pose = int(self.get_parameter("yolo.imgsz_pose").value)

        self._det_cfg = CascadeDetectorConfig(
            obb_model_path=str(obb_model),
            pose_model_path=str(pose_model),
            conf_threshold=conf,
            iou_threshold=iou,
            device=str(device),
            pad_ratio=pad_ratio,
            warp_size=warp_size,
            imgsz_obb=imgsz_obb,
            imgsz_pose=imgsz_pose,
        )

        self._pnp_cfg = PnPConfig(
            method=str(self.get_parameter("pnp.method").value),
            reproj_error_max_px=float(self.get_parameter("pnp.reproj_error_max_px").value),
            min_face_size_px=float(self.get_parameter("pnp.min_face_size_px").value),
            refine=str(self.get_parameter("pnp.refine").value),
            refine_iterations=int(self.get_parameter("pnp.refine_iterations").value),
            refine_eps=float(self.get_parameter("pnp.refine_eps").value),
            ba_window_size=int(self.get_parameter("pnp.ba_window_size").value),
        )

        self._camera_frame = self.get_parameter("frame_ids.camera_frame").value
        self._cube_center_frame = self.get_parameter("frame_ids.cube_center_frame").value

        self._pub_vis_enabled = bool(self.get_parameter("publish.visualization").value)
        self._pub_pose_enabled = bool(self.get_parameter("publish.pose").value)
        self._pub_corners_enabled = bool(self.get_parameter("publish.face_corners_3d").value)
        self._pub_tf_enabled = bool(self.get_parameter("publish.tf").value)
        self._pub_markers_enabled = bool(self.get_parameter("publish.markers").value)

        self._axes_len_mm = float(self.get_parameter("visualization.axes_length_mm").value)
        self._show_fps = bool(self.get_parameter("visualization.show_fps").value)

        self._markers_topic = str(self.get_parameter("markers.topic").value)
        self._cube_alpha = float(self.get_parameter("markers.cube_alpha").value)
        self._cube_color = [float(x) for x in self.get_parameter("markers.cube_color").value]
        self._center_color = [float(x) for x in self.get_parameter("markers.center_color").value]

        self._log_pnp_failures = bool(self.get_parameter("debug.log_pnp_failures").value)
        self._pnp_fail_log_interval = float(
            self.get_parameter("debug.pnp_failure_log_interval_sec").value
        )
        self._log_pnp_residuals = bool(self.get_parameter("debug.log_pnp_residuals").value)
        self._log_yolo_points = bool(self.get_parameter("debug.log_yolo_points").value)
        self._yolo_log_interval = float(self.get_parameter("debug.yolo_log_interval_sec").value)
        self._yolo_log_as_int = bool(self.get_parameter("debug.yolo_log_as_int").value)
        self._log_pnp_success = bool(self.get_parameter("debug.log_pnp_success").value)
        self._pnp_success_log_interval = float(
            self.get_parameter("debug.pnp_success_log_interval_sec").value
        )

        if not self._det_cfg.obb_model_path or not self._det_cfg.pose_model_path:
            self.get_logger().warn(
                "yolo.obb_model_path / yolo.pose_model_path is empty. "
                "Please set them in the params file."
            )

    def _start_usb_capture(self) -> None:
        backend = 0
        if self._usb_backend == "v4l2":
            backend = cv2.CAP_V4L2
        elif self._usb_backend == "any":
            backend = 0

        self._cap = cv2.VideoCapture(self._usb_camera_id, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera id={self._usb_camera_id}")

        # Best-effort camera controls.
        # NOTE: OpenCV uses backend-specific semantics for AUTO_EXPOSURE:
        # - For V4L2, values are often 0.25 (manual) / 0.75 (auto).
        # We'll try common values and ignore failures.
        try:
            if self._usb_auto_exposure:
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            else:
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        except Exception:
            pass

        if not self._usb_auto_exposure and self._usb_exposure >= 0.0:
            try:
                self._cap.set(cv2.CAP_PROP_EXPOSURE, float(self._usb_exposure))
            except Exception:
                pass

        if self._usb_gain >= 0.0:
            try:
                self._cap.set(cv2.CAP_PROP_GAIN, float(self._usb_gain))
            except Exception:
                pass

        if self._usb_width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._usb_width))
        if self._usb_height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._usb_height))
        if self._usb_fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, float(self._usb_fps))

        period = 1.0 / self._usb_fps if self._usb_fps > 0 else 1.0 / 30.0
        self._timer = self.create_timer(period, self._on_usb_timer)
        # Read back a few properties for observability (may return -1 if unsupported).
        try:
            ae = self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            exp = self._cap.get(cv2.CAP_PROP_EXPOSURE)
            gain = self._cap.get(cv2.CAP_PROP_GAIN)
        except Exception:
            ae, exp, gain = float("nan"), float("nan"), float("nan")

        self.get_logger().info(
            f"USB capture enabled: camera_id={self._usb_camera_id}, "
            f"req={self._usb_width}x{self._usb_height}@{self._usb_fps}Hz, backend={self._usb_backend}, "
            f"auto_exposure={self._usb_auto_exposure}, exposure={self._usb_exposure}, gain={self._usb_gain}, "
            f"readback(auto_exposure={ae:.3g}, exposure={exp:.3g}, gain={gain:.3g})"
        )

    def _on_usb_timer(self) -> None:
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.get_logger().warn("USB camera read failed.")
            return

        stamp = self.get_clock().now().to_msg()
        # Process without ROS Image input
        self._process_frame(frame, stamp)

    def _on_image(self, msg: Image) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return

        self._process_frame(bgr, msg.header.stamp)

    def _process_frame(self, bgr: np.ndarray, stamp_msg) -> None:
        now = time.perf_counter()
        dt = now - self._last_ts
        self._last_ts = now
        if dt > 1e-6:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        corners_xy = self._detector.detect_face_corners(bgr)
        if corners_xy is None:
            if self._pub_vis_enabled:
                self._publish_vis(bgr, stamp_msg)
            return

        # Update multi-frame BA window (stores only successful detections).
        try:
            self._corners_window.append(corners_xy.astype(np.float32))
        except Exception:
            pass

        if self._log_yolo_points:
            now = time.perf_counter()
            if now - float(self._last_yolo_log_ts) >= max(0.05, float(self._yolo_log_interval)):
                self._last_yolo_log_ts = now
                pts = corners_xy.astype(np.int32) if self._yolo_log_as_int else corners_xy.astype(np.float32)
                tl, tr, br, bl = pts.reshape(4, 2)
                if self._yolo_log_as_int:
                    self.get_logger().info(
                        "YOLO corners_xy(px) "
                        f"TL=({int(tl[0])},{int(tl[1])}) "
                        f"TR=({int(tr[0])},{int(tr[1])}) "
                        f"BR=({int(br[0])},{int(br[1])}) "
                        f"BL=({int(bl[0])},{int(bl[1])})"
                    )
                else:
                    self.get_logger().info(
                        "YOLO corners_xy(px) "
                        f"TL=({float(tl[0]):.2f},{float(tl[1]):.2f}) "
                        f"TR=({float(tr[0]):.2f},{float(tr[1]):.2f}) "
                        f"BR=({float(br[0]):.2f},{float(br[1]):.2f}) "
                        f"BL=({float(bl[0]):.2f},{float(bl[1]):.2f})"
                    )

        h, w = int(bgr.shape[0]), int(bgr.shape[1])

        window = None
        if int(self._pnp_cfg.ba_window_size) > 1 and len(self._corners_window) >= 2:
            window = tuple(self._corners_window)
        pnp_res = self._pnp.solve(corners_xy, corners_window=window)
        if pnp_res is None:
            if self._log_pnp_failures:
                now = time.perf_counter()
                if now - self._last_pnp_fail_log_ts >= max(0.1, self._pnp_fail_log_interval):
                    self._last_pnp_fail_log_ts = now
                    diag = self._pnp.diagnose(corners_xy)

                    msg = (
                        f"PnP failed: {diag.reason}, "
                        f"reproj_rmse={diag.reprojection_error_px:.2f}px, img={w}x{h}"
                    )
                    if self._log_pnp_residuals:
                        per_pt = ", ".join([f"{float(x):.1f}" for x in diag.per_point_error_px])
                        k = self._camera.camera_matrix
                        d = self._camera.dist_coeffs.reshape(-1)
                        msg += (
                            f", per_pt=[{per_pt}]px"
                            f", fx={float(k[0,0]):.1f}, fy={float(k[1,1]):.1f}, "
                            f"cx={float(k[0,2]):.1f}, cy={float(k[1,2]):.1f}"
                            f", dist=[{', '.join([f'{float(x):.3g}' for x in d])}]"
                            f", cube_size_mm={float(self._cube_size_mm):.1f}"
                        )
                    self.get_logger().warn(msg)

            if self._pub_vis_enabled:
                self._publish_vis(bgr, stamp_msg, corners_xy=corners_xy)
            return

        if self._log_pnp_success:
            now = time.perf_counter()
            if now - float(self._last_pnp_ok_log_ts) >= max(0.05, float(self._pnp_success_log_interval)):
                self._last_pnp_ok_log_ts = now
                t = pnp_res.tvec.reshape(3).astype(np.float64)
                c = pnp_res.cube_center_mm.reshape(3).astype(np.float64)
                self.get_logger().info(
                    "PnP ok "
                    f"reproj_rmse={float(pnp_res.reprojection_error_px):.2f}px, "
                    f"tvec_mm=({t[0]:.1f},{t[1]:.1f},{t[2]:.1f}), "
                    f"cube_center_mm=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})"
                )

        quat_xyzw = self._pnp.rvec_tvec_to_pose(pnp_res.rvec, pnp_res.tvec)[1]
        center_m = (pnp_res.cube_center_mm / 1000.0).astype(np.float64)

        if self._pub_pose_enabled:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp_msg
            pose_msg.header.frame_id = self._camera_frame
            pose_msg.pose.position.x = float(center_m[0])
            pose_msg.pose.position.y = float(center_m[1])
            pose_msg.pose.position.z = float(center_m[2])
            pose_msg.pose.orientation.x = float(quat_xyzw[0])
            pose_msg.pose.orientation.y = float(quat_xyzw[1])
            pose_msg.pose.orientation.z = float(quat_xyzw[2])
            pose_msg.pose.orientation.w = float(quat_xyzw[3])
            self._pub_pose.publish(pose_msg)

        if self._pub_tf_enabled:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp_msg
            tf_msg.header.frame_id = self._camera_frame
            tf_msg.child_frame_id = self._cube_center_frame
            tf_msg.transform.translation.x = float(center_m[0])
            tf_msg.transform.translation.y = float(center_m[1])
            tf_msg.transform.translation.z = float(center_m[2])
            tf_msg.transform.rotation.x = float(quat_xyzw[0])
            tf_msg.transform.rotation.y = float(quat_xyzw[1])
            tf_msg.transform.rotation.z = float(quat_xyzw[2])
            tf_msg.transform.rotation.w = float(quat_xyzw[3])
            self._tf_broadcaster.sendTransform(tf_msg)

        if self._pub_corners_enabled:
            corners_3d = self._compute_face_corners_3d(pnp_res.rvec, pnp_res.tvec)
            poly = PolygonStamped()
            poly.header.stamp = stamp_msg
            poly.header.frame_id = self._camera_frame
            poly.polygon.points = corners_3d
            self._pub_corners.publish(poly)

        if self._pub_markers_enabled:
            self._publish_markers(
                stamp_msg=stamp_msg,
                center_m=center_m,
                quat_xyzw=quat_xyzw,
                reproj_err=pnp_res.reprojection_error_px,
            )

        if self._pub_vis_enabled:
            self._publish_vis(
                bgr,
                stamp_msg,
                corners_xy=corners_xy,
                rvec=pnp_res.rvec,
                tvec=pnp_res.tvec,
                reproj_err=pnp_res.reprojection_error_px,
            )

    def _compute_face_corners_3d(self, rvec: np.ndarray, tvec: np.ndarray) -> List:
        l = float(self._cube_size_mm)
        obj = np.array(
            [
                [-l / 2.0, l / 2.0, 0.0],
                [l / 2.0, l / 2.0, 0.0],
                [l / 2.0, -l / 2.0, 0.0],
                [-l / 2.0, -l / 2.0, 0.0],
            ],
            dtype=np.float64,
        )  # (4,3) mm
        rotm, _ = cv2.Rodrigues(rvec)
        cam_pts = (rotm @ obj.T).T + tvec.reshape(1, 3)

        # geometry_msgs/Point32 requires float32.
        from geometry_msgs.msg import Point32

        pts: List[Point32] = []
        for p in cam_pts:
            pt = Point32()
            pt.x = float(p[0] / 1000.0)
            pt.y = float(p[1] / 1000.0)
            pt.z = float(p[2] / 1000.0)
            pts.append(pt)
        return pts

    def _publish_vis(
        self,
        bgr: np.ndarray,
        stamp_msg,
        corners_xy: np.ndarray | None = None,
        rvec: np.ndarray | None = None,
        tvec: np.ndarray | None = None,
        reproj_err: float | None = None,
    ) -> None:
        vis = bgr.copy()

        if corners_xy is not None and corners_xy.shape == (4, 2):
            pts = corners_xy.astype(np.int32)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # TL,TR,BR,BL
            for i, p in enumerate(pts):
                cv2.circle(vis, tuple(p), 5, colors[i], -1)
                cv2.putText(
                    vis,
                    str(i),
                    (int(p[0] + 6), int(p[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colors[i],
                    2,
                )
            for i in range(4):
                p1 = tuple(pts[i])
                p2 = tuple(pts[(i + 1) % 4])
                cv2.line(vis, p1, p2, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        if rvec is not None and tvec is not None:
            try:
                cv2.drawFrameAxes(
                    vis,
                    self._camera.camera_matrix.astype(np.float64),
                    self._camera.dist_coeffs.astype(np.float64),
                    rvec.astype(np.float64),
                    tvec.astype(np.float64),
                    self._axes_len_mm,
                )
            except Exception:
                # drawFrameAxes may fail for some OpenCV builds; ignore visualization only.
                pass

        lines: List[str] = []
        if self._show_fps:
            lines.append(f"FPS: {self._fps:.1f}")
        if reproj_err is not None:
            lines.append(f"Reproj RMSE: {reproj_err:.2f}px")

        y = 25
        for text in lines:
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y += 28

        out = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out.header.stamp = stamp_msg
        out.header.frame_id = self._camera_frame
        self._pub_vis.publish(out)

    def _publish_markers(
        self,
        stamp_msg,
        center_m: np.ndarray,
        quat_xyzw: np.ndarray,
        reproj_err: float,
    ) -> None:
        # Cube size in meters
        cube_size_m = float(self._cube_size_mm / 1000.0)

        def _mk_color(rgb, a: float):
            r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
            return r, g, b, float(a)

        cube_rgba = _mk_color(self._cube_color, self._cube_alpha)
        center_rgba = _mk_color(self._center_color, 1.0)

        arr = MarkerArray()

        # 1) Cube marker
        cube = Marker()
        cube.header.stamp = stamp_msg
        cube.header.frame_id = self._camera_frame
        cube.ns = "cube_pose"
        cube.id = 1
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position.x = float(center_m[0])
        cube.pose.position.y = float(center_m[1])
        cube.pose.position.z = float(center_m[2])
        cube.pose.orientation.x = float(quat_xyzw[0])
        cube.pose.orientation.y = float(quat_xyzw[1])
        cube.pose.orientation.z = float(quat_xyzw[2])
        cube.pose.orientation.w = float(quat_xyzw[3])
        cube.scale.x = cube_size_m
        cube.scale.y = cube_size_m
        cube.scale.z = cube_size_m
        cube.color.r = cube_rgba[0]
        cube.color.g = cube_rgba[1]
        cube.color.b = cube_rgba[2]
        cube.color.a = cube_rgba[3]
        arr.markers.append(cube)

        # 2) Center point marker
        center = Marker()
        center.header.stamp = stamp_msg
        center.header.frame_id = self._camera_frame
        center.ns = "cube_pose"
        center.id = 2
        center.type = Marker.SPHERE
        center.action = Marker.ADD
        center.pose.position.x = float(center_m[0])
        center.pose.position.y = float(center_m[1])
        center.pose.position.z = float(center_m[2])
        center.pose.orientation.w = 1.0
        center.scale.x = 0.02
        center.scale.y = 0.02
        center.scale.z = 0.02
        center.color.r = center_rgba[0]
        center.color.g = center_rgba[1]
        center.color.b = center_rgba[2]
        center.color.a = center_rgba[3]
        arr.markers.append(center)

        # 3) Text marker (distance + error)
        text = Marker()
        text.header.stamp = stamp_msg
        text.header.frame_id = self._camera_frame
        text.ns = "cube_pose"
        text.id = 3
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(center_m[0])
        text.pose.position.y = float(center_m[1])
        text.pose.position.z = float(center_m[2] + 0.06)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.05
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = (
            f"center: ({center_m[0]:.3f}, {center_m[1]:.3f}, {center_m[2]:.3f}) m\n"
            f"reproj: {reproj_err:.2f} px"
        )
        arr.markers.append(text)

        self._pub_markers.publish(arr)

    def destroy_node(self) -> bool:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        return super().destroy_node()


def main(args: List[str] | None = None) -> None:
    """Entrypoint for ros2 run."""
    rclpy.init(args=args)
    node = CubePoseEstimatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


