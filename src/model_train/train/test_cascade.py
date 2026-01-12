#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the cascaded detector.

This script demonstrates how to use the CascadeDetector class.
"""

import cv2
import numpy as np
from pathlib import Path
from inference import CascadeDetector, GeometryUtils

def test_geometry_utils():
    """Test GeometryUtils functions."""
    print("=" * 60)
    print("Testing GeometryUtils")
    print("=" * 60)
    
    # Test order_points
    print("\n1. Testing order_points...")
    test_points = np.array([
        [100, 50],   # top-right
        [50, 100],   # bottom-left
        [0, 0],      # top-left
        [150, 150]   # bottom-right
    ], dtype=np.float32)
    
    ordered = GeometryUtils.order_points(test_points)
    print(f"  输入点: {test_points}")
    print(f"  排序后: {ordered}")
    print(f"  顺序: [top-left, top-right, bottom-right, bottom-left]")
    
    # Test get_dilated_box_points
    print("\n2. Testing get_dilated_box_points...")
    obb = np.array([320, 240, 200, 100, 30], dtype=np.float32)  # [cx, cy, w, h, angle]
    dilated_points = GeometryUtils.get_dilated_box_points(obb, pad_ratio=1.2)
    print(f"  OBB: {obb}")
    print(f"  膨胀后角点: {dilated_points}")
    
    # Test warp_image
    print("\n3. Testing warp_image...")
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    src_pts = np.array([
        [100, 100],
        [500, 100],
        [500, 300],
        [100, 300]
    ], dtype=np.float32)
    warped, M = GeometryUtils.warp_image(test_img, src_pts, (256, 256))
    print(f"  原始尺寸: {test_img.shape}")
    print(f"  变换后尺寸: {warped.shape}")
    print(f"  变换矩阵 M:\n{M}")
    
    # Test map_points_back
    print("\n4. Testing map_points_back...")
    local_pts = np.array([
        [10, 10],
        [246, 10],
        [246, 118],
        [10, 118]
    ], dtype=np.float32)
    original_pts = GeometryUtils.map_points_back(local_pts, M)
    print(f"  局部坐标点: {local_pts}")
    print(f"  映射回原始坐标: {original_pts}")
    print(f"  原始源点: {src_pts}")
    print(f"  误差 (应该很小): {np.abs(original_pts - src_pts).max():.2f} pixels")
    
    print("\n" + "=" * 60)
    print("GeometryUtils 测试完成!")
    print("=" * 60)


def test_cascade_detector():
    """Test CascadeDetector with a sample image."""
    print("\n" + "=" * 60)
    print("Testing CascadeDetector")
    print("=" * 60)
    
    # Note: This requires actual model files.
    # We try to find the latest best.pt under ../output/ automatically.
    repo_model_train_dir = Path(__file__).resolve().parents[1]  # .../src/model_train
    output_dir = repo_model_train_dir / "output"

    def _latest_best(pattern: str) -> str:
        cands = list(output_dir.glob(pattern))
        if not cands:
            return ""
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(cands[0])

    obb_model = _latest_best("stage1_obb*/weights/best.pt")
    pose_model = _latest_best("stage2_pose*/weights/best.pt")
    
    print(f"\n注意: 需要提供实际的模型路径")
    print(f"  OBB 模型: {obb_model}")
    print(f"  Pose 模型: {pose_model}")
    
    # Uncomment to test with actual models:
    # detector = CascadeDetector(
    #     obb_model_path=obb_model,
    #     pose_model_path=pose_model,
    #     pad_ratio=1.2,
    #     warp_size=(256, 128),
    #     conf_threshold=0.25
    # )
    # 
    # # Load test image
    # image = cv2.imread("test_image.jpg")
    # result = detector.predict(image)
    # 
    # if result['success']:
    #     print("检测成功!")
    #     vis_image = detector.visualize(image, result)
    #     cv2.imshow('Result', vis_image)
    #     cv2.waitKey(0)
    # else:
    #     print("检测失败")


if __name__ == '__main__':
    test_geometry_utils()
    test_cascade_detector()
