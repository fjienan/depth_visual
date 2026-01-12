#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Calibration Tool

使用棋盘格标定相机内参和畸变系数
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path
import argparse
import yaml


def calibrate_camera(
    images_path: str,
    pattern_size: tuple = (9, 6),
    square_size: float = 25.0,
    visualize: bool = True,
    output_yaml: str = None
):
    """
    使用棋盘格标定相机
    
    Args:
        images_path: 标定图像文件夹路径
        pattern_size: 棋盘格内角点数 (width, height)
        square_size: 棋盘格方块边长 (mm)
        visualize: 是否可视化检测结果
        output_yaml: 输出配置文件路径
    
    Returns:
        camera_matrix: 相机内参矩阵 (3x3)
        dist_coeffs: 畸变系数 (1x5)
    """
    # 准备对象点 (在世界坐标系中的 3D 点)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 存储所有图像的对象点和图像点
    objpoints = []  # 3D 点（世界坐标系）
    imgpoints = []  # 2D 点（图像坐标系）
    
    # 读取所有标定图像
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(images_path, ext)))
    
    if not image_files:
        print(f"错误: 在 {images_path} 中没有找到图像文件")
        return None, None
    
    print(f"找到 {len(image_files)} 张标定图像")
    print(f"棋盘格尺寸: {pattern_size[0]}x{pattern_size[1]}")
    print(f"方块边长: {square_size} mm")
    print("-" * 60)
    
    successful_images = []
    failed_images = []
    
    for i, fname in enumerate(image_files):
        print(f"处理 {i+1}/{len(image_files)}: {os.path.basename(fname)}", end=' ')
        
        img = cv2.imread(fname)
        if img is None:
            print("❌ 无法读取")
            failed_images.append(fname)
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, 
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_FAST_CHECK + 
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # 精确化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            successful_images.append(fname)
            
            print("✓")
            
            # 可视化
            if visualize:
                img_vis = img.copy()
                cv2.drawChessboardCorners(img_vis, pattern_size, corners_refined, ret)
                
                # 添加文本
                cv2.putText(
                    img_vis, 
                    f"Image {i+1}/{len(image_files)} - Found {pattern_size[0]*pattern_size[1]} corners", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # 缩放显示
                h, w = img_vis.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    img_vis = cv2.resize(img_vis, (int(w*scale), int(h*scale)))
                
                cv2.imshow('Calibration', img_vis)
                key = cv2.waitKey(300)
                if key == ord('q'):
                    visualize = False
        else:
            print("❌ 未找到角点")
            failed_images.append(fname)
    
    if visualize:
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"成功: {len(successful_images)}/{len(image_files)} 张图像")
    
    if failed_images:
        print(f"\n失败的图像:")
        for fname in failed_images:
            print(f"  - {os.path.basename(fname)}")
    
    if len(successful_images) < 10:
        print("\n警告: 成功图像数量少于 10 张，标定结果可能不准确")
        print("建议: 添加更多标定图像（推荐 20-30 张）")
    
    if len(objpoints) == 0:
        print("\n错误: 没有成功检测到任何棋盘格，无法进行标定")
        return None, None
    
    print("\n开始标定...")
    
    # 执行标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, 
        imgpoints, 
        gray.shape[::-1], 
        None, 
        None
    )
    
    # 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_reprojected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    print("\n" + "=" * 60)
    print("标定完成！")
    print("=" * 60)
    print(f"RMS 重投影误差: {ret:.4f} 像素")
    print(f"平均重投影误差: {mean_error:.4f} 像素")
    
    if mean_error < 0.5:
        print("✓ 标定质量: 优秀")
    elif mean_error < 1.0:
        print("✓ 标定质量: 良好")
    elif mean_error < 2.0:
        print("⚠ 标定质量: 一般（建议重新标定）")
    else:
        print("❌ 标定质量: 较差（强烈建议重新标定）")
    
    print("\n相机内参矩阵 K:")
    print(camera_matrix)
    
    print("\n畸变系数 [k1, k2, p1, p2, k3]:")
    print(dist_coeffs[0])
    
    print("\n" + "=" * 60)
    print("配置文件格式（复制到 config/global.yaml）:")
    print("=" * 60)
    print("camera:")
    print("  intrinsics:")
    print(f"    fx: {camera_matrix[0, 0]:.1f}")
    print(f"    fy: {camera_matrix[1, 1]:.1f}")
    print(f"    cx: {camera_matrix[0, 2]:.1f}")
    print(f"    cy: {camera_matrix[1, 2]:.1f}")
    print(f"  distortion: [{dist_coeffs[0][0]:.6f}, {dist_coeffs[0][1]:.6f}, {dist_coeffs[0][2]:.6f}, {dist_coeffs[0][3]:.6f}, {dist_coeffs[0][4]:.6f}]")
    print("  resolution:")
    print(f"    width: {gray.shape[1]}")
    print(f"    height: {gray.shape[0]}")
    
    # 保存到 YAML 文件
    if output_yaml:
        calib_data = {
            'camera': {
                'intrinsics': {
                    'fx': float(camera_matrix[0, 0]),
                    'fy': float(camera_matrix[1, 1]),
                    'cx': float(camera_matrix[0, 2]),
                    'cy': float(camera_matrix[1, 2]),
                },
                'distortion': [float(x) for x in dist_coeffs[0]],
                'resolution': {
                    'width': int(gray.shape[1]),
                    'height': int(gray.shape[0]),
                },
                'calibration_info': {
                    'rms_error': float(ret),
                    'mean_reprojection_error': float(mean_error),
                    'num_images': len(successful_images),
                    'pattern_size': list(pattern_size),
                    'square_size_mm': float(square_size),
                }
            }
        }
        
        output_path = Path(output_yaml)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(calib_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n标定结果已保存到: {output_yaml}")
    
    return camera_matrix, dist_coeffs


def test_undistortion(
    images_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
):
    """
    测试畸变校正效果
    
    Args:
        images_path: 测试图像路径或文件夹
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
    """
    # 获取测试图像
    if os.path.isfile(images_path):
        test_images = [images_path]
    else:
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            test_images.extend(glob.glob(os.path.join(images_path, ext)))
    
    if not test_images:
        print("没有找到测试图像")
        return
    
    print(f"\n测试畸变校正 ({len(test_images)} 张图像)")
    print("按任意键查看下一张，按 'q' 退出")
    
    for fname in test_images:
        img = cv2.imread(fname)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # 获取最优相机矩阵
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # 去畸变
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # 裁剪
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]
        
        # 并排显示
        img_resized = cv2.resize(img, (640, 480))
        undistorted_resized = cv2.resize(undistorted, (640, 480))
        comparison = np.hstack([img_resized, undistorted_resized])
        
        # 添加文字
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "Undistorted", (650, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Undistortion Test', comparison)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()


def capture_calibration_images(
    output_dir: str,
    camera_id: int = 0,
    num_images: int = 20
):
    """
    从摄像头捕获标定图像
    
    Args:
        output_dir: 输出目录
        camera_id: 摄像头 ID
        num_images: 目标图像数量
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return
    
    print("\n" + "=" * 60)
    print("标定图像捕获")
    print("=" * 60)
    print(f"目标: {num_images} 张图像")
    print(f"保存到: {output_dir}")
    print("\n操作说明:")
    print("  - 按 SPACE 捕获图像")
    print("  - 按 'q' 退出")
    print("\n拍摄建议:")
    print("  1. 在不同角度拍摄（俯视、仰视、侧面）")
    print("  2. 在不同距离拍摄（近、中、远）")
    print("  3. 覆盖图像的各个区域")
    print("  4. 确保棋盘格清晰、无模糊")
    print("  5. 保持棋盘格平整")
    print("-" * 60)
    
    count = 0
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 显示
        display = frame.copy()
        cv2.putText(
            display, 
            f"Captured: {count}/{num_images} - Press SPACE to capture, 'q' to quit", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        cv2.imshow('Capture Calibration Images', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # 保存图像
            filename = output_path / f"calib_{count:03d}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"✓ 保存: {filename.name}")
            count += 1
            
            # 闪烁效果
            cv2.imshow('Capture Calibration Images', np.ones_like(frame) * 255)
            cv2.waitKey(100)
        
        elif key == ord('q'):
            print("\n用户中断")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n完成! 已捕获 {count} 张图像")


def main():
    parser = argparse.ArgumentParser(
        description="相机标定工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 捕获标定图像:
   python calibrate_camera.py --capture calib_images/ --num-images 25

2. 执行标定:
   python calibrate_camera.py --calibrate calib_images/ --pattern 9x6 --square-size 25

3. 测试畸变校正:
   python calibrate_camera.py --test-undistort calib_images/ --calib-file camera_calib.yaml

4. 一键标定（捕获 + 标定）:
   python calibrate_camera.py --capture calib_images/ --calibrate calib_images/ --pattern 9x6
        """
    )
    
    # 捕获图像模式
    parser.add_argument('--capture', type=str, help='捕获标定图像到指定目录')
    parser.add_argument('--camera', type=int, default=0, help='摄像头 ID (默认: 0)')
    parser.add_argument('--num-images', type=int, default=20, help='捕获图像数量 (默认: 20)')
    
    # 标定模式
    parser.add_argument('--calibrate', type=str, help='使用指定目录的图像进行标定')
    parser.add_argument('--pattern', type=str, default='9x6', help='棋盘格尺寸 (默认: 9x6)')
    parser.add_argument('--square-size', type=float, default=25.0, help='方块边长 mm (默认: 25)')
    parser.add_argument('--output', type=str, help='输出标定结果到 YAML 文件')
    parser.add_argument('--no-visualize', action='store_true', help='不显示可视化')
    
    # 测试模式
    parser.add_argument('--test-undistort', type=str, help='测试畸变校正')
    parser.add_argument('--calib-file', type=str, help='标定结果文件 (YAML)')
    
    args = parser.parse_args()
    
    # 捕获图像
    if args.capture:
        capture_calibration_images(args.capture, args.camera, args.num_images)
    
    # 执行标定
    if args.calibrate:
        pattern_size = tuple(map(int, args.pattern.split('x')))
        camera_matrix, dist_coeffs = calibrate_camera(
            args.calibrate,
            pattern_size,
            args.square_size,
            not args.no_visualize,
            args.output
        )
        
        if camera_matrix is None:
            return 1
        
        # 测试畸变校正
        if not args.no_visualize:
            print("\n是否测试畸变校正? (y/n): ", end='')
            if input().lower() == 'y':
                test_undistortion(args.calibrate, camera_matrix, dist_coeffs)
    
    # 仅测试畸变校正
    elif args.test_undistort:
        if not args.calib_file:
            print("错误: 需要指定 --calib-file")
            return 1
        
        with open(args.calib_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        intrinsics = calib_data['camera']['intrinsics']
        camera_matrix = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ])
        dist_coeffs = np.array([calib_data['camera']['distortion']])
        
        test_undistortion(args.test_undistort, camera_matrix, dist_coeffs)
    
    else:
        parser.print_help()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
