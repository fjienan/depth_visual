# 4-Point PnP Pose Estimation System

基于级联 YOLO 检测器的 6-DOF 姿态估计系统，用于实时获取目标物体的 3D 位置和姿态。

## 系统概述

本系统通过以下步骤实现目标物体的 6-DOF 姿态估计：

```
摄像头图像
    ↓
Stage 1: YOLOv8-OBB 粗定位
    ↓
Stage 2: YOLOv8-Pose 精确角点检测
    ↓
PnP 算法求解姿态
    ↓
6-DOF 姿态 (位置 + 旋转)
```

### 核心功能

- ✅ **实时检测**: 基于级联 YOLO 模型的高精度 4 角点检测
- ✅ **姿态估计**: cv2.solvePnP 求解 6-DOF 姿态（平移 + 旋转）
- ✅ **3D 可视化**: 实时显示 3D 坐标轴和重投影误差
- ✅ **灵活配置**: 所有参数通过 YAML 配置文件管理
- ✅ **结果保存**: 支持保存图像、视频、姿态数据

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install opencv-python numpy pyyaml ultralytics

# 确保已训练好 Stage1 和 Stage2 模型
# 模型路径在 config/global.yaml 中配置
```

### 2. 相机标定（重要！）

准确的相机内参是 PnP 求解的关键。使用 OpenCV 标定工具：

```bash
# 使用棋盘格标定相机
python calibrate_camera.py --images calib_images/ --pattern 9x6 --square-size 25

# 或使用在线工具
# https://calib.io/
```

标定后，更新 `config/global.yaml` 中的相机参数：

```yaml
camera:
  intrinsics:
    fx: 800.0    # 从标定结果获取
    fy: 800.0
    cx: 640.0
    cy: 360.0
  distortion: [k1, k2, p1, p2, k3]  # 从标定结果获取
```

### 3. 配置目标物体尺寸

测量你的目标物体的**真实尺寸**（单位：毫米），并更新配置：

```yaml
target_object:
  name: "Your Object"
  width_mm: 440.0    # 实际宽度
  height_mm: 140.0   # 实际高度
```

### 4. 运行程序

```bash
cd ws/src/four_pt_pnp/rgb_depth_detect

# 基础运行（使用默认配置）
python four_pt_pnp.py

# 使用自定义配置
python four_pt_pnp.py --config /path/to/config.yaml

# 指定摄像头
python four_pt_pnp.py --camera 0

# 覆盖模型路径
python four_pt_pnp.py \
    --stage1 /path/to/obb_model.pt \
    --stage2 /path/to/pose_model.pt
```

### 5. 交互控制

运行时按键控制：
- **`q`**: 退出程序
- **`p`**: 暂停/继续
- **`s`**: 保存当前帧快照

## 配置文件详解

配置文件: `config/global.yaml`

### 相机配置

```yaml
camera:
  source: 0              # 摄像头 ID 或视频路径
  intrinsics:
    fx: 800.0            # 焦距 X (像素)
    fy: 800.0            # 焦距 Y (像素)
    cx: 640.0            # 主点 X (像素)
    cy: 360.0            # 主点 Y (像素)
  distortion: [0, 0, 0, 0, 0]  # 畸变系数
  resolution:
    width: 1280
    height: 720
  fps: 30
```

**如何获取相机内参？**
1. **方法1**: OpenCV 标定（推荐）
   ```python
   import cv2
   # 使用 cv2.calibrateCamera() 标定
   ```
2. **方法2**: 使用制造商提供的参数
3. **方法3**: 使用典型值（精度较低）
   - 1080p 网络摄像头: fx=fy≈800-1000, cx=640, cy=360
   - 手机摄像头: fx=fy≈1000-1500

### 模型配置

```yaml
model:
  stage1_model: "path/to/obb_model.pt"    # Stage1 OBB 模型
  stage2_model: "path/to/pose_model.pt"   # Stage2 Pose 模型
  conf_threshold: 0.25                     # 置信度阈值
  device: "0"                              # GPU ID 或 "cpu"
  warp_size: [256, 256]                    # Stage2 输入尺寸
  pad_ratio: 1.1                           # OBB 裁剪膨胀比例
```

### 目标物体配置

```yaml
target_object:
  name: "License Plate"
  width_mm: 440.0       # 物体宽度（毫米）
  height_mm: 140.0      # 物体高度（毫米）
  
  corners_3d:
    use_auto: true      # 自动生成角点（推荐）
    
    # 如果 use_auto: false，手动指定角点坐标
    manual:
      - [-220.0, 70.0, 0.0]   # Top-Left
      - [220.0, 70.0, 0.0]    # Top-Right
      - [220.0, -70.0, 0.0]   # Bottom-Right
      - [-220.0, -70.0, 0.0]  # Bottom-Left
```

**物体坐标系定义**:
- 原点: 物体中心
- X 轴: 水平向右
- Y 轴: 垂直向上
- Z 轴: 垂直于物体平面（向外）

### PnP 算法配置

```yaml
pnp:
  method: "iterative"    # 方法: iterative, p3p, epnp, sqpnp
  use_ransac: false      # 是否使用 RANSAC
  ransac_reprojection_error: 8.0
  ransac_iterations: 100
```

**PnP 方法选择**:
- **`iterative`**: 迭代优化，推荐用于 4 点（最稳定）
- **`p3p`**: 3 点透视，速度快但需要额外点验证
- **`epnp`**: 适用于多点（>4）
- **`sqpnp`**: 最新方法，精度高

**何时使用 RANSAC？**
- 检测结果不稳定时
- 存在异常点时
- 需要更强的鲁棒性时

### 可视化配置

```yaml
visualization:
  show_window: true
  draw_detection_box: true   # 绘制检测框
  draw_keypoints: true       # 绘制关键点
  draw_axes: true            # 绘制 3D 坐标轴
  draw_reprojection: true    # 绘制重投影点
  draw_distance: true        # 显示距离
  
  show_fps: true
  show_coordinates: true
  show_rotation: true
  show_translation: true
  
  axes_length: 100.0         # 坐标轴长度（毫米）
```

### 输出配置

```yaml
output:
  save_results: false        # 是否保存结果
  save_dir: "output"         # 输出目录
  save_images: true          # 保存图像
  save_video: false          # 保存视频
  save_poses: true           # 保存姿态数据（CSV）
```

**输出格式**:
- **图像**: `output/images_TIMESTAMP/frame_XXXXXX.jpg`
- **姿态数据**: `output/poses_TIMESTAMP.csv`
  ```csv
  frame,timestamp,tx_mm,ty_mm,tz_mm,rx_deg,ry_deg,rz_deg,kp0_x,kp0_y,...
  ```

## 坐标系说明

### 1. 图像坐标系
```
(0,0) ───────── X (width)
  │
  │
  │
  Y (height)
```

### 2. 相机坐标系
```
      Z (光轴向前)
     /
    /
   o ─────── X (右)
   │
   │
   Y (下)
```

### 3. 物体坐标系
```
      Z (垂直物体向外)
     /
    /
   o ─────── X (右)
   │
   │
   Y (下)
```

## 姿态输出解释

程序输出的姿态包含两部分：

### 平移向量 (Translation Vector)
```
T: [tx, ty, tz] 单位：mm
```
- **tx**: 物体相对相机的左右位移（正值=右侧）
- **ty**: 物体相对相机的上下位移（正值=下方）
- **tz**: 物体到相机的距离（深度）

### 旋转角度 (Euler Angles)
```
R: [rx, ry, rz] 单位：度
```
- **rx**: 绕 X 轴旋转（俯仰）
- **ry**: 绕 Y 轴旋转（偏航）
- **rz**: 绕 Z 轴旋转（翻滚）

### 距离 (Distance)
```
Distance: sqrt(tx² + ty² + tz²) mm
```
物体中心到相机光学中心的欧氏距离。

## 常见问题

### Q1: 姿态抖动严重

**原因**:
1. 相机内参不准确
2. 目标物体尺寸测量不准
3. 关键点检测不稳定

**解决方案**:
```yaml
# 1. 重新标定相机
# 2. 使用卡尔曼滤波平滑姿态
# 3. 调整检测阈值
model:
  conf_threshold: 0.5  # 提高阈值

# 4. 启用 RANSAC
pnp:
  use_ransac: true
```

### Q2: PnP 求解失败

**症状**: 程序输出 "PnP failed" 或姿态异常

**检查清单**:
1. ✅ 相机内参是否正确？
2. ✅ 物体尺寸单位是否为毫米？
3. ✅ 检测到的 4 个点是否正确？
4. ✅ 点的顺序是否正确（TL, TR, BR, BL）？

**调试方法**:
```yaml
# 打开重投影可视化
visualization:
  draw_reprojection: true

# 检查重投影误差
# 误差应该 < 5 像素，如果 > 10 像素说明有问题
```

### Q3: 重投影误差大

**原因**:
- 相机内参不准
- 畸变校正不充分
- 物体尺寸测量误差

**解决方案**:
1. **重新标定相机**（最重要）
2. **精确测量物体尺寸**（使用卡尺，精确到 0.1mm）
3. **启用 RANSAC 过滤异常点**

### Q4: 距离估计不准

**原因**: 焦距参数不准确

**校准方法**:
1. 将物体放在已知距离 D（如 500mm）
2. 记录估计距离 D'
3. 计算缩放因子: scale = D / D'
4. 更新焦距: fx_new = fx * scale, fy_new = fy * scale

### Q5: 无法检测到物体

**检查**:
1. 模型路径是否正确？
2. 模型是否适配当前场景？
3. 置信度阈值是否过高？

**解决**:
```bash
# 降低阈值
python four_pt_pnp.py  # 然后在配置中修改 conf_threshold: 0.1

# 测试模型
cd ../../model_train/train
python inference.py --stage1-model xxx.pt --stage2-model yyy.pt --source test.jpg
```

## 相机标定工具

### 使用 OpenCV 标定脚本

创建 `calibrate_camera.py`:

```python
#!/usr/bin/env python3
import cv2
import numpy as np
import glob

def calibrate_camera(images_path, pattern_size=(9, 6), square_size=25.0):
    """
    标定相机
    
    Args:
        images_path: 标定图像文件夹路径
        pattern_size: 棋盘格内角点数 (width, height)
        square_size: 棋盘格方块边长 (mm)
    """
    # 准备对象点
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    objpoints = []  # 3D 点
    imgpoints = []  # 2D 点
    
    images = glob.glob(f"{images_path}/*.jpg")
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)
            
            # 可视化
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(100)
    
    cv2.destroyAllWindows()
    
    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    print("\n=== 相机标定结果 ===")
    print(f"RMS 误差: {ret:.3f}")
    print(f"\n相机内参矩阵 K:")
    print(mtx)
    print(f"\n畸变系数:")
    print(dist[0])
    
    print(f"\n复制到配置文件:")
    print(f"fx: {mtx[0, 0]:.1f}")
    print(f"fy: {mtx[1, 1]:.1f}")
    print(f"cx: {mtx[0, 2]:.1f}")
    print(f"cy: {mtx[1, 2]:.1f}")
    print(f"distortion: [{dist[0][0]:.6f}, {dist[0][1]:.6f}, {dist[0][2]:.6f}, {dist[0][3]:.6f}, {dist[0][4]:.6f}]")
    
    return mtx, dist

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='标定图像文件夹')
    parser.add_argument('--pattern', default='9x6', help='棋盘格尺寸，如 9x6')
    parser.add_argument('--square-size', type=float, default=25.0, help='方块边长(mm)')
    args = parser.parse_args()
    
    pattern = tuple(map(int, args.pattern.split('x')))
    calibrate_camera(args.images, pattern, args.square_size)
```

### 标定步骤

1. **打印标定板**
   - 下载棋盘格图案（OpenCV 官方）
   - 打印在平整硬纸板上
   - 测量实际方块尺寸

2. **拍摄标定图像**（20-30 张）
   ```bash
   # 不同角度、不同距离、覆盖图像各个区域
   # 保存到 calib_images/ 文件夹
   ```

3. **运行标定**
   ```bash
   python calibrate_camera.py --images calib_images/ --pattern 9x6 --square-size 25
   ```

4. **更新配置**
   将输出的参数复制到 `config/global.yaml`

## 性能优化

### 1. GPU 加速

```yaml
model:
  device: "0"  # 使用 GPU 0
```

### 2. 降低分辨率

```yaml
camera:
  resolution:
    width: 640   # 从 1280 降到 640
    height: 480  # 从 720 降到 480
```

### 3. 调整检测参数

```yaml
model:
  conf_threshold: 0.5  # 提高阈值减少误检
  warp_size: [128, 128]  # 降低 Stage2 输入尺寸（但会损失精度）
```

## 项目文件结构

```
four_pt_pnp/
├── config/
│   └── global.yaml          # 主配置文件
├── rgb_depth_detect/
│   ├── four_pt_pnp.py       # 主程序
│   └── __init__.py
├── output/                  # 输出目录（自动创建）
│   ├── images_TIMESTAMP/
│   └── poses_TIMESTAMP.csv
├── launch/
│   └── test.launch.py       # ROS2 launch 文件（可选）
├── README.md                # 本文档
└── calibrate_camera.py      # 相机标定工具（需创建）
```

## 相关文档

- [级联检测器文档](../../model_train/train/readme.md)
- [Stage1 训练](../../model_train/train/docs/README_TRAINING.md)
- [Stage2 数据准备](../../model_train/train/docs/README_STAGE2_DATA.md)
- [数据预处理](../../model_train/database/README.md)

## License

MIT License

## 作者

fjienan - 深度视觉项目组

---

**最后更新**: 2024-01-10
