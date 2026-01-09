# Cascaded 2-Stage Rigid Object Detection System

## 概述

这是一个高精度的两阶段级联检测系统，用于检测刚性矩形物体的4个精确角点。系统采用"LPR"（License Plate Recognition）范式，适用于车牌识别、工业零件检测等场景。

## 系统架构

```
输入图像
    ↓
[Stage 1: YOLOv8-OBB]
    ↓ 检测定向边界框 (OBB)
几何校正层
    ↓ 膨胀 OBB (1.2x) → 透视变换
[Stage 2: YOLOv8-Pose]
    ↓ 在变换后的图像上检测4个角点
逆映射
    ↓ 将点映射回原始图像坐标系
输出: 4个精确角点坐标
```

## 核心组件

### 1. GeometryUtils 类

提供几何变换工具函数：

- **`order_points(pts)`**: 将4个点排序为 [top-left, top-right, bottom-right, bottom-left]
- **`get_dilated_box_points(obb, pad_ratio)`**: 将 OBB 转换为4个膨胀后的角点
- **`warp_image(img, src_pts, dst_size)`**: 执行透视变换，返回变换后的图像和变换矩阵 M
- **`map_points_back(local_points, M)`**: 使用 M^(-1) 将点从局部坐标系映射回原始图像

### 2. CascadeDetector 类

主检测器类，实现完整的检测流程：

- **`__init__`**: 加载 Stage 1 (OBB) 和 Stage 2 (Pose) 模型
- **`predict(image)`**: 执行完整的检测流程
- **`visualize(image, result)`**: 可视化检测结果

## 数学原理

### 透视变换矩阵

透视变换使用 3×3 矩阵 M：

```
[x']   [m00 m01 m02] [x]
[y'] = [m10 m11 m12] [y]
[w ]   [m20 m21  1 ] [1]
```

变换后的坐标为：`(x'/w, y'/w)`

### 逆映射

使用逆矩阵 M^(-1) 将点从变换后的坐标系映射回原始图像：

```
[x_orig]   [M^(-1)] [x_local]
[y_orig] = [M^(-1)] [y_local]
[w      ]   [M^(-1)] [1      ]
```

## 安装要求

```bash
pip install ultralytics opencv-python numpy
```

## 使用方法

### 基本用法

```python
from inference import CascadeDetector
import cv2

# 初始化检测器
detector = CascadeDetector(
    obb_model_path="path/to/obb_model.pt",
    pose_model_path="path/to/pose_model.pt",
    pad_ratio=1.2,           # OBB 膨胀比例
    warp_size=(256, 128),     # Stage 2 输入尺寸
    conf_threshold=0.25,      # 置信度阈值
    device="0"                # GPU 设备
)

# 加载图像
image = cv2.imread("test_image.jpg")

# 执行检测
result = detector.predict(image)

# 检查结果
if result['success']:
    print("检测成功!")
    print(f"OBB 参数: {result['obb'].xywhr[0]}")
    print(f"关键点坐标: {result['keypoints']}")
    
    # 可视化
    vis_image = detector.visualize(image, result)
    cv2.imshow('Result', vis_image)
    cv2.waitKey(0)
```

### 命令行使用

```bash
# 检测单张图片
python inference.py \
    --obb-model path/to/obb_model.pt \
    --pose-model path/to/pose_model.pt \
    --source image.jpg \
    --show \
    --save

# 使用摄像头
python inference.py \
    --obb-model path/to/obb_model.pt \
    --pose-model path/to/pose_model.pt \
    --source 0 \
    --show

# 自定义参数
python inference.py \
    --obb-model path/to/obb_model.pt \
    --pose-model path/to/pose_model.pt \
    --source image.jpg \
    --pad-ratio 1.3 \
    --warp-size 256,128 \
    --conf 0.3 \
    --device 0
```

## 参数说明

### CascadeDetector 参数

- **`obb_model_path`**: YOLOv8-OBB 模型路径（必需）
- **`pose_model_path`**: YOLOv8-Pose 模型路径（必需）
- **`pad_ratio`**: OBB 膨胀比例，默认 1.2（20% 更大）
- **`warp_size`**: Stage 2 输入图像尺寸，默认 (256, 128)
- **`conf_threshold`**: 置信度阈值，默认 0.25
- **`device`**: 运行设备，默认自动选择

### 返回值结构

`predict()` 方法返回一个字典：

```python
{
    'obb': OBB 检测结果对象,
    'obb_points': np.ndarray,      # (4, 2) 膨胀后的 OBB 角点
    'warped_image': np.ndarray,    # 变换后的图像
    'keypoints': np.ndarray,       # (4, 2) 原始图像坐标系中的关键点
    'keypoints_local': np.ndarray, # (4, 2) 变换后图像坐标系中的关键点
    'transform_matrix': np.ndarray,# (3, 3) 透视变换矩阵 M
    'success': bool                # 是否检测成功
}
```

## 可视化

系统提供两种可视化：

1. **OBB (蓝色)**: Stage 1 检测的定向边界框
2. **关键点 (红色)**: Stage 2 检测的4个角点，带索引标签 (0, 1, 2, 3)

## 技术细节

### OBB 格式

ultralytics OBB 格式：`[center_x, center_y, width, height, angle]`
- `angle`: 角度（度），逆时针方向

### 关键点顺序

检测到的4个关键点按以下顺序排列：
- 0: 第一个关键点
- 1: 第二个关键点
- 2: 第三个关键点
- 3: 第四个关键点

### 膨胀策略

默认使用 1.2x 膨胀确保真实物理角点被包含在变换后的图像中。可根据实际情况调整 `pad_ratio`。

## 故障排除

### 问题 1: Stage 1 未检测到目标

- 检查 OBB 模型是否正确加载
- 降低 `conf_threshold`
- 确认输入图像包含目标物体

### 问题 2: Stage 2 未检测到关键点

- 检查 Pose 模型是否正确加载
- 确认模型训练时使用了正确的关键点数量（4个）
- 检查 `warp_size` 是否合适

### 问题 3: 关键点位置不准确

- 调整 `pad_ratio`（增大以确保包含所有角点）
- 检查透视变换是否正确
- 验证模型训练质量

## 示例代码

完整示例请参考 `test_cascade.py`。

## 许可证

本代码遵循项目主许可证。
