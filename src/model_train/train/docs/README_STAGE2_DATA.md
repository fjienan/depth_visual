# Stage 2 Data Preparation Script

## 概述

`prepare_stage2_data.py` 用于从原始数据集生成 Stage 2 模型的训练数据。该脚本通过模拟 Stage 1 (OBB) 的检测结果，创建裁剪后的图像和对应的关键点标注。

## 工作原理

### 数据流程

```
原始图像 + GT关键点
    ↓
计算最小外接矩形 (OBB)
    ↓
添加噪声 (模拟 Stage 1 误差)
    - 中心位置: ±5%
    - 尺寸缩放: 1.1 ~ 1.3x
    - 角度: ±5°
    ↓
透视变换到固定尺寸 (256x256)
    ↓
映射关键点到新坐标系
    ↓
保存裁剪图像和归一化标注
```

### 关键特性

1. **鲁棒性增强**: 通过添加噪声模拟 Stage 1 的检测误差，使 Stage 2 模型对输入变化更鲁棒
2. **精确映射**: 使用透视变换矩阵确保关键点坐标精确映射
3. **边界检查**: 自动过滤超出边界的关键点
4. **归一化**: 所有关键点坐标严格归一化到 [0, 1] 范围

## 输入数据格式

### 目录结构

```
source_dataset/
├── images/
│   └── train/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    └── train/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

### 标注格式 (YOLO Keypoints)

每行一个对象，格式为：
```
class_id x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
```

其中：
- `class_id`: 类别ID (整数)
- `x1, y1, ..., x4, y4`: 4个关键点的归一化坐标 (0.0-1.0)
- `v1, ..., v4`: 可见性标志 (0=不可见, 1=可见)

示例：
```
0 0.25 0.30 1 0.75 0.30 1 0.75 0.70 1 0.25 0.70 1
```

## 使用方法

### 基本用法

```bash
python prepare_stage2_data.py \
    --source ../database \
    --output ./stage2_dataset
```

### 完整参数示例

```bash
python prepare_stage2_data.py \
    --source ../database \
    --output ./stage2_dataset \
    --crop-size 256 256 \
    --num-variations 10 \
    --center-jitter 0.05 \
    --size-scale 1.1 1.3 \
    --angle-jitter 5.0 \
    --seed 42
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--source` | str | 必需 | 原始数据集目录路径 |
| `--output` | str | 必需 | 输出数据集目录路径 |
| `--crop-size` | int int | 256 256 | 裁剪图像尺寸 (宽度 高度) |
| `--num-variations` | int | 10 | 每个对象生成的变体数量 |
| `--center-jitter` | float | 0.05 | 中心位置抖动比例 (±5%) |
| `--size-scale` | float float | 1.1 1.3 | 尺寸缩放范围 (最小值 最大值) |
| `--angle-jitter` | float | 5.0 | 角度抖动 (度) |
| `--seed` | int | 42 | 随机种子 (用于可重复性) |

## 输出格式

### 目录结构

```
stage2_dataset/
├── images/
│   └── train/
│       ├── image1_obj0_var0.jpg
│       ├── image1_obj0_var1.jpg
│       ├── image1_obj1_var0.jpg
│       └── ...
└── labels/
    └── train/
        ├── image1_obj0_var0.txt
        ├── image1_obj0_var1.txt
        ├── image1_obj1_var0.txt
        └── ...
```

### 文件命名

- 格式: `{原图像名}_obj{对象索引}_var{变体索引}.jpg`
- 示例: `image001_obj0_var5.jpg` 表示 `image001.jpg` 中第0个对象的第5个变体

### 输出标注格式

与输入格式相同，但坐标是相对于裁剪图像的归一化坐标：

```
class_id x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
```

所有坐标值严格在 [0.0, 1.0] 范围内。

## 算法细节

### 1. OBB 计算

使用 `cv2.minAreaRect()` 从4个GT关键点计算最小外接矩形：

```python
rect = cv2.minAreaRect(keypoints)
# 返回: ((center_x, center_y), (width, height), angle)
```

### 2. 噪声添加

- **中心抖动**: `cx_new = cx * (1 + random(-jitter, +jitter))`
- **尺寸缩放**: `w_new = w * random(scale_min, scale_max)`
- **角度抖动**: `angle_new = angle + random(-jitter, +jitter)`

### 3. 透视变换

使用 `cv2.getPerspectiveTransform()` 计算变换矩阵：

```python
M = cv2.getPerspectiveTransform(src_corners, dst_corners)
warped = cv2.warpPerspective(image, M, (width, height))
```

### 4. 关键点映射

使用相同的变换矩阵映射关键点：

```python
# 齐次坐标
kpts_homogeneous = [x, y, 1]

# 变换
transformed = kpts_homogeneous @ M.T

# 归一化
x_new = transformed[0] / transformed[2]
y_new = transformed[1] / transformed[2]
```

## 质量保证

### 边界检查

脚本会自动检查变换后的关键点是否在裁剪图像边界内：

```python
valid = np.all((transformed_xy >= 0) & (transformed_xy < crop_size))
```

如果关键点超出边界，该变体会被跳过。

### 坐标归一化

所有输出坐标都严格归一化到 [0.0, 1.0]：

```python
x_normalized = x / crop_width
y_normalized = y / crop_height
# 然后 clamp 到 [0, 1]
```

## 性能考虑

- **内存**: 每个图像会生成 `num_objects × num_variations` 个裁剪图像
- **存储**: 确保有足够的磁盘空间
- **时间**: 处理时间与图像数量、对象数量和变体数量成正比

## 故障排除

### 问题 1: 没有生成任何裁剪图像

- 检查源图像和标注文件是否存在
- 确认标注格式正确（每行13个值：class + 4个关键点×3）
- 检查关键点是否在图像边界内

### 问题 2: 生成的裁剪图像中关键点位置不准确

- 检查原始标注是否正确
- 确认关键点顺序一致
- 验证透视变换矩阵计算

### 问题 3: 内存不足

- 减少 `--num-variations` 参数
- 分批处理图像
- 使用较小的 `--crop-size`

## 示例输出

```
Found 100 images
Generating 10 variations per object
Output directory: ./stage2_dataset
Crop size: (256, 256)
------------------------------------------------------------
Processing images: 100%|████████████| 100/100 [00:45<00:00,  2.22it/s]
------------------------------------------------------------
Processing complete!
  Total objects processed: 150
  Total crops generated: 1485
  Average crops per object: 9.90
  Output images: ./stage2_dataset/images/train
  Output labels: ./stage2_dataset/labels/train
```

## 注意事项

1. **数据增强**: 这个脚本创建的是"完美世界"数据集，但添加了足够的噪声来模拟 Stage 1 的误差
2. **可重复性**: 使用 `--seed` 参数确保结果可重复
3. **标注质量**: 确保原始标注准确，因为所有后续处理都基于这些标注
4. **存储空间**: 生成的裁剪图像数量 = 原始图像数 × 每图像对象数 × 变体数

## 后续步骤

生成数据后，可以使用标准 YOLO 训练流程训练 Stage 2 模型：

```bash
yolo pose train \
    data=stage2_dataset/data.yaml \
    model=yolov8n-pose.pt \
    epochs=100 \
    imgsz=256
```
