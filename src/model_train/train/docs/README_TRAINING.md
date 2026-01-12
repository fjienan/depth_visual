# LPR 两阶段训练指南

## 概述

LPR (License Plate Recognition) 训练系统采用两阶段级联检测架构：

1. **Stage 1: OBB 模型** - 粗定位，检测定向边界框
2. **Stage 2: Pose 模型** - 精细角点检测，检测4个精确角点

## 系统架构

```
原始图像
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

## 训练流程

### 方式1: 分步训练（推荐用于调试）

#### 步骤1: 训练 Stage 1 (OBB 模型)

```bash
python train_lpr.py \
    --stage 1 \
    --config stage1_config.yaml
```

#### 步骤2: 准备 Stage 2 数据

```bash
python prepare_stage2_data.py \
    --source ./database/KFS_splits \
    --output ./database/stage2_data \
    --crop-size 256 128 \
    --num-variations 10
```

#### 步骤3: 训练 Stage 2 (Pose 模型)

```bash
python train_lpr.py \
    --stage 2 \
    --config stage2_config.yaml
```

### 方式2: 完整流程（一键训练）

> 说明：该“一键训练（full-pipeline）”功能已从 `train_lpr.py` 移除。  
> 请按上面的 **方式1: 分步训练（推荐）** 执行（先 Stage1 → 再准备 Stage2 数据 → 再 Stage2）。

## 配置文件

### Stage 1 配置文件 (stage1_config.yaml)

```yaml
model:
  path: "yolov8n-obb.pt"
  pretrained: true

data:
  config: "./database/KFS_splits/data.yaml"
  imgsz: 640
  batch: 16
  workers: 8

training:
  epochs: 300
  optimizer: SGD
  lr0: 0.01
  patience: 50

output:
  project: "runs/obb"
  name: "stage1_obb"
```

### Stage 2 配置文件 (stage2_config.yaml)

```yaml
model:
  path: "yolov8n-pose.pt"
  pretrained: true

data:
  config: "./database/stage2_data/data.yaml"
  imgsz: 256
  batch: 32
  workers: 8

training:
  epochs: 200
  optimizer: SGD
  lr0: 0.01
  patience: 50

stage2_data:
  crop_size: [256, 128]
  num_variations: 10
  center_jitter: 0.05
  size_scale_range: [1.1, 1.3]
  angle_jitter: 5.0

output:
  project: "runs/pose"
  name: "stage2_pose"
```

## 数据准备

### 原始数据格式

原始数据应该是标准的 YOLO 格式，包含4点关键点标注：

```
database/
├── images/
│   └── train/
│       ├── img001.jpg
│       └── ...
└── labels/
    └── train/
        ├── img001.txt  # 格式: class x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
        └── ...
```

### Stage 2 数据生成

`prepare_stage2_data.py` 会：
1. 从原始数据中读取4点关键点
2. 计算 OBB（定向边界框）
3. 添加抖动模拟 Stage 1 的误差
4. 执行透视变换，将 OBB 区域变换为规范矩形
5. 将关键点映射到变换后的坐标系
6. 生成新的训练数据集

## 命令行参数

### train_lpr.py

#### 单阶段训练

```bash
# Stage 1
python train_lpr.py --stage 1 --config stage1_config.yaml

# Stage 2
python train_lpr.py --stage 2 --config stage2_config.yaml
```

#### 完整流程

> 说明：该“一键训练（full-pipeline）”功能已从 `train_lpr.py` 移除。  
> 请按本文前面的 **方式1: 分步训练（推荐）** 执行完整流程。

#### 常用参数

- `--stage`: 训练阶段 (1 或 2)
- `--config`: 配置文件路径
- `--model`: 模型路径（覆盖配置文件）
- `--data`: 数据配置文件路径（覆盖配置文件）
- `--epochs`: 训练轮数（覆盖配置文件）
- `--batch`: 批次大小（覆盖配置文件）
- `--device`: 训练设备（覆盖配置文件）
- `--resume`: 从检查点恢复训练

## 训练输出

### Stage 1 输出

```
model_train/output/stage1_obb__<dataset>_n1234/
├── weights/
│   ├── best.pt      # 最佳模型
│   └── last.pt      # 最新模型
├── results.png      # 训练曲线
└── ...
```

### Stage 2 输出

```
model_train/output/stage2_pose__<dataset>_n5678/
├── weights/
│   ├── best.pt      # 最佳模型
│   └── last.pt      # 最新模型
├── results.png      # 训练曲线
└── ...
```

## 使用训练好的模型

训练完成后，使用 `inference.py` 进行推理：

```bash
python inference.py \
    --obb-model output/<stage1_run_dir>/weights/best.pt \
    --pose-model output/<stage2_run_dir>/weights/best.pt \
    --source test_image.jpg \
    --show
```

## 参数调优建议

### Stage 1 (OBB)

- **图像尺寸**: 640 或更大，确保能检测到小目标
- **批次大小**: 根据GPU内存调整，通常 16-32
- **数据增强**: 可以较强，因为需要适应各种角度和位置
- **训练轮数**: 300-500 轮

### Stage 2 (Pose)

- **图像尺寸**: 256x128 或类似，因为已经裁剪过了
- **批次大小**: 可以较大，通常 32-64
- **数据增强**: 应该较弱，因为图像已经规范化了
- **训练轮数**: 200-300 轮

## 故障排除

### 问题1: Stage 1 训练失败

- 检查数据配置文件路径是否正确
- 确认数据格式正确（YOLO格式，包含4点关键点）
- 检查模型路径是否正确

### 问题2: Stage 2 数据准备失败

- 确认原始数据包含有效的关键点标注
- 检查 `crop_size` 是否合适
- 确认输出目录有写入权限

### 问题3: Stage 2 训练失败

- 检查 Stage 2 数据是否正确生成
- 确认数据配置文件中的路径正确
- 检查 `kpt_shape` 是否正确设置为 `[4, 3]`

## 最佳实践

1. **先训练 Stage 1**: 确保 OBB 模型性能良好
2. **验证 Stage 1**: 在验证集上测试 OBB 模型
3. **准备 Stage 2 数据**: 使用训练好的 Stage 1 模型（或真实标注）准备数据
4. **训练 Stage 2**: 在准备好的数据上训练 Pose 模型
5. **端到端测试**: 使用两个模型进行完整的推理测试

## 参考文档

- [级联检测系统说明](./docs/README_CASCADE.md)
- [Stage 2 数据准备说明](./docs/README_STAGE2_DATA.md)
