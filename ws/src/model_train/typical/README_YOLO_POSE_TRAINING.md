# YOLO Pose 模型训练完整指南

这是一个完整的YOLO Pose模型训练解决方案，支持从配置文件读取所有参数，也可以通过命令行参数覆盖配置。

## 📋 目录

- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [配置文件说明](#配置文件说明)
- [命令行参数](#命令行参数)
- [使用示例](#使用示例)
- [参数详解](#参数详解)
- [常见问题](#常见问题)

## 🔧 安装依赖

```bash
# 安装核心依赖
pip install ultralytics torch torchvision

# 如果需要使用YAML配置文件
pip install pyyaml
```

## 🚀 快速开始

### 1. 准备数据配置文件

首先，你需要准备一个数据配置文件（YAML格式），例如 `data.yaml`:

```yaml
# data.yaml
train: /path/to/train/images
val: /path/to/val/images
test: /path/to/test/images  # 可选

# 类别数量
nc: 1

# 类别名称
names:
  0: person

# Pose模型特有：关键点信息
kpt_shape: [17, 3]  # COCO格式：17个关键点，每个关键点有3个值(x, y, visibility)
```

### 2. 准备训练配置文件

复制并修改 `train_config.yaml` 文件，设置你的训练参数。

### 3. 开始训练

```bash
# 使用配置文件训练
python yolo_train.py --config train_config.yaml

# 或者直接使用命令行参数
python yolo_train.py \
    --model yolov8n-pose.pt \
    --data data.yaml \
    --epochs 500 \
    --batch 16 \
    --project ./runs/train \
    --name my_pose_model
```

## 📝 配置文件说明

配置文件 `train_config.yaml` 包含了所有可调节的训练参数，分为以下几个部分：

### 1. 模型配置 (`model`)

- `path`: 模型路径或模型名称
  - 预训练模型: `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, `yolov8l-pose.pt`, `yolov8x-pose.pt`
  - 自定义模型: `/path/to/your/model.pt`
- `pretrained`: 是否使用预训练权重
- `freeze`: 冻结层数（用于迁移学习）

### 2. 数据配置 (`data`)

- `config`: 数据配置文件路径（必需）
- `imgsz`: 输入图像尺寸（常见: 320, 416, 512, 640, 800, 1024）
- `batch`: 批次大小（根据GPU内存调整）
- `workers`: 数据加载工作进程数
- `rect`: 是否使用矩形训练
- `multi_scale`: 是否使用多尺度训练
- `augmentation`: 数据增强参数（详见配置文件）

### 3. 训练配置 (`training`)

- `epochs`: 训练轮数
- `optimizer`: 优化器类型（SGD, Adam, AdamW, RMSProp等）
- `lr0`: 初始学习率
- `lr_scheduler`: 学习率调度器（linear, cosine等）
- `patience`: 早停耐心值
- `amp`: 是否使用半精度训练（FP16）

### 4. 损失函数配置 (`loss`)

- `box`: 边界框损失权重
- `cls`: 类别损失权重
- `pose`: 关键点损失权重（Pose模型特有）
- `dfl`: DFL损失权重
- `kobj`: 关键点可见性损失权重

### 5. 输出配置 (`output`)

- `project`: 项目输出路径
- `name`: 训练运行名称
- `exist_ok`: 是否覆盖已有输出

### 6. 设备配置 (`device`)

- `device`: 训练设备（cpu, 0, 1, 2, ... 或 0,1,2,3）

### 7. Pose模型特有配置 (`pose`)

- `kpt_shape`: 关键点形状 `[关键点数量, 维度]`
- `kpt_visibility_threshold`: 关键点可见性阈值
- `kpt_loss_type`: 关键点损失类型（l1, l2, smooth_l1）

## 💻 命令行参数

所有配置文件中的参数都可以通过命令行参数覆盖：

```bash
python yolo_train.py [选项]

必需参数（如果未提供配置文件）:
  --model MODEL           模型路径或模型名称
  --data DATA             数据配置文件路径

可选参数:
  --config CONFIG         训练配置文件路径 (默认: train_config.yaml)
  --epochs EPOCHS         训练轮数
  --batch BATCH           批次大小
  --imgsz IMGSZ           输入图像尺寸
  --lr0 LR0               初始学习率
  --optimizer OPTIMIZER   优化器类型
  --device DEVICE         训练设备
  --project PROJECT       项目输出路径
  --name NAME            训练运行名称
  --resume RESUME         从检查点恢复训练
  --amp                   使用半精度训练
  --plots                 保存训练图表
  --help                  显示帮助信息
```

## 📚 使用示例

### 示例1: 基础训练（使用配置文件）

```bash
python yolo_train.py --config train_config.yaml
```

### 示例2: 覆盖配置文件中的部分参数

```bash
python yolo_train.py \
    --config train_config.yaml \
    --epochs 100 \
    --batch 32 \
    --lr0 0.001
```

### 示例3: 不使用配置文件，直接使用命令行参数

```bash
python yolo_train.py \
    --model yolov8n-pose.pt \
    --data /path/to/data.yaml \
    --epochs 500 \
    --batch 16 \
    --imgsz 640 \
    --project ./runs/train \
    --name my_pose_model \
    --device 0
```

### 示例4: 从检查点恢复训练

```bash
python yolo_train.py \
    --config train_config.yaml \
    --resume runs/train/yolov8n-pose/weights/last.pt
```

### 示例5: 快速测试（少量epochs，小batch）

```bash
python yolo_train.py \
    --config train_config.yaml \
    --epochs 10 \
    --batch 8 \
    --name quick_test
```

### 示例6: 高质量训练（大模型，长训练）

```bash
python yolo_train.py \
    --config train_config.yaml \
    --model yolov8m-pose.pt \
    --epochs 1000 \
    --batch 8 \
    --imgsz 1024 \
    --lr0 0.001 \
    --name high_quality_training
```

## 📖 参数详解

### 模型选择

YOLO Pose模型有多个尺寸可选：

- **yolov8n-pose**: 最小最快，适合实时应用
- **yolov8s-pose**: 小模型，平衡速度和精度
- **yolov8m-pose**: 中等模型，更好的精度
- **yolov8l-pose**: 大模型，高精度
- **yolov8x-pose**: 最大模型，最高精度

### 批次大小选择

根据GPU内存选择合适的批次大小：

| GPU内存 | 模型大小 | 推荐batch | imgsz |
|---------|---------|-----------|-------|
| 8GB     | yolov8n | 16-32     | 640   |
| 8GB     | yolov8s | 8-16      | 640   |
| 16GB    | yolov8m | 8-16      | 640   |
| 16GB    | yolov8l | 4-8       | 640   |
| 24GB+   | yolov8x | 4-8       | 640   |

### 学习率设置

- **初始学习率 (lr0)**: 
  - 从零训练: 0.01
  - 迁移学习: 0.001 - 0.0001
  - 微调: 0.0001 - 0.00001

- **学习率调度器**:
  - `linear`: 线性衰减
  - `cosine`: 余弦衰减（推荐）
  - `lrf`: 最终学习率比例（默认0.01）

### 数据增强

数据增强可以提升模型泛化能力，但要注意：

- **Mosaic**: 提高小目标检测能力
- **Mixup**: 提高模型鲁棒性
- **HSV增强**: 提高对不同光照条件的适应性
- **旋转/平移**: 提高对姿态变化的适应性

### 关键点配置

对于Pose模型，关键点配置很重要：

- **COCO格式**: 17个关键点 `[17, 3]`
- **自定义格式**: 根据你的数据集调整

关键点标注格式（YOLO Pose）:
```
<class_id> <x_center> <y_center> <width> <height> <x1> <y1> <v1> <x2> <y2> <v2> ...
```

其中 `v` 是关键点可见性：
- `0`: 不可见
- `1`: 遮挡但可见
- `2`: 完全可见

## ❓ 常见问题

### 1. 内存不足 (CUDA out of memory)

**解决方案**:
- 减小批次大小: `--batch 8` 或 `--batch 4`
- 减小图像尺寸: `--imgsz 512` 或 `--imgsz 416`
- 使用更小的模型: `yolov8n-pose` 而不是 `yolov8m-pose`
- 启用半精度训练: `--amp`

### 2. 训练速度慢

**解决方案**:
- 增加 `workers` 数量（但不要超过CPU核心数）
- 使用 `rect=True` 矩形训练
- 使用更小的图像尺寸
- 使用GPU训练（确保 `device` 设置为GPU）

### 3. 验证指标不改善

**解决方案**:
- 检查学习率是否合适（可能太大或太小）
- 增加训练轮数
- 检查数据质量
- 调整损失函数权重
- 使用数据增强

### 4. 关键点检测不准确

**解决方案**:
- 检查关键点标注是否正确
- 调整 `pose` 损失权重
- 增加关键点相关的数据增强
- 使用更大的模型
- 增加训练数据量

### 5. 配置文件加载失败

**解决方案**:
- 检查YAML文件格式是否正确
- 确保文件路径正确
- 检查文件编码（应为UTF-8）
- 使用命令行参数代替配置文件

### 6. 如何选择最佳参数？

**建议流程**:
1. 从小模型开始: `yolov8n-pose`
2. 使用默认参数训练少量epochs（如10-20）
3. 观察训练曲线，调整学习率
4. 逐步增加模型大小和训练轮数
5. 根据验证集表现调整超参数

## 📊 训练输出

训练完成后，输出目录结构：

```
project/name/
├── weights/
│   ├── best.pt          # 最佳模型权重（验证集表现最好）
│   ├── last.pt          # 最新模型权重
│   └── epoch*.pt        # 每个epoch的权重（如果设置了save_period）
├── results.csv          # 训练指标CSV文件
├── confusion_matrix.png # 混淆矩阵
├── F1_curve.png        # F1曲线
├── P_curve.png         # 精确率曲线
├── R_curve.png         # 召回率曲线
├── PR_curve.png        # PR曲线
├── results.png         # 训练结果图表
├── train_batch*.jpg    # 训练批次可视化
└── val_batch*.jpg      # 验证批次可视化
```

## 🔍 监控训练

### TensorBoard

如果启用了TensorBoard（`tensorboard: true`），可以使用：

```bash
tensorboard --logdir runs/train
```

### 查看训练日志

训练过程中的日志会显示：
- 每个epoch的训练损失
- 验证指标（mAP, precision, recall等）
- 关键点检测指标（Pose模型特有）
- 学习率变化
- 训练时间

## 🎯 最佳实践

1. **数据准备**:
   - 确保数据标注质量高
   - 数据分布要均衡
   - 训练集和验证集要合理划分（通常8:2或7:3）

2. **训练策略**:
   - 先用小模型快速验证
   - 逐步增加模型复杂度
   - 使用学习率调度器
   - 设置合理的早停策略

3. **参数调优**:
   - 从默认参数开始
   - 一次只调整一个参数
   - 记录每次实验的结果
   - 使用验证集评估，不要只看训练集

4. **模型评估**:
   - 使用验证集评估模型
   - 关注关键点检测的准确性
   - 检查不同姿态下的表现
   - 测试在真实场景中的表现

## 📞 获取帮助

如果遇到问题：

1. 检查配置文件格式是否正确
2. 查看训练日志中的错误信息
3. 参考Ultralytics官方文档: https://docs.ultralytics.com
4. 检查数据格式是否符合要求

## 📄 许可证

本脚本基于Ultralytics YOLO，遵循相应的开源许可证。

---

**祝训练顺利！** 🚀
