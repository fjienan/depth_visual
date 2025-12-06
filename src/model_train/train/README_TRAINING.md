# YOLO 训练脚本使用说明

这是一个完整的YOLO训练脚本，包含了所有必要的训练参数和模型导出功能。

## 安装依赖

```bash
pip install ultralytics torch torchvision
```

## 基本使用方法

### 1. 准备数据配置文件

首先创建数据配置YAML文件，参考 `data_config_example.yaml`：

```yaml
# my_dataset.yaml
train: /path/to/train/images
val: /path/to/val/images
nc: 3  # 类别数量
names:
  0: cat
  1: dog
  2: bird
```

### 2. 基本训练命令

```bash
python yolo_train.py \
    --data-config my_dataset.yaml \
    --export-path ./output \
    --epochs 100 \
    --batch-size 16
```

## 详细参数说明

### 模型参数
- `--model-size`: 模型大小 (yolov8n/s/m/l/x, yolov9t/s/m/c/e, yolov10n/s/m/b/l)
- `--model-path`: 自定义模型文件路径
- `--pretrained`: 使用预训练权重
- `--freeze-layers`: 冻结层数量

### 数据参数
- `--data-config`: 数据配置文件路径 (必需)
- `--img-size`: 训练图像尺寸 (默认: 640)
- `--batch-size`: 批次大小 (默认: 16)
- `--data-fraction`: 使用数据集的比例 (0.0-1.0)
- `--single-class`: 单类别训练
- `--rect-training`: 矩形训练

### 训练参数
- `--epochs`: 训练轮数 (默认: 100)
- `--optimizer`: 优化器 (SGD/Adam/AdamW/RMSProp)
- `--learning-rate`: 初始学习率 (默认: 0.01)
- `--warmup-epochs`: 预热轮数 (默认: 3)
- `--patience`: 早停耐心值 (默认: 50)
- `--cosine-lr`: 余弦学习率调度器

### 导出参数
- `--export-path`: 模型导出路径 (必需)
- `--export-formats`: 导出格式 (onnx,torchscript,coreml,tensorrt,openvino)
- `--half-precision`: 半精度导出
- `--int8-quantization`: INT8量化导出
- `--dynamic-axes`: 动态轴导出

## 使用示例

### 示例1: 基础训练

```bash
python yolo_train.py \
    --data-config ./configs/coco.yaml \
    --export-path ./runs/train \
    --model-size yolov8s \
    --epochs 100 \
    --batch-size 32 \
    --img-size 640
```

### 示例2: 高质量训练

```bash
python yolo_train.py \
    --data-config ./configs/my_dataset.yaml \
    --export-path ./runs/high_quality \
    --model-size yolov8m \
    --epochs 200 \
    --batch-size 16 \
    --img-size 1024 \
    --learning-rate 0.001 \
    --optimizer AdamW \
    --cosine-lr \
    --patience 100
```

### 示例3: 快速训练测试

```bash
python yolo_train.py \
    --data-config ./configs/my_dataset.yaml \
    --export-path ./runs/quick_test \
    --model-size yolov8n \
    --epochs 10 \
    --batch-size 8 \
    --img-size 320 \
    --data-fraction 0.1
```

### 示例4: 多格式导出

```bash
python yolo_train.py \
    --data-config ./configs/my_dataset.yaml \
    --export-path ./runs/multi_format \
    --model-size yolov8s \
    --epochs 100 \
    --batch-size 16 \
    --export-formats "onnx,torchscript,tensorrt" \
    --half-precision \
    --dynamic-axes
```

### 示例5: 断点续训

```bash
python yolo_train.py \
    --data-config ./configs/my_dataset.yaml \
    --export-path ./runs/resume_training \
    --model-size yolov8s \
    --resume \
    --epochs 200 \
    --batch-size 16
```

### 示例6: 迁移学习

```bash
python yolo_train.py \
    --data-config ./configs/custom_dataset.yaml \
    --export-path ./runs/transfer_learning \
    --model-path ./pretrained/yolov8s.pt \
    --freeze-layers 10 \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 0.0001
```

## 输出结构

训练完成后，导出目录将包含以下结构：

```
export_path/
├── weights/              # 训练权重文件
│   ├── best.pt          # 最佳权重
│   ├── last.pt          # 最新权重
│   └── ...
├── results/              # 训练结果
│   ├── results.csv      # 训练指标
│   ├── confusion_matrix.png
│   ├── PR_curve.png
│   └── ...
├── logs/                 # 训练日志
├── exported_models/      # 导出的模型
│   ├── model.onnx       # ONNX格式
│   ├── model.torchscript # TorchScript格式
│   └── ...
└── training_run/         # 完整训练记录
```

## 数据集格式要求

### 图像格式
- 支持格式: JPG, PNG, BMP, TIFF
- 文件命名: 任意，但需要与标注文件对应

### 标注格式
YOLO格式标注文件，每个图像对应一个 `.txt` 文件：

```
<class_id> <x_center> <y_center> <width> <height>
```

坐标值都是归一化的 (0-1)。

### 数据集结构示例

```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

## 性能优化建议

### GPU训练
- 使用GPU训练：脚本会自动检测并使用可用的GPU
- 指定GPU ID：`--gpu-id 0`

### 内存优化
- 减小批次大小：`--batch-size 8`
- 减小图像尺寸：`--img-size 512`
- 使用梯度累积：`--nominal-batch-size 64`

### 速度优化
- 使用小模型：`--model-size yolov8n`
- 多尺度训练：`--multi-scale`
- 矩形训练：`--rect-training`

## 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减小批次大小
   python yolo_train.py --batch-size 8 ...
   ```

2. **数据路径错误**
   ```bash
   # 检查数据配置文件中的路径是否正确
   python yolo_train.py --data-config /path/to/config.yaml ...
   ```

3. **CUDA内存不足**
   ```bash
   # 使用CPU或减小批次大小
   python yolo_train.py --batch-size 4 ...
   ```

4. **权限错误**
   ```bash
   # 确保导出路径有写权限
   python yolo_train.py --export-path /path/with/permission ...
   ```

## 高级配置

### 自定义超参数
可以通过修改脚本中的默认参数来调整训练行为。

### 分布式训练
对于大规模训练，可以使用PyTorch的分布式训练功能。

### 自定义数据增强
可以通过修改数据配置文件或脚本来实现自定义数据增强。