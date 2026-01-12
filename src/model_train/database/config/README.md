# 数据预处理配置文件说明

本目录包含所有数据预处理脚本的配置文件，用于规范化和简化数据处理流程。

---

## 📁 目录结构

```
config/
├── json2obb_config.yaml          # JSON → OBB 转换配置
├── obb_augment_config.yaml       # OBB 数据增强配置
├── stage2_prepare_config.yaml    # Stage2 数据准备配置
├── dataset_split_config.yaml     # 数据集划分配置
└── README.md                     # 本文档
```

---

## 🚀 快速开始

### 完整数据处理流程

```bash
cd /home/fjienan/Desktop/workspace/depth_visual/ws/src/model_train/database

# Step 1: JSON 转 YOLO-OBB
python json2txt.py \
    --config config/json2obb_config.yaml

# Step 2: OBB 数据增强（可选）
python augment_obb_data.py \
    --config config/obb_augment_config.yaml

# Step 3: 划分 Stage1 数据集
python split.dataset.py \
    --config config/dataset_split_config.yaml

# Step 4: 准备 Stage2 数据
python prepare_stage2_data.py \
    --config config/stage2_prepare_config.yaml

# Step 5: 划分 Stage2 数据集
python split.dataset.py \
    --source stage2_<your_dataset>
```

---

## 📋 配置文件详解

### 1. `json2obb_config.yaml`

**作用**：将 LabelMe JSON 标注转换为 YOLO-OBB 格式

**关键参数**：
```yaml
io:
  input_dir: "../KFS-1"           # LabelMe 数据目录
  output_dir: null                # 默认: <input>_yolo_obb

processing:
  move: false                     # 是否移动（而非复制）
  include_unlabeled: true         # 是否包含无标注图片
  strict: false                   # 是否严格模式
```

**使用示例**：
```bash
# 基础用法
python json2txt.py KFS-1

# 使用配置文件（推荐）
python json2txt.py --config config/json2obb_config.yaml

# 命令行覆盖配置
python json2txt.py --config config/json2obb_config.yaml --move --strict
```

**输出格式**：
```
output_dir/
├── images/  (所有图片)
└── labels/  (YOLO-OBB txt: class x1 y1 x2 y2 x3 y3 x4 y4)
```

---

### 2. `obb_augment_config.yaml`

**作用**：对 OBB 数据进行几何和光度增强

**关键参数**：
```yaml
augmentation:
  num_augments: 5                 # 每张图生成的增强版本数

geometric:
  rotation_min: -15.0             # 旋转范围
  rotation_max: 15.0
  scale_min: 0.8                  # 缩放范围
  scale_max: 1.2
  flip_horizontal: 0.0            # 翻转概率

photometric:
  hsv_s: 0.7                      # 饱和度增强
  brightness_min: 1.0             # 亮度范围
  brightness_max: 1.0
```

**使用场景**：

```bash
# 场景1: 保守增强（数据量已足够）
# 修改配置：num_augments: 3, rotation: [-10, 10]
python augment_obb_data.py --config config/obb_augment_config.yaml

# 场景2: 激进增强（数据量很少）
# 修改配置：num_augments: 10, rotation: [-30, 30], scale: [0.7, 1.3]
python augment_obb_data.py --config config/obb_augment_config.yaml

# 场景3: 只做光度增强（不改变几何）
# 修改配置：rotation: [0, 0], scale: [1.0, 1.0], hsv_s: 0.9
python augment_obb_data.py --config config/obb_augment_config.yaml
```

**预期效果**：
- 355 张原图 × 5 增强 = 2,130 张（含原图）
- 输出：`<source>_augmented/`

---

### 3. `stage2_prepare_config.yaml`

**作用**：从 Stage1 (OBB) 数据生成 Stage2 (Pose) 训练数据

**关键参数**：
```yaml
crop:
  width: 256                      # 裁剪宽度
  height: 256                     # 裁剪高度

augmentation:
  num_variations: 10              # 每个对象的变化数

jitter:
  center_jitter: 0.05             # 中心抖动 ±5%
  size_scale_min: 1.1             # 尺寸缩放范围
  size_scale_max: 1.3
  angle_jitter: 5.0               # 角度抖动 ±5°
```

**重要说明**：
- `crop_size` 应与目标长宽比匹配：
  - 正方形目标 → `[256, 256]`
  - 2:1 长条目标 → `[256, 128]`
- `num_variations` 决定数据量：
  - 355 张 × 10 = 3,550 张
  - 355 张 × 15 = 5,325 张（推荐）

**使用示例**：
```bash
# 基础用法
python prepare_stage2_data.py --config config/stage2_prepare_config.yaml

# 命令行覆盖参数
python prepare_stage2_data.py \
    --config config/stage2_prepare_config.yaml \
    --num-variations 15 \
    --crop-size 256 256
```

---

### 4. `dataset_split_config.yaml`

**作用**：将数据集划分为 train / val / test

**关键参数**：
```yaml
split_ratio:
  train: 0.8                      # 80% 训练集
  val: 0.15                       # 15% 验证集
  test: 0.05                      # 5% 测试集

processing:
  move: false                     # 是否移动（清空源目录）
  seed: 42                        # 随机种子
  shuffle: true                   # 是否打乱
```

**使用示例**：
```bash
# 基础用法
python split.dataset.py --config config/dataset_split_config.yaml

# 快速命令（使用默认配置）
python split.dataset.py --source KFS-1_yolo_obb

# 自定义划分比例
python split.dataset.py \
    --source KFS-1_yolo_obb \
    --train-ratio 0.85 \
    --val-ratio 0.15 \
    --test-ratio 0.0
```

**输出结构**：
```
<source>_split/
├── images/
│   ├── train/  (80%)
│   ├── val/    (15%)
│   └── test/   (5%)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

## 📊 完整工作流示例

### 场景1：从零开始的 LPR 级联检测系统

```bash
cd /home/fjienan/Desktop/workspace/depth_visual/ws/src/model_train/database

# 1. 准备原始数据（LabelMe JSON + 图片）
ls KFS-1/
# → 1_frame_00000.json, 1_frame_00000.jpg, ...

# 2. 转换为 YOLO-OBB
python json2txt.py KFS-1
# → 生成 KFS-1_yolo_obb/

# 3. （可选）增强 OBB 数据
vim config/obb_augment_config.yaml  # 修改 num_augments: 5
python augment_obb_data.py --source KFS-1_yolo_obb
# → 生成 KFS-1_yolo_obb_augmented/ (2,130张)

# 4. 划分 Stage1 数据
python split.dataset.py --source KFS-1_yolo_obb_augmented
# → 生成 KFS-1_yolo_obb_augmented_split/

# 5. 训练 Stage1 (OBB)
cd ../train/LPR
vim data_1.yaml  # path: ../../database/KFS-1_yolo_obb_augmented_split
python train_lpr.py --stage 1 --config stage1_config_example.yaml
# → 得到 stage1_obb/weights/best.pt

# 6. 准备 Stage2 数据
cd ../../database
vim config/stage2_prepare_config.yaml  # num_variations: 15
python prepare_stage2_data.py \
    --source KFS-1_yolo_obb_augmented \
    --config config/stage2_prepare_config.yaml
# → 生成 stage2_KFS-1_yolo_obb_augmented/ (5,325张)

# 7. 划分 Stage2 数据
python split.dataset.py --source stage2_KFS-1_yolo_obb_augmented
# → 生成 stage2_KFS-1_yolo_obb_augmented_split/

# 8. 训练 Stage2 (Pose)
cd ../train/LPR
vim data_2.yaml  # path: ../../database/stage2_KFS-1_yolo_obb_augmented_split
python train_lpr.py --stage 2 --config stage2_config_example.yaml
# → 得到 stage2_pose/weights/best.pt

# 9. 级联推理
python inference.py \
    --obb-model output/stage1_obb/weights/best.pt \
    --pose-model output/stage2_pose/weights/best.pt \
    --source test.jpg \
    --show
```

---

## 🎯 参数调优指南

### OBB 数据增强

| 数据量 | num_augments | rotation | scale | flip |
|--------|-------------|----------|-------|------|
| 很少(<100) | 10-15 | [-30, 30] | [0.7, 1.3] | 0.5 |
| 中等(100-500) | 5-8 | [-15, 15] | [0.8, 1.2] | 0.5 |
| 充足(>500) | 3-5 | [-10, 10] | [0.9, 1.1] | 0.5 |

### Stage2 数据准备

| 原图数量 | num_variations | 生成数量 | 训练时间 |
|---------|----------------|---------|---------|
| 355 | 10 | 3,550 | ~2-3h |
| 355 | 15 | 5,325 | ~3-5h |
| 355 | 20 | 7,100 | ~5-7h |

**推荐**：
- **快速验证**：`num_variations: 8-10`
- **生产环境**：`num_variations: 15` ⭐
- **高精度要求**：`num_variations: 20` 或采集更多原图

### 数据集划分

| 总样本数 | train | val | test | 说明 |
|---------|-------|-----|------|------|
| <1000 | 0.85 | 0.15 | 0.0 | 不留测试集 |
| 1000-5000 | 0.8 | 0.15 | 0.05 | 标准划分 ⭐ |
| >5000 | 0.7 | 0.2 | 0.1 | 增大验证集 |

---

## ⚠️ 常见问题

### Q1: 为什么要用配置文件？

**A**: 
- ✅ **可重复性**：记录所有参数，确保实验可复现
- ✅ **版本管理**：可以 git 管理配置文件
- ✅ **团队协作**：统一参数标准
- ✅ **快速切换**：不同场景用不同配置

### Q2: 命令行参数和配置文件冲突怎么办？

**A**: 命令行参数**优先级更高**，会覆盖配置文件
```bash
# 配置文件里 num_augments: 5
# 命令行指定 --num-augments 10
# 最终使用 10
```

### Q3: 配置文件参数不全怎么办？

**A**: 未指定的参数会使用**脚本的默认值**

### Q4: 如何调试配置文件？

**A**: 
```bash
# 1. 小数据量测试
python augment_obb_data.py \
    --source test_small_dataset \
    --config config/obb_augment_config.yaml

# 2. 检查输出
ls -lh test_small_dataset_augmented/images/ | wc -l

# 3. 查看生成的标签
head test_small_dataset_augmented/labels/*.txt
```

---

## 📚 扩展阅读

- [OBB 数据增强详细说明](../README_AUGMENT_OBB.md)
- [Stage2 数据准备说明](../../train/LPR/docs/README_STAGE2_DATA.md)
- [完整训练流程](../../train/LPR/README_TRAINING.md)

---

## 🔧 维护说明

**添加新配置文件时**：
1. 创建 `<script_name>_config.yaml`
2. 在本 README 添加说明
3. 在对应脚本添加 `--config` 参数支持

**修改配置格式时**：
1. 更新示例配置文件
2. 更新本 README
3. 确保向后兼容

---

**最后更新**: 2024-01-09  
**维护者**: 深度视觉项目组
