# 数据预处理工具集

本目录包含完整的数据预处理工具链，用于将原始标注数据转换、增强、划分为可直接训练的 YOLO 格式数据集。

---

## 📋 目录结构

```
database/
├── config/                        # 配置文件目录
│   ├── json2obb_config.yaml      # JSON转换配置
│   ├── obb_augment_config.yaml   # OBB增强配置
│   ├── stage2_prepare_config.yaml # Stage2数据准备配置
│   ├── dataset_split_config.yaml # 数据集划分配置
│   └── README.md                 # 配置文件说明
├── json2txt.py                   # LabelMe JSON → YOLO-OBB 转换
├── augment_obb_data.py           # OBB 数据增强
├── prepare_stage2_data.py        # Stage2 数据准备
├── split.dataset.py              # 数据集划分（train/val/test）
└── README.md                     # 本文档
```

---

## 🚀 工具链概览

### 完整数据处理流程

```
原始 LabelMe 标注
    ↓ [json2txt.py]
YOLO-OBB 格式数据
    ↓ [augment_obb_data.py] (可选)
增强后的 OBB 数据
    ↓ [split.dataset.py]
划分后的 Stage1 数据集
    ↓ [训练 Stage1]
Stage1 训练完成
    ↓ [prepare_stage2_data.py]
Stage2 训练数据
    ↓ [split.dataset.py]
划分后的 Stage2 数据集
    ↓ [训练 Stage2]
完整训练完成
```

### 工具速查表

| 脚本 | 功能 | 输入 | 输出 | 详细说明 |
|------|------|------|------|---------|
| `json2txt.py` | JSON→OBB转换 | LabelMe JSON | YOLO-OBB | [查看](#1-json2txtpy-labelme-json--yolo-obb-转换) |
| `augment_obb_data.py` | OBB数据增强 | YOLO-OBB | 增强的OBB | [查看](#2-augment_obb_datapy-obb-数据增强) |
| `prepare_stage2_data.py` | Stage2数据生成 | YOLO-OBB/Pose | Stage2数据 | [查看](#3-prepare_stage2_datapy-stage2-数据准备) |
| `split.dataset.py` | 数据集划分 | YOLO数据 | train/val/test | [查看](#4-splitdatasetpy-数据集划分) |

---

## 1. `json2txt.py`: LabelMe JSON → YOLO-OBB 转换

### 功能

将 LabelMe 标注的 JSON 文件转换为 YOLO-OBB 格式：
- **输入**: 图片 + LabelMe JSON（混放在同一目录）
- **输出**: `images/` + `labels/` 分离结构

### 支持的标注类型

| 类型 | 点数 | 处理方式 |
|------|------|---------|
| polygon | 4 | 直接作为四角点 |
| polygon | ≠4 | 拟合最小外接旋转矩形 (需要 OpenCV) |
| rectangle | 2 | 转换为轴对齐的四角点 |

### 输出格式

**labels/*.txt** 每行格式（归一化 0-1）:
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

### 使用示例

```bash
# 基础用法（默认输出到 <input_dir>_yolo_obb）
python json2txt.py KFS-1

# 指定输出目录
python json2txt.py KFS-1 --output-dir KFS-1_yolo_obb

# 移动文件（而非复制）
python json2txt.py KFS-1 --move

# 不包含无标注图片
python json2txt.py KFS-1 --no-unlabeled

# 严格模式（遇到错误立即停止）
python json2txt.py KFS-1 --strict
```

### 输出结构

```
KFS-1_yolo_obb/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── labels/
    ├── img001.txt
    ├── img002.txt
    └── ...
```

---

## 2. `augment_obb_data.py`: OBB 数据增强

### 功能

对 YOLO-OBB 格式数据进行几何和光度增强，自动调整四角点标签。

### 支持的增强类型

#### 几何变换
- ✅ **旋转** (Rotation): 默认 ±15°（可调）
- ✅ **缩放** (Scale): 默认 0.8-1.2x（可调）
- ✅ **平移** (Translation): 默认 0%（可调）
- ✅ **透视变换** (Perspective): 默认关闭
- ✅ **水平翻转** (Horizontal Flip): 默认 0%（可调）
- ✅ **垂直翻转** (Vertical Flip): 默认 0%（可调）

#### 光度变换
- ✅ **HSV 饱和度**: 默认 ±70%
- ✅ **HSV 色调**: 默认关闭
- ✅ **HSV 明度**: 默认关闭
- ✅ **亮度调整**: 默认关闭
- ✅ **对比度调整**: 默认关闭

### 快速开始

```bash
# 基础用法（自动输出到 <source>_augmented）
python augment_obb_data.py --source KFS-1_yolo_obb

# 指定输出目录
python augment_obb_data.py \
    --source KFS-1_yolo_obb \
    --output KFS-1_augmented

# 不复制原图（只生成增强版本）
python augment_obb_data.py \
    --source KFS-1_yolo_obb \
    --no-copy-original
```

### 参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | 必需 | 源数据集目录 |
| `--output` | `<source>_augmented` | 输出目录 |
| `--num-augments` | 5 | 每张图生成的增强版本数 |
| `--rotation MIN MAX` | `-15 15` | 旋转角度范围（度） |
| `--scale MIN MAX` | `0.8 1.2` | 缩放倍数范围 |
| `--translate` | `0.0` | 平移比例 |
| `--perspective` | `0.0` | 透视变换强度（0=关闭） |
| `--flip-horizontal` | `0.0` | 水平翻转概率 |
| `--flip-vertical` | `0.0` | 垂直翻转概率 |
| `--hsv-h` | `0.0` | HSV色调增强 |
| `--hsv-s` | `0.7` | HSV饱和度增强 |
| `--hsv-v` | `0.0` | HSV明度增强 |
| `--brightness MIN MAX` | `1.0 1.0` | 亮度范围（关闭） |
| `--contrast MIN MAX` | `1.0 1.0` | 对比度范围（关闭） |

### 使用场景

#### 场景1: 大幅增强数据量（适合数据少）

```bash
python augment_obb_data.py \
    --source KFS-1_yolo_obb \
    --num-augments 10 \
    --rotation -30 30 \
    --scale 0.7 1.3 \
    --translate 0.15
```
**效果**: 355张 → 3,905张

#### 场景2: 保守增强（适合数据较多）

```bash
python augment_obb_data.py \
    --source KFS-1_yolo_obb \
    --num-augments 3 \
    --rotation -10 10 \
    --scale 0.95 1.05
```
**效果**: 355张 → 1,420张

#### 场景3: 只做光度增强

```bash
python augment_obb_data.py \
    --source KFS-1_yolo_obb \
    --num-augments 5 \
    --rotation 0 0 \
    --scale 1.0 1.0 \
    --hsv-s 0.9 \
    --brightness 0.6 1.4
```

### 注意事项

⚠️ **数据质量 vs 数量**: 不是越多越好，过度增强会引入不真实的变换  
⚠️ **磁盘空间**: 355张 × 6倍 ≈ 2GB  
⚠️ **角度范围**: 有方向性的目标（如车牌）不要用垂直翻转  
⚠️ **透视变换**: 建议值 ≤ 0.03，太大会扭曲四边形

---

## 3. `prepare_stage2_data.py`: Stage2 数据准备

### 功能

从 Stage1 (OBB) 数据生成 Stage2 (Pose) 训练数据：
- 计算 OBB 并添加噪声（模拟 Stage1 误差）
- 透视变换到固定尺寸
- 映射关键点到新坐标系

### 快速开始

```bash
# 基础用法（自动输出到 stage2_<source>）
python prepare_stage2_data.py --source KFS-1_yolo_obb

# 完整参数
python prepare_stage2_data.py \
    --source KFS-1_yolo_obb \
    --output stage2_KFS-1 \
    --crop-size 256 256 \
    --num-variations 15
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | 必需 | 源 OBB 数据集 |
| `--output` | `stage2_<source>` | 输出目录 |
| `--crop-size W H` | `256 256` | 裁剪尺寸 |
| `--num-variations` | 10 | 每对象变化数 |
| `--center-jitter` | 0.05 | 中心抖动 ±5% |
| `--size-scale MIN MAX` | `1.1 1.3` | 尺寸缩放范围 |
| `--angle-jitter` | 5.0 | 角度抖动 ±5° |

**详细说明**: 见 `../train/LPR/docs/README_STAGE2_DATA.md`

---

## 4. `split.dataset.py`: 数据集划分

### 功能

将 YOLO 数据集划分为 train / val / test 三个子集。

### 支持的输入结构

**方式1: 分离结构（推荐）**
```
dataset/
├── images/
└── labels/
```

**方式2: 混合结构**
```
dataset/
├── img001.jpg
├── img001.txt
└── ...
```

### 使用示例

```bash
# 基础用法（默认 80/15/5 划分）
python split.dataset.py --source KFS-1_yolo_obb

# 自定义比例
python split.dataset.py \
    --source KFS-1_yolo_obb \
    --train-ratio 0.8 \
    --val-ratio 0.15 \
    --test-ratio 0.05

# 不留测试集
python split.dataset.py \
    --source KFS-1_yolo_obb \
    --train-ratio 0.85 \
    --val-ratio 0.15 \
    --test-ratio 0.0

# 移动文件（而非复制）
python split.dataset.py --source KFS-1_yolo_obb --move

# 指定输出目录
python split.dataset.py \
    --source KFS-1_yolo_obb \
    --output-dir KFS-1_split
```

### 输出结构

```
KFS-1_yolo_obb_split/
├── images/
│   ├── train/  (80%)
│   ├── val/    (15%)
│   └── test/   (5%)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 划分建议

| 总样本数 | train | val | test | 说明 |
|---------|-------|-----|------|------|
| <1000 | 0.85 | 0.15 | 0.0 | 不留测试集 |
| 1000-5000 | 0.8 | 0.15 | 0.05 | 标准划分 ⭐ |
| >5000 | 0.7 | 0.2 | 0.1 | 增大验证集 |

---

## 📊 完整工作流示例

### 场景1: 从 LabelMe 标注到 Stage1 训练

```bash
cd /path/to/database

# Step 1: 转换 LabelMe JSON 为 YOLO-OBB
python json2txt.py KFS-1
# → 生成 KFS-1_yolo_obb/

# Step 2: （可选）数据增强
python augment_obb_data.py --source KFS-1_yolo_obb --num-augments 5
# → 生成 KFS-1_yolo_obb_augmented/

# Step 3: 划分数据集
python split.dataset.py --source KFS-1_yolo_obb_augmented
# → 生成 KFS-1_yolo_obb_augmented_split/

# Step 4: 训练 Stage1
cd ../train/LPR
# 修改 data_1.yaml: path: ../../database/KFS-1_yolo_obb_augmented_split
python train_lpr.py --stage 1 --config stage1_config_example.yaml
```

### 场景2: 准备 Stage2 数据并训练

```bash
cd /path/to/database

# Step 1: 准备 Stage2 数据
python prepare_stage2_data.py \
    --source KFS-1_yolo_obb_augmented \
    --crop-size 256 256 \
    --num-variations 15
# → 生成 stage2_KFS-1_yolo_obb_augmented/

# Step 2: 划分数据集
python split.dataset.py --source stage2_KFS-1_yolo_obb_augmented
# → 生成 stage2_KFS-1_yolo_obb_augmented_split/

# Step 3: 训练 Stage2
cd ../train/LPR
# 修改 data_2.yaml: path: ../../database/stage2_KFS-1_yolo_obb_augmented_split
python train_lpr.py --stage 2 --config stage2_config_example.yaml
```

### 场景3: 使用配置文件（推荐）

```bash
cd /path/to/database

# 所有脚本都支持配置文件
python json2txt.py --config config/json2obb_config.yaml
python augment_obb_data.py --config config/obb_augment_config.yaml
python prepare_stage2_data.py --config config/stage2_prepare_config.yaml
python split.dataset.py --config config/dataset_split_config.yaml
```

**配置文件详细说明**: 见 `config/README.md`

---

## 🎯 数据量建议

### Stage 1 (OBB)

| 原始数据 | 增强倍数 | 最终数量 | 训练时间 | 效果 |
|---------|---------|---------|---------|------|
| 355 | 无增强 | 355 | ~1-2h | ⚠️ 可能欠拟合 |
| 355 | 5倍 | 2,130 | ~3-4h | ✅ 基本可用 |
| 355 | 10倍 | 3,905 | ~5-7h | ✅ 良好 |

### Stage 2 (Pose)

| 原始OBB | num_variations | 最终数量 | 训练时间 | 效果 |
|---------|----------------|---------|---------|------|
| 355 | 10 | 3,550 | ~2-3h | ⚠️ 可能过拟合 |
| 355 | 15 | 5,325 | ~3-5h | ✅ 推荐 ⭐ |
| 355 | 20 | 7,100 | ~5-7h | ✅ 高精度 |

---

## ⚙️ 高级用法

### 批量处理多个数据集

```bash
#!/bin/bash
for dataset in KFS-1 KFS-2 KFS-3; do
    python json2txt.py ${dataset}
    python augment_obb_data.py --source ${dataset}_yolo_obb --num-augments 5
    python split.dataset.py --source ${dataset}_yolo_obb_augmented
done
```

### 验证数据质量

```bash
# 检查生成的样本数
ls -1 KFS-1_yolo_obb_augmented/images/ | wc -l

# 检查标签格式
head KFS-1_yolo_obb_augmented/labels/*.txt

# 统计角度分布（需要自定义脚本）
python analyze_angle_distribution.py KFS-1_yolo_obb_augmented
```

### 清理中间文件

```bash
# 只保留最终的划分后数据，删除中间产物
rm -rf KFS-1_yolo_obb KFS-1_yolo_obb_augmented
# 保留 KFS-1_yolo_obb_augmented_split/
```

---

## 🔧 故障排除

### Q1: json2txt.py 报错 "No images found"

**A**: 检查输入目录是否包含图片和JSON文件

### Q2: augment_obb_data.py 生成样本数少于预期

**A**: 
- 减小 `--rotation` 和 `--translate` 范围
- 检查是否有大量样本因超出边界被过滤

### Q3: prepare_stage2_data.py 生成数据少

**A**:
- 降低 `size_scale_range`
- 检查原始关键点标注是否正确

### Q4: split.dataset.py 划分比例不对

**A**:
- 确保三个比例之和 = 1.0
- 检查是否有空标签文件

---

## 📚 相关文档

- [配置文件说明](./config/README.md)
- [OBB增强详细说明](./README_AUGMENT_OBB.md) *(可选保留)*
- [Stage2数据准备详解](../train/LPR/docs/README_STAGE2_DATA.md)
- [完整训练流程](../train/LPR/docs/README_TRAINING.md)

---

**最后更新**: 2024-01-09  
**维护者**: 深度视觉项目组
