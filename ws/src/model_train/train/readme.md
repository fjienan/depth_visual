# LPR 级联检测系统 - 文档总览

本目录包含 LPR (License Plate Recognition) 两阶段级联检测系统的完整文档。

---

## 📚 文档索引

| 文档 | 内容 | 适用人群 |
|------|------|----------|
| [README_TRAINING.md](./docs/README_TRAINING.md) | 完整训练流程和配置 | 训练工程师 |
| [README_CASCADE.md](./docs/README_CASCADE.md) | 系统架构和推理使用 | 开发者、用户 |
| [README_STAGE2_DATA.md](./docs/README_STAGE2_DATA.md) | Stage2 数据准备详解 | 数据工程师 |

---

## 🎯 系统核心概念

### 两阶段级联架构

```
原始图像
    ↓
┌─────────────────────────┐
│  Stage 1: YOLOv8-OBB    │  粗定位
│  检测定向边界框 (OBB)     │  • 找到目标的大致位置和方向
└─────────────────────────┘  • 输出: [cx, cy, w, h, angle]
    ↓
┌─────────────────────────┐
│  几何校正层              │  桥接层
│  • 膨胀 OBB (1.1x)      │  • 确保角点在裁剪范围内
│  • 透视变换到固定尺寸    │  • 规范化为标准视图
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Stage 2: YOLOv8-Pose   │  精细定位
│  检测 4 个精确角点       │  • 在规范化图像上回归角点
└─────────────────────────┘  • 输出: [(x,y)] × 4
    ↓
┌─────────────────────────┐
│  逆映射                 │  坐标还原
│  M^(-1) 变换           │  • 映射回原始图像坐标系
└─────────────────────────┘
    ↓
输出: 4个精确角点坐标
```

### 关键设计思想

1. **分而治之**: Stage1 负责粗定位，Stage2 负责精确回归
2. **几何规范化**: 透视变换消除旋转和透视影响，简化 Stage2 任务
3. **模拟误差**: Stage2 训练时故意加入噪声，模拟 Stage1 的检测误差
4. **端到端**: 两个阶段无缝衔接，形成完整的检测流水线

---

## 📖 快速导航

### 1️⃣ 我想训练模型

**入口**: [README_TRAINING.md](./docs/README_TRAINING.md)

**关键章节**:
- **训练流程** → 分步训练 vs 一键训练
- **配置文件** → stage1_config.yaml / stage2_config.yaml
- **命令行参数** → 所有可用参数说明
- **参数调优** → 针对不同阶段的优化建议
- **故障排除** → 常见问题和解决方案

**快速开始**:
```bash
# 方式1: 分步训练（推荐）
python train_lpr.py --stage 1 --config stage1_config.yaml
python prepare_stage2_data.py --source ./database/KFS_splits
python train_lpr.py --stage 2 --config stage2_config.yaml

# 方式2: 一键训练
python train_lpr.py --full-pipeline \
    --stage1-config stage1_config.yaml \
    --stage2-config stage2_config.yaml
```

---

### 2️⃣ 我想理解系统架构

**入口**: [README_CASCADE.md](./docs/README_CASCADE.md)

**关键章节**:
- **系统架构** → 完整数据流图
- **核心组件** → GeometryUtils / CascadeDetector 类
- **数学原理** → 透视变换矩阵和逆映射
- **可视化** → 检测结果展示
- **技术细节** → OBB格式、关键点顺序、膨胀策略

**快速开始**:
```python
from inference import CascadeDetector

detector = CascadeDetector(
    obb_model_path="stage1_obb/best.pt",
    pose_model_path="stage2_pose/best.pt"
)

result = detector.predict(image)
if result['success']:
    keypoints = result['keypoints']  # 4个角点
```

---

### 3️⃣ 我想准备 Stage2 数据

**入口**: [README_STAGE2_DATA.md](./docs/README_STAGE2_DATA.md)

**关键章节**:
- **工作原理** → 数据生成流程图
- **输入数据格式** → YOLO Keypoints 格式说明
- **使用方法** → 完整命令行参数
- **算法细节** → OBB计算、噪声添加、透视变换
- **质量保证** → 边界检查、坐标归一化

**快速开始**:
```bash
python prepare_stage2_data.py \
    --source ../database/KFS-1_yolo_obb \
    --crop-size 256 256 \
    --num-variations 15
```

---

## 🔑 核心概念解释

### OBB (Oriented Bounding Box)
- **定义**: 定向边界框，可以旋转的矩形框
- **格式**: `[center_x, center_y, width, height, angle]`
- **作用**: Stage1 输出，用于粗定位目标

### 透视变换 (Perspective Transform)
- **定义**: 3×3 矩阵变换，可以处理旋转、缩放、透视
- **作用**: 将倾斜的目标"摆正"到标准视图
- **公式**: `[x', y', w] = M @ [x, y, 1]`，最终坐标 `(x'/w, y'/w)`

### 几何校正层 (Geometric Rectification)
- **作用**: 连接 Stage1 和 Stage2 的桥梁
- **步骤**:
  1. 膨胀 OBB (默认 1.1x)
  2. 计算透视变换矩阵 M
  3. 变换图像到固定尺寸

### 逆映射 (Inverse Mapping)
- **作用**: 将 Stage2 输出的角点映射回原始图像
- **方法**: 使用逆矩阵 M^(-1)
- **重要性**: 确保最终输出坐标正确

### 模拟误差 (Error Simulation)
- **目的**: 让 Stage2 对 Stage1 的误差具有鲁棒性
- **方法**:
  - 中心位置抖动: ±5%
  - 尺寸缩放: 1.1x ~ 1.3x
  - 角度抖动: ±5°
- **效果**: Stage2 能适应 Stage1 的不完美检测

---

## 📊 数据流详解

### 完整数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 原始数据                                                     │
│ • 图片: xxx.jpg                                              │
│ • 标签: xxx.txt (YOLO Keypoints: class x1 y1 v1 ... x4 y4 v4) │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1 训练                                                 │
│ • 输入: 原始图像 (640×640)                                   │
│ • 模型: YOLOv8-OBB                                           │
│ • 输出: best.pt                                              │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2 数据准备 (prepare_stage2_data.py)                   │
│ 1. 从GT关键点计算 OBB                                        │
│ 2. 添加噪声 (中心±5%, 尺寸1.1-1.3x, 角度±5°)                │
│ 3. 透视变换到 256×256                                        │
│ 4. 映射关键点到新坐标                                        │
│ 5. 生成: 355张 × 15变化 = 5,325张                           │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2 训练                                                 │
│ • 输入: 规范化图像 (256×256)                                 │
│ • 模型: YOLOv8-Pose (4 keypoints)                            │
│ • 输出: best.pt                                              │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ 级联推理 (inference.py)                                      │
│ 1. Stage1 检测 OBB                                          │
│ 2. 膨胀 + 透视变换                                           │
│ 3. Stage2 检测角点                                           │
│ 4. 逆映射到原图                                              │
│ → 输出: 4个精确角点                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ 关键参数速查

### Stage 1 (OBB)
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| imgsz | 640 | 输入图像尺寸 |
| batch | 16 | 批次大小 |
| epochs | 300 | 训练轮数 |
| lr0 | 0.01 | 初始学习率 |

### Stage 2 (Pose)
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| imgsz | 256 | 输入图像尺寸 |
| batch | 32 | 批次大小 |
| epochs | 300 | 训练轮数 |
| lr0 | 0.005 | 初始学习率（降低防止过拟合） |
| weight_decay | 0.002 | 正则化（增强防止过拟合） |

### Stage 2 数据准备
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| crop_size | 256×256 | 正方形目标 |
| num_variations | 15 | 每对象变化数 |
| center_jitter | 0.08 | 中心抖动 |
| size_scale | [1.05, 1.35] | 尺寸范围 |

### 推理
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| pad_ratio | 1.1 | OBB 膨胀比例 |
| warp_size | (256, 256) | 必须与训练一致 |
| conf_threshold | 0.25 | 置信度阈值 |

---

## 🔧 常见问题速查

### Q1: Stage1 训练收敛，但 Stage2 过拟合严重？
**A**: 
- 增加 `weight_decay`: 0.0005 → 0.002
- 降低学习率: `lr0: 0.01` → `0.005`
- 增强数据: `num_variations: 10` → `15`
- 更强增强: `mosaic: 0.5` → `1.0`

### Q2: 推理时关键点位置不准？
**A**:
- 检查 `warp_size` 是否与训练时一致
- 调整 `pad_ratio`: 太小会裁掉角点，太大会引入噪声
- 检查 `flip_idx` 配置是否正确: `[1, 0, 3, 2]`

### Q3: Stage2 数据准备生成样本太少？
**A**:
- 降低 `size_scale_range`: [1.1, 1.3] → [1.05, 1.2]
- 增加 `keypoint_margin`: 允许更多样本保留
- 增加 `num_variations`

### Q4: 两个模型如何协同工作？
**A**: 
- Stage1 负责"在哪里"（定位）
- Stage2 负责"是什么形状"（精确角点）
- 几何校正层确保它们无缝对接

---

## 📈 性能优化建议

### 数据量
- **原始数据**: 至少 200-500 张高质量标注
- **Stage2 数据**: 5,000-8,000 张（推荐 15 variations）
- **验证集**: 至少 500 张，确保验证稳定

### 训练策略
1. **先训练 Stage1**: 确保 mAP@50 > 0.9
2. **验证 Stage1**: 可视化检测结果，确认旋转角度正确
3. **准备 Stage2 数据**: 使用足够的 variations
4. **训练 Stage2**: 关注 Train/Val gap，防止过拟合
5. **端到端测试**: 在真实场景测试完整流水线

### 硬件建议
- **GPU**: 至少 8GB 显存（推荐 RTX 3060 或以上）
- **存储**: Stage2 数据约 2-5GB
- **训练时间**: Stage1 约 3-5h，Stage2 约 3-5h

---

## 📝 快速命令参考

```bash
# 完整训练流程
cd /path/to/model_train/train/LPR

# 1. 训练 Stage1
python train_lpr.py --stage 1 --config stage1_config_example.yaml

# 2. 准备 Stage2 数据
cd ../../database
python prepare_stage2_data.py \
    --source KFS-1_yolo_obb \
    --crop-size 256 256 \
    --num-variations 15

# 3. 划分数据集
python split.dataset.py --source stage2_KFS-1_yolo_obb

# 4. 训练 Stage2
cd ../train/LPR
python train_lpr.py --stage 2 --config stage2_config_example.yaml

# 5. 推理测试
python inference.py \
    --obb-model output/stage1_obb/weights/best.pt \
    --pose-model output/stage2_pose/weights/best.pt \
    --source test.jpg \
    --show
```

---

## 🎓 学习路径

### 初学者
1. 阅读 [README_CASCADE.md](./docs/README_CASCADE.md) 了解系统架构
2. 运行推理示例，理解输入输出
3. 阅读 [README_TRAINING.md](./docs/README_TRAINING.md) 的"快速开始"章节

### 进阶用户
1. 详细阅读 [README_STAGE2_DATA.md](./docs/README_STAGE2_DATA.md)
2. 理解透视变换和逆映射的数学原理
3. 调整参数优化性能

### 专家用户
1. 修改 `inference.py` 适配自定义任务
2. 优化几何校正层的膨胀策略
3. 改进数据增强策略

---

**最后更新**: 2024-01-09  
**维护者**: 深度视觉项目组
