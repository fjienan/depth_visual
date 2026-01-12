# depth_visual

一个将 **两阶段级联 YOLO（OBB → Pose）** 用于立方体角点检测，并在 ROS 2 中完成 **PnP 位姿解算 +（可选）BA 优化 + IMU 融合** 的项目。

本仓库本身就是一个 **ROS 2 colcon workspace**（根目录包含 `build/ install/ log/ src/`）。

## 快速导航

- **从这里开始（仓库结构总览）**：本 README 的「[目录结构](#目录结构)」
- **训练与推理入口**：`src/model_train/`
  - [训练说明](src/model_train/train/readme.md)
  - [训练脚本](src/model_train/train/train_lpr.py)
  - [推理脚本](src/model_train/train/inference.py)
  - [数据工具链说明](src/model_train/database/README.md)
- **ROS2 实时定位入口**：`src/cube_pose_estimator/`
  - [包内 README（topics/params/运行）](src/cube_pose_estimator/README.md)
  - [默认参数：PnP/BA](src/cube_pose_estimator/config/cube_pose_estimator.yaml)
  - [默认参数：IMU 融合](src/cube_pose_estimator/config/cube_pose_fusion.yaml)
- **关键约定：四点顺序 TL/TR/BR/BL**：见「[数据格式约定（非常重要：四点顺序）](#数据格式约定非常重要四点顺序)」与 [`src/model_train/train/config/data_2.yaml`](src/model_train/train/config/data_2.yaml)

## 目录（建议第一次按这个顺序读）

- [项目内容](#项目内容)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [快速开始（训练与推理）](#快速开始训练与推理)
- [数据格式约定（非常重要：四点顺序）](#数据格式约定非常重要四点顺序)
- [ROS 2：实时定位（PnP + BA + IMU 融合）](#ros-2实时定位pnp--ba--imu-融合)
- [关键参数说明（定位相关）](#关键参数说明定位相关)
- [常见问题](#常见问题)
- [License](#license)

## 项目内容

- **模型训练与推理（Python）**：位于 `src/model_train/`
  - Stage 1：YOLO-OBB 旋转框粗定位
  - Stage 2：YOLO-Pose 4 角点精定位
  - 配套的数据处理工具链：LabelMe → YOLO-OBB/Pose、增强、划分、Stage2 数据生成
- **ROS 2 实时定位（Python，ament_python）**：位于 `src/cube_pose_estimator/`
  - `cube_pose_node`：YOLO 检测 4 角点 → PnP → 输出立方体中心位姿
  - `pose_fusion_node`：视觉位姿 + IMU 轻量融合（位置 KF、姿态 slerp）

## 目录结构

```text
depth_visual/
├── src/
│   ├── model_train/                 # 训练/推理 + 数据工具链
│   │   ├── train/
│   │   └── database/
│   └── cube_pose_estimator/         # ROS2 包（PnP + BA + IMU 融合）
├── build/                           # colcon build 产物（已被 .gitignore 忽略）
├── install/
├── log/
└── README.md
```

## 环境要求

- **Python**：3.8+（建议 3.10）
- **OpenCV / numpy / ultralytics**：用于推理与几何计算
- **ROS 2**：Humble（推荐）/ Foxy

> 说明：训练依赖（PyTorch/CUDA）与推理依赖请按你的机器环境自行安装。

## 快速开始（训练与推理）

### 1) 安装 Python 依赖（示例）

```bash
python3 -m pip install -U pip
pip install ultralytics opencv-python numpy pyyaml
```

### 2) 数据准备（LabelMe → YOLO）

```bash
cd src/model_train/database

# LabelMe JSON → YOLO-OBB
python json2txt.py --source /abs/path/to/labelme_dir

# (可选) OBB 数据增强
python augment_obb_data.py --source /abs/path/to/labelme_dir_yolo_obb --num-augments 5

# 划分 train/val/test
python split.dataset.py --source /abs/path/to/labelme_dir_yolo_obb_augmented
```

#### 标注工具：XAnyLabeling

当前数据标注/导出以 **XAnyLabeling** 为主。为了复用本仓库现有的数据工具链，推荐你在 XAnyLabeling 中：

- 导出为 **LabelMe JSON（或兼容的 JSON）** → 走 `src/model_train/database/json2txt.py`（YOLO-OBB）
- 或直接导出为 **YOLO 格式**（如果你已经导出为 YOLO-OBB / YOLO-Pose，也可以跳过部分转换步骤）

> 只要导出的 JSON 字段兼容（包含图像尺寸 + 4 角点/多边形），即可接入本仓库的转换与增强脚本。

更多细节见：`src/model_train/database/README.md`

## 数据格式约定（非常重要：四点顺序）

本项目的核心标签是“**带顺序的四点坐标**”，该顺序会贯穿：

- **Stage2（Pose）训练标签**（4 keypoints）
- **推理输出的 4 角点**（像素坐标）
- **PnP 解算**（要求 2D 点与 3D 模型点顺序一致）

### 四点顺序（统一约定）

四个点的顺序固定为：

- **0 = TL**（Top-Left，左上）
- **1 = TR**（Top-Right，右上）
- **2 = BR**（Bottom-Right，右下）
- **3 = BL**（Bottom-Left，左下）

你当前的标注工具是 **XAnyLabeling**，请务必保证导出数据在进入训练/推理前已经符合以上顺序（或在转换脚本中完成一致化）。

### 相关配置/文档链接（建议都看一遍）

- `src/model_train/train/config/data_2.yaml`：Pose 数据集的 `kpt_shape` 与 **水平翻转映射 `flip_idx: [1, 0, 3, 2]`**（对应 TL↔TR，BL↔BR）
- `src/model_train/database/README.md`：数据转换/增强/划分工具链说明
- `src/model_train/train/readme.md`：训练与推理入口说明
- `src/cube_pose_estimator/README.md`：ROS2 运行、PnP/BA 参数与输出 topics

### 3) 训练 Stage1 / Stage2

```bash
cd ../train

# Stage1 (OBB)
python train_lpr.py --stage 1 --config config/stage1_config_example.yaml

# 准备 Stage2 数据（用数据工具链脚本）
cd ../database
python prepare_stage2_data.py --source /abs/path/to/<your_stage1_dataset_dir>
python split.dataset.py --source /abs/path/to/<your_stage2_dataset_dir>
cd ../train

# Stage2 (Pose)
python train_lpr.py --stage 2 --config config/stage2_config_example.yaml
```

更多细节见：`src/model_train/train/readme.md`

### （规划）合成数据集

后续如果要引入 **生成/合成数据集**（例如程序化渲染 + 自动标注），建议仍输出到与训练脚本兼容的目录结构（YOLO images/labels），这样可以无缝接入 `src/model_train/train/` 的训练流程。

## ROS 2：实时定位（PnP + BA + IMU 融合）

### 1) 编译

在仓库根目录：

```bash
colcon build --packages-select cube_pose_estimator
source install/setup.bash
```

### 2) 运行（视觉 PnP）

```bash
ros2 launch cube_pose_estimator cube_pose_estimator.launch.py
```

默认参数文件：`src/cube_pose_estimator/config/cube_pose_estimator.yaml`

### 3) 运行（视觉 + IMU 融合）

```bash
ros2 launch cube_pose_estimator cube_pose_estimator_with_fusion.launch.py
```

默认融合参数：`src/cube_pose_estimator/config/cube_pose_fusion.yaml`

### 4) 快速验证

- 看视觉中心点输出：

```bash
ros2 topic echo /cube_pose/pose
```

- 看融合输出：

```bash
ros2 topic echo /cube_pose/fused_pose
```

- 看 IMU（默认）：

```bash
ros2 topic echo /imu/data
```

- 确认参数确实加载：

```bash
ros2 param get /cube_pose_estimator pnp.refine
ros2 param get /cube_pose_estimator pnp.ba_window_size
ros2 param get /cube_pose_fusion imu.topic
```

## 关键参数说明（定位相关）

参数文件：`src/cube_pose_estimator/config/cube_pose_estimator.yaml`

- **`target.cube_size_mm`**：立方体边长（mm）
- **`camera.intrinsics.*` / `camera.distortion`**：相机内参/畸变（OpenCV 5 参数）
- **`yolo.obb_model_path` / `yolo.pose_model_path`**：两阶段模型路径（建议绝对路径）

PnP / BA：

- **`pnp.refine`**：单帧 BA（LM 重投影优化）`NONE|LM`
- **`pnp.ba_window_size`**：多帧 BA 窗口（<=1 关闭；>1 用最近 N 帧角点共同优化同一位姿）
  - 适合：目标相对静止/缓慢运动
  - 代价：窗口越大越“稳”，但会引入滞后，运动快时可能拉偏

IMU 融合：

参数文件：`src/cube_pose_estimator/config/cube_pose_fusion.yaml`

- **`imu.topic`**：IMU 输入 topic（默认 `/imu/data`）
- **姿态融合注意**：默认假设 IMU `frame_id` 与视觉 pose 的 `frame_id` 表达同一坐标系；若不一致，需要做 TF/外参对齐或关闭姿态融合。

## 常见问题

- **`git push` 401 / SSH publickey**：建议使用 GitHub PAT（HTTPS）或正确配置 SSH key。
- **中心点抖动**：优先打开 `pnp.refine=LM`，再尝试 `pnp.ba_window_size=3~5`；同时检查相机内参/畸变与 `cube_size_mm` 是否正确。

## License

- `src/cube_pose_estimator/` 内含 `LICENSE`（以该包内为准）。

