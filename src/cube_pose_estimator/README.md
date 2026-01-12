# cube_pose_estimator

一个 ROS 2（`ament_python`）包，包含两个节点：

1. **`cube_pose_node`**：YOLO 检测立方体可见面 4 角点 → PnP 解算 → 输出立方体中心位姿
2. **`pose_fusion_node`**：将视觉位姿与 IMU 进行轻量级融合（位置用 Kalman Filter，姿态用 slerp）

核心流程（`cube_pose_node`）：

- **输入图像**：支持订阅 ROS 图像话题（topic），也支持直接读取 USB 摄像头（OpenCV `VideoCapture`）
- **角点识别**：Ultralytics YOLO 两阶段级联（Stage1 OBB → 透视矫正 warp → Stage2 Pose）输出“可见面 4 个角点”
- **位姿解算**：对平面正方形执行 **PnP** 求解该面位姿
- **中心推算**：沿该面法向偏移 \(L/2\)（\(L\)=立方体边长），得到**立方体中心**在相机坐标系下的位置/位姿

融合流程（`pose_fusion_node`，轻量但实用）：

- **位置（position）**：3D 常速度 Kalman Filter，测量来自视觉位置；可选把 IMU 线加速度作为预测输入（默认关闭）
- **姿态（orientation）**：视觉与 IMU 的四元数做 slerp（权重可调）

## Topics

- **`cube_pose_node` 订阅（仅 `input.mode:=topic` 时）**
  - **`image_topic`**（默认：`/camera/color/image_raw`）：`sensor_msgs/Image`

- **`cube_pose_node` 发布**
  - **`/cube_pose/pose`**：`geometry_msgs/PoseStamped`（立方体中心在相机坐标系下）
  - **`/cube_pose/face_corners_3d`**：`geometry_msgs/PolygonStamped`（该可见面 4 角点在相机坐标系下的 3D 坐标）
  - **`/cube_pose/visualization`**：`sensor_msgs/Image`（可视化叠加图）
  - **`/cube_pose/markers`**（可配置）：`visualization_msgs/MarkerArray`（用于 RViz 里显示立方体与中心点）

- **`pose_fusion_node` 订阅**
  - **`input.pose_topic`**（默认：`/cube_pose/pose`）：`geometry_msgs/PoseStamped`
  - **`imu.topic`**（默认：`/imu/data`）：`sensor_msgs/Imu`（topic 输入）

- **`pose_fusion_node` 发布**
  - **`output.pose_topic`**（默认：`/cube_pose/fused_pose`）：`geometry_msgs/PoseStamped`

## Params

完整参数见：

- `config/cube_pose_estimator.yaml`（`cube_pose_node`）
- `config/cube_pose_fusion.yaml`（`pose_fusion_node`）

### cube_pose_node 参数（重点）

重点关注：

- **`yolo.obb_model_path`**：Stage 1 OBB 模型 `.pt` 的绝对路径
- **`yolo.pose_model_path`**：Stage 2 Pose 模型 `.pt` 的绝对路径
- **`target.cube_size_mm`**：立方体边长 \(L\)（单位：mm）
- **`camera.intrinsics.*`** 与 **`camera.distortion`**：OpenCV 针孔相机内参 + 畸变参数

PnP / 优化相关：

- **`pnp.method`**：PnP 方法（默认 `IPPE_SQUARE`）
- **`pnp.refine`**：单帧 BA（重投影 LM refine）开关：`NONE` / `LM`（推荐先开 `LM`）
- **`pnp.refine_iterations` / `pnp.refine_eps`**：LM 迭代与收敛阈值
- **`pnp.ba_window_size`**：多帧 BA 窗口大小（<=1 关闭；>1 表示用最近 N 帧角点共同优化“同一位姿”）
  - **适用**：目标相对静止/缓慢运动
  - **代价**：会引入少量“滞后”（窗口越大越明显）

Debug / 观测：

- **`debug.log_yolo_points`**：打印 YOLO 角点像素坐标（TL/TR/BR/BL）
- **`debug.log_pnp_success`**：打印 PnP 成功时 `tvec/cube_center/reproj_rmse`
- **`debug.log_pnp_failures` / `debug.log_pnp_residuals`**：PnP 失败诊断（建议打开便于定位）

### 参数文件（`config/cube_pose_estimator.yaml`）各字段作用说明

> 说明：参数文件里使用 ROS 2 的 `ros__parameters` 格式，节点名为 `cube_pose_estimator`。

- **`input.mode`**：输入模式
  - **`topic`**：订阅 `image_topic` 获取图像
  - **`usb`**：节点内部用 OpenCV 直连 USB 摄像头抓图（不需要 ROS Image topic）

- **`image_topic`**：当 `input.mode:=topic` 时生效，订阅的图像话题名（`sensor_msgs/Image`）。

- **`usb.*`**：当 `input.mode:=usb` 时生效
  - **`usb.camera_id`**：摄像头编号（通常 0/1/2…）
  - **`usb.width` / `usb.height`**：期望采集分辨率（实际值取决于摄像头是否支持）
  - **`usb.fps`**：期望采集帧率（同时决定节点内部定时器频率）
  - **`usb.backend`**：OpenCV 后端选择
    - **`any`**：让 OpenCV 自己选
    - **`v4l2`**：Linux 下常用（更稳定时可用）

- **`camera.intrinsics.*`**：相机内参（OpenCV 针孔模型）
  - **`fx, fy, cx, cy`**：用于 PnP/投影/画坐标轴等，数值来自相机标定

- **`camera.distortion`**：镜头畸变参数（OpenCV 标准 5 参数）
  - 格式为 **`[k1, k2, p1, p2, k3]`**
  - 来自 `cv2.calibrateCamera()` 标定输出；不准会显著影响距离/姿态

- **`target.cube_size_mm`**：立方体边长 \(L\)，单位 **mm**
  - 节点会先用这个面（边长 \(L\) 的正方形）做 PnP，再沿面法向偏移 \(L/2\) 得到立方体中心

- **`yolo.*`**：Ultralytics YOLO 两阶段级联检测参数
  - **`yolo.obb_model_path`**：Stage 1 OBB 模型路径（粗定位旋转框）
  - **`yolo.pose_model_path`**：Stage 2 Pose 模型路径（输出 4 个角点关键点）
  - **`yolo.conf_threshold`**：置信度阈值（过高会漏检，过低会误检）
  - **`yolo.device`**：运行设备（如 `cuda` / `cpu` / `0`）
  - **`yolo.pad_ratio`**：Stage1 的 OBB 膨胀比例（膨胀后再 warp，通常更稳）
  - **`yolo.warp_size`**：透视矫正后 Stage2 输入的固定尺寸 `[width, height]`

- **`pnp.*`**：PnP 求解相关参数
  - **`pnp.method`**：PnP 方法（优先推荐 `IPPE_SQUARE`，适合平面正方形，且可处理多解）
  - **`pnp.reproj_error_max_px`**：最大允许重投影误差（超过则认为本帧解算失败）
  - **`pnp.min_face_size_px`**：面在图像上太小时直接拒绝（避免远距离小目标导致深度/中心跳变）；0 表示禁用
  - **`pnp.refine`**：单帧 BA（重投影 LM refine）开关：`NONE` / `LM`
  - **`pnp.ba_window_size`**：多帧 BA 窗口大小（<=1 关闭；>1 启用）

- **`frame_ids.*`**：坐标系命名
  - **`frame_ids.camera_frame`**：发布姿态/TF 时使用的相机坐标系（建议 optical frame）
  - **`frame_ids.cube_center_frame`**：TF 子坐标系名（立方体中心）

- **`publish.*`**：开关项（是否发布某些输出）
  - **`publish.pose`**：发布 `/cube_pose/pose`
  - **`publish.face_corners_3d`**：发布 `/cube_pose/face_corners_3d`
  - **`publish.visualization`**：发布 `/cube_pose/visualization`
  - **`publish.tf`**：广播 TF（`camera_frame` → `cube_center_frame`）

- **`visualization.*`**：可视化参数
  - **`visualization.axes_length_mm`**：叠加坐标轴长度（单位 mm）
  - **`visualization.show_fps`**：是否在叠加图上显示 FPS

## Run

编译：

```bash
cd /home/fjienan/Desktop/workspace/depth_visual/ws
colcon build --packages-select cube_pose_estimator
source install/setup.bash
```

启动 `cube_pose_node`（默认参数文件）：

```bash
ros2 launch cube_pose_estimator cube_pose_estimator.launch.py
```

启动“检测 + IMU 融合”（默认参数文件）：

```bash
ros2 launch cube_pose_estimator cube_pose_estimator_with_fusion.launch.py
```

### 快速验证（推荐）

- 看视觉中心点输出：

```bash
ros2 topic echo /cube_pose/pose
```

- 看融合后的输出：

```bash
ros2 topic echo /cube_pose/fused_pose
```

- 检查 IMU 是否有数据（默认）：

```bash
ros2 topic echo /imu/data
```

- 确认参数是否真的加载成功（避免“改了 yaml 但 launch 没用到”）：

```bash
ros2 param get /cube_pose_estimator pnp.ba_window_size
ros2 param get /cube_pose_estimator pnp.refine
ros2 param get /cube_pose_fusion imu.topic
```

### 通过参数文件配置（推荐方式）

该包的 launch 文件默认只接收 **params_file 路径**（不逐项透传参数），因此推荐你直接复制/修改 YAML：

- `config/cube_pose_estimator.yaml`
- `config/cube_pose_fusion.yaml`

然后通过 launch 参数指定：

```bash
ros2 launch cube_pose_estimator cube_pose_estimator_with_fusion.launch.py \
  estimator_params_file:=/abs/path/to/cube_pose_estimator.yaml \
  fusion_params_file:=/abs/path/to/cube_pose_fusion.yaml
```

### USB 摄像头直连（不走 ROS Image topic）

```bash
# 1) 复制 config/cube_pose_estimator.yaml 到你的工作目录
# 2) 修改其中：
#   input.mode: "usb"
#   usb.camera_id / usb.width / usb.height / usb.fps / usb.backend
#   yolo.obb_model_path / yolo.pose_model_path (绝对路径)
# 3) 启动：
ros2 launch cube_pose_estimator cube_pose_estimator.launch.py \
  params_file:=/abs/path/to/cube_pose_estimator.yaml
```

### 在 RViz 中显示“立方体中心 + 立方体”

1. 打开 RViz：`rviz2`
2. **Fixed Frame** 选择为你的相机坐标系（对应参数 `frame_ids.camera_frame`，例如 `rgb_camera_link` 或 `camera_color_optical_frame`）
3. Add → **MarkerArray**
   - Topic 选择 **`/cube_pose/markers`**（或你在参数 `markers.topic` 里配置的 topic）
4. （可选）Add → **TF**，查看 `camera_frame` → `cube_center_frame` 的变换是否在刷新

也可以直接运行（并指定参数文件）：

```bash
ros2 run cube_pose_estimator cube_pose_node --ros-args --params-file \
  /home/fjienan/Desktop/workspace/depth_visual/ws/src/cube_pose_estimator/config/cube_pose_estimator.yaml
```

`pose_fusion_node` 也可以直接运行：

```bash
ros2 run cube_pose_estimator pose_fusion_node --ros-args --params-file \
  /home/fjienan/Desktop/workspace/depth_visual/ws/src/cube_pose_estimator/config/cube_pose_fusion.yaml
```

## Python deps

本节点运行时依赖：

- `opencv-python`
- `numpy`
- `ultralytics`

请按你的环境（系统 / venv / conda）自行安装。

## Notes / 常见坑

- **IMU 融合的 frame 对齐**：`pose_fusion_node` 的姿态融合默认假设 IMU `frame_id` 与视觉 pose 的 `frame_id` 表达的是同一坐标系；如果不一致，需要 TF/外参对齐，或先关闭姿态融合。
- **位置融合使用 IMU 加速度**：默认 `fuse.position.use_imu_accel=false`，因为加速度的坐标系对齐与重力处理更容易踩坑；如果你已经做了对齐/去重力，可以打开它来增强预测。
- **多帧 BA 的适用范围**：`pnp.ba_window_size>1` 默认假设窗口内“位姿几乎不变”，适合静止目标；若目标快速运动，窗口过大可能会造成滞后甚至拉偏，建议先从 3~5 帧试起。