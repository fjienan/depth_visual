# 立方体单面四角点 → PnP位姿 → 立方体中心坐标 解算（ROS 2）设计文档

## 1. 项目概述

### 1.1 功能描述
本项目实现一个基于 ROS 2 的实时视觉定位系统，用 **YOLO 识别立方体某一可见面的四个角点**，再用已知 **立方体边长** 进行 PnP 解算，最终输出 **立方体中心** 的 3D 坐标/位姿。

核心流程：
1. 从摄像头获取 RGB 图像
2. YOLO（可选两阶段级联：OBB → Pose）检测到该“面”的 **4 个角点像素坐标（2D）**
3. 以“该面”为平面正方形模型，通过 PnP 求解该面坐标系相对相机的位姿 \((R, t)\)
4. 由 \((R, t)\) 将“面中心”沿法向偏移 \(\frac{L}{2}\) 得到 **立方体中心**（\(L\)=立方体边长）
5. 发布：立方体中心位姿（`PoseStamped`）、TF、可视化图像等

### 1.2 技术栈
- **ROS 2**: 系统框架
- **OpenCV**: 图像处理和PnP求解
- **Ultralytics YOLO**: 角点检测（Keypoint Detection）
- **cv_bridge**: ROS与OpenCV图像格式转换
- **Python 3**: 主要编程语言

---

## 2. 系统架构

### 2.1 模块结构
```
rgb_depth_detect/
├── rgb_depth_detect/
│   ├── __init__.py
│   ├── four_pt_pnp.py          # 主ROS节点
│   ├── yolo_detector.py        # YOLO检测器封装（可选两阶段级联）
│   └── pnp_solver.py           # PnP求解器
├── config/
│   └── four_pt_pnp.yaml        # 参数配置文件
├── launch/
│   └── four_pt_pnp.launch.py   # 启动文件
├── package.xml
├── setup.py
└── setup.cfg
```

> 说明：**YOLO 模型文件不强制放在包内**（您已说明模型在别处保存）。因此设计里不再依赖 `models/` 目录，改为参数文件中配置 **模型的绝对路径**（或网络共享路径/挂载路径）。

### 2.2 数据流程
```
摄像头 → ROS Image Topic
   ↓
YOLO检测器（可选：Stage1 OBB → 透视矫正 → Stage2 Pose）
   ↓
四角点坐标(2D, TL/TR/BR/BL)
   ↓                         相机内参 + 畸变
PnP求解（面坐标系 → 相机坐标系）
   ↓
由“面位姿”推立方体中心位姿/坐标
   ↓
发布 Pose / TF / 可视化
```

---

## 3. 详细设计

### 3.1 主节点类（建议命名：`CubeCenterPnPNode`）

#### 3.1.1 订阅话题
- **话题名**: `/camera/color/image_raw` (可配置)
- **消息类型**: `sensor_msgs/Image`
- **频率**: 30Hz (取决于相机)

#### 3.1.2 发布话题
- **检测结果可视化**
  - 话题名: `/cube_pose/visualization`
  - 消息类型: `sensor_msgs/Image`
  - 内容: 标注了角点和坐标轴的图像

- **位姿信息**
  - 话题名: `/cube_pose/pose`
  - 消息类型: `geometry_msgs/PoseStamped`
  - 内容: **立方体中心**在相机坐标系下的位姿

- **角点坐标**
  - 话题名: `/cube_pose/face_corners_3d`
  - 消息类型: `geometry_msgs/PolygonStamped`
  - 内容: 被检测到的“立方体面”四个角点在相机坐标系下的 3D 坐标

#### 3.1.3 TF广播
- **父坐标系**: `camera_color_optical_frame` (可配置)
- **子坐标系**: `cube_center`（可配置）
- **内容**: **立方体中心**相对于相机的变换

#### 3.1.4 参数列表
```yaml
# 相机话题
image_topic: "/camera/color/image_raw"

# 相机内参（从标定文件读取）
camera:
  intrinsics:
    fx: 615.0
    fy: 615.0
    cx: 320.0
    cy: 240.0
  distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
  
# 目标物体参数
target:
  # 说明：我们检测的是“立方体某一个面”，该面是边长为 cube_size 的正方形
  cube_size: 100.0  # 立方体边长 L（单位：mm）
  # （可选）如果你想支持“非立方体”的某个正方形板，也可以保留 square_size
  # square_size: 100.0  # 正方形边长（单位：mm）
  
# YOLO模型参数（模型文件在别处保存：建议用绝对路径）
yolo:
  # 方案A：单模型直接输出4个角点（Pose）
  # pose_model_path: "/abs/path/to/corner_pose.pt"
  #
  # 方案B（推荐，与你的 inference.py 一致）：两阶段级联
  obb_model_path: "/abs/path/to/stage1_obb.pt"
  pose_model_path: "/abs/path/to/stage2_pose.pt"
  pad_ratio: 1.2         # Stage1 OBB 膨胀比例（用于透视矫正更稳）
  warp_size: [256, 256]  # Stage2 输入的归一化尺寸（width,height）
  confidence_threshold: 0.5
  device: "cuda"  # cuda / cpu
  
# PnP求解参数
pnp:
  # 对“正方形平面”目标，建议优先考虑 IPPE / IPPE_SQUARE（更适合平面姿态，且能处理两解）
  method: "IPPE_SQUARE"  # ITERATIVE / SQPNP / IPPE / IPPE_SQUARE ...
  use_extrinsic_guess: false
  refine_iterations: 10
  
# 坐标系定义
frame_ids:
  camera_frame: "camera_color_optical_frame"
  target_frame: "cube_center"
  
# 可视化设置
visualization:
  enable: true
  draw_axes: true
  axes_length: 50.0  # mm
  draw_corners: true
  draw_corner_ids: true
  fps_display: true
  
# 调试选项
debug:
  log_detection_time: true
  log_pnp_solution: true
  save_failed_images: false
  failed_images_dir: "/tmp/pnp_failed"
```

#### 3.1.5 `distortion` 是什么参数？（必须搞清楚）
`camera.distortion` 是 **相机镜头畸变系数**，用于把理想针孔模型与真实镜头的成像偏差联系起来。最常见（OpenCV `calibrateCamera`）给出的 5 个参数为：

- **k1, k2, k3**：径向畸变（越靠近画面边缘越明显，常见桶形/枕形）
- **p1, p2**：切向畸变（镜头装配/光轴偏心导致）

所以你在标定里通常会看到（和你现有 `calibrate_camera.py` 输出一致）：
`distortion: [k1, k2, p1, p2, k3]`

要点：
- 如果畸变参数不准，PnP 的距离（Z）和姿态会明显漂。
- 对于广角镜头/鱼眼镜头，可能需要 fisheye 模型（那就不是这 5 个参数的形式了）。

---

### 3.2 YOLO检测器模块 (`YoloDetector`)

#### 3.2.1 功能
- 加载YOLO关键点检测模型
- 对输入图像进行推理
- 返回检测到的四个角点的像素坐标

#### 3.2.2 输入/输出
```python
class YoloDetector:
    def __init__(self, model_path, confidence_threshold, device):
        """初始化YOLO检测器"""
        
    def detect_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        检测四个角点
        
        Args:
            image: BGR图像，shape=(H, W, 3)
            
        Returns:
            corners: 角点坐标数组，shape=(4, 2)，格式为[[x1,y1], [x2,y2], ...]
                    顺序: 左上、右上、右下、左下
                    如果检测失败返回None
        """
```

#### 3.2.3 角点顺序约定
```
0: 左上角 (Top-Left)
1: 右上角 (Top-Right)
2: 右下角 (Bottom-Right)
3: 左下角 (Bottom-Left)

(0)-------(1)
 |         |
 |    +    |  (中心点)
 |         |
(3)-------(2)
```

#### 3.2.4 参考你现有 `model_train/train/inference.py` 的识别思路（两阶段级联）
你当前的推理思路（我已阅读 `depth_visual/ws/src/model_train/train/inference.py`）是典型的 **“粗定位→几何矫正→精回归角点”**，目的是在小目标、倾斜、透视强时仍保持角点精度：

- **Stage 1（OBB）**：用 YOLO-OBB 找到目标的旋转框（中心/宽高/角度）
- **几何矫正**：把 OBB 按 `pad_ratio` 膨胀后取 4 个角点，排序为 TL/TR/BR/BL，然后做透视变换把目标“拉正”到固定 `warp_size`
- **Stage 2（Pose）**：在拉正后的图上跑 YOLO-Pose 输出 4 个关键点（角点）
- **反变换**：把关键点从 warped 图坐标用逆矩阵映射回原图坐标
- **角点顺序稳定化**：你代码里用 `GeometryUtils.order_points_indices()` 再按几何关系重排为 TL/TR/BR/BL（这一点对 PnP 很重要）

结论：如果你实际场景里“角点很小/目标倾斜大/背景复杂”，这套级联会比单阶段 pose 稳很多；本 ROS 方案建议直接按此思路实现（后续开始写代码时再落地）。

---

### 3.3 PnP求解器模块 (`PnPSolver`)

#### 3.3.1 功能
- 根据2D角点坐标和3D物体模型求解相机到物体的变换
- 支持多种PnP算法
- 处理畸变校正

#### 3.3.2 3D物体模型定义
以“被检测到的那个立方体面”的中心为原点建立 **面坐标系**（face frame）：
```python
# 假设立方体边长为 L (单位: mm)，该面为边长 L 的正方形
# face 坐标系约定：
# - X: 面内向右
# - Y: 面内向下
# - Z: 面法向（关键！后续用于推立方体中心）
#   建议约定：+Z 指向“远离立方体中心的外侧”（即面外法向）
object_points_3d = np.array([
    [-L/2,  L/2, 0],  # 左上
    [ L/2,  L/2, 0],  # 右上
    [ L/2, -L/2, 0],  # 右下
    [-L/2, -L/2, 0],  # 左下
], dtype=np.float32)
```

#### 3.3.2.1 立方体中心怎么从“任意一个面”推出来？
PnP 解出来的是 **face 坐标系到 camera 坐标系** 的刚体变换：
\[
\mathbf{p}_c = \mathbf{R}\,\mathbf{p}_f + \mathbf{t}
\]
其中 \(\mathbf{t}\) 是 **face 原点（面中心）在相机坐标系下的位置**。

立方体中心在 face 坐标系里，位于面中心沿 **指向立方体内部** 的方向偏移 \(\frac{L}{2}\)：
- 如果你约定 **+Z 是外法向**（远离立方体内部），那么“指向内部”就是 \(-Z\)
- 所以：
\[
\mathbf{p}^{cube\_center}_c = \mathbf{t} + \mathbf{R}\,[0,\,0,\,-L/2]^T
\]

这条公式解决了你提出的第 6 点：**面不一定是正面**也没关系，只要四角点顺序一致，PnP 得到的 \(\mathbf{R}\) 会正确反映该面的空间姿态，中心偏移就自然成立。

#### 3.3.2.2 平面 PnP 的“两解”问题（需要在方案里提前考虑）
对平面正方形，PnP 可能存在镜像/翻转的多解（尤其噪声大时）。建议方案层面加入“解的判别”：
- **优先用** `SOLVEPNP_IPPE_SQUARE`（或 IPPE）获取候选解
- 通过以下方式选正确解：
  - **Cheirality**：四个角点投影回去应在相机前方（Z>0）
  - **重投影误差最小**：用 `projectPoints` 计算误差，选最小
  - （可选）结合相机朝向先验：例如目标一般在画面前方、法向不应指向相机背后等

#### 3.3.3 核心接口
```python
class PnPSolver:
    def __init__(self, camera_matrix, dist_coeffs, square_size, method):
        """初始化PnP求解器"""
        
    def solve(self, corners_2d: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        求解PnP问题
        
        Args:
            corners_2d: 2D角点坐标，shape=(4, 2)
            
        Returns:
            success: 求解是否成功
            rvec: 旋转向量，shape=(3, 1)
            tvec: 平移向量，shape=(3, 1)，单位mm
        """
        
    def compute_corner_positions_3d(self, rvec, tvec) -> np.ndarray:
        """
        计算角点在相机坐标系下的3D坐标
        
        Returns:
            corners_3d: shape=(4, 3)，单位mm
        """
        
    def rvec_tvec_to_pose(self, rvec, tvec) -> Tuple[np.ndarray, np.ndarray]:
        """
        将旋转向量和平移向量转换为位置和四元数
        
        Returns:
            position: [x, y, z]，单位米
            quaternion: [qx, qy, qz, qw]
        """
```

#### 3.3.4 坐标系转换
- **图像坐标系**: 原点在左上角，u右，v下
- **相机坐标系**: 原点在光心，X右，Y下，Z朝前（光轴方向）
- **物体坐标系**: 原点在正方形中心，X右，Y下，Z垂直平面向外

---

### 3.4 主节点执行流程

```python
def image_callback(self, msg):
    """图像回调函数"""
    
    # 1. 转换图像格式
    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    
    # 2. YOLO检测角点
    corners_2d = self.detector.detect_corners(cv_image)
    if corners_2d is None:
        self.get_logger().warn("未检测到角点")
        return
    
    # 3. PnP求解
    success, rvec, tvec = self.solver.solve(corners_2d)
    if not success:
        self.get_logger().warn("PnP求解失败")
        return
    
    # 4. 计算3D角点坐标
    corners_3d = self.solver.compute_corner_positions_3d(rvec, tvec)
    
    # 5. 转换为位姿
    position, quaternion = self.solver.rvec_tvec_to_pose(rvec, tvec)
    
    # 6. 发布结果
    self.publish_pose(position, quaternion, msg.header)
    self.publish_corners_3d(corners_3d, msg.header)
    self.broadcast_tf(position, quaternion, msg.header)
    
    # 7. 可视化
    if self.visualization_enabled:
        vis_image = self.draw_visualization(cv_image, corners_2d, rvec, tvec)
        self.publish_visualization(vis_image, msg.header)
```

---

## 4. 可视化设计

### 4.1 可视化元素
1. **角点标记**: 在图像上绘制四个角点，不同颜色区分
   - 角点0（左上）: 红色
   - 角点1（右上）: 绿色
   - 角点2（右下）: 蓝色
   - 角点3（左下）: 黄色

2. **角点编号**: 在每个角点旁标注编号 (0, 1, 2, 3)

3. **坐标轴**: 使用`cv2.drawFrameAxes`绘制3D坐标轴
   - X轴: 红色
   - Y轴: 绿色
   - Z轴: 蓝色

4. **中心点**: 标记正方形中心

5. **信息文本**: 显示关键信息
   - FPS
   - 距离（Z坐标）
   - 姿态角度（Roll, Pitch, Yaw）
   - 置信度

### 4.2 示例可视化效果
```
┌────────────────────────────────────┐
│ FPS: 30.5 | Distance: 450mm       │
│ Roll: 2.3° Pitch: -5.1° Yaw: 15.6°│
├────────────────────────────────────┤
│                                    │
│     ●0────────────●1               │
│      │            │                │
│      │      +     │                │
│      │     /|\    │                │
│      │    Z Y X   │                │
│     ●3────────────●2               │
│                                    │
└────────────────────────────────────┘
```

---

## 5. Launch文件设计

```python
# four_pt_pnp.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取配置文件路径
    pkg_dir = get_package_share_directory('rgb_depth_detect')
    config_file = os.path.join(pkg_dir, 'config', 'four_pt_pnp.yaml')
    
    # 声明启动参数
    return LaunchDescription([
        # 参数：配置文件路径
        DeclareLaunchArgument(
            'config_file',
            default_value=config_file,
            description='Path to config file'
        ),
        
        # 参数：是否启用可视化
        DeclareLaunchArgument(
            'visualization',
            default_value='true',
            description='Enable visualization'
        ),
        
        # 参数：相机话题
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/color/image_raw',
            description='Camera image topic'
        ),
        
        # 主节点
        Node(
            package='rgb_depth_detect',
            executable='four_pt_pnp',
            name='four_pt_pnp_node',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'visualization.enable': LaunchConfiguration('visualization'),
                    'image_topic': LaunchConfiguration('image_topic'),
                }
            ],
            remappings=[
                ('image_raw', LaunchConfiguration('image_topic')),
            ]
        ),
        
        # （可选）启动相机驱动
        # Node(
        #     package='realsense2_camera',
        #     executable='realsense2_camera_node',
        #     name='camera',
        #     ...
        # ),
    ])
```

---

## 6. 使用流程

### 6.1 准备工作

#### 步骤1: 相机标定
```bash
# 已有标定工具 calibrate_camera.py
python3 calibrate_camera.py --capture calib_images/ --num-images 20
python3 calibrate_camera.py --calibrate calib_images/ --output config/camera_calib.yaml
```

#### 步骤2: 准备YOLO模型
- 使用Ultralytics YOLO训练角点检测模型（Pose模型，4个关键点）
- 或者使用已训练好的模型
- 将模型放在 `models/corner_detector.pt`

#### 步骤3: 测量正方形边长
- 精确测量目标正方形的边长（单位：mm）
- 更新到配置文件 `target.square_size`

### 6.2 配置参数
编辑 `config/four_pt_pnp.yaml`，填入相机内参和目标参数

### 6.3 启动系统
```bash
# 方式1: 使用launch文件（推荐）
ros2 launch rgb_depth_detect four_pt_pnp.launch.py

# 方式2: 指定自定义配置文件
ros2 launch rgb_depth_detect four_pt_pnp.launch.py config_file:=/path/to/config.yaml

# 方式3: 直接运行节点
ros2 run rgb_depth_detect four_pt_pnp --ros-args --params-file config/four_pt_pnp.yaml

# 方式4: 指定参数
ros2 run rgb_depth_detect four_pt_pnp --ros-args \
    -p image_topic:=/camera/color/image_raw \
    -p target.square_size:=100.0
```

### 6.4 查看结果
```bash
# 查看可视化
ros2 run rqt_image_view rqt_image_view /four_pt_pnp/visualization

# 查看位姿
ros2 topic echo /four_pt_pnp/pose

# 查看TF树
ros2 run tf2_tools view_frames

# 使用RViz可视化
rviz2 -d config/four_pt_pnp.rviz
```

---

## 7. 测试与验证

### 7.1 单元测试
- **YOLO检测器测试**: 测试角点检测准确率
- **PnP求解器测试**: 使用模拟数据验证求解精度
- **坐标转换测试**: 验证各坐标系转换的正确性

### 7.2 集成测试
1. **静态测试**: 将正方形固定在不同位置和角度，验证位姿估计准确性
2. **动态测试**: 移动正方形，验证实时跟踪性能
3. **鲁棒性测试**: 测试光照变化、遮挡、模糊等情况

### 7.3 性能指标
- **检测成功率**: > 95%（正常光照条件下）
- **位置精度**: < 5mm（距离 < 1m）
- **角度精度**: < 2°
- **处理延迟**: < 50ms（包含检测+PnP）
- **帧率**: > 20 FPS

---

## 8. 错误处理

### 8.1 常见错误及处理
1. **未检测到角点**
   - 记录警告日志
   - 可选：保存失败图像用于调试
   - 不发布位姿信息

2. **PnP求解失败**
   - 检查角点顺序是否正确
   - 检查角点是否共线
   - 记录错误日志

3. **相机话题无数据**
   - 定时检查订阅状态
   - 输出错误提示

4. **模型加载失败**
   - 节点启动时检查模型文件
   - 给出明确错误信息和解决建议

### 8.2 异常恢复策略
- 检测失败不影响后续帧处理
- 提供心跳话题，指示节点运行状态
- 支持动态重新加载模型和参数

---

## 9. 扩展功能（可选）

### 9.1 多目标跟踪
- 支持同时检测和跟踪多个正方形标靶
- 为每个目标分配ID
- 发布目标数组

### 9.2 时间滤波
- 使用卡尔曼滤波平滑位姿估计
- 减少抖动，提高稳定性

### 9.3 标定验证
- 提供自动标定验证功能
- 移动标靶到已知位置，对比估计结果

### 9.4 深度融合
- 如果使用RGB-D相机，融合深度信息
- 提高距离估计精度

### 9.5 性能优化
- 模型量化（INT8/FP16）
- TensorRT加速
- 多线程处理

---

## 10. 依赖项

### 10.1 系统依赖
```bash
# ROS 2
ros-humble-desktop
ros-humble-cv-bridge
ros-humble-image-transport

# Python包
pip install ultralytics opencv-python numpy pyyaml scipy
```

### 10.2 package.xml 依赖
```xml
<depend>rclpy</depend>
<depend>sensor_msgs</depend>
<depend>geometry_msgs</depend>
<depend>cv_bridge</depend>
<depend>tf2_ros</depend>
<depend>tf2_geometry_msgs</depend>
```

---

## 11. 目录结构总结

```
rgb_depth_detect/
├── config/
│   ├── four_pt_pnp.yaml          # 主配置文件
│   ├── camera_calib.yaml         # 相机标定结果
│   └── four_pt_pnp.rviz          # RViz配置
├── launch/
│   └── four_pt_pnp.launch.py     # 启动文件
├── models/
│   └── corner_detector.pt        # YOLO模型
├── rgb_depth_detect/
│   ├── __init__.py
│   ├── four_pt_pnp.py            # 主节点（核心）
│   ├── yolo_detector.py          # YOLO检测器
│   └── pnp_solver.py             # PnP求解器
├── test/
│   ├── test_yolo_detector.py
│   └── test_pnp_solver.py
├── README.md                      # 用户文档
├── DESIGN.md                      # 本设计文档
├── package.xml
├── setup.py
└── setup.cfg
```

---

## 11.1 是否要新建 ROS 仓库/包？（按你的建议给出取舍）
你提的建议非常合理：当前 `ws/src` 下已有相机驱动、训练代码、以及这个 `rgb_depth_detect` 包。为了后期维护清晰，通常有两种组织方式：

### 方案 1：继续复用现有包 `rgb_depth_detect`（改名/不改名都可）
- **优点**：改动最少；现有 `calibrate_camera.py`、`launch/`、配置机制可复用
- **缺点**：包名 `rgb_depth_detect` 与“立方体中心位姿估计”语义不完全匹配；后续功能多了会变杂

### 方案 2（推荐）：新建一个独立 ROS 2 Python 包（例如 `cube_pose_estimator`）
- **优点**：职责单一；参数/话题/TF 命名更干净；与训练工程解耦（训练在 `model_train/`，推理节点在新包）
- **缺点**：需要新建 `package.xml / setup.py / launch / config` 等基础结构

**我的建议**：如果你预计后面还会加滤波、多目标、标定验证、不同相机适配等功能，建议走 **方案 2** 新建包；如果只是快速跑通、临时验证，则先走方案 1。

> 等你确认后，我再开始真正创建新包/代码（你没说写，我绝不动代码）。

## 12. 开发计划

### Phase 1: 核心功能开发（第1-2天）
- [ ] PnP求解器模块（pnp_solver.py）
- [ ] YOLO检测器封装（yolo_detector.py）
- [ ] 主节点框架（four_pt_pnp.py）

### Phase 2: ROS集成（第3天）
- [ ] 配置文件（four_pt_pnp.yaml）
- [ ] Launch文件（four_pt_pnp.launch.py）
- [ ] 话题发布与订阅
- [ ] TF广播

### Phase 3: 可视化（第4天）
- [ ] 图像标注
- [ ] 信息叠加
- [ ] RViz配置

### Phase 4: 测试与优化（第5天）
- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能优化
- [ ] 文档完善

---

## 13. 注意事项

### 13.1 坐标系注意事项
⚠️ **相机光学坐标系** vs **相机坐标系**
- ROS中相机通常有两个坐标系
- `camera_link`: 相机物理坐标系
- `camera_color_optical_frame`: 光学坐标系（用于视觉算法）
- PnP结果应在光学坐标系下

### 13.2 单位注意事项
⚠️ **一致的单位系统**
- 配置文件中边长单位：**mm**
- PnP内部计算：**mm**
- ROS消息发布：**m**（米）
- 注意转换！

### 13.3 YOLO模型要求
⚠️ **关键点模型**
- 使用 YOLO-Pose 架构
- 4个关键点（不是边界框检测）
- 训练时确保角点顺序一致

### 13.4 相机标定
⚠️ **准确的标定至关重要**
- PnP精度严重依赖标定质量
- 建议定期重新标定
- 保存多组标定结果备用

---

## 14. 常见问题FAQ

**Q1: 检测不到角点怎么办？**
A: 检查光照、模型置信度阈值、相机焦距、标靶大小

**Q2: 位姿抖动严重？**
A: 考虑加卡尔曼滤波、检查角点检测精度、增加PnP迭代次数

**Q3: 距离估计不准？**
A: 重新标定相机、精确测量正方形边长、检查畸变校正

**Q4: 如何提高帧率？**
A: 使用更小的YOLO模型（nano）、降低图像分辨率、使用GPU加速

**Q5: 支持哪些YOLO版本？**
A: YOLOv8-pose、YOLOv11-pose、或任何Ultralytics格式的pose模型

---

## 15. 参考资料

1. OpenCV PnP文档: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
2. Ultralytics YOLO: https://docs.ultralytics.com/
3. ROS 2 tf2: https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Tf2-Main.html
4. 相机标定理论: Zhang's calibration method

---

**设计文档版本**: v1.0  
**最后更新**: 2026-01-10  
**作者**: AI Assistant  
**审核状态**: 待审核 ✓
