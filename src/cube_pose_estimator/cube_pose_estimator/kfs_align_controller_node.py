#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        if dt <= 0.0:
            return 0.0
        self.integral += error * dt
        # 积分限幅抗饱和
        self.integral = np.clip(self.integral, -self.output_limit, self.output_limit)
        
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        return np.clip(output, -self.output_limit, self.output_limit)

class KFSAlignController(Node):
    def __init__(self):
        super().__init__('kfs_align_controller')

        # === 1. 声明并读取参数 (方便你在比赛时通过 YAML 或命令行动态调参) ===
        self.declare_parameter("target.distance", 1.2)     # 目标停止距离 (米)
        self.declare_parameter("target.yaw", 0.0)          # 目标偏航角设定为 0.0
        
        # 限制底盘与升降机构的最大速度 (安全第一)
        self.declare_parameter("limit.linear_x", 0.8)      # m/s
        self.declare_parameter("limit.linear_y", 0.8)      # m/s
        self.declare_parameter("limit.linear_z", 0.5)      # m/s (升降机构)
        self.declare_parameter("limit.angular_z", 1.0)     # rad/s

        # PID 参数 (默认给一组经验保守值，赛场上再微调)
        self.declare_parameter("pid.x", [1.5, 0.0, 0.1])
        self.declare_parameter("pid.y", [1.5, 0.0, 0.15])
        self.declare_parameter("pid.z", [2.0, 0.0, 0.2])
        self.declare_parameter("pid.yaw", [1.2, 0.0, 0.1])

        self._load_parameters()

        # === 2. 初始化 ROS 接口 ===
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/cube_pose/pose', self.pose_callback, 10)

        # === 3. 时间与状态管理 ===
        self.last_time = self.get_clock().now()
        self.last_pose_time = self.get_clock().now()
        
        # 安全看门狗 (Watchdog)：如果超过 0.5 秒没看到目标，立刻停车
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info("KFS 对齐控制节点已启动！")

    def _load_parameters(self):
        self.target_dist = self.get_parameter("target.distance").value
        self.target_yaw = self.get_parameter("target.yaw").value
        
        lim_x = self.get_parameter("limit.linear_x").value
        lim_y = self.get_parameter("limit.linear_y").value
        lim_z = self.get_parameter("limit.linear_z").value
        lim_yaw = self.get_parameter("limit.angular_z").value

        pid_x_params = self.get_parameter("pid.x").value
        pid_y_params = self.get_parameter("pid.y").value
        pid_z_params = self.get_parameter("pid.z").value
        pid_yaw_params = self.get_parameter("pid.yaw").value

        self.pid_x = PIDController(*pid_x_params, lim_x)
        self.pid_y = PIDController(*pid_y_params, lim_y)
        self.pid_z = PIDController(*pid_z_params, lim_z)
        self.pid_yaw = PIDController(*pid_yaw_params, lim_yaw)

    def pose_callback(self, msg: PoseStamped):
        # 记录收到位姿的时间（用于看门狗）
        current_time = self.get_clock().now()
        self.last_pose_time = current_time

        # 计算时间差 dt
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        if dt <= 0:
            return

        # 1. 提取位置误差 (视觉节点已经输出了 FLU 坐标系)
        current_x = msg.pose.position.x
        current_y = msg.pose.position.y
        current_z = msg.pose.position.z

        error_x = current_x - self.target_dist 
        error_y = current_y - 0.0 
        error_z = current_z - 0.0 

        # 2. 提取姿态误差 (将四元数转为欧拉角)
        q = [msg.pose.orientation.x, msg.pose.orientation.y, 
             msg.pose.orientation.z, msg.pose.orientation.w]
        r = R.from_quat(q)
        
        # 【改动 1】使用 zyx 顺规提取，对应返回顺序为 yaw, pitch, roll
        yaw, pitch, roll = r.as_euler('zyx', degrees=False)
        
        # 【改动 2】强行抹平 PnP 算法自带的 90 度 (1.5708 弧度) 坐标系夹角
        # 注：实车测试时，如果发现机器人正对着 KFS 时，这里打印出来的 RawYaw 是 -1.57，
        # 则请将下面的 `yaw - 1.5708` 更改为 `yaw + 1.5708`
        current_yaw = yaw + np.pi / 2  # 将原始偏航角顺时针旋转 90 度，修正坐标系夹角问题
        current_roll = roll - np.pi / 2  # 将原始滚转角逆时针旋转 90 度，修正坐标系夹角问题 
        
        self.get_logger().info(
            f"Pose-> roll:{current_roll:.2f}，Pitch:{pitch:.2f} Yaw:{current_yaw:.2f} | ")

        # 计算原始偏航误差
        error_yaw = self.target_yaw - current_yaw 

        # 【改动 3】角度归一化，强制走最短路径
        error_yaw = (error_yaw + np.pi) % (2 * np.pi) - np.pi

        # 3. PID 计算输出速度
        cmd = Twist()
        cmd.linear.x = self.pid_x.compute(error_x, dt)
        cmd.linear.y = self.pid_y.compute(error_y, dt)
        cmd.linear.z = self.pid_z.compute(error_z, dt)
        cmd.angular.z = self.pid_yaw.compute(error_yaw, dt)

        # 4. 发布速度
        self.cmd_pub.publish(cmd)

        # 【改动 4】打开调试日志，方便观察真实角度和修正后的误差
        # self.get_logger().info(
        #     f"Err-> X:{error_x:.2f} Y:{error_y:.2f} Z:{error_z:.2f} | "
        #     f"RawYaw:{yaw:.2f} YawErr:{error_yaw:.2f}"
        # )

    def watchdog_callback(self):
        # 超过 0.5 秒没有收到 YOLO 视觉发来的目标位姿，认为目标丢失
        time_since_last_pose = (self.get_clock().now() - self.last_pose_time).nanoseconds / 1e9
        if time_since_last_pose > 0.5:
            stop_cmd = Twist()
            self.cmd_pub.publish(stop_cmd)
            # self.get_logger().warn("目标丢失，触发安全停车！", throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = KFSAlignController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()