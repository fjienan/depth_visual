"""Launch cube_pose_estimator node with a params file only.

This launch file intentionally does NOT override parameters from CLI by default.
Edit the params YAML (or pass a different params_file) to configure the node.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_dir = get_package_share_directory("cube_pose_estimator")
    default_params = os.path.join(pkg_dir, "config", "cube_pose_estimator.yaml")
    default_rviz = os.path.join(pkg_dir, "config", "cube_pose_estimator.rviz")

    params_file = LaunchConfiguration("params_file")
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to the ROS2 parameters YAML file.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Whether to start RViz2 with a pre-configured layout.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="Path to an RViz2 config file.",
            ),
            Node(
                package="cube_pose_estimator",
                executable="cube_pose_node",
                name="cube_pose_estimator",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(rviz),
            ),
        ]
    )

