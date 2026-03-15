from setuptools import find_packages, setup

package_name = 'cube_pose_estimator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/cube_pose_estimator.yaml']),
        ('share/' + package_name + '/config', ['config/cube_pose_fusion.yaml']),
        ('share/' + package_name + '/config', ['config/cube_pose_estimator.rviz']),
        ('share/' + package_name + '/launch', ['launch/cube_pose_estimator.launch.py']),
        ('share/' + package_name + '/launch', ['launch/cube_pose_estimator_with_fusion.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fjienan',
    maintainer_email='fjienan@example.com',
    description='Cube face corner detection + PnP cube center pose estimation (ROS 2)',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cube_pose_node = cube_pose_estimator.cube_pose_node:main',
            'pose_fusion_node = cube_pose_estimator.pose_fusion_node:main',
        ],
    },
)
