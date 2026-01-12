from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rgb_depth_detect'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Install config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jienan',
    maintainer_email='fjienan@example.com',
    description='4-Point PnP Pose Estimation using Cascade YOLO Detector',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'four_pt_pnp_node = rgb_depth_detect.four_pt_pnp:main',
            'test_ui = test.test_ui:main',
            'save_image = test.save_service:main',
        ],
    },
)
