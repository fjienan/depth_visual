from setuptools import find_packages, setup

package_name = 'rgb_depth_detect'

setup(
    name=package_name,
    version='0.0.0',
    # include all python packages under this package so console entry points
    # that live in the `test` module are installed. Avoid using a package
    # name like `test` for production code (it commonly conflics with
    # testing/tools); consider renaming the module later.
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jienan',
    maintainer_email='fjienan@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_ui =test.test_ui:main',
            'save_image = test.save_service:main',
        ],
    },
)
