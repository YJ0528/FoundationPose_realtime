import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'yolov8_image_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yj0528',
    maintainer_email='tyjun0528@gmail.com',
    description='Image Segmentation using yolov8',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov8_image_segmentation = yolov8_image_segmentation.yolov8_image_segmentation:main',
        ],
    },
)
