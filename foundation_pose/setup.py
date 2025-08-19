import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'foundation_pose'

def get_data_files():
    data_files = []
    
    # Standard ROS 2 files
    data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
    data_files.append(('share/' + package_name, ['package.xml']))
    data_files.append((os.path.join('share', package_name, 'config'), glob('config/*.yaml')))

    # Automatically find and install test_realtime directories
    test_realtime_path = 'test_realtime'
    if os.path.exists(test_realtime_path):
        for root, dirs, files in os.walk(test_realtime_path):
            if files:  # Only add directories that contain files
                # Convert source path to install path
                install_path = os.path.join('share', package_name, root)
                # Get all files in this directory
                file_paths = [os.path.join(root, f) for f in files]
                data_files.append((install_path, file_paths))
    
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files= get_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yj0528',
    maintainer_email='tyjun0528@gmail.com',
    description='Foundation Pose object position tracking',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'foundation_pose = foundation_pose.foundation_pose:main',
        ],
    },
)

