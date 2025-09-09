
###########################################################################################################################

# PLEASE REFER TO: src/realsense-ros/realsense2_camara/examples/launch_params_from_file/rs_launch_get_params_from_yaml

###########################################################################################################################

from launch import LaunchDescription
import launch_ros.actions
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
import os
from ament_index_python.packages import get_package_share_directory
sys.path.append(os.path.join(get_package_share_directory('realsense2_camera'), 'launch'))
import rs_launch

package_name= 'yolov8_image_segmentation'
package_dir= get_package_share_directory(package_name) 

local_parameters = [{'name': 'camera_name',         'default': 'camera', 'description': 'camera unique name'},
                    {'name': 'camera_namespace',    'default': 'camera', 'description': 'camera namespace'},
                    {'name': 'config_file',         'default': [package_dir, "/config/realsense_camera_config.yaml"], 'description': 'yaml config file'},
                   ]

def set_configurable_parameters(local_params):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in local_params])


def generate_launch_description():
    
    yolov8_config_file = os.path.join(
        package_dir, 
        'config', 
        'yolov8_config.yaml'
        )

    yolo_multi_object_config_file = os.path.join(
        package_dir, 
        'config', 
        'yolo_image_segmentation_multi_object_config.yaml'
        )

    params = rs_launch.configurable_parameters
    declare_realsense_local_params = rs_launch.declare_configurable_parameters(local_parameters)
    declare_realsense_params = rs_launch.declare_configurable_parameters(params)

    rs_launch_setup = OpaqueFunction(function=rs_launch.launch_setup,
                kwargs = {'params' : set_configurable_parameters(params)}
        )

    # YOLOv8 node
    yolov8_node = launch_ros.actions.Node(
        package='yolov8_image_segmentation',
        executable='yolov8_image_segmentation',
        name='yolov8_image_segmentation',
        parameters=[
            {
                'camera_name': LaunchConfiguration('camera_name'),
                'camera_namespace': LaunchConfiguration('camera_namespace'),
            },
            yolov8_config_file
        ],
        output='screen'
    )

    yolo_multi_object_node = launch_ros.actions.Node(
        package='yolov8_image_segmentation',
        executable='yolo_image_segmentation_multi_object',
        name='yolo_image_segmentation_multi_object',
        parameters=[
            {
                'camera_name': LaunchConfiguration('camera_name'),
                'camera_namespace': LaunchConfiguration('camera_namespace'),
            },
            yolo_multi_object_config_file
        ],
        output='screen'
    )

    # Create LaunchDescription and add actions
    ld = LaunchDescription()

    # Params is a list so need for loop:
    for action in declare_realsense_local_params: ld.add_action(action)
    for action in declare_realsense_params: ld.add_action(action)
    
    ld.add_action(rs_launch_setup)
    # ld.add_action(yolov8_node)
    ld.add_action(yolo_multi_object_node)


    return ld