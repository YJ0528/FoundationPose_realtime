import argparse
import os
import time
import torch
import json
import cv2
import sys
import numpy as np
from typing import List
import trimesh
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs
import logging
from ultralytics import YOLO
from std_msgs.msg import String
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
import sensor_msgs.msg
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from std_srvs.srv import Trigger



class yolov8ImageSegmentationMultiObject(Node):

    def __init__(self):

        super().__init__('yolo_image_segmentation_multi_object')
        self.bridge = CvBridge()
                
        self.get_logger().info('Initializing YOLO Image Segmentation Node...')
        
        #################################################
        # Initialize parameters
        #################################################
        # Declare and get camera parameters:
        self.declare_parameter('camera_name', 'camera')
        self.declare_parameter('camera_namespace', 'camera')

        self.declare_parameter('yolo_image_segmentation_multi_object.default_class_name', 'blue_tube')
        self.declare_parameter('yolo_image_segmentation_multi_object.yolo_model_name', 'best.pt')
        self.declare_parameter('yolo_image_segmentation_multi_object.confidence_threshold', 0.9)
        self.declare_parameter('yolo_image_segmentation_multi_object.visualize', True)
        self.declare_parameter('yolo_image_segmentation_multi_object.timeout_seconds', -1)
        
        self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        self.camera_namespace = self.get_parameter('camera_namespace').get_parameter_value().string_value
        self.topic_name = f"/{self.camera_namespace}/{self.camera_name}/color/image_raw"
        self.default_class_name = self.get_parameter('yolo_image_segmentation_multi_object.default_class_name').get_parameter_value().string_value
        self.yolo_model_name = self.get_parameter('yolo_image_segmentation_multi_object.yolo_model_name').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('yolo_image_segmentation_multi_object.confidence_threshold').get_parameter_value().double_value
        self.visualize = self.get_parameter('yolo_image_segmentation_multi_object.visualize').get_parameter_value().bool_value
        self.timeout_seconds = self.get_parameter('yolo_image_segmentation_multi_object.timeout_seconds').get_parameter_value().integer_value

        #################################################
        # Initialize YOLO model
        #################################################
        self.yolo_model_path = os.path.join(get_package_share_directory('yolov8_image_segmentation'), 'models', self.yolo_model_name)
        self.get_logger().info(f'model path: {self.yolo_model_path}')
        self.yolo_model = YOLO(self.yolo_model_path)
        self.get_logger().info(f'Model classes: {list(self.yolo_model.names.values())}')
        
        
        self.start_time = time.time()
        self.i = None  # Initialize i to ensure it exists
        self.is_init_mask = True

        ################################################################
        # Initialize ROS2 api members (publisher, subsrciber, services)
        ################################################################
        self.class_publishers = {}
        self.class_services = {}
        self.class_masks = {}

        init_mask_qos_profile = QoSProfile(
            reliability = QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth = 1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        camera_qos_profile = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth = 1,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        self.model_classes_publisher = self.create_publisher(
            String,  
            '/yolo_model/available_classes',
            init_mask_qos_profile  #share share~
        )

        #################################
        # Publish class names once
        #################################
        try:
            classes_msg = String()
            classes_msg.data = json.dumps(list(self.yolo_model.names.values()))
            self.model_classes_publisher.publish(classes_msg)
            self.get_logger().info(f'Published model classes: {list(self.yolo_model.names.values())}')
            
        except Exception as e:
            self.get_logger().error(f"Error publishing model classes: {str(e)}")

        ###############################################################
        # Bulk create publishers and services for each object class
        ################################################################
        for class_name in self.yolo_model.names.values():
            
            
            topic_name = f"/foundation_pose/init_mask/{class_name.replace(' ', '_').replace('-', '_')}"
            service_name = f'publish_init_mask_{class_name.replace(" ", "_").replace("-", "_")}'
            self.class_masks[class_name] = None

            publisher = self.create_publisher(
                sensor_msgs.msg.Image,
                topic_name,
                init_mask_qos_profile
            )
            self.class_publishers[class_name] = publisher
            
            service = self.create_service(
                Trigger,
                service_name,
                lambda req, resp, cn=class_name: self.pub_init_mask(req, resp, cn)
            )
            self.class_services[class_name] = service

            self.get_logger().info(f'Created publisher for class "{class_name}" on topic: {topic_name}')
            self.get_logger().info(f'Created service: {service_name}')

        self.rgb_data_subscriber = self.create_subscription(
            sensor_msgs.msg.Image,
            self.topic_name,
            self.subscriber_callback,
            camera_qos_profile)

        self.service = self.create_service(
            Trigger,
            'publish_init_mask',
            self.pub_init_mask
        )
        
        self.get_logger().info(f'Initialization completed ')

    def pub_init_mask(self, request, response, class_name=None):

        try:   
            if class_name is None:
                self.get_logger().warn(f'No class name provided, using default class: {self.default_class_name}')
                class_name = self.default_class_name

            # Publish initial mask to ros2 topic:
            if class_name in self.class_masks and self.class_masks[class_name] is not None and np.any(self.class_masks[class_name]):
                
                outMsg = self.bridge.cv2_to_imgmsg(self.class_masks[class_name], encoding='mono8')
                outMsg.header.stamp = self.get_clock().now().to_msg()
                outMsg.header.frame_id = "camera_frame"
                
                # Publish to the specific class publisher
                self.class_publishers[class_name].publish(outMsg)
                self.get_logger().info(f"Mask for '{class_name}' detected and published")
                
                response.success = True
                response.message = f"Successfully published mask for {class_name}"
                
            else:
                self.get_logger().warn(f"No valid mask available for class '{class_name}'")
                response.success = False
                response.message = f"No mask detected for {class_name} in current frame"
                
        except Exception as e:
            self.get_logger().error(f"Error publishing mask for {class_name}: {str(e)}")
            response.success = False
            response.message = f"Error: {str(e)}"

        return response

    def subscriber_callback(self, inMsg):

        try:
            frame = self.bridge.imgmsg_to_cv2(inMsg, desired_encoding='bgr8')

            # Yolo Image Segmentation
            self.get_object_mask_from_frame(
                    frame=frame,
                    yolo_model= self.yolo_model,
                    confidence_threshold= self.confidence_threshold,
                    visualize= self.visualize 
                )

            if self.timeout_seconds < 0: return
            if time.time() - self.start_time > self.timeout_seconds:

                self.get_logger().error("Timeout reached! Stop capturing image")
                rclpy.shutdown()

        except KeyboardInterrupt:
            self.get_logger().info("Stopping...")

    def get_object_mask_from_frame(self, frame, yolo_model, confidence_threshold=0.9, visualize=True):

        frame_h, frame_w = frame.shape[:2]
        for class_name in self.yolo_model.names.values():
            self.class_masks[class_name] = np.zeros((frame_h, frame_w), dtype=np.uint8)

        ################################################################
        # Predict mask the object
        ################################################################
        results = yolo_model(frame)  # Simplified call

        for result in results:
            classes_names = result.names
            if result.masks is not None:
                masks = result.masks.xy
                for mask, box in zip(masks, result.boxes):
                    cls = int(box.cls[0])
                    class_name = classes_names[cls]
                    conf = float(box.conf[0])
                    if conf > confidence_threshold:
                        mask_np = np.array(mask, dtype=np.int32)
                        cv2.fillPoly(self.class_masks[class_name], [mask_np], 255)

                        if visualize:
                            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                            overlay_color = colors[cls % len(colors)]
                            cv2.polylines(frame, [mask_np], isClosed=True, color=overlay_color, thickness=2)
                            cv2.putText(frame, f'{class_name} {conf:.2f}',
                                        (mask_np[0][0], mask_np[0][1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2)

        if visualize:
            cv2.imshow('YOLO Segmentation', frame)
            cv2.waitKey(1)

        return 

def main(args=None):
    rclpy.init(args=args)

    yolo_image_segmentation_multi_object = yolov8ImageSegmentationMultiObject()
    rclpy.spin(yolo_image_segmentation_multi_object)
    yolo_image_segmentation_multi_object.destroy_node()
    rclpy.shutdown()

    torch.cuda.empty_cache()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()