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
from std_msgs.msg import Bool  
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
import sensor_msgs.msg
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from std_srvs.srv import Trigger



class yolov8ImageSegmentation(Node):

    def __init__(self):

        super().__init__('yolov8_image_segmentation')
        self.bridge = CvBridge()
                
        self.get_logger().info('Initializing YOLO Image Segmentation Node...')
        
        # Declare and get camera parameters:
        self.declare_parameter('camera_name', 'camera')
        self.declare_parameter('camera_namespace', 'camera')

        self.declare_parameter('yolov8_image_segmentation.class_name', 'blue_tube')
        self.declare_parameter('yolov8_image_segmentation.yolo_model_name', 'best.pt')
        self.declare_parameter('yolov8_image_segmentation.confidence_threshold', 0.9)
        self.declare_parameter('yolov8_image_segmentation.visualize', True)
        self.declare_parameter('yolov8_image_segmentation.timeout_seconds', -1)
        
        self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        self.camera_namespace = self.get_parameter('camera_namespace').get_parameter_value().string_value

        self.class_name = self.get_parameter('yolov8_image_segmentation.class_name').get_parameter_value().string_value
        self.yolo_model_name = self.get_parameter('yolov8_image_segmentation.yolo_model_name').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('yolov8_image_segmentation.confidence_threshold').get_parameter_value().double_value
        self.visualize = self.get_parameter('yolov8_image_segmentation.visualize').get_parameter_value().bool_value
        self.timeout_seconds = self.get_parameter('yolov8_image_segmentation.timeout_seconds').get_parameter_value().integer_value

        topic_name = f"/{self.camera_namespace}/{self.camera_name}/color/image_raw"
        self.yolo_model_path = os.path.join(get_package_share_directory('yolov8_image_segmentation'), 'models', self.yolo_model_name)
        self.yolo_model = YOLO(self.yolo_model_path)
        self.start_time = time.time()
        self.i = None  # Initialize i to ensure it exists
        self.get_logger().info(f'model path: {self.yolo_model_path}')
        self.mask = None

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


        self.is_init_mask = True

        self.init_mask_publisher_ = self.create_publisher(
            sensor_msgs.msg.Image,
            '/foundation_pose/init_mask', 
            init_mask_qos_profile)

        self.rgb_data_subscriber = self.create_subscription(
            sensor_msgs.msg.Image,
            topic_name,
            self.subscriber_callback,
            camera_qos_profile)

        self.service = self.create_service(
            Trigger,
            'publish_init_mask',
            self.pub_init_mask
        )

        self.get_logger().info(f'Initialization completed ')


    def pub_init_mask(self, request, response):
        try:

            # Publish initial mask to ros2 topic:
            if hasattr(self, 'mask_full') and self.mask_full is not None and np.any(self.mask_full):
                
                outMsg = self.bridge.cv2_to_imgmsg(self.mask_full, encoding='mono8')
                outMsg.header.stamp = self.get_clock().now().to_msg()
                outMsg.header.frame_id = "camera_frame"
                
                self.init_mask_publisher_.publish(outMsg)
                self.get_logger().info(f"/{self.class_name} detected and published")
                
                response.success = True
                response.message = f"Successfully published mask for {self.class_name}"
                
            else:
                self.get_logger().warn("No valid mask available to publish")
                response.success = False
                response.message = "No mask detected in current frame"
                
        except Exception as e:
            self.get_logger().error(f"Error publishing mask: {str(e)}")
            response.success = False
            response.message = f"Error: {str(e)}"
        
        return response

    def subscriber_callback(self, inMsg):
      
        try:
            frame = self.bridge.imgmsg_to_cv2(inMsg, desired_encoding='bgr8')

            # Yolov8 Image Segmentation
            self.mask_full = self.get_object_mask_from_frame(
                    frame=frame,
                    target_class_name= self.class_name,  
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

    def get_object_mask_from_frame(self, frame, target_class_name, yolo_model, confidence_threshold=0.9, visualize=True):
        frame_h, frame_w = frame.shape[:2]
        mask_full = np.zeros((frame_h, frame_w), dtype=np.uint8)

        # Predict
        results = yolo_model(frame)  # Simplified call

        for result in results:
            classes_names = result.names
            if result.masks is not None:
                masks = result.masks.xy
                for mask, box in zip(masks, result.boxes):
                    cls = int(box.cls[0])
                    class_name = classes_names[cls]
                    conf = float(box.conf[0])

                    if class_name.lower() == target_class_name.lower() and conf > confidence_threshold:
                        mask_np = np.array(mask, dtype=np.int32)
                        cv2.fillPoly(mask_full, [mask_np], 255)

                        if visualize:
                            overlay_color = (255, 0, 0)
                            cv2.polylines(frame, [mask_np], isClosed=True, color=overlay_color, thickness=2)
                            cv2.putText(frame, f'{class_name} {conf:.2f}',
                                        (mask_np[0][0], mask_np[0][1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2)

        if visualize:
            cv2.imshow('YOLO Segmentation', frame)
            cv2.waitKey(1)

        return mask_full

def main(args=None):
    rclpy.init(args=args)

    yolov8_image_segmentation = yolov8ImageSegmentation()
    rclpy.spin(yolov8_image_segmentation)
    yolov8_image_segmentation.destroy_node()
    rclpy.shutdown()

    torch.cuda.empty_cache()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()