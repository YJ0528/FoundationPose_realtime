import argparse
import os
import time
import torch
import json
import cv2
import sys
import numpy as np
import multiprocessing as mp
from typing import List
import imageio.v2 as imageio  
import trimesh
from scipy.spatial.transform import Rotation
from .VOT import Cutie, Tracker_2D  
from .kalman_filter_6d import KalmanFilter6D
import pyrealsense2 as rs
import logging
from ultralytics import YOLO
from hydra.core.global_hydra import GlobalHydra

from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
import sensor_msgs.msg
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PointStamped, PoseStamped
from ament_index_python.packages import get_package_share_directory
import message_filters
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation
import math


class foundationPoseMultiObject(Node):

    def __init__(self):

        super().__init__('foundation_pose_multi_object')
        self.bridge = CvBridge()
                
        self.get_logger().info('Initializing Foundation Pose Node...')

        # 0.82, 0.12,0.3
        self.T_base_tool0 = np.array([
            [ 1.000,  0.000,  0.000,  0.000],
            [ 0.000,  1.000,  0.000,  0.000],
            [ 0.000,  0.000,  1.000,  0.000],
            [ 0.000,  0.000,  0.000,  1.000]
        ])
        # tool position of home pose

        #matrix got from hand eye calibration

        self.T_cam2gripper = np.array([
           [ -0.71451674,  -0.69947352, 0.01423476,   0.04551642],
           [ 0.69960197, -0.71449134,  0.00769545,  0.04991512],
           [ 0.00478785,  0.01545719,  0.99986907,  0.04579313],
           [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])



        # self.T_cam2gripper = np.array([
        #     [ 0.57123,      0.81900076, -0.05416683,  0.06254428],
        #     [-0.82046694,   0.56790884, -0.06567765,  0.03732711],
        #     [-0.02302823,   0.08195913,  0.99636961,  0.01224618],
        #     [ 0.0,          0.0,         0.0,         1.0]
        # ])

        # Declare parameters
        self.declare_parameter('package_name', 'foundation_pose')
        self.declare_parameter('init_mask_topic', '/foundation_pose/init_mask')
        self.declare_parameter('color_camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('aligned_depth_camera_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('class_name', 'blue_tube')
        self.declare_parameter('apply_scale', 0.001)
        self.declare_parameter('force_apply_color', False)
        self.declare_parameter('apply_color', [0, 159, 237])
        self.declare_parameter('foundation_pose_library_directory', 'Moveit_ur5e_ros2-with-cartesian-path-planning/ur5e_ws/src/vision_based_position_estimatior/foundation_pose/foundation_pose/lib')
        self.declare_parameter('foundation_pose_src_directory', 'Moveit_ur5e_ros2-with-cartesian-path-planning/ur5e_ws/src/vision_based_position_estimatior/foundation_pose/foundation_pose/lib/FoundationPose')
        self.declare_parameter('cutie_library_directory', 'Moveit_ur5e_ros2-with-cartesian-path-planning/ur5e_ws/src/vision_based_position_estimatior/foundation_pose/foundation_pose/lib/Cutie')
        self.declare_parameter('activate_2d_tracker', True)
        self.declare_parameter('activate_kalman_filter', False)
        self.declare_parameter('kf_measurement_noise_scale', 0.05)
        self.declare_parameter('est_refine_iter', 10)
        self.declare_parameter('track_refine_iter', 5)
        self.declare_parameter('cam_K', [387.88845825, 0.0, 323.28192139, 0.0, 387.46902466, 237.11705017, 0.0, 0.0, 1.0])
        self.declare_parameter('publish_once', True)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'tool0')

        # Get parameters 
        self.package_name = self.get_parameter('package_name').get_parameter_value().string_value
        self.init_mask_topic = self.get_parameter('init_mask_topic').get_parameter_value().string_value
        self.color_camera_topic = self.get_parameter('color_camera_topic').get_parameter_value().string_value
        self.aligned_depth_camera_topic = self.get_parameter('aligned_depth_camera_topic').get_parameter_value().string_value
        self.class_name = self.get_parameter('class_name').get_parameter_value().string_value
        self.apply_scale = self.get_parameter('apply_scale').get_parameter_value().double_value
        self.force_apply_color = self.get_parameter('force_apply_color').get_parameter_value().bool_value
        self.apply_color = self.get_parameter('apply_color').get_parameter_value().integer_array_value
        self.foundation_pose_library_directory = self.get_parameter('foundation_pose_library_directory').get_parameter_value().string_value
        self.foundation_pose_src_directory = self.get_parameter('foundation_pose_src_directory').get_parameter_value().string_value
        self.cutie_library_directory = self.get_parameter('cutie_library_directory').get_parameter_value().string_value
        self.activate_2d_tracker = self.get_parameter('activate_2d_tracker').get_parameter_value().bool_value
        self.activate_kalman_filter = self.get_parameter('activate_kalman_filter').get_parameter_value().bool_value
        self.kf_measurement_noise_scale = self.get_parameter('kf_measurement_noise_scale').get_parameter_value().double_value
        self.est_refine_iter = self.get_parameter('est_refine_iter').get_parameter_value().integer_value
        self.track_refine_iter = self.get_parameter('track_refine_iter').get_parameter_value().integer_value
        cam_K_flat = self.get_parameter('cam_K').get_parameter_value().double_array_value
        self.cam_K = np.array(cam_K_flat, dtype=np.float64).reshape(3, 3)
        self.publish_once = self.get_parameter('publish_once').get_parameter_value().bool_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.tool_frame = self.get_parameter('tool_frame').get_parameter_value().string_value

        # Initialize as None
        self.est = {}
        self.tracker_2D = {}
        self.kf = {}
        self.pose_seq = [] 
        self.frame_times = []
        self.kf_mean = {}
        self.kf_covariance = {}
        self.mask = None
        self.frame_count = 0
        self.draw_posed_3d_box = None
        self.draw_xyz_axis = None
        self.frame_start_time = time.time()
        self.to_origin = {}
        self.bbox = {}
        self.classes_name = []
        self.init_mask_subscriber = {}
        self.vis_color = None

        #################################################
        # Declare subsrciber and publisher
        #################################################
        classes_name_qos_profile = QoSProfile(
            reliability = QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth = 1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.init_mask_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,  # Better for networks
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        camera_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE  # No persistence needed
        )

        self.classes_name_subscriber = self.create_subscription(
            String,  
            '/yolo_model/available_classes',
            self.classes_name_callback,
            classes_name_qos_profile  
        )

        self.color_subscriber = self.create_subscription(
            sensor_msgs.msg.Image, 
            self.color_camera_topic,
            self.color_callback,
            camera_qos_profile
        )

        self.depth_subscriber = self.create_subscription(
            sensor_msgs.msg.Image, 
            self.aligned_depth_camera_topic,
            self.depth_callback,
            camera_qos_profile
        )

        self.publisher = self.create_publisher(
            PointStamped,
            'target_position',
            10  # Queue size
        )

        # Store latest messages
        self.latest_color = None
        self.latest_depth = None
        self.depth_updated = False
        self.color_updated = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.timer = self.create_timer(0.02, self.process_latest_data)  # 50 Hz?
        

    def classes_name_callback(self, msg):
        try:
            received_classes = json.loads(msg.data)        
            self.classes_name = received_classes.copy()

            for i, class_name in enumerate(self.classes_name):
                self.get_logger().info(f"  {i}: {class_name}")
            
            for class_name in self.classes_name:
                
                topic_name = f"{self.init_mask_topic}/{class_name.replace(' ', '_').replace('-', '_')}"

                self.init_mask_subscriber[class_name] = self.create_subscription(
                    sensor_msgs.msg.Image,  
                    topic_name, 
                    lambda msg, cn=class_name: self.init_mask_callback(msg, cn),
                    self.init_mask_qos_profile
                )
            
            self.init_foundation_pose()
            # self.destroy_subscription(self.classes_name_subscriber)
            self.get_logger().info(f'Initialization completed ')
            
        except Exception as e:
            self.get_logger().error(f"Error in classes_name_callback: {e}")        
        

    def init_mask_callback(self, inMsg, class_name):
        
        self.mask = None    
        self.vis_color = None
        self.class_name = class_name
        # if not (self.depth_updated and self.color_updated): return
        
        self.frame_start_time = time.time()
        self.depth_updated = False
        self.color_updated = False
        
        try:
            #################################################
            # Extract initial mask
            #################################################
            self.mask = self.bridge.imgmsg_to_cv2(inMsg, desired_encoding='mono8')
            self.get_logger().info(f"Received mask with shape: {self.mask.shape}")
            
            #################################################
            # Extract rgbd data
            #################################################
            # Convert ROS Image messages to OpenCV format
            color = self.bridge.imgmsg_to_cv2(self.latest_color, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='16UC1')
            depth = depth.astype(np.float32) / 1000.0  # Convert mm to meters
            depth[(depth < 0.001) | (depth >= np.inf)] = 0
            
            # Resize if needed (though aligned frames should already match)
            color = cv2.resize(color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            #################################################
            # Foundation pose estimation
            #################################################
            pose = self.est[self.class_name].register(K=self.cam_K, rgb=color, depth=depth, ob_mask=self.mask, iteration=self.est_refine_iter)

            pose_arr = self.get_6d_pose_arr_from_mat(pose)
            position = pose_arr[:3]
            euler = pose_arr[3:]
            quaternion = Rotation.from_euler('xyz', euler).as_quat() 
            print("Position:", position)
            pose_matrix = self.get_mat_from_6d_pose_arr(pose_arr)

            # print(f"Frame {self.frame_count}:")
            # print("Position:", position)
            # print("Orientation:", euler)
            # print("Pose:\n", pose)
            print("Pose Matrix:\n", pose_matrix)
            if self.activate_kalman_filter:
                self.kf_mean[self.class_name], self.kf_covariance[self.class_name] = self.kf[self.class_name].initiate(pose_arr)

            if self.activate_2d_tracker:
                self.tracker_2D[self.class_name].initialize(color, init_info={"mask": self.mask})
            #################################################
            # Update frame process time
            #################################################
            self.pose_seq.append(pose.reshape(4, 4))
            frame_end_time = time.time()
            self.get_logger().info(f"frame_time{frame_end_time - self.frame_start_time}\n\n")
            self.frame_count += 1
            self.frame_times.append(frame_end_time - self.frame_start_time)


            # if self.frame_count > 1 and self.publish_once: return
            #################################################
            # Compute position from base frame
            #################################################
            self.update_base_tool0_transform()
            T_obj2base = self.T_base_tool0 @ self.T_cam2gripper @ pose_matrix 
            global_position = T_obj2base[:3, 3]
            # position = T_obj2base[:3, 3].copy()
            global_position[2] += 0.27
            print("object position in base frame:", global_position)

            #################################################
            # Publish target position
            #################################################
            # output_msg = PoseStamped()
            # output_msg.header.stamp = self.get_clock().now().to_msg()
            # output_msg.header.frame_id = "base_link"

            # output_msg.pose.position.x = float(global_position[0])
            # output_msg.pose.position.y = float(global_position[1])
            # output_msg.pose.position.z = float(global_position[2])
            # output_msg.pose.orientation.x = quaternion[0]
            # output_msg.pose.orientation.y = quaternion[1]
            # output_msg.pose.orientation.z = quaternion[2]
            # output_msg.pose.orientation.w = quaternion[3]
            # self.publisher.publish(output_msg)
            
            output_msg = PointStamped()
            output_msg.header.stamp = self.get_clock().now().to_msg()
            output_msg.header.frame_id = "base_link" 
            output_msg.point.x = float(global_position[0])
            output_msg.point.y = float(global_position[1])
            output_msg.point.z = float(global_position[2])
            self.publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing images: {str(e)}")
    
    def color_callback(self, msg):

        self.latest_color = msg
        self.color_updated = True

    def depth_callback(self, msg):

        self.latest_depth = msg
        self.depth_updated = True

    def process_latest_data(self):
        
        
        if self.mask is None or self.mask.size == 0: return
        if not (self.depth_updated and self.color_updated): return
        # print(self.class_name)
        self.frame_start_time = time.time()
        self.depth_updated = False
        self.color_updated = False
        
        try:
            #################################################
            # Extract rgbd data
            #################################################
            # Convert ROS Image messages to OpenCV format
            color = self.bridge.imgmsg_to_cv2(self.latest_color, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='16UC1')
            depth = depth.astype(np.float32) / 1000.0  # Convert mm to meters
            depth[(depth < 0.001) | (depth >= np.inf)] = 0
            
            # Resize if needed (though aligned frames should already match)
            color = cv2.resize(color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            #################################################
            # Foundation pose estimation
            #################################################
            if self.activate_2d_tracker:
                bbox_2d = self.tracker_2D[self.class_name].track(color)

                if not self.activate_kalman_filter:
                    self.est[self.class_name].pose_last = self.adjust_pose_to_image_point(
                        ob_in_cam=self.est[self.class_name].pose_last, K=self.cam_K,
                        x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2
                    )
                else:
                    self.kf_mean[self.class_name], self.kf_covariance[self.class_name] = self.kf[self.class_name].update(self.kf_mean[self.class_name], self.kf_covariance[self.class_name], self.get_6d_pose_arr_from_mat(self.est[self.class_name].pose_last))
                    measurement_xy = np.array(self.get_pose_xy_from_image_point(
                        ob_in_cam=self.est[self.class_name].pose_last, K=self.cam_K,
                        x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2
                    ))
                    self.kf_mean[self.class_name], self.kf_covariance[self.class_name] = self.kf[self.class_name].update_from_xy(self.kf_mean[self.class_name], self.kf_covariance[self.class_name], measurement_xy)
                    self.est[self.class_name].pose_last = torch.from_numpy(self.get_mat_from_6d_pose_arr(self.kf_mean[self.class_name][:6])).unsqueeze(0).to(self.est[self.class_name].pose_last.device)

                pose = self.est[self.class_name].track_one(rgb=color, depth=depth, K=self.cam_K, iteration= self.track_refine_iter)
                pose_arr = self.get_6d_pose_arr_from_mat(pose)
                position = pose_arr[:3]
                euler = pose_arr[3:]
                pose_matrix = self.get_mat_from_6d_pose_arr(pose_arr)
                
                # print(f"Frame {i}:")
                # print("Position:", position)
                # print("Orientation:", euler)
                # print("Pose:\n", pose)
                print("Pose Matrix:\n", pose_matrix)

                if self.activate_2d_tracker and self.activate_kalman_filter:
                    self.kf_mean[self.class_name], self.kf_covariance[self.class_name] = self.kf[self.class_name].predict(self.kf_mean[self.class_name], self.kf_covariance[self.class_name])
            
           

            #################################################
            # Draw box in cv2 visualization
            #################################################
            center_pose = pose @ np.linalg.inv(self.to_origin[self.class_name])
            self.vis_color = self.draw_posed_3d_box(self.cam_K, img=color, ob_in_cam= center_pose, bbox = self.bbox[self.class_name])
            self.vis_color = self.draw_xyz_axis(
                self.vis_color,
                ob_in_cam= center_pose,
                scale=0.1,
                K = self.cam_K,
                thickness=2,
                transparency=0,
                is_input_rgb=True,
            )
            cv2.imshow("Pose Tracking", self.vis_color)
            # cv2.imshow("Pose Tracking", cv2.cvtColor(self.vis_color, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'): return

            #################################################
            # Update frame process time
            #################################################
            self.pose_seq.append(pose.reshape(4, 4))
            frame_end_time = time.time()
            self.get_logger().info(f"frame_time{frame_end_time - self.frame_start_time}\n\n")
            self.frame_count += 1
            self.frame_times.append(frame_end_time - self.frame_start_time)

        except Exception as e:
            self.get_logger().error(f"Error processing images: {str(e)}")
       
    def init_foundation_pose(self):

        sys.path.append(os.path.join(os.path.expanduser('~'),self.cutie_library_directory))
        sys.path.append(os.path.join(os.path.expanduser('~'),self.foundation_pose_library_directory))
        sys.path.append(os.path.join(os.path.expanduser('~'),self.foundation_pose_src_directory))        
        
        self.get_logger().info(f'Initializatiing Estimators and utilities from FoundationPose...')
        from FoundationPose.estimater import trimesh_add_pure_colored_texture
        from FoundationPose.estimater import (
            ScorePredictor,
            PoseRefinePredictor,
            dr,
            FoundationPose,
            logging,
            draw_posed_3d_box,
            draw_xyz_axis,
        )

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.draw_posed_3d_box = draw_posed_3d_box
        self.draw_xyz_axis = draw_xyz_axis

        for class_name in self.classes_name:
            mesh_file_dir = os.path.join(
                get_package_share_directory(self.package_name), 
                "test_realtime", 
                class_name, 
                "mesh", 
                f"{class_name}.stl")          
            try:
                self.get_logger().info(f'Loading mesh from {mesh_file_dir}...')
                mesh = trimesh.load(os.path.abspath(mesh_file_dir))
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                mesh.apply_scale(self.apply_scale)
                if self.force_apply_color:
                    mesh = trimesh_add_pure_colored_texture(mesh, color=np.array(self.apply_color), resolution=10)
                self.to_origin[class_name], extents = trimesh.bounds.oriented_bounds(mesh)
                self.bbox[class_name] = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

                self.est[class_name] = FoundationPose(
                    model_pts=mesh.vertices,
                    model_normals=mesh.vertex_normals,
                    mesh=mesh,
                    scorer=scorer,
                    refiner=refiner,
                    glctx=glctx,
                )
 
                #################################################
                # Instantiate the 2D tracker
                #################################################

                if self.activate_2d_tracker:     # Default using Cutie as a 2D tracker
                    if GlobalHydra().is_initialized():
                        GlobalHydra.instance().clear()
                    self.tracker_2D[class_name] = Cutie()
                else:
                    self.tracker_2D[class_name] = Tracker_2D()

                #################################################
                # 6D pose tracking
                #################################################

                if self.activate_kalman_filter:
                    self.kf[class_name] = KalmanFilter6D(self.kf_measurement_noise_scale)
              

            except Exception as e:
                self.get_logger().warn(f'Failed to load mesh for {class_name}: {str(e)}')
                continue

    def adjust_pose_to_image_point(
            self,
            ob_in_cam: torch.Tensor,
            K: torch.Tensor,
            x: float = -1.,
            y: float = -1.,
    ) -> torch.Tensor:
        """
        Adjusts the 6D pose(s) so that the projection matches the given 2D coordinate (x, y).

        Parameters:
        - ob_in_cam: Original 6D pose(s) as [4,4] or [B,4,4] tensor.
        - K: Camera intrinsic matrix (3x3 tensor).
        - x, y: Desired 2D coordinates on the image plane.

        Returns:
        - ob_in_cam_new: Adjusted pose(s) in same shape as input (tensor).
        """
        device = ob_in_cam.device
        dtype = ob_in_cam.dtype

        is_batched = ob_in_cam.ndim == 3
        if not is_batched:
            ob_in_cam = ob_in_cam.unsqueeze(0)  # [1, 4, 4]

        B = ob_in_cam.shape[0]
        ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)

        for i in range(B):
            R = ob_in_cam[i, :3, :3]
            t = ob_in_cam[i, :3, 3]

            tx, ty = self.get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
            t_new = torch.tensor([tx, ty, t[2]], device=device, dtype=dtype)

            ob_in_cam_new[i, :3, :3] = R
            ob_in_cam_new[i, :3, 3] = t_new

        return ob_in_cam_new if is_batched else ob_in_cam_new[0]

    def get_pose_xy_from_image_point(
            self,
            ob_in_cam: torch.Tensor, 
            K: torch.Tensor, 
            x: float = -1., 
            y: float = -1.,
    ) -> tuple:
        """
        Computes new (tx, ty) in camera space such that the projection matches image point (x, y).

        Parameters:
        - ob_in_cam: 4x4 pose tensor.
        - K: 3x3 intrinsic matrix tensor.
        - x, y: Desired image coordinates.

        Returns:
        - tx, ty: New x/y in camera coordinate system.
        """

        is_batched = ob_in_cam.ndim == 3
        if is_batched:
            ob_in_cam_new = ob_in_cam[0].cpu()  # [1, 4, 4]
        else:
            ob_in_cam_new = ob_in_cam.cpu()

        if x == -1. or y == -1.:
            return x, y
        
        t = ob_in_cam_new[:3, 3]

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        tz = t[2]

        tx = (x - cx) * tz / fx
        ty = (y - cy) * tz / fy

        return tx, ty

    def get_mat_from_6d_pose_arr(self, pose_arr):
        # get (xyz) translation
        xyz = pose_arr[:3]
        
        # get euler angles
        euler_angles = pose_arr[3:]
        
        # generate rotation matirx
        rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
        rotation_matrix = rotation.as_matrix()
        
        # generate 4*4 tansformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = xyz
        
        return transformation_matrix

    def get_6d_pose_arr_from_mat(self, pose):
        if torch.is_tensor(pose):
            is_batched = pose.ndim == 3
            if is_batched:
                pose_np = pose[0].cpu().numpy()
            else:
                pose_np = pose.cpu().numpy()
        else:
            pose_np = pose

        xyz = pose_np[:3, 3]
        rotation_matrix = pose_np[:3, :3]
        euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
        return np.r_[xyz, euler_angles]
    
    def update_base_tool0_transform(self):
        """Update T_base_tool0 with live robot pose"""
        try:
            # Get transform from base_link to tool0
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, 
                self.tool_frame,  
                rclpy.time.Time() 
            )
            
            # Extract translation
            trans = transform.transform.translation
            translation = np.array([trans.x, trans.y, trans.z])
            
            # Extract rotation and convert to matrix
            rot = transform.transform.rotation
            quaternion = [rot.x, rot.y, rot.z, rot.w]
            r = Rotation.from_quat(quaternion)
            rotation_matrix = r.as_matrix()
            
            # Update the transformation matrix
            self.T_base_tool0[:3, :3] = rotation_matrix
            self.T_base_tool0[:3, 3] = translation
            
            # Optional: uncomment to see live updates
            # self.get_logger().info(f"Updated T_base_tool0: position=[{trans.x:.3f}, {trans.y:.3f}, {trans.z:.3f}]")
            
        except TransformException as ex:
            self.get_logger().info("Failed to obtain coordinate")


def main(args=None):
    rclpy.init(args=args)

    foundation_pose_multi_object = foundationPoseMultiObject()
    try:
        rclpy.spin(foundation_pose_multi_object)
    except KeyboardInterrupt:
        print('Received keyboard interrupt!')
    except ExternalShutdownException:
        print('Received external shutdown request!')

    print('Exiting...')

    foundation_pose_multi_object.destroy_node()
    rclpy.try_shutdown()
    torch.cuda.empty_cache()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()