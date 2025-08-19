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
import pyrealsense2 as rs
import logging
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

package_name= 'foundation_pose'
package_dir= get_package_share_directory(package_name) 
test_case = 'blue_tube'
apply_scale = 0.01
force_apply_color = False
apply_color = [0, 159, 237]

src_path = os.path.join(
    package_dir, "..", "..", "..", "..", 
    "src", 
    package_name, 
    "foundation_pose", 
    "lib")

foundationpose_path = os.path.join(
    src_path, 
    "FoundationPose")

mesh_file_dir = os.path.join(
    src_path, 
    "..", "..",
    "test_realtime", 
    test_case, 
    "mesh", 
    f"{test_case}.stl")

if src_path not in sys.path:
    sys.path.append(src_path)
    
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)

print(os.path.abspath(src_path))
print(os.path.abspath(foundationpose_path))
print(os.path.abspath(mesh_file_dir))


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


mesh = trimesh.load(os.path.abspath(mesh_file_dir))
if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)
mesh.apply_scale(apply_scale)
if force_apply_color:
    mesh = trimesh_add_pure_colored_texture(mesh, color=np.array(apply_color), resolution=10)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    glctx=glctx,
)
logging.info("Estimator initialization done")