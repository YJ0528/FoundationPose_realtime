## Jetson Orin Nano setup:

1.  For Intel realsense camera conflict with Jetpack 6, see: [https://github.com/IntelRealSense/librealsense/issues/12831#issuecomment-2041140925](https://github.com/IntelRealSense/librealsense/issues/12831#issuecomment-2041140925)
    *   In your terminal, enter: 
        `rs-enumerate-device` to check if your device is connected to the realsense
        `realsense-viewer` to check if the the camera is streaming data.

2.  To download pytorch and pyvision for Jetson Nano (ARM64 architecture): see [https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) 


3. Download the dependencies using pip:
    ```
    pip install scikit-image meshcat webdataset omegaconf pypng roma seaborn opencv-contrib-python openpyxl wandb imgaug Ninja xlsxwriter timm albumentations xatlas rtree nodejs jupyterlab objaverse g4f ultralytics==8.0.120 pycocotools videoio numba
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 &&\
    pip install pytorch3d &&\
    pip install scipy joblib scikit-learn ruamel.yaml trimesh pyyaml opencv-python imageio open3d transformations warp-lang einops kornia pyrender
    ```
    *   For pytorch, pytorchvision, make sure the version is compatible with your system's cuda version.
    **Note: you mate need to troubleshoot them one by one.**
4. At `/vision_based_position_estimation/foundation_pose/config/foundation_pose_config.yaml`, configure the `foundation_pose_library_directory`, `foundation_pose_src_directory`, and `cutie_library_directory` accordingly. 
5. At `/vision_based_position_estimation/foundation_pose/foundation_pose/lib/FoundationPose`, run `bash build_all.sh`
6.  Install ur robot driver using `sudo apt-get install ros-humble-ur-robot-driver`
7.  Download ur5e_moveit_config from [
Moveit_ur5e_ros2-with-cartesian-path-planning
Public](https://github.com/loggcc/Moveit_ur5e_ros2-with-cartesian-path-planning) 

## Data Prepare:
see [here](https://github.com/loggcc/FoundationPose_realtime) for initial post.
1.  Download the customized [YOLOv8 segmentation model](https://drive.google.com/drive/folders/1d5r2kKmLp0LrwcIyJ-ldvjb4zBZkXhIE) best.pt, put it under `/vision_based_position_estimation/yolov8_image_segmentation/models`
2.  Download foundation pose [demo data](https://drive.google.com/drive/folders/1d5r2kKmLp0LrwcIyJ-ldvjb4zBZkXhIE) test_realtime and extract them under the folder `/vision_based_position_estimation/foundation_pose/models`
3.  Download the foundation pose [network weights](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing),put it under `/vision_based_position_estimation/foundation_pose/foundation_pose/lib/FoundationPose/weights`
4. Download the cutie [weights](https://drive.google.com/file/d/1Eja4IAnOW6Xi0ONiz8QTU18XEYZ1xJTC/view?usp=sharing), put it under `/vision_based_position_estimation/foundation_pose/foundation_pose/lib/cutie/weights`