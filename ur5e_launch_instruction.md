# Launch Instruction
The packages can be run on the desktop workstation directly. Alternatively, user can choose to run `yolov8_image_segmentation` on the edge device.

## Setting up Jetson Orin Nano
To connect to Jetson Nano via ssh connection: make sure all the device are in the same local network (e.g. via router).

1.  To search the edge device ip address remotely, in your desktop connected to the router:
    *   Check your ip address using `ip addr`
        *   It should be under section that start with `eth` or `enp`
    *   Search the network using `nmap -sn (your router address)`, where your router address shuold be your ipaddress but with `0` for the value before `/24`

2.  Connects to the host using ssh:
    *   `ssh -Y host_user_name@host_ip_addr`

3.  Synchornise the system time (Every once when Jetson Orin Nano is reboot or boot):
To ensure the ur5e robot can be controlled via moveit2.
    *   `sudo nmcli radio wifi on` to turn on wifi
    *   Sync the time with the desktop
        ```
        sudo systemctl unmask systemd-timesyncd
        sudo systemctl enable systemd-timesyncd    
        sudo systemctl start systemd-timesyncd
        ```
        or
        ```
        sudo systemctl unmask chrony
        sudo systemctl enable chronyd
        sudo systemctl start chronyd
        ```
    *   `timedatectl status` to check your current system time
    *   `sudo nmcli radio wifi off` to turn off wifi **(Important)**

4.  Initialize the system (pick one):
    *  Jetson Orin Nano without realsense camera issue related to JetPack:
        ```
        cd ur5e_ws
        . install/setup.bash
        . /opt/ros/humble/setup.bash
        export ROS_DOMAIN_ID=25
    *   Jetson Orin Nano with realsense camera issue related to JetPack:  
        ```
        cd ur5e_ws
        . install/setup.bash
        sudo udevadm control --reload-rules && sudo udevadm trigger
        export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH.
        . /opt/ros/humble/setup.bash
        export ROS_DOMAIN_ID=25
        ```
## Launch the system:

#### A. At Jetson Orin Nano and UR5e robot:
1.  Follow the section Setting up Jetson Orin Nano - 4. Initialize the system to initialise the device everytime connected to it via ssh.
2.  Launch the ur5e driver:
    *   ```
        ros2 launch ur_robot_driver ur5e.launch.py  robot_ip:=192.168.0.100 initial_joint_controller:=joint_trajectory_controller launch_rviz:=true use_sim_time:=false
        ```
3.  Set up URCaps connection:
    *   Ensure the IP of your PC is on the same subnet as the UR5e Example robot IP: 192.168.0.100.
    *   Turn on and initialize ur5e robot.
    *   Go to Installation → URCaps → External Control, change the Host Ip to match the Jetson Orin Nano's IP.
    *   Go to Program → URCaps, press to select External Control
    *   Press play and select External Control.
    **Note: 
    Only play the external control after you launch ur5e driver.
    You need to stop(not pause) and play the External Control again if you relaunch your ur5e driver.**

4.  Launch the yolov8_image_segmentation nodes:
     *  ```
        ros2 launch yolov8_image_segmentation yolov8_image_segmentation.launch.py
        ```
#### B. At Desktop:
1.  Enter the following each time when a new window is opend in terminal:
    *   ```
        . install/setup.bash
        . /opt/ros/humble/setup.bash
        export ROS_DOMAIN_ID=25
        ```
2.  Launch moveit control and rviz in seperated terminal
    *   ```
        ros2 launch ur5e_moveit_config move_group.launch.py
        ```
    *   ```
        ros2 launch ur5e_moveit_config moveit_rviz.launch.py
        ```
3.  Launch `foundation_pose` node:
    *   ```
        ros2 run foundation_pose foundation_pose --ros-args --params-file src/vision_based_position_estimation/foundation_pose/config/foundation_pose_config.yaml
        ```
4.  Launch `cartesian_control` node to send goal position to moveit2
    *   ```
        ros2 run moveit_control_pkg cartesian_control
        ```

## Run the system:
Open 2 Jetson Orin Nano ssh terminal
1.  Publish segmented image to foundation pose
    *   ```
        ros2 service call /publish_init_mask std_srvs/srv/Trigger
        ```
2.  Control the gripper using pyserial miniterm
    *   ```
        python3 -m serial.tools.miniterm /dev/ttyACM0 115200 --eol LF 
        ```
    *   Enter pos:{position} to control the position (e.g. pos:100)
    *   Enter move:{distance} to move the gripper a relative distance (e.g. move:100)

## (Alternative) Move the Robot without Moveit:
1.  With the `ur_robot_driver` turned on and connected via URCaps, turn on the `scaled_joint_trajectory_controller`
    *   ```
        ros2 control switch_controllers --activate scaled_joint_trajectory_controller
        ```
    Check activated controller using:
    *   ```
        ros2 control list_controllers
        ```
2.  Control the joint angle via the topic
    * via terminal (for example):
        ```
        ros2 topic pub /scaled_joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
        header: {stamp: {sec: 0, nanosec: 0}},
        joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
        points: [{
            positions: [0.6515496373176575, -0.7259034675410767, 0.5065854231463831, 1.7610785204121093, 1.5376534461975098, 0.5],
            velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            time_from_start: {sec: 2, nanosec: 0}
        }]
        }"
        ```

