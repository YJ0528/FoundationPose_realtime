#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
// #include <ur_msgs/ur_msgs/srv/set_io.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <rclcpp/rclcpp.hpp>
#include <mutex>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_demo");

geometry_msgs::msg::PoseStamped latest_target;
bool has_new_target = false;
std::mutex target_mutex;

void target_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(target_mutex);
  latest_target = *msg;
  has_new_target = true;
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  auto move_group_node = rclcpp::Node::make_shared("move_group_interface_tutorial", node_options);

  auto sub = move_group_node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "target_position", 10, target_callback);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(move_group_node);
  std::thread([&executor]() { executor.spin(); }).detach();

  static const std::string PLANNING_GROUP_ARM = "ur5e_arm";
  moveit::planning_interface::MoveGroupInterface move_group(move_group_node, PLANNING_GROUP_ARM);
  move_group.setEndEffectorLink("tool0");

  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  std::string frame_id = move_group.getPlanningFrame();

  RCLCPP_INFO(LOGGER, "Waiting for target position...");
  while (rclcpp::ok()) {
    {
      std::lock_guard<std::mutex> lock(target_mutex);
      if (!has_new_target) continue;
    }

    geometry_msgs::msg::Pose target_pose1;
    {
      std::lock_guard<std::mutex> lock(target_mutex);
      target_pose1.position = latest_target.pose.position;
      // target_pose1.orientation = latest_target.pose.orientation;
      has_new_target = false;
    }

    // Extract yaw from the received object orientation
    tf2::Quaternion object_q;
    tf2::fromMsg(latest_target.pose.orientation, object_q);

    tf2::Matrix3x3 object_matrix(object_q);
    double object_roll, object_pitch, object_yaw;
    object_matrix.getRPY(object_roll, object_pitch, object_yaw);

    // RCLCPP_INFO(LOGGER, "Object RPY: roll=%.3f, pitch=%.3f, yaw=%.3f", 
    //     object_roll, object_pitch, object_yaw);

    // Method 1: Create downward-facing gripper with object's yaw rotation
    tf2::Quaternion gripper_q;
    gripper_q.setRPY(M_PI, 0.0, object_yaw);  // Roll=180° (downward), Pitch=0°, Yaw=object's yaw

    // Convert to geometry_msgs and set the orientation
    target_pose1.orientation.x = gripper_q.x();
    target_pose1.orientation.y = gripper_q.y();
    target_pose1.orientation.z = gripper_q.z();
    target_pose1.orientation.w = gripper_q.w();

    // RCLCPP_INFO(LOGGER, "Gripper yaw aligned to: %.3f rad (%.1f deg)", 
    //     object_yaw, object_yaw * 180.0 / M_PI);

    RCLCPP_INFO(LOGGER, "Target pose: pos(%.3f, %.3f, %.3f) orient(%.3f, %.3f, %.3f, %.3f)",
        target_pose1.position.x,
        target_pose1.position.y,
        target_pose1.position.z,
        target_pose1.orientation.x,
        target_pose1.orientation.y,
        target_pose1.orientation.z,
        target_pose1.orientation.w);


    std::vector<geometry_msgs::msg::Pose> waypoints;
    geometry_msgs::msg::Pose start_pose = move_group.getCurrentPose().pose;
    
    // target_pose1.position.x = -0.499;
    // target_pose1.position.y = 0.055;
    // target_pose1.position.z = 0.468;
    geometry_msgs::msg::Pose mid_pose = target_pose1;
    mid_pose.position.z += 0.10;
    
    // waypoints.push_back(start_pose);
    waypoints.push_back(mid_pose);
    waypoints.push_back(target_pose1);

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double eef_step = 0.01;
    const double jump_threshold = 0.0;
    double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

    if (fraction > 0.99) {
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      robot_trajectory::RobotTrajectory rt(move_group.getRobotModel(), move_group.getName());
      rt.setRobotTrajectoryMsg(*move_group.getCurrentState(), trajectory);
      trajectory_processing::IterativeParabolicTimeParameterization iptp;
      iptp.computeTimeStamps(rt, 0.03, 0.03);
      rt.getRobotTrajectoryMsg(trajectory);
      plan.trajectory_ = trajectory;

      move_group.execute(plan);

      // auto client = move_group_node->create_client<ur_msgs::srv::SetIO>("/io_and_status_controller/set_io");
      // auto request = std::make_shared<ur_msgs::srv::SetIO::Request>();
      // request->fun = 1;
      // request->pin = 1;
      // request->state = 1.0;

      // if (client->wait_for_service(std::chrono::seconds(1))) {
      //   auto future = client->async_send_request(request);
      //   auto status = future.wait_for(std::chrono::seconds(5));
      //   if (status == std::future_status::ready) {
      //     auto result = future.get();
      //     if (result->success) {
      //       RCLCPP_INFO(LOGGER, "Digital output pin 1 set to HIGH");
      //     } else {
      //       RCLCPP_WARN(LOGGER, "Failed to set digital output");
      //     }
      //   } else {
      //     RCLCPP_WARN(LOGGER, "Service call timeout");
      //   }
      // } else {
      //   RCLCPP_WARN(LOGGER, "Service /io_and_status_controller/set_io not available");
      // }

    } 
    else {
      RCLCPP_WARN(LOGGER, "Cartesian path planning failed, only achieved %.2f%% of the path", fraction * 100.0);
    }
  }

  rclcpp::shutdown();
  return 0;
}
