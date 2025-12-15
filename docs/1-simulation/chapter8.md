---
id: simulation-chapter8
title: Module-1.simulation-chapter-8
sidebar_position: 4
slug: /simulation/chapter8
---

# Chapter 8: Controlling Simulated Robots with ROS2

Having a robot model in Gazebo is only the first step; to make it perform useful tasks, you need to control its joints. ROS2 provides `ros2_control`, a powerful framework that offers a standardized way to interface with robot hardware (or simulated hardware) and control its various components.

## Introduction to `ros2_control`

`ros2_control` is a set of packages that provides a generic and reusable architecture for robot control. It decouples the robot's control logic from its specific hardware interfaces, making it easier to switch between different robots or between physical and simulated environments without changing your high-level control code.

Key concepts in `ros2_control`:

*   **Hardware Interface**: A software component that directly interacts with the robot's actuators and sensors (e.g., motor controllers, joint encoders). In simulation, this is provided by Gazebo plugins.
*   **Controller**: A software component that implements a control strategy (e.g., a position controller, a velocity controller, an impedance controller). Controllers run in the `controller_manager`.
*   **Controller Manager**: A central node responsible for loading, starting, stopping, and switching controllers.
*   **URDF/XACRO extensions**: Your robot's description file (URDF/XACRO) is extended to include `<ros2_control>` tags, specifying which hardware interfaces and controllers are available for each joint.

## Integrating `ros2_control` with our Robot

Let's modify `my_robot.xacro` to include `ros2_control` configuration for our differential drive robot's wheels.

1.  **Modify `~/ros2_ws/src/my_robot_pkg/urdf/my_robot.xacro`:**

    We need to add a `<ros2_control>` block and Gazebo-specific plugins.

    ```xml
    <?xml version="1.0"?>
    <robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

      <xacro:macro name="default_inertial" params="mass">
        <inertial>
          <mass value="${mass}"/>
          <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
        </inertial>
      </xacro:macro>

      <link name="base_link">
        <visual>
          <geometry>
            <box size="0.3 0.2 0.1"/>
          </geometry>
          <material name="blue">
            <color rgba="0 0 0.8 1"/>
          </material>
        </visual>
        <collision>
          <geometry>
            <box size="0.3 0.2 0.1"/>
          </geometry>
        </collision>
        <xacro:default_inertial mass="1.0"/>
      </link>

      <link name="left_wheel_link">
        <visual>
          <origin xyz="0 0 0" rpy="${pi/2} 0 0"/>
          <geometry>
            <cylinder radius="0.05" length="0.02"/>
          </geometry>
          <material name="black">
            <color rgba="0 0 0 1"/>
          </material>
        </visual>
        <collision>
          <origin xyz="0 0 0" rpy="${pi/2} 0 0"/>
          <geometry>
            <cylinder radius="0.05" length="0.02"/>
          </geometry>
        </collision>
        <xacro:default_inertial mass="0.1"/>
      </link>

      <link name="right_wheel_link">
        <visual>
          <origin xyz="0 0 0" rpy="${pi/2} 0 0"/>
          <geometry>
            <cylinder radius="0.05" length="0.02"/>
          </geometry>
          <material name="black">
            <color rgba="0 0 0 1"/>
          </material>
        </visual>
        <collision>
          <origin xyz="0 0 0" rpy="${pi/2} 0 0"/>
          <geometry>
            <cylinder radius="0.05" length="0.02"/>
          </geometry>
        </collision>
        <xacro:default_inertial mass="0.1"/>
      </link>

      <joint name="base_to_left_wheel" type="continuous">
        <parent link="base_link"/>
        <child link="left_wheel_link"/>
        <origin xyz="0.1 0.11 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
      </joint>

      <joint name="base_to_right_wheel" type="continuous">
        <parent link="base_link"/>
        <child link="right_wheel_link"/>
        <origin xyz="0.1 -0.11 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
      </joint>

      <!-- ROS2 Control configuration -->
      <ros2_control name="GazeboSystem" type="system">
        <hardware>
          <plugin>gazebo_ros2_control/GazeboSystem</plugin>
        </hardware>
        <joint name="base_to_left_wheel">
          <command_interface name="velocity">
            <param name="min">-10</param>
            <param name="max">10</param>
          </command_interface>
          <state_interface name="velocity"/>
          <state_interface name="position"/>
        </joint>
        <joint name="base_to_right_wheel">
          <command_interface name="velocity">
            <param name="min">-10</param>
            <param name="max">10</param>
          </command_interface>
          <state_interface name="velocity"/>
          <state_interface name="position"/>
        </joint>
      </ros2_control>

      <!-- Gazebo Diff Drive Plugin (for physics and cmd_vel interface) -->
      <gazebo>
        <plugin name="gazebo_ros_ackermann_diff_drive" filename="libgazebo_ros_ackermann_diff_drive.so">
          <ros> <!-- ROS properties -->
            <namespace>/</namespace>
            <command_topic>cmd_vel</command_topic>
            <odom_topic>odom</odom_topic>
            <odom_frame>odom</odom_frame>
            <base_frame>base_link</base_frame>
          </ros>
          <left_joint>base_to_left_wheel</left_joint>
          <right_joint>base_to_right_wheel</right_joint>
          <wheel_separation>0.22</wheel_separation> <!-- from model geometry -->
          <wheel_radius>0.05</wheel_radius> <!-- from model geometry -->
          <publish_odom>true</publish_odom>
          <publish_odom_tf>true</publish_odom>
          <publish_wheel_tf>true</publish_wheel_tf>
          <legacy_mode>false</legacy_mode>
          <max_wheel_torque>200</max_wheel_torque>
          <max_wheel_acceleration>10.0</max_wheel_acceleration>
        </plugin>
      </gazebo>

    </robot>
    ```
    *Note: The `gazebo_ros_ackermann_diff_drive` plugin is used here as a common example for differential drive robots to translate `cmd_vel` to joint commands. You might need to install `ros-humble-gazebo-ros-pkgs` for this.* If you prefer `diff_drive` over `ackermann_diff_drive` you may need to adjust the plugin name in the `.xacro` accordingly and install `ros-humble-gazebo-ros2-control` which should provide `libgazebo_ros_diff_drive.so`.

2.  **Create a `my_robot_pkg/config/` directory:**

    ```bash
    mkdir -p ~/ros2_ws/src/my_robot_pkg/config
    ```

3.  **Create `~/ros2_ws/src/my_robot_pkg/config/my_robot_controllers.yaml`:**

    This YAML file defines the controllers to be loaded by the `controller_manager`.

    ```yaml
    controller_manager:
      ros__parameters:
        update_rate: 100 # Hz

        joint_state_broadcaster:
          type: joint_state_broadcaster/JointStateBroadcaster

        diff_drive_controller:
          type: diff_drive_controller/DiffDriveController

    diff_drive_controller:
      ros__parameters:
        left_wheel_names: ['base_to_left_wheel']
        right_wheel_names: ['base_to_right_wheel']
        wheel_separation: 0.22
        wheel_radius: 0.05
        publish_rate: 50.0
        velocity_rolling_window_size: 10
        linear_tolerance: 0.0
        angular_tolerance: 0.0

        base_frame_id: base_link
        odom_frame_id: odom

        # Optional: kinematics parameters for a differential drive robot
        # The plugin takes care of this, but it's good to define if not using plugin
        # wheel_separation: 0.22
        # wheel_radius: 0.05
    ```

4.  **Update `setup.py` to install the config file:**

    Add the `config` directory to `data_files` in `~/ros2_ws/src/my_robot_pkg/setup.py`:

    ```python
    # ... inside setup()
    data_files=[
        ...
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    # ... rest of setup()
    ```

5.  **Modify `~/ros2_ws/src/my_robot_pkg/launch/gazebo_launch.py`:**

    We need to:
    *   Load the `robot_controllers.yaml` file.
    *   Launch the `controller_manager` node.
    *   Load and start the `joint_state_broadcaster` and `diff_drive_controller`.

    ```python
    import os
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch.substitutions import LaunchConfiguration, Command
    from launch_ros.actions import Node

    def generate_launch_description():
        # Declare arguments
        pkg_name = 'my_robot_pkg'
        world_file_name = 'empty.world'
        robot_file_name = 'my_robot.xacro'
        controller_file_name = 'my_robot_controllers.yaml'

        # Get package directory
        pkg_share_dir = get_package_share_directory(pkg_name)
        world_path = os.path.join(pkg_share_dir, 'worlds', world_file_name)
        robot_path = os.path.join(pkg_share_dir, 'urdf', robot_file_name)
        controller_config_path = os.path.join(pkg_share_dir, 'config', controller_file_name)

        # Process the xacro file into a URDF string
        robot_description_content = Command(['xacro ', robot_path])

        # Robot State Publisher Node
        robot_state_publisher_node = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}],
        )

        # Launch Gazebo
        gazebo_ros_package_dir = get_package_share_directory('gazebo_ros')
        gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_package_dir, 'launch', 'gazebo.launch.py')),
            launch_arguments={'world': world_path}.items(),
        )

        # Spawn Entity Node (from ros_gz_sim package)
        spawn_entity = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-name', 'my_robot',
                       '-topic', 'robot_description',
                       '-x', '0.0',
                       '-y', '0.0',
                       '-z', '0.1'],
            output='screen',
        )

        # ROS2 Control Nodes
        control_node = Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[robot_description_content, controller_config_path],
            output={'stdout': 'screen', 'stderr': 'screen'},
        )

        # Load controllers
        load_joint_state_broadcaster = Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
            output={'stdout': 'screen', 'stderr': 'screen'},
        )

        load_diff_drive_controller = Node(
            package='controller_manager',
            executable='spawner',
            arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
            output={'stdout': 'screen', 'stderr': 'screen'},
        )

        return LaunchDescription([
            gazebo_launch,
            robot_state_publisher_node,
            spawn_entity,
            control_node,
            load_joint_state_broadcaster,
            load_diff_drive_controller
        ])
    ```
    *Don't forget to add necessary imports, e.g., `from launch_ros.actions import Node`.* And ensure you have `ros-humble-controller-manager` and `ros-humble-diff-drive-controller` installed (`sudo apt install ros-humble-controller-manager ros-humble-diff-drive-controller`).

6.  **Build and source your package:**

    ```bash
    cd ~/ros2_ws
    colcon build --packages-select my_robot_pkg
    source install/setup.bash
    ```

7.  **Launch Gazebo with controlled robot:**

    ```bash
    ros2 launch my_robot_pkg gazebo_launch.py
    ```

    You should see Gazebo launch, and in the terminal output, messages indicating that the controllers are loaded and started.

## Sending Commands to the Robot

With the `diff_drive_controller` loaded, your robot should now be listening for `geometry_msgs/msg/Twist` messages on the `/diff_drive_controller/cmd_vel_unstamped` topic (or `/cmd_vel` if configured differently in your YAML). You can send velocity commands to make your robot move.

1.  **Open a new terminal and source your workspace.**

2.  **Publish a `Twist` message:**

    To make the robot move forward:

    ```bash
    ros2 topic pub /diff_drive_controller/cmd_vel_unstamped geometry_msgs/msg/Twist \
        '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' --once
    ```

    To make the robot turn:

    ```bash
    ros2 topic pub /diff_drive_controller/cmd_vel_unstamped geometry_msgs/msg/Twist \
        '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}' --once
    ```

    You can observe your robot moving in Gazebo. Use `ros2 topic echo /odom` to see the odometry messages published by the controller.

This chapter concludes the Robot Simulation module by demonstrating how to define a controlled robot in Gazebo using `ros2_control` and how to send commands to it via ROS2 topics. You now have the fundamental knowledge to set up sophisticated robot simulations and control them programmatically. In the next module, we will explore the AI-Robot Brain with NVIDIA Isaac.