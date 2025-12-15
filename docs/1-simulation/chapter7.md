---
id: simulation-chapter7
title: Module-1.simulation-chapter-7
sidebar_position: 3
slug: /simulation/chapter7
---

# Chapter 7: Defining and Spawning Robots in Gazebo

To simulate a robot in Gazebo, you first need a detailed description of the robot itself. In the ROS/ROS2 ecosystem, robots are commonly described using **URDF** (Unified Robot Description Format) or **XACRO** (XML Macros for ROS). These formats allow you to define the robot's kinematics (links and joints), dynamics (mass, inertia), visuals (3D models, colors), and collision properties.

## Understanding URDF and XACRO

*   **URDF (Unified Robot Description Format)**:
    *   An XML format for representing robot models.
    *   Defines rigid bodies (links) and their connections (joints).
    *   Can include visual, collision, and inertial properties.
    *   Static and generally less flexible for complex, modular robots.

*   **XACRO (XML Macros for ROS)**:
    *   An XML macro language that extends URDF.
    *   Allows for parameterized and modular robot descriptions.
    *   Reduces repetition and makes robot models easier to manage and modify.
    *   XACRO files are processed to generate a URDF file.

For complex robots, XACRO is almost always preferred over raw URDF due to its modularity and readability.

## Creating a Simple Robot Model (XACRO)

Let's create a simple differential drive robot model using XACRO. This robot will have a base link and two wheel links.

1.  **Create a `urdf` directory in your `my_robot_pkg`:**

    ```bash
    mkdir -p ~/ros2_ws/src/my_robot_pkg/urdf
    ```

2.  **Create `~/ros2_ws/src/my_robot_pkg/urdf/my_robot.xacro`:**

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

    </robot>
    ```

## Spawning the Robot in Gazebo

To get our robot into the Gazebo simulation, we typically use a ROS2 node called `robot_state_publisher` to publish the robot's URDF to the `/robot_description` topic, and then a Gazebo plugin or a specific ROS2 package to spawn the model.

We'll modify our `gazebo_launch.py` to achieve this.

1.  **Update `setup.py` to install URDF files:**

    Add the `urdf` directory to the `data_files` list in `~/ros2_ws/src/my_robot_pkg/setup.py`:

    ```python
    # ... inside setup()
    data_files=[
        ...
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*.xacro'))),
    ],
    # ... rest of setup()
    ```

2.  **Modify `~/ros2_ws/src/my_robot_pkg/launch/gazebo_launch.py`:**

    We need to:
    *   Process the XACRO file into a URDF string.
    *   Launch `robot_state_publisher` to broadcast the robot's state.
    *   Launch a `spawn_entity.py` node from `ros_gz_sim` to inject the robot into Gazebo.

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

        # Get package directory
        pkg_share_dir = get_package_share_directory(pkg_name)
        world_path = os.path.join(pkg_share_dir, 'worlds', world_file_name)
        robot_path = os.path.join(pkg_share_dir, 'urdf', robot_file_name)

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
                       '-z', '0.1'], # Lift slightly off ground
            output='screen',
        )

        return LaunchDescription([
            gazebo_launch,
            robot_state_publisher_node,
            spawn_entity
        ])
    ```

    *Don't forget to add `from launch_ros.actions import Node` and `from launch.substitutions import Command` at the top of the file.*

3.  **Ensure `xacro` and `ros_gz_sim` are installed:**

    ```bash
    sudo apt install ros-humble-xacro ros-humble-ros-gz-sim
    ```

4.  **Build and source your package:**

    ```bash
    cd ~/ros2_ws
    colcon build --packages-select my_robot_pkg
    source install/setup.bash
    ```

5.  **Launch Gazebo with your robot:**

    ```bash
    ros2 launch my_robot_pkg gazebo_launch.py
    ```

    You should now see your simple differential drive robot spawned in the Gazebo world!

This chapter provided the foundation for bringing your robot designs into a Gazebo simulation. You've learned how to define robot kinematics and visuals using XACRO and how to launch and spawn these models using ROS2 launch files. In the next chapter, we will explore how to control the simulated robot.