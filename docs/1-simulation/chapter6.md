---
id: simulation-chapter6
title: Module-1.simulation-chapter-6
sidebar_position: 2
slug: /simulation/chapter6
---

# Chapter 6: Gazebo for Robot Simulation

Gazebo is one of the most powerful and widely used 3D robot simulators in the robotics community. It offers a robust physics engine, high-quality graphics, and convenient interfaces for integrating with robotics middleware like ROS/ROS2. Gazebo allows you to accurately simulate complex robot mechanisms, sensor data, and environmental interactions.

## Key Features of Gazebo

*   **Physics Engine**: Gazebo integrates with advanced physics engines (like ODE, Bullet, Simbody, DART) to provide realistic rigid-body dynamics, gravity, friction, and collision detection.
*   **3D Visualization**: It includes a sophisticated 3D rendering engine (OGRE) for visualizing robots, environments, and sensor feedback.
*   **Sensor Simulation**: Gazebo can accurately simulate a wide range of sensors, including cameras (RGB, depth, monochrome), LiDAR, IMUs, force/torque sensors, and more.
*   **Robot and World Models**: Robots and environments are described using SDF (Simulation Description Format) files, which can define links, joints, sensors, materials, and plugins.
*   **Plugins**: Gazebo's functionality is highly extensible through plugins, allowing users to customize physics, sensors, control, and user interfaces.
*   **ROS/ROS2 Integration**: A key strength of Gazebo is its tight integration with ROS/ROS2, enabling seamless communication between simulated robots and ROS2 nodes.

## Installing Gazebo (with ROS2 Humble)

If you've followed Chapter 1 and installed ROS2 Humble on Ubuntu 22.04, Gazebo Garden (the version typically integrated with Humble) should be relatively easy to set up. Ensure your ROS2 environment is sourced.

1.  **Install Gazebo and ROS2 Gazebo packages:**

    ```bash
    sudo apt update
    sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
    ```

    *   `ros-humble-gazebo-ros-pkgs`: Provides the bridge between ROS2 and Gazebo, allowing ROS2 nodes to interact with simulated robots and sensors.
    *   `ros-humble-gazebo-ros2-control`: Enables the use of `ros2_control` (a hardware abstraction layer) with Gazebo, which is crucial for controlling robot joints.

2.  **Verify Gazebo installation:**

    You can launch Gazebo's GUI directly to see if it's working:

    ```bash
    gazebo
    ```

    This should open the Gazebo simulator with an empty world. Close it to proceed.

## Creating a Simple Gazebo World

Gazebo worlds are defined in `.world` files using SDF (Simulation Description Format). Let's create a very basic world.

1.  **Create a `worlds` directory in your `my_robot_pkg`:**

    ```bash
    mkdir -p ~/ros2_ws/src/my_robot_pkg/worlds
    ```

2.  **Create `~/ros2_ws/src/my_robot_pkg/worlds/empty.world`:**

    ```xml
    <?xml version="1.0" ?>
    <sdf version="1.6">
      <world name="empty_world">
        <gravity>0 0 -9.8</gravity>
        <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
        <gui>
          <camera name="user_camera">
            <pose>0 0 5 0 0.9 0</pose>
          </camera>
        </gui>
        <light type="directional" name="sun">
          <cast_shadows>1</cast_shadows>
          <pose>0 0 10 0 -0 0</pose>
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
          <attenuation>
            <range>1000</range>
            <constant>0.9</constant>
            <linear>0.01</linear>
            <quadratic>0.001</quadratic>
          </attenuation>
          <direction>-0.5 0.1 -0.9</direction>
        </light>
        <model name="ground_plane">
          <static>true</static>
          <link name="link">
            <collision name="collision">
              <geometry>
                <plane>
                  <normal>0 0 1</normal>
                  <size>100 100</size>
                </plane>
              </geometry>
              <surface>
                <friction>
                  <ode>
                    <mu>1.0</mu>
                    <mu2>1.0</mu2>
                  </ode>
                </friction>
              </surface>
            </collision>
            <visual name="visual">
              <geometry>
                <plane>
                  <normal>0 0 1</normal>
                  <size>100 100</size>
                </plane>
              </geometry>
              <material>
                <ambient>0.8 0.8 0.8 1</ambient>
                <diffuse>0.8 0.8 0.8 1</diffuse>
                <specular>0.8 0.8 0.8 1</specular>
              </material>
            </visual>
          </link>
        </model>
      </world>
    </sdf>
    ```

3.  **Launch Gazebo with your custom world:**

    ```bash
    gazebo --verbose ~/ros2_ws/src/my_robot_pkg/worlds/empty.world
    ```

    You should now see Gazebo launched with a flat ground plane and a directional light.

## Integrating with ROS2

The real power of Gazebo comes from its integration with ROS2. We use Gazebo plugins to expose simulated robot components and sensor data as ROS2 topics, services, and actions.

### Using `ros2_launch` to Start Gazebo

It's common to use `ros2_launch` files to start Gazebo and load robots. This allows for a more structured and reproducible setup.

1.  **Create a `launch` directory in your `my_robot_pkg`:**

    ```bash
    mkdir -p ~/ros2_ws/src/my_robot_pkg/launch
    ```

2.  **Create `~/ros2_ws/src/my_robot_pkg/launch/gazebo_launch.py`:**

    ```python
    import os

    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource


    def generate_launch_description():
        # Get the launch directory
        gazebo_ros_package_dir = get_package_share_directory('gazebo_ros')

        # Launch Gazebo
        gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_package_dir, 'launch', 'gazebo.launch.py')),
            launch_arguments={'world': os.path.join(get_package_share_directory('my_robot_pkg'), 'worlds', 'empty.world')}.items(),
        )

        return LaunchDescription([
            gazebo_launch
        ])
    ```

3.  **Update `setup.py` to install the launch file:**

    Add the following to the `data_files` list in `~/ros2_ws/src/my_robot_pkg/setup.py`. If `data_files` doesn't exist, create it.

    ```python
    # ... inside setup()
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add these lines:
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'worlds'), glob(os.path.join('worlds', '*.world'))),
    ],
    # ... rest of setup()
    ```
    *Don't forget to add `import os` and `from glob import glob` at the top of `setup.py`.*

4.  **Build and source your package:**

    ```bash
    cd ~/ros2_ws
    colcon build --packages-select my_robot_pkg
    source install/setup.bash
    ```

5.  **Launch Gazebo using `ros2_launch`:**

    ```bash
    ros2 launch my_robot_pkg gazebo_launch.py
    ```

    This will now launch Gazebo with your `empty.world` using the ROS2 launch system.

In the next chapter, we'll learn how to define robots using URDF/XACRO and spawn them into our Gazebo world.