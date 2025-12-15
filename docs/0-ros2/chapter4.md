---
id: ros2-chapter4
title: "Module-0.ros2-chapter-4"
sidebar_position: 4
slug: /ros2/chapter4
---

# Chapter 4: Defining Custom ROS2 Message Types

While `std_msgs` provides basic data types like `String` and `Int32`, real-world robotics applications often require more complex and application-specific data structures. ROS2 allows you to define your own custom message types, services, and actions using simple `.msg`, `.srv`, and `.action` files.

In this chapter, we'll focus on defining custom **message types** to convey structured data between nodes.

## Why Custom Messages?

Consider a robot that needs to report its current position (x, y, z coordinates) and orientation (roll, pitch, yaw angles). While you could send six separate `Float32` messages, packaging them into a single `RobotPose` message is far more efficient, readable, and less prone to synchronization issues. It clearly defines a single, coherent piece of information.

## Defining a Custom Message

Custom messages are defined in `.msg` files, which are plain text files containing field type and field name pairs.

### Message Definition Syntax

A `.msg` file uses a simple syntax:

```
field_type1 field_name1
field_type2 field_name2
...
```

*   **`field_type`**: Can be a built-in ROS2 primitive type (e.g., `int8`, `float32`, `string`, `bool`, `time`, `duration`), another custom message type, or an array of these types.
*   **`field_name`**: The name of the field, which should be descriptive.

### Example: `my_robot_pkg/msg/RobotPose.msg`

Let's create a `RobotPose` message to store a robot's 3D position and orientation.

1.  **Create a `msg` directory inside your package:**

    ```bash
    mkdir -p ~/ros2_ws/src/my_robot_pkg/msg
    ```

2.  **Create `~/ros2_ws/src/my_robot_pkg/msg/RobotPose.msg`:**

    ```
    # RobotPose.msg
    float32 x
    float32 y
    float32 z

    float32 roll
    float32 pitch
    float32 yaw
    ```

## Building and Using Custom Messages

For ROS2 to recognize and generate code for your custom messages, you need to configure your package's build system.

1.  **Modify `package.xml`:**

    Add the following build and execution dependencies to `~/ros2_ws/src/my_robot_pkg/package.xml`:

    ```xml
    <!-- ... inside <package> tag ... -->

    <buildtool_depend>ament_cmake</buildtool_depend>
    <buildtool_depend>ament_cmake_python</buildtool_depend>
    <build_depend>rosidl_default_generators</build_depend>
    <exec_depend>rosidl_default_runtime</exec_depend>
    <member_of_group>rosidl_interface_packages</member_of_of_group>

    <!-- ... rest of package.xml ... -->
    ```
    *Note: `ament_cmake` and `ament_cmake_python` are build system dependencies. `rosidl_default_generators` is needed to generate code for your messages, and `rosidl_default_runtime` is needed to use them at runtime.* `member_of_group` helps with discoverability.

2.  **Modify `CMakeLists.txt`:**

    Open `~/ros2_ws/src/my_robot_pkg/CMakeLists.txt` and add these lines:

    *   Find the `rosidl_default_generators` package:
        ```cmake
        # ... inside find_package(ament_cmake REQUIRED)
        find_package(rosidl_default_generators REQUIRED)
        # ...
        ```
    *   Add your message files:
        ```cmake
        # ... after ament_export_dependencies()
        rosidl_generate_interfaces(${PROJECT_NAME}
        "msg/RobotPose.msg"
        # "srv/MyService.srv" # Uncomment and add for services
        # "action/MyAction.action" # Uncomment and add for actions
        )
        # ...
        ```
    *   Export dependencies (if using `ament_cmake_python`):
        ```cmake
        # ... after rosidl_generate_interfaces
        ament_export_dependencies(rosidl_default_runtime)
        # ...
        ```

3.  **Build your package:**

    ```bash
    cd ~/ros2_ws
    colcon build --packages-select my_robot_pkg
    ```

4.  **Source your workspace:**

    ```bash
    source install/setup.bash
    ```

## Using Custom Messages in Nodes

Now that the custom message `RobotPose` is built, we can use it in our publisher and subscriber nodes.

### Publisher using `RobotPose`

Let's modify `my_robot_pkg/my_robot_pkg/simple_publisher.py` to publish `RobotPose` messages instead of `String`.

```python
# ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/custom_pose_publisher.py

import rclpy
from rclpy.node import Node
from my_robot_pkg.msg import RobotPose  # Import our custom message

class CustomPosePublisher(Node):

    def __init__(self):
        super().__init__('custom_pose_publisher')
        self.publisher_ = self.create_publisher(RobotPose, 'robot_pose', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def timer_callback(self):
        msg = RobotPose() # Create an instance of our custom message
        msg.x = self.x
        msg.y = self.y
        msg.z = self.z
        msg.roll = self.roll
        msg.pitch = self.pitch
        msg.yaw = self.yaw

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing RobotPose: x={msg.x:.2f}, y={msg.y:.2f}, yaw={msg.yaw:.2f}')

        # Simulate movement
        self.x += 0.1
        self.y += 0.05
        self.yaw += 0.01

def main(args=None):
    rclpy.init(args=args)
    custom_pose_publisher = CustomPosePublisher()
    rclpy.spin(custom_pose_publisher)
    custom_pose_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber using `RobotPose`

Similarly, modify `my_robot_pkg/my_robot_pkg/simple_subscriber.py` to subscribe to `RobotPose` messages.

```python
# ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/custom_pose_subscriber.py

import rclpy
from rclpy.node import Node
from my_robot_pkg.msg import RobotPose  # Import our custom message

class CustomPoseSubscriber(Node):

    def __init__(self):
        super().__init__('custom_pose_subscriber')
        self.subscription = self.create_subscription(
            RobotPose,  # Subscribe to our custom message type
            'robot_pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I received RobotPose: x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}, ' +
                               f'roll={msg.roll:.2f}, pitch={msg.pitch:.2f}, yaw={msg.yaw:.2f}')

def main(args=None):
    rclpy.init(args=args)
    custom_pose_subscriber = CustomPoseSubscriber()
    rclpy.spin(custom_pose_subscriber)
    custom_pose_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Update `setup.py` for new nodes

Remember to update `~/ros2_ws/src/my_robot_pkg/setup.py` to include the new publisher and subscriber nodes:

```python
# ... inside setup()
entry_points={
    'console_scripts': [
        'simple_publisher = my_robot_pkg.simple_publisher:main',
        'simple_subscriber = my_robot_pkg.simple_subscriber:main',
        'custom_pose_publisher = my_robot_pkg.custom_pose_publisher:main', # Add this line
        'custom_pose_subscriber = my_robot_pkg.custom_pose_subscriber:main', # Add this line
    ],
},
# ... rest of setup()
```

This chapter has guided you through creating and using custom ROS2 message types, which is essential for building more sophisticated robot applications. You now have the tools to define structured data and enable nodes to communicate complex information effectively. This concludes the ROS2 Fundamentals module. In the next module, we will explore robot simulation.

```mermaid
graph TD
    A[Define .msg file (e.g., RobotPose.msg)] --> B{Update package.xml}
    B --> C{Update CMakeLists.txt}
    C --> D[colcon build]
    D --> E[Source install/setup.bash]
    E --> F[Import in Python/C++ Node]
    F --> G[Publish/Subscribe Custom Message]
```

### Exercise: Defining and Using a Custom Sensor Data Message

**Goal:** Create a custom ROS2 message to represent more complex sensor data (e.g., IMU readings) and integrate it into a publisher-subscriber pair.

**Steps:**

1.  **Define Custom Message `SensorData.msg`:**
    *   Navigate to `~/ros2_ws/src/my_robot_pkg/msg`.
    *   Create a file named `SensorData.msg` with the following content:
        ```
        # SensorData.msg
        std_msgs/Header header
        float32 linear_acceleration_x
        float32 linear_acceleration_y
        float32 linear_acceleration_z
        float32 angular_velocity_x
        float32 angular_velocity_y
        float32 angular_velocity_z
        ```
        *(Note: `std_msgs/Header` is a commonly used built-in message type for timestamping and frame information.)*

2.  **Update Build Configuration:**
    *   Ensure `package.xml` has the necessary build and execution dependencies (from this chapter).
    *   Modify `~/ros2_ws/src/my_robot_pkg/CMakeLists.txt` to include `msg/SensorData.msg` in the `rosidl_generate_interfaces` call. Make sure `std_msgs` is also found if you haven't already: `find_package(std_msgs REQUIRED)`. If you added `std_msgs/Header`, you'll need to add `<depend>std_msgs</depend>` to `package.xml` and `ament_target_dependencies(${PROJECT_NAME} std_msgs)` to `CMakeLists.txt`.

3.  **Build and Source:**
    *   Navigate to `~/ros2_ws` and build your `my_robot_pkg` using `colcon build --packages-select my_robot_pkg`.
    *   Source your workspace: `source install/setup.bash`.

4.  **Create Custom Sensor Publisher:**
    *   In `~/ros2_ws/src/my_robot_pkg/my_robot_pkg/`, create `sensor_publisher.py`.
    *   Implement a ROS2 node that publishes `my_robot_pkg/msg/SensorData` messages on a topic named `/imu_data` at 2 Hz. Populate the fields with simulated values (e.g., incrementing numbers).

5.  **Create Custom Sensor Subscriber:**
    *   In the same directory, create `sensor_subscriber.py`.
    *   Implement a ROS2 node that subscribes to `/imu_data` and prints the `linear_acceleration_x` and `angular_velocity_z` values.

6.  **Update `setup.py`:**
    *   Add entry points for `sensor_publisher` and `sensor_subscriber` to `~/ros2_ws/src/my_robot_pkg/setup.py`.

7.  **Run and Verify:**
    *   Open two terminals, source your workspace in each.
    *   Run the publisher: `ros2 run my_robot_pkg sensor_publisher`.
    *   Run the subscriber: `ros2 run my_robot_pkg sensor_subscriber`.
    *   Verify the subscriber receives and prints the custom sensor data.
    *   Use `ros2 topic echo /imu_data` in a third terminal to observe the raw message data.

**Expected Outcome:**
You will have successfully created, built, and used a custom ROS2 message type to exchange complex sensor information between nodes. You should see the subscriber printing the relevant data and `ros2 topic echo` displaying the full `SensorData` messages.