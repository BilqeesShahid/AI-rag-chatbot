---
id: ros2-chapter3
title: "Module-0.ros2-chapter-3"
sidebar_position: 3
slug: /ros2/chapter3
---

# Chapter 3: Writing Your First ROS2 Nodes (Publishers & Subscribers)

In ROS2, the fundamental building blocks of a robot application are **nodes**. Nodes are independent executable processes that perform a specific task, such as reading sensor data, controlling a motor, or processing an image. Nodes communicate with each other using **topics**, **services**, and **actions**.

In this chapter, we'll focus on the most common communication pattern: **topics**, using **publishers** and **subscribers**.

## Publishers: Sending Data on a Topic

A publisher node is responsible for sending (publishing) messages on a specific topic. Imagine a temperature sensor publishing temperature readings on a "/temperature" topic.

Let's create a simple Python publisher node within our `my_robot_pkg`.

1.  **Create a `my_robot_pkg/my_robot_pkg/` directory for our Python scripts:**

    ```bash
    mkdir -p ~/ros2_ws/src/my_robot_pkg/my_robot_pkg
    ```

    This nested directory structure is a Python convention for packages.

2.  **Create `my_robot_pkg/my_robot_pkg/simple_publisher.py`:**

    ```python
    # ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/simple_publisher.py

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class SimplePublisher(Node):

        def __init__(self):
            super().__init__('simple_publisher')
            self.publisher_ = self.create_publisher(String, 'chatter', 10)
            timer_period = 0.5  # seconds
            self.timer = self.create_timer(timer_period, self.timer_callback)
            self.i = 0

        def timer_callback(self):
            msg = String()
            msg.data = f'Hello ROS2! Count: {self.i}'
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: "{msg.data}"')
            self.i += 1


    def main(args=None):
        rclpy.init(args=args)

        simple_publisher = SimplePublisher()

        rclpy.spin(simple_publisher)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        simple_publisher.destroy_node()
        rclpy.shutdown()


    if __name__ == '__main__':
        main()
    ```

3.  **Update `setup.py` to make the node executable:**

    Edit `~/ros2_ws/src/my_robot_pkg/setup.py` and add the following inside the `setup()` function's `entry_points` dictionary. If `entry_points` doesn't exist, create it.

    ```python
    # ... inside setup()
    entry_points={
        'console_scripts': [
            'simple_publisher = my_robot_pkg.simple_publisher:main',
        ],
    },
    # ... rest of setup()
    ```

4.  **Build your package:**

    ```bash
    cd ~/ros2_ws
    colcon build --packages-select my_robot_pkg
    ```

5.  **Source your workspace:**

    ```bash
    source install/setup.bash
    ```

6.  **Run your publisher node:**

    Open a new terminal (and source your workspace in it):

    ```bash
    ros2 run my_robot_pkg simple_publisher
    ```

    You should see messages being published in your terminal.

## Subscribers: Receiving Data from a Topic

A subscriber node is responsible for listening to (subscribing to) a specific topic and receiving messages published on it. Following our example, another node might subscribe to the "/temperature" topic to display the current temperature.

1.  **Create `my_robot_pkg/my_robot_pkg/simple_subscriber.py`:**

    ```python
    # ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/simple_subscriber.py

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class SimpleSubscriber(Node):

        def __init__(self):
            super().__init__('simple_subscriber')
            self.subscription = self.create_subscription(
                String,
                'chatter',
                self.listener_callback,
                10)
            self.subscription  # prevent unused variable warning

        def listener_callback(self, msg):
            self.get_logger().info(f'I heard: "{msg.data}"')


    def main(args=None):
        rclpy.init(args=args)

        simple_subscriber = SimpleSubscriber()

        rclpy.spin(simple_subscriber)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        simple_subscriber.destroy_node()
        rclpy.shutdown()


    if __name__ == '__main__':
        main()
    ```

2.  **Update `setup.py` for the new subscriber node:**

    Add the new `simple_subscriber` entry point to the `console_scripts` list in `~/ros2_ws/src/my_robot_pkg/setup.py`:

    ```python
    # ... inside setup()
    entry_points={
        'console_scripts': [
            'simple_publisher = my_robot_pkg.simple_publisher:main',
            'simple_subscriber = my_robot_pkg.simple_subscriber:main', # Add this line
        ],
    },
    # ... rest of setup()
    ```

3.  **Build your package again:**

    ```bash
    cd ~/ros2_ws
    colcon build --packages-select my_robot_pkg
    ```

4.  **Source your workspace:**

    ```bash
    source install/setup.bash
    ```

5.  **Run your subscriber node:**

    Open another new terminal (and source your workspace in it). Run the subscriber node while the publisher is still running in the first terminal:

    ```bash
    ros2 run my_robot_pkg simple_subscriber
    ```

    You should now see the subscriber terminal receiving and printing the messages published by the `simple_publisher`.

## Inspecting ROS2 Topics

ROS2 provides command-line tools to inspect your running system:

*   **List active topics:**

    ```bash
    ros2 topic list
    ```

*   **Show topic information (type, publishers, subscribers):**

    ```bash
    ros2 topic info /chatter
    ```

*   **Echo messages on a topic (to see data directly):**

    ```bash
    ros2 topic echo /chatter
    ```

This chapter has walked you through creating your first ROS2 publisher and subscriber nodes, establishing basic communication between them. In the next chapter, we'll explore how to define custom message types for more complex data exchange.

```mermaid
graph TD
    Publisher[Publisher Node] -- Publishes messages on /chatter topic --> Topic[/chatter Topic (std_msgs/msg/String)]
    Topic --> Subscriber[Subscriber Node]
```

### Exercise: Implementing and Inspecting ROS2 Publisher/Subscriber

**Goal:** Reinforce your understanding of ROS2 publisher and subscriber nodes by implementing them and using ROS2 command-line tools to inspect their communication.

**Steps:**

1.  **Create Publisher Node:**
    *   Navigate to your `~/ros2_ws/src/my_robot_pkg/my_robot_pkg/` directory.
    *   Create a Python file named `custom_publisher.py`.
    *   Implement a ROS2 publisher node that publishes `std_msgs.msg.String` messages on a topic named `/my_custom_topic` at a rate of 1 Hz. The messages should contain a simple counter.

2.  **Create Subscriber Node:**
    *   In the same directory, create a Python file named `custom_subscriber.py`.
    *   Implement a ROS2 subscriber node that subscribes to `/my_custom_topic` and prints the received messages to the console.

3.  **Update `setup.py`:**
    *   Modify `~/ros2_ws/src/my_robot_pkg/setup.py` to include entry points for both `custom_publisher` and `custom_subscriber`.

4.  **Build and Source:**
    *   Navigate to `~/ros2_ws` and build your `my_robot_pkg` using `colcon build --packages-select my_robot_pkg`.
    *   Source your workspace: `source install/setup.bash`.

5.  **Run and Verify:**
    *   Open two separate terminals. In the first terminal, run your publisher: `ros2 run my_robot_pkg custom_publisher`.
    *   In the second terminal, run your subscriber: `ros2 run my_robot_pkg custom_subscriber`.
    *   Verify that the subscriber is receiving and printing messages from the publisher.

6.  **Inspect with ROS2 Tools:**
    *   Open a third terminal (and source your workspace).
    *   Use `ros2 topic list` to see if `/my_custom_topic` is active.
    *   Use `ros2 topic info /my_custom_topic` to inspect its type, publishers, and subscribers.
    *   Use `ros2 topic echo /my_custom_topic` to directly view the messages being published.

**Expected Outcome:**
You should see messages flowing from your custom publisher to your custom subscriber, and you should be able to inspect this communication using the ROS2 command-line tools.