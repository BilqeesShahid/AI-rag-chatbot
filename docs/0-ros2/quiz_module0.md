---
id: ros2-quiz-module0
title: "Quiz for Module 0: ROS2 Fundamentals (Chapters 1-4)"
sidebar_position: 5
slug: /ros2/quiz_module0
---

## Quiz for Module 0: ROS2 Fundamentals (Chapters 1-4)

This quiz is designed to test your understanding of the foundational concepts of ROS2, including its architecture, workspaces, basic node communication, and custom message types. Choose the best answer for each question.

---

### Question 1: What is the primary purpose of a ROS2 `Node`?

a) To manage global parameters for the entire robot system.

b) An independent process that performs a specific computation.

c) A communication channel for high-bandwidth sensor data.

d) A graphical visualization tool for debugging.

**Answer**: b

---

### Question 2: Which directory within a ROS2 workspace typically contains the source code of your ROS2 packages?

a) `build/`

b) `install/`

c) `log/`

d) `src/`

**Answer**: d

---

### Question 3: How do ROS2 `Publisher` and `Subscriber` nodes typically communicate with each other?

a) Through `Services` for request/reply interactions.

b) By directly calling functions in each other's code.

c) Via `Topics` by exchanging `Messages`.

d) Using `Actions` for long-running, feedback-enabled tasks.

**Answer**: c

---

### Question 4: What is the main advantage of defining **custom message types** in ROS2 (e.g., using a `.msg` file) compared to using only standard message types?

a) Custom messages are encrypted for secure communication.

b) They allow for more efficient use of network bandwidth by compressing data.

c) They enable the creation of structured, application-specific data types for complex information exchange.

d) Custom messages automatically provide built-in error checking and recovery mechanisms.

**Answer**: c

---

### Question 5: Which command is used to build a ROS2 workspace containing your packages?

a) `ros2 build`

b) `catkin_make`

c) `colcon build`

d) `ament build`

**Answer**: c

---

### Question 6: After creating a new ROS2 package or building your workspace, what essential step must you perform in each new terminal session to make your packages discoverable by ROS2?

a) Run `ros2 install_packages`.

b) Execute `sudo apt update`.

c) Source the workspace's `install/setup.bash` (or equivalent).

d) Restart your computer.

**Answer**: c

---

### Question 7: You want to check the active topics in your running ROS2 system. Which command would you use?

a) `ros2 node list`

b) `ros2 topic list`

c) `ros2 interface show`

d) `ros2 param list`

**Answer**: b

---

### Question 8: A `.msg` file for a custom message type contains:

a) Python or C++ source code for the message class.

b) XML definitions of the message structure and metadata.

c) Field type and field name pairs defining the message's data structure.

d) Launch instructions for publishing and subscribing to the message.

**Answer**: c