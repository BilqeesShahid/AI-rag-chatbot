---
id: capstone-quiz-module4
title: "Quiz for Module 4: Capstone Project (Chapters 17-20)"
sidebar_position: 5
---

## Quiz for Module 4: Capstone Project (Chapters 17-20)

This quiz tests your understanding of the Capstone Project's design, architecture, simulation setup, and testing/evaluation strategies for an autonomous humanoid robot.

---

### Question 1: What is the primary objective of the "Fetch and Tidy" scenario in the Capstone Project?

a) To develop a robot that can play chess.

b) To design an autonomous humanoid robot capable of understanding natural language commands to perform household tasks like fetching and tidying objects.

c) To create a robot that only responds to predefined button presses.

d) To simulate robot movements in a purely virtual environment without AI.

**Answer**: b

---

### Question 2: In the Capstone Project's system architecture, which module is primarily responsible for interpreting natural language commands and decomposing them into simpler sub-goals?

a) Low-Level Control Module.

b) Perception Module.

c) High-Level Reasoning & Task Planning (VLA) Module.

d) Manipulation Module.

**Answer**: c

---

### Question 3: What role does USD (Universal Scene Description) play when setting up the simulated home environment using NVIDIA Isaac Sim?

a) It is a text-based programming language for robot control.

b) It provides a universal framework for describing 3D scenes, assets, and animations, foundational for building realistic simulation worlds.

c) It is a specific type of sensor used for depth perception.

d) It is a protocol for wireless communication between robots.

**Answer**: b

---

### Question 4: When integrating a humanoid robot model with ROS2 `ros2_control` for simulation, what crucial information must the robot's URDF/XACRO description include?

a) The robot's favorite color.

b) The `<ros2_control>` tag, specifying hardware interfaces for each joint and simulator plugins.

c) A list of all human operators.

d) The robot's preferred navigation algorithm.

**Answer**: b

---

### Question 5: Which of the following is considered a "System-Level (End-to-End) Testing" purpose for the Capstone Project?

a) Verifying the correctness of individual inverse kinematics functions.

b) Ensuring that different ROS2 nodes communicate correctly.

c) Evaluating the robot's performance on the full "Fetch and Tidy" scenarios in the simulated environment.

d) Rerunning previous tests after making changes.

**Answer**: c

---

### Question 6: What is the primary purpose of `robot_state_publisher` in a ROS2 Gazebo simulation setup, as mentioned in the Capstone project?

a) To publish sensor data from the simulated robot.

b) To publish the robot's URDF/XACRO to the `/robot_description` topic.

c) To manage the loading and unloading of Gazebo plugins.

d) To translate `cmd_vel` commands to joint efforts.

**Answer**: b

---

### Question 7: When evaluating the Capstone Project, which metric focuses on how well the robot handles unexpected variations in object placement, lighting, or minor environmental changes?

a) Task Success Rate.

b) Efficiency.

c) Robustness.

d) Accuracy.

**Answer**: c

---

### Question 8: Which of the following is NOT a suggested avenue for "Future Work and Extensions" for the Capstone Project?

a) Real-World Deployment.

b) Advanced VLA Models.

c) Basic PID control tuning (as this should be covered in core implementation).

d) Multi-Robot Collaboration.

**Answer**: c