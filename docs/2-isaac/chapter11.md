---
id: isaac-chapter11
title: Module-2.isaac-chapter-11
sidebar_position: 3
slug: /isaac/chapter11
---

# Chapter 11: Working with USD and Robot Assets in Isaac Sim

NVIDIA Isaac Sim leverages **USD (Universal Scene Description)** as its core scene representation format. Understanding USD is fundamental to creating, modifying, and integrating robot assets and environments within Isaac Sim. USD is a powerful, extensible, open-source 3D scene description technology developed by Pixar, designed for collaborative workflows and physically accurate simulation.

## Understanding Universal Scene Description (USD)

USD is more than just a file format; it's an ecosystem for describing 3D scenes. Key concepts include:

*   **Layers**: USD scenes are composed of layers that can override each other, allowing for non-destructive editing and collaboration. You can add details without modifying the original asset.
*   **Composing Prims**: Scenes are built from "Prims" (primitives), which represent objects, lights, cameras, materials, or even other USD files. Prims can be composed together to form complex hierarchies.
*   **References and Variants**: USD supports referencing other USD files (e.g., a robot model referenced into a world scene) and defining "variants" for different configurations of an asset (e.g., a robot with different end-effectors).
*   **Strong Typing**: Prims have a strong type system, defining what properties they can have (e.g., `Sphere`, `Cube`, `Xform`).

Isaac Sim extends USD with specific schemas for robotics, such as rigid body physics, joints, and sensors.

## Importing Existing Robot Models

Isaac Sim supports importing various 3D model formats, which are then converted to USD. Common formats include:

*   **URDF/XACRO**: The standard for ROS/ROS2 robot descriptions. Isaac Sim has built-in tools to convert these into USD for simulation.
*   **OBJ, FBX, STEP, STL**: General 3D model formats. These are typically imported as static meshes.

### Importing URDF into Isaac Sim

1.  **Launch Isaac Sim.**
2.  **Access the URDF Importer**: In Isaac Sim, go to `Create -> Robotics -> URDF Importer`.
3.  **Load URDF File**: Browse to your URDF/XACRO file (e.g., `~/ros2_ws/src/my_robot_pkg/urdf/my_robot.xacro`).
4.  **Configure Import Options**: The importer provides options for physics materials, joint drive parameters, and more. For now, use defaults.
5.  **Import**: Click `Import`. Your robot model will appear in the Isaac Sim stage.

## Creating Custom Robot Assets in USD

For more complex or native Isaac Sim robots, you might want to build them directly in USD or modify imported assets. This typically involves using the "Stage" and "Property" windows in Isaac Sim or Omniverse USD Composer.

### Basic Steps for USD Asset Creation:

1.  **Create a New Stage**: `File -> New` in Isaac Sim.
2.  **Add Primitives**: Use `Create -> Primitives` to add basic shapes (Cube, Sphere, Cylinder) to represent robot links.
3.  **Transform Prims**: Use the `Transform` tools (move, rotate, scale) to position and size your links in relation to each other.
4.  **Define Joints**: To connect links, you need to add joints. Select two links, then go to `Create -> Physics -> Joint` and choose a joint type (e.g., `Revolute Joint`, `Prismatic Joint`). Position the joint correctly between the links.
5.  **Add Physics Components**: For realistic simulation, add physics properties to your links. Select a link, then go to `Add -> Physics -> Rigid Body` (for dynamic links) or `Add -> Physics -> Colliders` (for collision detection).
6.  **Materials**: Assign materials for visual appearance. Drag and drop materials from the `Content` browser onto your links.
7.  **Save as USD**: Save your robot model as a `.usd` or `.usda` (ASCII USD) file. This allows you to easily reference it into other worlds.

### Example: Manually Creating a Simple Link and Joint

Imagine creating a base link and a rotating arm:

1.  **Create a `Cube`** (for base_link). Name it `/World/MyRobot/BaseLink`.
2.  **Create another `Cube`** (for arm_link). Name it `/World/MyRobot/ArmLink`.
3.  **Add `Rigid Body` physics to both `BaseLink` and `ArmLink`** (`Add -> Physics -> Rigid Body`).
4.  **Add a `Revolute Joint`**: Select `/World/MyRobot/BaseLink`, then `Create -> Physics -> Revolute Joint`. Set its `Parent Prim` to `/World/MyRobot/BaseLink` and `Child Prim` to `/World/MyRobot/ArmLink`. Position the joint's `Local Pos` and `Axis` to define the pivot point.

## Adding Sensors to Your Robot

Isaac Sim provides various sensor components that can be attached to your robot's links to simulate real-world sensor data.

*   **Camera**: `Create -> Sensor -> Camera`. Attach it to a link and configure its properties (resolution, FOV, update rate).
*   **LiDAR**: `Create -> Sensor -> LiDAR`. Configure scan parameters, range, and update rate.
*   **IMU**: `Create -> Sensor -> IMU`. Simulates inertial measurement data.

These simulated sensors can then publish data to ROS2 topics using Isaac ROS, which we will cover in the next chapter.

This chapter has equipped you with the knowledge to work with USD, import existing robot models, and begin building custom robot assets within Isaac Sim. Mastering USD is key to unlocking the full potential of the NVIDIA Isaac platform for your robotics projects. In the next chapter, we will delve into Isaac ROS and building AI perception pipelines.