---
id: isaac-chapter10
title: Module-2.isaac-chapter-10
sidebar_position: 2
slug: /isaac/chapter10
---

# Chapter 10: Setting Up Isaac Sim and Omniverse

To begin developing with NVIDIA Isaac, the first crucial step is to set up your development environment, primarily involving NVIDIA Omniverse and Isaac Sim. Omniverse acts as the foundational platform, while Isaac Sim is the specialized application built on top of it for robotics simulation.

## System Requirements

Before installation, ensure your system meets the minimum requirements:

*   **Operating System**: Ubuntu 20.04 or 22.04 LTS, or Windows 10/11.
*   **GPU**: NVIDIA RTX series GPU (e.g., RTX 2060, 3060, 4060 or newer) with up-to-date NVIDIA drivers. Isaac Sim relies heavily on GPU acceleration.
*   **CPU**: Intel Core i7 / AMD Ryzen 7 (or newer) equivalent.
*   **RAM**: 32 GB or more.
*   **Storage**: 100 GB or more free SSD space for Omniverse Launcher, Cache, and Isaac Sim assets.
*   **Internet Connection**: Required for downloading and updates.

## Installing NVIDIA Omniverse Launcher

NVIDIA Omniverse is managed through the Omniverse Launcher, which helps you install and manage Omniverse applications, connectors, and extensions.

1.  **Download Omniverse Launcher**: Visit the [NVIDIA Omniverse website](https://www.nvidia.com/omniverse/) and download the Omniverse Launcher for your operating system.
2.  **Install Launcher**: Run the installer and follow the on-screen instructions. You might need to create an NVIDIA account if you don't have one.
3.  **Launch Omniverse Launcher**: Once installed, open the Omniverse Launcher.

## Installing Isaac Sim via Omniverse Launcher

After setting up the Omniverse Launcher, you can proceed to install Isaac Sim.

1.  **Navigate to the `Exchange` tab in the Omniverse Launcher.**
2.  **Search for `Isaac Sim`**: Use the search bar to find Isaac Sim. You'll see an entry for "Isaac Sim".
3.  **Install Isaac Sim**: Click on the Isaac Sim application and then click the `Install` button. The Launcher will download and install all necessary components.
4.  **Wait for Installation**: The installation process can take some time, as Isaac Sim is a large application with many assets. Ensure you have a stable internet connection.

## Essential Omniverse Extensions and Connectors

To maximize your Isaac Sim experience, ensure you have the following installed/updated via the Omniverse Launcher (under the `Library` -> `Extensions` or `Connectors` tabs):

*   **Cache**: Essential for efficient data streaming and asset management. Ensure it's running and has sufficient disk space.
*   **Code**: Provides a development environment for Omniverse-based applications.
*   **VS Code Extension**: If you use VS Code, install the Omniverse VS Code extension for better integration.
*   **USD Composer (Optional)**: A powerful application for assembling, lighting, simulating, and rendering USD scenes. Useful for advanced scene creation.

## Launching Isaac Sim

Once installed, you can launch Isaac Sim directly from the Omniverse Launcher:

1.  **Navigate to the `Library` -> `Apps` tab in the Omniverse Launcher.**
2.  **Launch Isaac Sim**: Click the `Launch` button next to Isaac Sim.

Isaac Sim will open, presenting you with a welcome screen or a default empty stage. You are now ready to start creating and simulating robotic applications.

## Verifying Your Installation

To ensure everything is working correctly:

*   **Load a Sample Scene**: In Isaac Sim, go to `File -> Open` and browse through the sample projects provided (e.g., under `omni.isaac.sim.demos/demos/`). Try loading a simple robot scene or an environment.
*   **Check Console Output**: Monitor the Isaac Sim console for any error messages during startup or scene loading.
*   **Check Drivers**: Ensure your NVIDIA GPU drivers are up-to-date, as outdated drivers are a common source of issues.

This chapter has covered the foundational steps for setting up your NVIDIA Isaac development environment. With Isaac Sim and Omniverse correctly installed, you are now prepared to dive into creating robot assets and developing AI-powered robotics applications. In the next chapter, we will explore how to work with USD and create custom robot assets within Isaac Sim.