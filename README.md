# DRL-PPo-for-Autonomous-Drone-Navigation
Unstructured Environment

# PPO-Based Autonomous Drone Navigation

This project implements **Autonomous Drone Navigation** using **Proximal Policy Optimization (PPO)** in a simulation environment built with **ROS Noetic**, **Gazebo11**, and **PX4-Autopilot**.

The drone must fly through a narrow corridor (`corridor_static.world`) containing obstacles, starting from the origin and navigating towards a goal location at **(23, 0, 0)** using **LiDAR-based perception** and **reinforcement learning**.

---

## 🧠 Core Features

- **Deep Reinforcement Learning** with PPO for autonomous navigation.
- **Gazebo-based simulation** with realistic physics and 3D obstacles.
- **LiDAR-based collision avoidance** using `/scan` topic.
- Compatible with **PX4 OFFBOARD mode** and MAVROS APIs.
- Includes training visualization via **TensorBoard**.

---

## 📂 Project Structure

```
drone_ppo_nav/
├── launch/
│   └── simulation.launch             # Launches PX4 SITL + Gazebo + world + drone
├── models/
│   └── iris.sdf                      # Modified Iris drone with LiDAR
├── scripts/
│   ├── drone_env.py                  # Gym environment (PPO-compatible)
│   └── train_ppo.py                  # Training loop using Stable-Baselines3
├── worlds/
│   └── corridor_static.world         # Custom Gazebo world with obstacles
├── ppo_tensorboard/                  # TensorBoard logs
└── README.md
```

---

## 🚀 How to Run

### 1. Start Simulation

```bash
roslaunch drone_ppo_nav simulation.launch
```

### 2. Start MAVROS

```bash
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
```

### 3. Start Training

```bash
rosrun drone_ppo_nav train_ppo.py
```

---

## 🛠 Dependencies

- **Ubuntu 20.04 (WSL or native)**
- **ROS Noetic**
- **PX4-Autopilot**
- **Gazebo 11**
- **MAVROS**
- **Stable-Baselines3**
- **Gym / Gymnasium**
- **TensorFlow** (for PPO)
- **TensorBoard** (for training logs)

---

## 📈 Training Output

Training results (reward, goal distance, success count) are logged to TensorBoard:

```bash
tensorboard --logdir=ppo_tensorboard
```

---

## 📍 Goal

Enable a drone to autonomously fly in unknown, cluttered environments with only onboard LiDAR, using learned PPO policies.

---

## 🤖 Acknowledgments

This project integrates multiple robotics and AI frameworks. Inspired by work in autonomous navigation, drone reinforcement learning, and PX4 open-source simulation.

