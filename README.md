# 🛸 Drone Navigation using Deep Reinforcement Learning (PPO)

This project implements **autonomous drone navigation** in a simulated environment using **Deep Reinforcement Learning** with the **Proximal Policy Optimization (PPO)** algorithm. The drone learns to fly through a corridor filled with obstacles and reach a target point while avoiding collisions — all in simulation using **PX4**, **ROS Noetic**, and **Gazebo 11**.

---

## 📁 Package Name

```bash
drone_ppo_nav
```

---

## 📍 Simulation Overview

- **Simulator**: Gazebo 11  
- **Flight Controller**: PX4-Autopilot (SITL)  
- **Middleware**: MAVROS (ROS Noetic)  
- **DRL Algorithm**: Stable-Baselines3 PPO  
- **Sensor**: 2D 360° LiDAR (`/scan` topic)  
- **Goal**: Navigate from (0, 0, 0) → (23, 0, 0) through a corridor with obstacles

---

## 🧠 Folder Structure

```
drone_ppo_nav/
├── launch/
│   └── simulation.launch            # Launch PX4 SITL + Gazebo + world
├── models/
│   └── iris.sdf                     # Drone SDF model with LiDAR
├── worlds/
│   └── corridor_static.world       # Gazebo world with corridor and obstacles
├── scripts/
│   ├── drone_env.py                 # Gym-compatible environment
│   └── train_ppo.py                 # PPO training script
```

---

## ⚙️ Dependencies

Make sure the following are installed:

- [PX4-Autopilot](https://github.com/PX4/PX4-Autopilot)
- ROS Noetic
- Gazebo 11
- `mavros` & `mavros_extras`
- `stable-baselines3`
- `gym` or `gymnasium`
- `rospy`, `numpy`, `sensor_msgs`, `geometry_msgs`

Install Python packages (inside virtualenv recommended):

```bash
pip install stable-baselines3[extra] torch tensorboard
```

---

## 🚀 How to Launch the Simulation

### 1. Start PX4 + Gazebo + Drone

```bash
roslaunch drone_ppo_nav simulation.launch
```

### 2. Start MAVROS (in a new terminal)

```bash
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
```

---

## 🧠 How to Train the PPO Agent

### Run training script:

```bash
rosrun drone_ppo_nav train_ppo.py
```

The drone will:

- Arm and take off
- Enter OFFBOARD mode
- Navigate through the corridor
- Learn from reward signals

---

## 📊 Visualize Training with TensorBoard

After training starts, run:

```bash
tensorboard --logdir ppo_tensorboard/
```

Open [http://localhost:6006](http://localhost:6006) in your browser to monitor reward curves and losses.

---

## 🎯 PPO Objective

The PPO reward function encourages:

- Forward movement toward the goal
- Avoiding collisions using LiDAR
- Staying within corridor boundaries

The training uses continuous or discrete action space (configurable inside `drone_env.py`).

---

## 📌 Notes

- The LiDAR topic is `/scan` and provides 360° 2D laser data.
- The target point is defined as `(23, 0, 0)` at the end of the corridor.
- Collision triggers episode reset.

---

## 📃 License

This project is released under the [Nan](LICENSE).

---

## 🙌 Acknowledgements

- PX4 Autopilot
- ROS & MAVROS
- Stable-Baselines3
- OpenAI Gym

---

## 👨‍💻 Author

**Dhayaa Khudher**

> For questions or collaborations, feel free to reach out via GitHub or open an issue!
