
#!/usr/bin/env python3
import time
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from mavros_msgs.srv import CommandBool, SetMode
from gym import spaces, Env
from stable_baselines3.common.logger import configure

class DroneEnv(Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        rospy.init_node('drone_training_ppo', anonymous=True)
        rospy.set_param('/use_sim_time', True)

        # === Subscribers ===
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.cmd_callback)

        # === Publishers ===
        self.cmd_publisher = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)

        # === Services ===
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/gazebo/set_model_state')
        self.arming_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.set_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.logger = configure(folder="ppo_tensorboard", format_strings=["stdout", "csv", "tensorboard"])

        self.steps = 0
        self.max_steps = 10000
        self.position = np.array([0.0, 0.0, 0.0])
        self.prev_position = self.position.copy()
        self.velocity = np.zeros(3)
        self.yaw = 0.0
        self.lidar_data = np.ones(60) * 6.0

        self.goal_position = np.array([23.0, 0.0, 0.0])
        self.boundary_limits = np.array([[-1.0, 40.5], [-10.0, 10.0], [0.1, 3.0]])

        self.action_space = spaces.Box(
            low=np.array([-1.0, -0.5, -0.5, -1.0]),
            high=np.array([1.0, 0.5, 0.5, 1.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(60 + 3 + 3 + 1,), dtype=np.float32
        )

        self.prev_obs = None
        self.same_obs_counter = 0
        self.success_counter = 0

    def lidar_callback(self, data):
        lidar = np.array(data.ranges[:60], dtype=np.float32)
        lidar = np.nan_to_num(lidar, nan=6.0, posinf=6.0, neginf=0.1)
        self.lidar_data = np.clip(lidar, 0.1, 6.0)

    def position_callback(self, data):
        self.position = np.array([
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z
        ])

    def cmd_callback(self, data):
        self.velocity = np.array([data.twist.linear.x, data.twist.linear.y, data.twist.linear.z])
        self.yaw = data.twist.angular.z

    def reset(self):
        self.steps = 0
        self.velocity = np.zeros(3)
        self.yaw = 0.0
        self.lidar_data = np.ones(60) * 6.0
        self.same_obs_counter = 0
        self.prev_obs = None

        reset_position = [0.1, 0.0, 0.3]

        try:
            state_msg = ModelState()
            state_msg.model_name = 'iris'
            state_msg.pose.position.x = reset_position[0]
            state_msg.pose.position.y = reset_position[1]
            state_msg.pose.position.z = reset_position[2]
            state_msg.pose.orientation.w = 1.0
            self.set_state_srv(state_msg)
            time.sleep(0.5)

            # Stream velocity before switching to OFFBOARD
            for _ in range(20):
                cmd = TwistStamped()
                self.cmd_publisher.publish(cmd)
                time.sleep(0.05)

            self.set_mode_srv(custom_mode='OFFBOARD')
            self.arming_srv(True)

            # Boost up to takeoff
            boost_cmd = TwistStamped()
            boost_cmd.twist.linear.z = 0.5
            self.cmd_publisher.publish(boost_cmd)
            time.sleep(1.5)

        except rospy.ServiceException as e:
            print("Service call failed:", e)

        self.prev_position = self.position.copy()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.lidar_data, self.position, self.velocity, [self.yaw]])

    def step(self, action):
        self.steps += 1

        cmd = TwistStamped()
        cmd.twist.linear.x = action[0]
        cmd.twist.linear.y = action[1]
        cmd.twist.linear.z = action[2]
        cmd.twist.angular.z = action[3]
        self.cmd_publisher.publish(cmd)
        time.sleep(0.1)

        obs = self._get_obs()
        reward = 0
        done = False

        distance_to_goal = np.linalg.norm(self.goal_position - self.position)
        progress = self.position[0] - self.prev_position[0]
        min_lidar = np.min(self.lidar_data)

        reward += self.position[0] * 3.0

        if progress > 0.01:
            reward += progress * 80
        else:
            reward -= 3

        if min_lidar > 1.5:
            reward += 100

        if distance_to_goal < 0.5:
            reward += 200
            done = True
            self.success_counter += 1
            self.logger.record("custom/success_count", self.success_counter)

        if min_lidar < 1.0:
            reward -= 50

        if not self._within_boundaries():
            reward -= 100
            done = True

        if self.steps >= self.max_steps:
            done = True

        if self.prev_obs is not None and np.allclose(obs, self.prev_obs, rtol=0.0, atol=1e-4):
            self.same_obs_counter += 1
        else:
            self.same_obs_counter = 0

        self.prev_obs = obs.copy()

        if self.same_obs_counter > 150:
            reward -= 50
            done = True

        self.prev_position = self.position.copy()

        self.logger.record("custom/reward", reward)
        self.logger.record("custom/distance_to_goal", distance_to_goal)
        self.logger.dump(self.steps)

        return obs, reward, done, {}

    def _within_boundaries(self):
        x, y, z = self.position
        return (self.boundary_limits[0][0] <= x <= self.boundary_limits[0][1] and
                self.boundary_limits[1][0] <= y <= self.boundary_limits[1][1] and
                self.boundary_limits[2][0] <= z <= self.boundary_limits[2][1])
