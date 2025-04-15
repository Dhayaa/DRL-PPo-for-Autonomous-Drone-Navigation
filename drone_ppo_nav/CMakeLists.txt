cmake_minimum_required(VERSION 3.0.2)
project(drone_ppo_nav)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  gazebo_ros
  mavros_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Make Python scripts executable
catkin_install_python(PROGRAMS
  scripts/drone_env.py
  scripts/train_ppo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
