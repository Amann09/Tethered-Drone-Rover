#!/bin/bash
# Terminal #1
cd ~/Micro-XRCE-DDS-Agent && MicroXRCEAgent udp4 -p 8888 &


cd ~/ros2_ws && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 run ros_gz_image image_bridge /camera &

# & ros2 run ros_gz_image image_bridge /camera &

# Terminal #2
xterm -e "cd ~/PX4-Autopilot && PX4_SYS_AUTOSTART=4002  PX4_GZ_MODEL=x500_depth PX4_GZ_MODEL_POSE="1,0,0,0,0,0" ./build/px4_sitl_default/bin/px4 ;/bin/bash" &


# Terminal #3
# source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && cd ~/scripts &&  python3 aruco_detect.py &


# Terminal #4
# xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && sleep 40 && cd ~/scripts && python3 controller.py --takeoff & exec bash" &


# # # Terminal # -1 - 1
# xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && 
# cd ~/ros2_ws && ros2 run ros_gz_bridge parameter_bridge /model/tugbot/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist" &


# # # Terminal # -1 - 2
# xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && 
# cd ~/ros2_ws && ros2 run ros_gz_bridge parameter_bridge /model/tugbot/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry" &

# # Terminal # -1 - 3
xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && 
cd ~/ros2_ws && ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/root/ros2_ws/yaml_files/bridge.yaml ; /bin/bash" &

# Terminal # -2
# xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && sleep 20 && cd ~/ros2_ws && ros2 topic pub /model/tugbot/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}' ;/bin/bash" &


# Terminal #5
xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && cd ~/scripts ; /bin/bash" &


# Terminal #5
xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && cd ~/ros2_ws ; /bin/bash" &

# Terminal #5.1
xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && cd ~/ros2_ws ; /bin/bash" &

# Terminal #6
# xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && cd ~/ros2_ws && ros2 launch clearpath_gz robot_spawn.launch.py ; /bin/bash" &
# xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && cd ~/ros2_ws && ros2 launch clearpath_gz robot_spawn.launch.py setup_path:=/root/ros2_ws/ ; /bin/bash" &
# xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && cd ~/ros2_ws && ros2 launch clearpath_gz robot_spawn.launch.py setup_path:=/root/ros2_ws/ x:=1.0 y:=1.0 ; /bin/bash" &

# # # Terminal to bridge topic for Husky A200
# xterm -e "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && 
# cd ~/ros2_ws && ros2 run ros_gz_bridge parameter_bridge /model/a200_0000/robot/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist" &

# Keep the script running to keep the terminals open
while :; do sleep 1; done
