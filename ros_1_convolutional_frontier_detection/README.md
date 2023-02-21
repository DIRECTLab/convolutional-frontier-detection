# Launching

roscore
roslaunch yahboomcar_nav laser_astrapro_bringup.launch
roslaunch yahboomcar_nav yahboomcar_map.launch


# Launching on Black Robot

roslaunch xrrobot bringup.launch
roslaunch xrrobot lidar_slam.launch
rosrun frontier_detection cfd.py
