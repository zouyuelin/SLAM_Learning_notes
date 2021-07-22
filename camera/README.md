# 发布话题供ORB_SLAM2 使用, 专用双目

订阅者：
```shell
camera/left/image_raw
camera/right/image_raw
```

发布者：
Node [/camera_stereo]
Publications: 
 * /camera/left/image_raw [sensor_msgs/Image]
 * /camera/right/image_raw [sensor_msgs/Image]

# 构建方法
复制到 ~/catkin_ws/src/
cd ~/catkin_ws
cakin_make

# 运行
source ~/catkin_ws/devel/setup.bash
rosrun camera camera_stereo_node
