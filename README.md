# 发布话题供ORB_SLAM2 使用, 专用双目

```shell
camera/left/image_raw
camera/right/image_raw
```

# 构建方法
复制到 ~/catkin_ws/src/
cd ~/catkin_ws
cakin_make

# 运行
source ~/catkin_ws/devel/setup.bash
rosrun camera camera_stereo_node 0
