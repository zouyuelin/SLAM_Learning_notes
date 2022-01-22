# 深度学习模型的训练
在images/pose.txt中给出了示例的数据集，同时把图像文件给到images文件夹下，修改posenet.py中的dataset即可训练

训练方法：
```shell
python posenet.py
```

将模型转换为 onnx
```shell
python -m tf2onnx.convert --saved-model kerasTempModel --output "model.onnx" --opset 14
```
# C++上部署模型
需要的库：
onnxruntime : cuda tensorRT(没有也可以不用)
g2o
pangolin
boost
opencv 4.3.0 以上
eigen
sophus

编译方法：
```shell
mkdir build
cd build
cmake ..
make -j
```
运行方法：
```shell
./PoseImageNet /PATH_TO_YOUR_DATASETS(TUM)
```
# 运行结果
DeepLearning poseEstimation:
![Deeplearning_CameraPoseEstimate](https://user-images.githubusercontent.com/58660028/150632332-c39dba97-ae38-4157-8653-72eb5b9302d0.jpg)
pnp solver
![PNPSolver](https://user-images.githubusercontent.com/58660028/150632337-5da6f2de-467c-447f-a6ed-1d6ee633160f.jpg)
