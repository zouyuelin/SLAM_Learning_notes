# 深度学习模型的训练
在images/pose.txt中给出了示例的数据集，同时把图像文件给到images文件夹下，修改posenet.py中的dataset即可训练

训练方法：
'''shell
python posenet.py
'''

将模型转换为 onnx
'''shell
python -m tf2onnx.convert --saved-model kerasTempModel --output "model.onnx" --opset 14
'''
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
mkdir build
cd build
cmake ..
make -j

运行方法：
./PoseImageNet /PATH_TO_YOUR_DATASETS(TUM)
