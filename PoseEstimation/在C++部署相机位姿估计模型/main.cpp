//onnxruntime
#include <core/session/onnxruntime_cxx_api.h>
#include <core/providers/cuda/cuda_provider_factory.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/providers/tensorrt/tensorrt_provider_factory.h>
#include "vo_slam.hpp"

Sophus::SE3 computePoseDNN(Mat img_1, Mat img_2, Ort::Session &session, Ort::MemoryInfo &memory_info);
void printModelInfo(Ort::Session &session,Ort::AllocatorWithDefaultOptions &allocator);

//输入网络的维度
static constexpr const int width = 512;
static constexpr const int height = 512;
static constexpr const int channel = 3;
std::array<int64_t, 4> input_shape_{ 1,height, width,channel};

using namespace cv;
using namespace std;

int main(int argc,char**argv)
{
    std::string dataset_dir = argv[1]; //dataset/rgbd_dataset_freiburg1_desk
    std::string assPath = dataset_dir+"/associate.txt";

    std::ifstream txt;
    txt.open(assPath.data());
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while (!txt.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        txt>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( txt.good() == false )
            break;
    }

    //*********************************************DNN Pose Estimate*******************************//

    //-------------------------------------------------------------onnxruntime-------------------------------------------------
    //图片和模型位置
    string model_path = "../model.onnx";

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "PoseEstimate");
    Ort::SessionOptions session_options;
    //CUDA加速开启
    OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0); //tensorRT
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::AllocatorWithDefaultOptions allocator;
    //加载ONNX模型
    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    //打印模型的信息
    printModelInfo(session,allocator);
    //********************************************DNN POSE Estimate********************************//

    Sophus::SE3 currentPose(Eigen::Matrix3d::Identity(),Eigen::Vector3d(0,0,0));//Eigen::Quaterniond(-0.3879,0.7907,0.4393,-0.1770),Eigen::Vector3d(1.2334, -0.0113, 1.6941)

    //*****************************************利用pnp的视觉里程计******************************
//    VO_slam* slam = new VO_slam;
//    for(size_t i=0;i<rgb_files.size()-2;i++)
//    {
//    //---------------------------------------------------------------tracing------------------------------------------------------//
//        boost::timer timer;
//        Mat img = imread(rgb_files[i],IMREAD_COLOR);
//        Mat depth = imread(depth_files[i], IMREAD_UNCHANGED);

//        slam->tracking(img,depth);

//        cout<<"VO cost time is:"<<timer.elapsed()<<endl;
//        cv::imshow("frame",img);
//        cv::waitKey(10);

//     }

//    return 0;
//    delete slam;

    //*****************************************利用深度学习的视觉里程计******************************
    visualMap::Ptr map_(new visualMap());

    for(size_t i=0;i<rgb_files.size()-2;i++)
    {
    //---------------------------------------------------------------tracing------------------------------------------------------//
        Mat img_1 = imread(rgb_files[i],IMREAD_COLOR);
        Mat img_2 = imread(rgb_files[i+1],IMREAD_COLOR);

        boost::timer timer;//boost计时
            Sophus::SE3 pose = computePoseDNN(img_1,img_2,session,memory_info);
        cout<<"cost time: "<<timer.elapsed()<<endl;//boost计时

        currentPose = pose*currentPose;

        Eigen::Vector3d rpy = camera::ToEulerAngles(pose.unit_quaternion());

        map_->keyposes.push_back(Eigen::Isometry3d(currentPose.matrix().inverse()));
        map_->drawing(Eigen::Isometry3d(currentPose.matrix().inverse()));
        pangolin::FinishFrame();
        cv::imshow("frame",img_2);
        cv::waitKey(10);
        //绘画出轨迹图
    }
    txt.close();

    return 0;
}


Sophus::SE3 computePoseDNN(Mat img_1, Mat img_2, Ort::Session &session,Ort::MemoryInfo &memory_info)
{
    Mat Input_1,Input_2;
    resize(img_1,Input_1,Size(512,512));
    resize(img_2,Input_2,Size(512,512));
    std::vector<const char*> input_node_names = {"input1","input2"};
    std::vector<const char*> output_node_names = {"Output"};

    //将图像存储到uchar数组中，BGR--->RGB
    std::array<float, width * height *channel> input_image_1{};
    std::array<float, width * height *channel> input_image_2{};

    float* input_1 =  input_image_1.data();
    float* input_2 =  input_image_2.data();

    for (int i = 0; i < Input_1.rows; i++) {
        for (int j = 0; j < Input_1.cols; j++) {
            for (int c = 0; c < 3; c++)
            {
                //NHWC 格式
                if(c==0)
                    input_1[i*Input_1.cols*3+j*3+c] = Input_1.ptr<uchar>(i)[j*3+2]/255.0;
                if(c==1)
                    input_1[i*Input_1.cols*3+j*3+c] = Input_1.ptr<uchar>(i)[j*3+1]/255.0;
                if(c==2)
                    input_1[i*Input_1.cols*3+j*3+c] = Input_1.ptr<uchar>(i)[j*3+0]/255.0;
                //NCHW 格式
//                if (c == 0)
//                     input_1[c*imgSource.rows*imgSource.cols + i * imgSource.cols + j] = imgSource.ptr<uchar>(i)[j * 3 + 2]/255.0;
//                if (c == 1)
//                     input_1[c*imgSource.rows*imgSource.cols + i * imgSource.cols + j] = imgSource.ptr<uchar>(i)[j * 3 + 1]/255.0;
//                if (c == 2)
//                     input_1[c*imgSource.rows*imgSource.cols + i * imgSource.cols + j] = imgSource.ptr<uchar>(i)[j * 3 + 0]/255.0;


            }
        }
    }
    for (int i = 0; i < Input_2.rows; i++) {
        for (int j = 0; j < Input_2.cols; j++) {
            for (int c = 0; c < 3; c++)
            {
                //NHWC 格式
                if(c==0)
                    input_2[i*Input_2.cols*3+j*3+c] = Input_2.ptr<uchar>(i)[j*3+2]/255.0;
                if(c==1)
                    input_2[i*Input_2.cols*3+j*3+c] = Input_2.ptr<uchar>(i)[j*3+1]/255.0;
                if(c==2)
                    input_2[i*Input_2.cols*3+j*3+c] = Input_2.ptr<uchar>(i)[j*3+0]/255.0;
            }
        }
    }

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, input_1, input_image_1.size(), input_shape_.data(), input_shape_.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, input_2, input_image_2.size(), input_shape_.data(), input_shape_.size()));
    //不知道输入维度时
    //input_tensors.push_back(Ort::Value::CreateTensor<uchar>(
    //        memory_info, input, input_image_.size(), input_dims.data(), input_dims.size()));

    std::vector<Ort::Value> output_tensors;

    output_tensors = session.Run(Ort::RunOptions { nullptr },
                                    input_node_names.data(), //输入节点名
                                    input_tensors.data(),     //input tensors
                                    input_tensors.size(),     //2
                                    output_node_names.data(), //输出节点名
                                    output_node_names.size()); //1

//    cout<<output_tensors.size()<<endl;//输出的维度
    float* output = output_tensors[0].GetTensorMutableData<float>();
    Eigen::Vector3d t(output[0],output[1],output[2]);
    Eigen::Vector3d r(output[3],output[4],output[5]);

    // 初始化旋转向量，绕z轴旋转，y轴，x轴；
    Eigen::AngleAxisd R_z(r[2], Eigen::Vector3d(0,0,1));
    Eigen::AngleAxisd R_y(r[1], Eigen::Vector3d(0,1,0));
    Eigen::AngleAxisd R_x(r[0], Eigen::Vector3d(1,0,0));
    // 转换为旋转矩阵
    Eigen::Matrix3d R_matrix_xyz  = R_z.toRotationMatrix()*R_y.toRotationMatrix()*R_x.toRotationMatrix();

    return Sophus::SE3(R_matrix_xyz,t);
}

void printModelInfo(Ort::Session &session, Ort::AllocatorWithDefaultOptions &allocator)
{
    //输出模型输入节点的数量
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    cout<<"Number of input node is:"<<num_input_nodes<<endl;
    cout<<"Number of output node is:"<<num_output_nodes<<endl;

    //获取输入输出维度
    for(auto i = 0; i<num_input_nodes;i++)
    {
        std::vector<int64_t> input_dims = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout<<endl<<"input "<<i<<" dim is: ";
        for(auto j=0; j<input_dims.size();j++)
            cout<<input_dims[j]<<" ";
    }
    for(auto i = 0; i<num_output_nodes;i++)
    {
        std::vector<int64_t> output_dims = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout<<endl<<"output "<<i<<" dim is: ";
        for(auto j=0; j<output_dims.size();j++)
            cout<<output_dims[j]<<" ";
    }
    //输入输出的节点名
    cout<<endl;//换行输出
    for(auto i = 0; i<num_input_nodes;i++)
        cout<<"The input op-name "<<i<<" is:"<<session.GetInputName(i, allocator)<<endl;
    for(auto i = 0; i<num_output_nodes;i++)
        cout<<"The output op-name "<<i<<" is:"<<session.GetOutputName(i, allocator)<<endl;

    //input_dims_2[0] = input_dims_1[0] = output_dims[0] = 1;//batch size = 1
}
