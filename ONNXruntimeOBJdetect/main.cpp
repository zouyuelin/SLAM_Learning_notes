#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <string>

//onnxruntime
#include <core/session/onnxruntime_cxx_api.h>
#include <core/providers/cuda/cuda_provider_factory.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/providers/tensorrt/tensorrt_provider_factory.h>

using namespace std;

class threshold
{
public:
    int OtsuAlgThreshold(cv::Mat image);
    cv::Mat MoveDetect(cv::Mat frame, cv::Mat object, int thred=100);
private:
    float offset = -70;
};
void objectPosition(cv::Mat &frame);
void objectPositionONNX(cv::Mat &imgSource,Ort::Session &session,Ort::MemoryInfo &memory_info );
void calcHistR(cv::Mat frame, cv::Mat object, int thred);
cv::dnn::Net net;


//---------------------------网络部署------------------------
//输入网络的维度
static constexpr const int width = 600;
static constexpr const int height = 600;
static constexpr const int channel = 3;
std::array<int64_t, 4> input_shape_{ 1,height, width,channel};
std::vector<const char*> input_node_names = {"image_tensor:0"};
std::vector<const char*> output_node_names = {"detection_boxes:0","detection_scores:0","detection_classes:0","num_detections:0" };


int main(int argc,char ** argv)
{

    //模型位置
    string model_path = "onnx/model_ssd_resnet101.onnx";//model_ssd_resnet101.onnx   model_ssd_resnet50.onnx

    Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
    Ort::SessionOptions session_options;
    //CUDA加速开启
    //OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0);
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::AllocatorWithDefaultOptions allocator;

    //加载ONNX模型
    Ort::Session session(env, model_path.c_str(), session_options);
    cout<<"load the model successful"<<endl;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    //利用chrono 检测时间---------
//    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//    chrono::duration<double> delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1); //milliseconds ms
//    cout<<"耗时:"<<delay_time.count()<<"ms"<<endl;


    string tes = "p1000=100";
    bool findIt = false;
    string arg;
    for (int i = 0;i<tes.length();i++) {
        if(tes[i]=='=')
        {
            findIt = true;
            continue;
        }
        if(findIt)
            arg.push_back(tes[i]);
    }
    std::cout<<arg<<endl;


    cv::String pb_path="./cnnDIR_resnet101/frozen_inference_graph.pb"; //faster-rcnn cnnDIR_resnet101
    cv::String pbtxt_path="./cnnDIR_resnet101/frozen_inference_graph.pbtxt"; //cnnDIR_resnet101
    cv::String videoPath = "./elec.mp4"; //elec.mp4

    net=cv::dnn::readNetFromTensorflow(pb_path,pbtxt_path);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);


    //相机内参
    string img = argv[1];
    cv::Mat cameraMatrix = cv::Mat::zeros(3,3,CV_64F);
    cameraMatrix.at<double>(0,0) = 399.23;
    cameraMatrix.at<double>(1,1) = 396.99;
    cameraMatrix.at<double>(0,2) = 301.175;
    cameraMatrix.at<double>(1,2) = 306.12;
    cameraMatrix.at<double>(2,2) = 1.00;

    cv::Mat distCoeffs = cv::Mat::zeros(5,1,CV_64F);
    distCoeffs.at<double>(0,0) = -0.0876;
    distCoeffs.at<double>(0,1) = -0.1972;
    distCoeffs.at<double>(0,4) = 0.1358;

    cv::Mat image_ = cv::imread(img);
    cv::Mat image,imageONNX;
    cv::Mat objecImage = cv::imread("./4.png"); //3.png
    //消除畸变
    cv::undistort(image_,image,cameraMatrix,distCoeffs,cameraMatrix);
    imageONNX = image_.clone();

    //Opencv dnn 检测
    objectPosition(image_);

    //onnxruntime 检测
    objectPositionONNX(imageONNX,session,memory_info);
    cv::imshow(img,image_);
    cv::imshow("ONNX",imageONNX);
    cv::waitKey(0);


    threshold obj;
    cv::Mat gray = obj.MoveDetect(image,objecImage);
    cv::imshow(img+"gray",gray);
    cv::waitKey(0);
    cv::destroyAllWindows();

    //加载视频测试-----
    cv::VideoCapture capture(0);
    capture.set(cv::CAP_PROP_FRAME_WIDTH,1920);//1920*1080
    capture.set(cv::CAP_PROP_FRAME_HEIGHT,1080);

    while (capture.isOpened()) {

        cv::TickMeter meter;
        meter.start();
        cv::Mat frame;
        capture>>frame;
        if (frame.empty())
        {
            capture.release();
            break;
        }
        cv::Rect rect(frame.cols/2-355,frame.rows/2-355,750,750);
        frame=frame(rect);
        cv::resize(frame,frame,cv::Size(600,600));

        cv::Mat distortImage;
        cv::undistort(frame,distortImage,cameraMatrix,distCoeffs);

        //深度学习目标检测
        //objectPosition(frame);
        //ONNXRunTime 检测
        objectPositionONNX(frame,session,memory_info);

        meter.stop();
        cv::putText(frame,cv::format("FPS: %.2f ; time: %.2f ms", 1000.f / meter.getTimeMilli(), meter.getTimeMilli()),cv::Point(10, 20), 0, 0.5, cv::Scalar(0, 0, 255),1);

        //阈值分割
        cv::Mat threshImage = obj.MoveDetect(distortImage,objecImage);

        cv::imshow("Image",frame);
        cv::imshow("Image_thresh",threshImage);

        cv::Mat mergeMat;
        cv::cvtColor(threshImage,threshImage,cv::COLOR_GRAY2BGR);
        cv::addWeighted(frame,0.6,threshImage,0.5,0.2,mergeMat);
        cv::imshow("Merge",mergeMat);
        cv::waitKey(1);
    }



    return 0;
}

cv::Mat threshold::MoveDetect(cv::Mat frame,cv::Mat object,int thred)
{
    //canny
    /*cv::Mat canny_;
    cv::Canny(frame,canny_,50,150,3);
    cv::imshow("canny",canny_);*/

    //差分
    //cv::Mat diff;
    //cv::absdiff(backgroundGray, frame, diff);

    cv::Mat gray;

    cv::medianBlur(frame,frame,5);

    cv::Mat r(gray.rows, gray.cols, CV_8UC1);
    cv::Mat g(gray.rows, gray.cols, CV_8UC1);
    cv::Mat b(gray.rows, gray.cols, CV_8UC1);
    cv::Mat out[]={b,g,r};
    cv::split(frame, out);
    b=out[0];
    g=out[1];
    r=out[2];

    gray = r.clone();
    //cv::equalizeHist( gray, gray );

    cv::imshow("rgb",r);
    calcHistR(gray,object,thred);
    //cv::waitKey(0);

    //int thred = OtsuAlgThreshold(gray);

    cv::threshold(gray,
              gray,
              thred,   //85 20 15
              255,
              cv::THRESH_BINARY);//cv::THRESH_BINARY_INV cv::THRESH_OTSU 60

    //cv::copyMakeBorder(gray,gray,10,10,10,10,cv::BORDER_CONSTANT,cv::Scalar(255,255,255));

    cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));

    /*----先腐蚀后膨胀: 开运算（morph_open）
        消除小物体,在纤细点分离物体，在平滑处不明显改变面积
      ----先膨胀后腐蚀: 闭运算（morph_close）
        消除小黑点
      ----膨胀图与腐蚀图之差: 形态学梯度（gradient）
        保留和突出边缘
      ----闭运算结果与原图作差: 顶帽（morph_tophat）
        分离比临近点亮的斑块（较大斑块）
      ----原图与开运算结果作差: 黑帽（morph_blacktop）
        分离比临近点暗的斑块
    */
    //高级形态学 处理
    //cv::morphologyEx(diff,diff,cv::MORPH_OPEN,kernel);

    //腐蚀
    cv::erode(gray,gray,kernel_erode);
    //膨胀
    cv::dilate(gray,gray,kernel_dilate);

    return gray;//
}

int threshold::OtsuAlgThreshold(cv::Mat image)
{
        int T = 0; //Otsu算法阈值
        double varValue = 0; //类间方差中间值保存
        double w0 = 0; //前景像素点数所占比例
        double w1 = 0; //背景像素点数所占比例
        double u0 = 0; //前景平均灰度
        double u1 = 0; //背景平均灰度
        double Histogram[256] = { 0 }; //灰度直方图，下标是灰度值，保存内容是灰度值对应的像素点总数
        uchar *data = image.data;

        double totalNum = image.rows*image.cols; //像素总数

        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                if (image.at<uchar>(i, j) != 0) Histogram[data[i*image.step + j]]++;
            }
        }
        int minpos, maxpos;
        for (int i = 0; i < 255; i++)
        {
            if (Histogram[i] != 0)
            {
                minpos = i;
                break;
            }
        }
        for (int i = 255; i > 0; i--)
        {
            if (Histogram[i] != 0)
            {
                maxpos = i;
                break;
            }
        }

        for (int i = minpos; i <= maxpos; i++)
        {
            //每次遍历之前初始化各变量
            w1 = 0;       u1 = 0;       w0 = 0;       u0 = 0;
            //***********背景各分量值计算**************************
            for (int j = 0; j <= i; j++) //背景部分各值计算
            {
                w1 += Histogram[j];   //背景部分像素点总数
                u1 += j*Histogram[j]; //背景部分像素总灰度和
            }
            if (w1 == 0) //背景部分像素点数为0时退出
            {
                break;
            }
            u1 = u1 / w1; //背景像素平均灰度
            w1 = w1 / totalNum; // 背景部分像素点数所占比例
            //***********背景各分量值计算**************************

            //***********前景各分量值计算**************************
            for (int k = i + 1; k < 255; k++)
            {
                w0 += Histogram[k];  //前景部分像素点总数
                u0 += k*Histogram[k]; //前景部分像素总灰度和
            }
            if (w0 == 0) //前景部分像素点数为0时退出
            {
                break;
            }
            u0 = u0 / w0; //前景像素平均灰度
            w0 = w0 / totalNum; // 前景部分像素点数所占比例
            //***********前景各分量值计算**************************

            //***********类间方差计算******************************
            double varValueI = w0*w1*(u1 - u0)*(u1 - u0); //当前类间方差计算
            if (varValue < varValueI)
            {
                varValue = varValueI;
                T = i;
            }
        }
        T+=offset;
        return T;
}

void objectPosition(cv::Mat &frame)
{
    cv::Mat inputblob = cv::dnn::blobFromImage(frame,
                                   1.,
                                   cv::Size(600, 600),true,true);//,false,true
    net.setInput(inputblob);
    cv::Mat output = net.forward();//1*1*100*7

    cv::Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    float confidenceThreshold = 0.36;

//------------------循环遍历显示检测框--------------------------
    for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

            if (confidence >= confidenceThreshold && objectClass == 0)
            {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
                std::ostringstream ss;
                ss << confidence;
                cv::String conf(ss.str());

                if(xRightTop - xLeftBottom>400||yRightTop - yLeftBottom>400)
                    continue;

                //cacl the theta and length
                 cv::Point center_C;
                 center_C.x = (xLeftBottom+xRightTop)/2.F;
                 center_C.y = (yLeftBottom+yRightTop)/2.F;

                //显示检测框
                cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                    (int)(xRightTop - xLeftBottom),
                    (int)(yRightTop - yLeftBottom));


                cv::rectangle(frame, object, cv::Scalar(0,0,255), 2);
                cv::String label = cv::String("confidence") + ": " + conf;
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.3, 1, &baseLine);
                cv::rectangle(frame, cv::Rect(cv::Point(xLeftBottom, yLeftBottom - labelSize.height),
                    cv::Size(labelSize.width, labelSize.height + baseLine)),
                    cv::Scalar(255, 255, 0), cv::FILLED);
                cv::putText(frame, label, cv::Point(xLeftBottom, yLeftBottom),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));

            }
        }
}


void objectPositionONNX(cv::Mat &imgSource,Ort::Session &session,Ort::MemoryInfo &memory_info )
{
    //-------------------------------------------------------------onnxruntime-------------------------------------------------
    //获取输入输出的维度
//    std::vector<int64_t> input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
//    std::vector<int64_t> output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
//    input_dims[0] = output_dims[0] = 1;//batch size = 1

    /*
        session.GetOutputName(1, allocator);
        session.GetInputName(1, allocator);
        //输出模型输入节点的数量
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();
    */

    std::vector<Ort::Value> input_tensors;

    //将图像存储到uchar数组中，BGR--->RGB
    std::array<uchar, width * height *channel> input_image_{};
    uchar* input =  input_image_.data();
    for (int i = 0; i < imgSource.rows; i++) {
        for (int j = 0; j < imgSource.cols; j++) {
            for (int c = 0; c < 3; c++)
            {
                //NHWC 格式
                if(c==0)
                    input[i*imgSource.cols*3+j*3+c] = imgSource.ptr<uchar>(i)[j*3+2];
                if(c==1)
                    input[i*imgSource.cols*3+j*3+c] = imgSource.ptr<uchar>(i)[j*3+1];
                if(c==2)
                    input[i*imgSource.cols*3+j*3+c] = imgSource.ptr<uchar>(i)[j*3+0];
                //NCHW 格式
//                if (c == 0)
//                     input[c*imgSource.rows*imgSource.cols + i * imgSource.cols + j] = imgSource.ptr<uchar>(i)[j * 3 + c];
//                if (c == 1)
//                     input[c*imgSource.rows*imgSource.cols + i * imgSource.cols + j] = imgSource.ptr<uchar>(i)[j * 3 + c];
//                if (c == 2)
//                     input[c*imgSource.rows*imgSource.cols + i * imgSource.cols + j] = imgSource.ptr<uchar>(i)[j * 3 + c];


            }
        }
    }

    input_tensors.push_back(Ort::Value::CreateTensor<uchar>(
            memory_info, input, input_image_.size(), input_shape_.data(), input_shape_.size()));
    //不知道输入维度时
    //input_tensors.push_back(Ort::Value::CreateTensor<uchar>(
    //        memory_info, input, input_image_.size(), input_dims.data(), input_dims.size()));

    std::vector<Ort::Value> output_tensors;
    output_tensors = session.Run(Ort::RunOptions { nullptr },
                                                       input_node_names.data(), //输入节点名
                                                       input_tensors.data(),     //input tensors
                                                       input_tensors.size(),     //1
                                                       output_node_names.data(), //输出节点名
                                                       output_node_names.size()); //4


    float* boxes_ = output_tensors[0].GetTensorMutableData<float>();
    float* scores_ = output_tensors[1].GetTensorMutableData<float>();
    float* class_ = output_tensors[2].GetTensorMutableData<float>();
    float* num_detection = output_tensors[3].GetTensorMutableData<float>();

    //------------------循环遍历显示检测框--------------------------

        for (int i = 0; i < num_detection[0]; i++)
            {
                float confidence = scores_[i];
                size_t objectClass = (size_t)class_[i];

                if (confidence >= 0.05)
                {
                    int xLeftBottom = static_cast<int>(boxes_[i*4 + 1] * imgSource.cols);
                    int yLeftBottom = static_cast<int>(boxes_[i*4 + 0] * imgSource.rows);
                    int xRightTop = static_cast<int>(boxes_[i*4 + 3] * imgSource.cols);
                    int yRightTop = static_cast<int>(boxes_[i*4 + 2]* imgSource.rows);

                    //显示检测框
                    cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));

                    cv::rectangle(imgSource, object, cv::Scalar(0,0,255), 2);
                    cv::String label = cv::String("confidence :") +to_string(confidence);
                    int baseLine = 0;
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.3, 1, &baseLine);
                    cv::rectangle(imgSource, cv::Rect(cv::Point(xLeftBottom, yLeftBottom - labelSize.height),
                        cv::Size(labelSize.width, labelSize.height + baseLine)),
                        cv::Scalar(255, 255, 0), cv::FILLED);
                    cv::putText(imgSource, label, cv::Point(xLeftBottom, yLeftBottom),
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));
                }
            }
            //-------------------------------------------------------------onnxruntime-------------------------------------------------
}


void calcHistR(cv::Mat frame, cv::Mat object, int thred )
{
    int histsize = 256;
    float range[] = { 0,256 };
    const float*histRanges = { range };
    cv::Mat hist;
    cv::calcHist(&frame, 1, 0, cv::Mat(), hist, 1, &histsize, &histRanges, true, false);


        //归一化
    int hist_h = 400;//直方图的图像的高
    int hist_w = 512; //直方图的图像的宽
    int bin_w = hist_w / histsize;//直方图的等级
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));//绘制直方图显示的图像
    cv::normalize(hist, hist, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat());//归一化


        //步骤三：绘制直方图（render histogram chart）
        for (int i = 1; i < histsize; i++)
        {   //绘制红色分量直方图
            cv::line(histImage, cv::Point((i - 1)*bin_w, hist_h - cvRound(hist.at<float>(i - 1))),
                cv::Point((i)*bin_w, hist_h - cvRound(hist.at<float>(i))), cv::Scalar(0, 0, 255), 2);
        }
        cv::line(histImage, cv::Point(thred*bin_w, hist_h),
                 cv::Point(thred*bin_w, 0), cv::Scalar(255, 0, 0), 2);

    //直方图反投影
    vector<cv::Mat> objectBGR;
    cv::split(object,objectBGR);
    cv::Mat histObject;
    cv::calcHist(&objectBGR[2], 1, 0, cv::Mat(), histObject, 1, &histsize, &histRanges, true, false);
    cv::imshow("object",object);
    cv::normalize(histObject, histObject, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat());//归一化

    cv::Mat bpoutput;
    cv::calcBackProject(&frame,1,0,histObject,bpoutput,&histRanges,1,true);
    cv::imshow("calcBackProject",bpoutput);
    cv::imshow("HistImage_channel_Red", histImage);
}
