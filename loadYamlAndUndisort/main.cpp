#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>

using namespace std;

class configYaml
{
    configYaml(){};

public:
    static cv::FileStorage fseting;
    static cv::Mat cameraMatrix_Mat;
    static Eigen::Matrix3d cameraMatrix_Eigen;
    static cv::Mat distCoeffs;
    static bool isOpened;
    static void loadYaml(std::string path)
    {
        fseting.open(path,cv::FileStorage::READ);
        if(!fseting.isOpened())
        {
            isOpened = false;
            cout<<"please confirm the path of yaml!\n";
            return ;
        }
        isOpened = true;

        double fx = configYaml::fseting["Camera.fx"];
        double fy = configYaml::fseting["Camera.fy"];
        double cx = configYaml::fseting["Camera.cx"];
        double cy = configYaml::fseting["Camera.cy"];

        double k1 = configYaml::fseting["Camera.k1"];
        double k2 = configYaml::fseting["Camera.k2"];
        double p1 = configYaml::fseting["Camera.p1"];
        double p2 = configYaml::fseting["Camera.p2"];
        double k3 = configYaml::fseting["Camera.k3"];

        cameraMatrix_Mat = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

        cameraMatrix_Eigen<<fx,   0,   cx,
                             0,  fy,   cy,
                             0,   0,    1;

        distCoeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
    }
    inline static void undistort(cv::Mat &frame,cv::Mat &output)
    {
        cv::undistort(frame ,output ,configYaml::cameraMatrix_Mat,configYaml::distCoeffs);
    }
};

cv::FileStorage configYaml::fseting = cv::FileStorage();
cv::Mat configYaml::cameraMatrix_Mat = cv::Mat::zeros(3,3,CV_64F);
Eigen::Matrix3d configYaml::cameraMatrix_Eigen = Eigen::Matrix3d::Identity(3,3);
cv::Mat configYaml::distCoeffs = cv::Mat::zeros(5,1,CV_64F);
bool configYaml::isOpened = false;

int main(int argc,char**argv)
{
    if(argc <3)
    {
        cout<<"please input a image!\n";
        return -1;
    }
    configYaml::loadYaml(argv[1]);
    if(!configYaml::isOpened)
    {
        cout<<"Can not open the file yaml!\n";
        return -1;
    }

    cv::Mat frame = cv::imread(argv[2]);
    cv::Mat output;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //undistort
    configYaml::undistort(frame,output);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<float> delay_time = t2 - t1; //milliseconds 毫秒
    cout<<"time_cost:"<<delay_time.count()<<" /seconds"<<endl;

    cv::imshow("source ",frame);
    cv::imshow("distort ",output);
    cv::waitKey(0);

    return 0;
}
