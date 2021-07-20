#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;
using namespace cv ;

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

void computeBM(Mat &left,Mat &right,Mat &disparity);
void computeSGBM(Mat &left,Mat &right,Mat &disparity);

// 在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main(int argc, char **argv) {

    int type = 0; //type BM

    if(argc <5)
    {
        cout<<"输入参数方法：**exe left.png right.png **.yaml type(0 fo BM or 1 for SGBM)\n";
        return 0;
    }
    if(argc>=5)
    {
        type = atoi(argv[4]); //C_str or string translate to int
    }

    configYaml::loadYaml(argv[3]);
    string left_file = argv[1];
    string right_file = argv[2];

    // 基线
    double b = 0.573;

    // load the image
    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);

    cv::Mat disparity;

    //两种方法BM或者SGBM
    if(type == 0)
        computeBM(left,right,disparity);
    else
        computeSGBM(left,right,disparity);

    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

    // 如果你的机器慢，请把后面的v++和u++改成v+=2, u+=2
    for (int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++) {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色

            // 根据双目模型计算 point 的位置
            double x = (u - configYaml::cameraMatrix_Eigen(0,2)) / configYaml::cameraMatrix_Eigen(0,0);
            double y = (v - configYaml::cameraMatrix_Eigen(1,2)) / configYaml::cameraMatrix_Eigen(1,1);
            double depth = configYaml::cameraMatrix_Eigen(0,0) * b / (disparity.at<float>(v, u));
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            pointcloud.push_back(point);
        }

    cv::imshow("disparity", disparity /8);
    cv::waitKey(0);

    // 画出点云
    showPointCloud(pointcloud);

    return 0;
}

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}

void computeBM(Mat &left,Mat &right,Mat &disparity)
{
    cv::cuda::GpuMat left_g,right_g;
    left_g.upload(left);
    right_g.upload(right);
    //------use GPU to compute------
    cv::Ptr<cv::cuda::StereoBM> sgbm = cv::cuda::createStereoBM();
    //sgbm->setPreFilterSize(9);
    //sgbm->setPreFilterCap(31);
    //sgbm->setMinDisparity(-16);
    //sgbm->setSpeckleRange(32);      //视差变化阈值，大于阈值窗口视差清零
    //影响较大的参数
    sgbm->setBlockSize(7);           //BlockSize 取值必须是奇数(3 5 7 9)
    sgbm->setSpeckleWindowSize(100);
    sgbm->setNumDisparities(128);    //NumDisparities 越大视差量数越大,必须是16的整数
    sgbm->setTextureThreshold(3);   //TextureThreshold 越小平滑度越高，纹理阈值。
    sgbm->setUniquenessRatio(5);

    cv::cuda::GpuMat disparity_sgbm_g, disparity_g;
    sgbm->compute(left_g, right_g, disparity_sgbm_g);
    disparity_sgbm_g.convertTo(disparity_g, CV_32F, 1.0 / 16.0f);

    disparity_g.download(disparity);
}

void computeSGBM(Mat &left,Mat &right,Mat &disparity)
{
    //------use cpu to compute------
    //cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 神奇的参数
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
    //预处理
    sgbm->setPreFilterCap(63);
    //代价参数
    sgbm->setSpeckleWindowSize(200);
    sgbm->setNumDisparities(96);
    sgbm->setMinDisparity(0);
    //动态参数
    sgbm->setP1(8 * 3 * 3);
    sgbm->setP2(32 * 9 * 9);
    //后处理
    sgbm->setBlockSize(3);
    sgbm->setSpeckleRange(32);
    sgbm->setUniquenessRatio(3);
    sgbm->setDisp12MaxDiff(1);     //左右视差

    cv::Mat disparity_sgbm;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
}
