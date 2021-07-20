#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;

// 在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main(int argc, char **argv) {

    if(argc <3)
    {
        cout<<"输入参数方法：**exe left.png right.png\n";
        return 0;
    }

    string left_file = argv[1];
    string right_file = argv[2];
    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    //double fx = 399.23, fy = 396.99, cx = 301.175, cy = 306.12;
    // 基线
    double b = 0.573;

    // 读取图像
    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);
    cv::cuda::GpuMat left_g,right_g;
    left_g.upload(left);
    right_g.upload(right);


    //------use cpu to compute------
    /*cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 神奇的参数
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);*/


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

    cv::Mat disparity;
    disparity_g.download(disparity);
    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

    // 如果你的机器慢，请把后面的v++和u++改成v+=2, u+=2
    for (int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++) {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色

            // 根据双目模型计算 point 的位置
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            pointcloud.push_back(point);
        }

    //cv::imshow("disparity", disparity /96);//cpu

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
