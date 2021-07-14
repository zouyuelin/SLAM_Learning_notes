#include <iostream>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>

#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>

#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>

#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3quat.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <Eigen/Core>

#include <vector>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches_GPU(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches);
void find_feature_matches_CPU(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches);
void BASolver(vector<Point2f> &points_1,vector<Point2f> &points_2,Mat &K);

//相机内参
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

int main(int argc,char**argv)
{
    std::string path1 = argv[1];
    std::string path2 = argv[2];
    if(argc !=3)
    {
        cout<<"输入参数方法：**/testOptimizerG2o ./1.png ./2.png\n";
        return 0;
    }
    Mat img_1 = imread(path1);
    Mat img_2 = imread(path2);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<Point2f> points_1,points_2;
    vector<DMatch> matches;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //use GPU
    //find_feature_matches_GPU(img_1,img_2,keypoints_1,keypoints_2,matches);
    //use cpu
    find_feature_matches_CPU(img_1,img_2,keypoints_1,keypoints_2,matches);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1); //milliseconds 毫秒

    cout<<"匹配耗时:"<<delay_time.count()<<"秒"<<endl;


    for(auto m:matches)
    {
        points_1.push_back(keypoints_1[m.queryIdx].pt);
        points_2.push_back(keypoints_2[m.trainIdx].pt);
    }

    cout<<"特征点的数量："<<points_1.size()<<"  "<<points_2.size()<<endl;

    //BA优化
    BASolver(points_1,points_2,K);

    Mat img_keypoints;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_keypoints);
    imshow("matches",img_keypoints);
    waitKey(0);

    return 0;
}

void find_feature_matches_GPU(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {

  cuda::Stream myStream;
  // 利用GPU加速提取特征点
  cuda::GpuMat d_img1,d_img2;
  d_img1.upload(img_1,myStream);
  d_img2.upload(img_2,myStream);

  cuda::GpuMat d_img1Gray,d_img2Gray;
  cuda::cvtColor(d_img1,d_img1Gray,COLOR_BGR2GRAY,0,myStream);
  cuda::cvtColor(d_img2,d_img2Gray,COLOR_BGR2GRAY,0,myStream);


  //-- 初始化
  cuda::GpuMat d_keypoints1, d_keypoints2;
  cuda::GpuMat descriptors_1, descriptors_2,descriptors_1_32F,descriptors_2_32F;

  //创建ORB检测
  Ptr<cuda::ORB> detector_ORB = cuda::ORB::create();

  //-- 第一步:检测 Oriented FAST 角点位置 计算 BRIEF 描述子
  detector_ORB->detectAndComputeAsync(d_img1Gray,cuda::GpuMat(),d_keypoints1,descriptors_1,false,myStream);
  detector_ORB->convert(d_keypoints1,keypoints_1);
  descriptors_1.convertTo(descriptors_1_32F,CV_32F);

  detector_ORB->detectAndComputeAsync(d_img2Gray,cuda::GpuMat(),d_keypoints2,descriptors_2,false,myStream);
  detector_ORB->convert(d_keypoints2,keypoints_2);
  descriptors_2.convertTo(descriptors_2_32F,CV_32F);

  //-- 第二步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离

  //*******************************************利用Match方法进行匹配************************************/
  //可取消注释选择此方法
  /*
  vector<DMatch> match;
  Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1_32F, descriptors_2_32F, match);

  //-- 第三步:匹配点对筛选
  double min_dist = 1000, max_dist = 0;


  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  cout<<"-- Max dist : "<< max_dist<<endl;
  cout<<"-- Min dist : "<< min_dist<<endl;


  for (int i = 0; i < descriptors_1.rows; i++)
  {
    if (match[i].distance <= 0.7*max_dist)
    {
      matches.push_back(match[i]);
    }
  }
  */



  //*******************************************利用KnnMatch方法进行匹配（效果较好）**************************/
  vector< vector<DMatch>> knnmatch;
  Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->knnMatch(descriptors_1_32F, descriptors_2_32F, knnmatch,2);

  //-- 第三步:匹配点对筛选

    for (int i = 0; i < knnmatch.size(); i++)
    {
        if (knnmatch[i][0].distance <= 0.8*knnmatch[i][1].distance)
        {
            matches.push_back(knnmatch[i][0]);
        }
    }


}

void find_feature_matches_CPU(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {

    //-- 初始化
    Mat descriptors_1, descriptors_2;
    Mat imgGray_1,imgGray_2;
    cvtColor(img_1,imgGray_1,COLOR_BGR2GRAY);
    cvtColor(img_2,imgGray_2,COLOR_BGR2GRAY);

    //创建ORB检测
    Ptr<ORB> detector_ORB = ORB::create();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置 计算 BRIEF 描述子
    detector_ORB->detectAndCompute(imgGray_1,Mat(),keypoints_1,descriptors_1);

    detector_ORB->detectAndCompute(imgGray_2,Mat(),keypoints_2,descriptors_2);


    //-- 第二步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第三步:匹配点对筛选
    double min_dist = 1000, max_dist = 0;

    for (int i = 0; i < descriptors_1.rows; i++)
    {
      double dist = match[i].distance;
      if (dist < min_dist) min_dist = dist;
      if (dist > max_dist) max_dist = dist;
    }

    cout<<"-- Max dist : "<< max_dist<<endl;
    cout<<"-- Min dist : "<< min_dist<<endl;


  for (int i = 0; i < descriptors_1.rows; i++)
  {
    if (match[i].distance <= 0.4*max_dist)
    {
      matches.push_back(match[i]);
    }
  }

}

void BASolver(vector<Point2f> &points_1,vector<Point2f> &points_2,Mat &K)
{
    //****************************BA优化过程*********************
    //构造求解器
    g2o::SparseOptimizer optimizer;
    //线性方程求解器
    //g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
    //矩阵块求解器
    g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3(linearSolver);
    //L-M优化算法
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(block_solver);
    //
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(true);

    //添加顶点就是添加优化的参数，这里位姿和特征点都要优化；
    //边实际上就是两个参数之间的关系，在这里是两者参数映射的关系
    //添加位姿顶点
    for(int i = 0;i<2;i++)
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap;
        v->setId(i);
        v->setFixed(i==0);
        v->setEstimate(g2o::SE3Quat());
        optimizer.addVertex(v);
    }

    //添加特征点顶点
    for(int i=0;i<points_1.size();i++)
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        Point2d cam = pixel2cam(points_1[i],K);
        v->setId(i+2); //已经添加过两个位姿的顶点了
        v->setEstimate(Eigen::Vector3d(cam.x,cam.y,1));
        v->setMarginalized(true);//把矩阵块分成两个部分，分别求解微量
        optimizer.addVertex(v);
    }

    //添加相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double>(0, 0),Eigen::Vector2d(K.at<double>(0, 2),K.at<double>(1, 2)),0);
    camera->setId(0);
    optimizer.addParameter(camera);

    //添加边,第一帧和第二帧
    for(size_t i = 0;i<points_1.size();i++)
    {

        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2)));
        edge->setVertex(1,dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(0)));
        edge->setMeasurement(Eigen::Vector2d(points_1[i].x,points_1[i].y));
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0,0);//这句必要
        optimizer.addEdge(edge);
    }

    for(size_t i = 0;i<points_2.size();i++)
    {

        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2)));
        edge->setVertex(1,dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(1)));
        edge->setMeasurement(Eigen::Vector2d(points_2[i].x,points_2[i].y));
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0,0);
        optimizer.addEdge(edge);
    }
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    //变换矩阵
    g2o::VertexSE3Expmap * v1 = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0)) ;
    g2o::VertexSE3Expmap * v2 = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1)) ;
    Eigen::Isometry3d pose1 = v1->estimate();
    Eigen::Isometry3d pose2 = v2->estimate();
    cout<<"The Pose1 from fram 1=\n"<<pose1.matrix()<<endl;
    cout<<"The Pose2 from fram 2(or the frame1 to frame2)=\n"<<pose2.matrix()<<endl;

    //****************************BA优化过程*********************
}
