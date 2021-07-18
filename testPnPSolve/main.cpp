

/*
 * 本程序不对特征点的位置优化，只考虑优化相机的位姿
 * 作者：zyl
 */


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
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Core>

#include <sophus/se3.h>

#include <vector>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches);

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

void solvePnPWithG2o(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts_2d_eigen,
              Mat &K,Sophus::SE3 &pose);

void BASolver(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts_2d_eigen,
              Mat &K,Sophus::SE3 &pose);


int main(int argc,char**argv)
{
    if(argc <4)
    {
        cout<<"输入参数方法：**/testOptimizerG2o ./1.png ./2.png ./1_depth.png\n";
        return 0;
    }
    Mat img_1 = imread(argv[1],IMREAD_COLOR);
    Mat img_2 = imread(argv[2],IMREAD_COLOR);
    //深度图加载
    Mat depth_1 = imread(argv[3], IMREAD_UNCHANGED);

    vector<KeyPoint> keypoints_1, keypoints_2;
    //匹配得到的路标点
    vector<Point2f> points_1,points_2;
    vector<Point3f> points_1_3d;
    vector<DMatch> matches;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1); //milliseconds 毫秒
    cout<<"匹配耗时:"<<delay_time.count()<<"秒"<<endl;

    for(auto m:matches)
    {
       ushort d = depth_1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
       if (d == 0)   // 去除差的深度点
              continue;
       float dd = d / 5000.0;
       Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
       points_1_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));

       points_1.push_back(keypoints_1[m.queryIdx].pt);
       points_2.push_back(keypoints_2[m.trainIdx].pt);
    }

    //转化为Eigen,传入到G2o中
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pts_3d_eigen;
    vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts_2d_eigen;
    for (size_t i = 0; i < points_1_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(points_1_3d[i].x, points_1_3d[i].y, points_1_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(points_2[i].x, points_2[i].y));
      }
    cout<<"特征点的数量："<<pts_3d_eigen.size()<<"  "<<pts_2d_eigen.size()<<endl;


    //*******************************G2o优化--->EdgeSE3ProjectXYZOnlyPose*******************************/
    t1 = chrono::steady_clock::now();
    Sophus::SE3 pose;
    solvePnPWithG2o(pts_3d_eigen,pts_2d_eigen,K,pose);
    t2 = chrono::steady_clock::now();
    delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by using g2o(EdgeSE3ProjectXYZOnlyPose) time_cost: " << delay_time.count() << " seconds." << endl;
    cout << "pose estimated by g2o(EdgeSE3ProjectXYZOnlyPose) =\n" << pose.matrix() << endl;

    //*******************************G2o优化--->EdgeProjectXYZ2UV*******************************/
    t1 = chrono::steady_clock::now();
    BASolver(pts_3d_eigen,pts_2d_eigen,K,pose);
    t2 = chrono::steady_clock::now();
    delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by using g2o(EdgeProjectXYZ2UV) time_cost: " << delay_time.count() << " seconds." << endl;
    cout << "pose estimated by g2o(EdgeProjectXYZ2UV) =\n" << pose.matrix() << endl;

    //********************************opencv优化***************************/
    t1 = chrono::steady_clock::now();
    Mat r, t, R;
    cv::solvePnP(points_1_3d, points_2, K, Mat(), r, t, false);     // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::Rodrigues(r, R);                                            // r为旋转向量形式，用Rodrigues公式转换为矩阵
    t2 = chrono::steady_clock::now();
    delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp in opencv time_cost: " << delay_time.count() << " seconds." << endl;
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R,R_eigen);
    cv::cv2eigen(t,t_eigen);
    pose = Sophus::SE3(R_eigen,t_eigen);
    cout << "pose estimated by opencv =\n" << pose.matrix() << endl;

    //*********输出匹配图3s*************//
    Mat img_keypoints;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_keypoints);
    imshow("matches",img_keypoints);
    waitKey(3000);
    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {

    //-- 初始化
    Mat descriptors_1, descriptors_2;

    //创建ORB检测
    Ptr<Feature2D> detector_ORB = ORB::create();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置 计算 BRIEF 描述子
    detector_ORB->detect(img_1,keypoints_1);
    detector_ORB->detect(img_2,keypoints_2);

    detector_ORB->compute(img_1,keypoints_1,descriptors_1);
    detector_ORB->compute(img_2,keypoints_2,descriptors_2);


    //-- 第二步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第三步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

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

void solvePnPWithG2o(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts_2d_eigen,
              Mat &K,Sophus::SE3 &pose)
{
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    //构造求解器
    g2o::SparseOptimizer optimizer;
    //线性方程求解器
    //g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
    //矩阵块求解器
    g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3(linearSolver);
    //L-M优化算法
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(block_solver);
    //
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(true);

    //顶点
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(g2o::SE3Quat());
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    //边
    for(size_t i = 0;i<pts_2d_eigen.size();i++)
    {
        g2o::EdgeSE3ProjectXYZOnlyPose* edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setMeasurement(pts_2d_eigen[i]);
        edge->fx = fx;
        edge->fy = fy;
        edge->cx = cx;
        edge->cy = cy;
        edge->Xw = Eigen::Vector3d(pts_3d_eigen[i][0],pts_3d_eigen[i][1],pts_3d_eigen[i][2]);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        optimizer.addEdge(edge);
    }

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    pose = Sophus::SE3(vSE3->estimate().rotation(),vSE3->estimate().translation());
}

void BASolver(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts_2d_eigen,
              Mat &K,Sophus::SE3 &pose)
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


    //添加位姿顶点

    g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap;
    v->setId(0);
    v->setFixed(false);
    v->setEstimate(g2o::SE3Quat());
    optimizer.addVertex(v);

    //添加特征点顶点
    for(int i=1;i<pts_3d_eigen.size();i++)
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId(i); //已经添加过两个位姿的顶点了
        v->setEstimate(pts_3d_eigen[i]);
        v->setFixed(true); //固定，不优化
        v->setMarginalized(true);//把矩阵块分成两个部分，分别求解微量
        optimizer.addVertex(v);
    }

    //添加相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double>(0, 0),Eigen::Vector2d(K.at<double>(0, 2),K.at<double>(1, 2)),0);
    camera->setId(0);
    optimizer.addParameter(camera);

    //添加边,第一帧和第二帧
    for(size_t i = 1;i<pts_3d_eigen.size();i++)
    {

        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i)));
        edge->setVertex(1,v);
        edge->setMeasurement(pts_2d_eigen[i]);
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0,0);//这句必要
        optimizer.addEdge(edge);
    }

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    pose = Sophus::SE3(v->estimate().rotation(),v->estimate().translation());

    //****************************BA优化过程*********************
}
