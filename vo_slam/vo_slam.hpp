//g2o
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

#include <g2o/types/sim3/sim3.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3quat.h>
//opencv
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
//eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//pangolin
#include <pangolin/pangolin.h>
#include <sophus/se3.h>
//boost
#include <boost/timer.hpp>
//std
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <fstream>
#include <omp.h>

using namespace std;
using namespace cv;

class camera
{
public:
    typedef shared_ptr<camera> Ptr;
    camera(){}

    static cv::FileStorage configfile;
    static cv::Mat K;
    static Eigen::Matrix3d K_Eigen;
    static cv::Mat distCoeffs;

    static void loadYaml(std::string path);
    static void undistort(cv::Mat &frame,cv::Mat &output);

    static Point3f pixel2cam(const Point2d &p, const Mat &K);
    static Point2f cam2pixel(const Point3f &p, const Mat &K);
    static Eigen::Vector3d ToEulerAngles(Eigen::Quaterniond q);

};
cv::FileStorage camera::configfile = cv::FileStorage();
cv::Mat camera::K = (cv::Mat_<double>(3, 3) << 517.3, 0, 325.1, 0, 516.5, 249.7, 0, 0, 1);
//K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);//fr2
Eigen::Matrix3d camera::K_Eigen = Eigen::Matrix3d::Identity(3,3);
cv::Mat camera::distCoeffs = cv::Mat::zeros(5,1,CV_64F);

class Frame
{
public:
    Frame(cv::Mat img,cv::Mat depth);
    inline void detectKeyPoints();
    inline void computeDescriptors();
    inline void getkeyPoints_3d_cam();
    Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat frame;
    cv::Mat depth_;
    std::vector<cv::Point3f> pts_3d; //相机坐标系下的点P_c,P_w = T_wc*P_c
    static cv::Ptr<ORB> detector_ORB;//
    static cv::FlannBasedMatcher matcher;// = cv::flann::LshIndexParams(5,10,2);// = DescriptorMatcher::create("BruteForce-Hamming")
};

class Map
{
public:
    Map(){}
    typedef std::shared_ptr<Map> Ptr; //定义智能指针，简化了实例化过程
    void addKeyFrames(Frame frame);
    void addKeyPoses(Sophus::SE3 frame_pose_Tcw);
    void add3Dpts_world(std::vector<cv::Point3f> pts_3d_Pc);

    unordered_map<unsigned long,vector<cv::Point3f> > map_points;
    vector<Frame> keyFrames;
    vector<Sophus::SE3> KeyPoses;

    Sophus::SE3 pose_cur_Tcw;
};

class visualMap
{
public:
    typedef std::shared_ptr<visualMap> Ptr;
    visualMap();
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;

    const float w = 0.06;
    const float h = w*0.75;
    const float z = w*0.6;
    const bool drawline = true;
    const long int drawGaps = 5*100;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> keyposes;
    //画当前位姿
    void drawCurrentFrame(Eigen::Isometry3d poses);
    //画关键帧
    void drawKeyFrame(std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> keyframs);
    //画动态坐标轴
    void drawAxis(Eigen::Isometry3d poses);
    //画原点坐标系
    void drawCoordinate(float scale);
    //画轨迹线
    void drawLine(size_t i, vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses, bool drawLine);
    //求解移动速度
    void getVelocity(Eigen::Isometry3d &pose_last,Eigen::Isometry3d &pose_next,double &time_used,Eigen::Vector4d &trans_velocity,Eigen::Vector3d &angluar_velocity);

    void drawing(Eigen::Isometry3d pose);
    void drawPoints(vector<cv::Point3f> &points_world);
};


class VO_slam
{
public:
    typedef shared_ptr<VO_slam> Ptr;
    enum VOstate{
        INITIALIZING = 0,
        OK = 1,
        LOST = -1
    };
    VOstate state_;

    VO_slam();

    void tracking(Mat img,Mat depth);

    void featureMatch(Mat descriptors_ref,Mat descriptors_cur);

    void setPts3d_refAndPts2d_cur(Frame frame_ref, Frame frame_cur,
                           vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &pts_3d_eigen,
                           vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &pts_2d_eigen);

    void solvePnP_ProjectToPose(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,
                                vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &pts_2d_eigen,
                                Mat &K,Sophus::SE3 &pose);
    void solvePnP_BA(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,
                     vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &pts_2d_eigen,
                     Mat &K,Sophus::SE3 &pose);
    void solvePnP_opencv(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,
                         vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &pts_2d_eigen,
                         Mat &K,Sophus::SE3 &pose);

    Sophus::SE3 pnpSolver(Frame frame_ref, Frame frame_cur);
    bool checkEstimatedPose(Sophus::SE3 &pose);
    bool checkKeyFrame(Sophus::SE3 &pose);

    Map::Ptr map_;
    visualMap::Ptr visual_;
    vector<DMatch> featureMatches;
    double pose_min_trans;  //keyFrame 和 pose的筛选机制差不多，只是pose的为了抑制较大的误差，条件较小，keyFrame 尽量位移大一些；
    double pose_max_trans;
    double keyFrame_min_trans;
    double KeyFrame_max_trans;

};

cv::FlannBasedMatcher Frame::matcher = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(5,10,2));
cv::Ptr<ORB> Frame::detector_ORB = ORB::create(800,1.2,4);

void camera::loadYaml(std::string path)
{
    configfile.open(path,cv::FileStorage::READ);
    if(!configfile.isOpened())
    {
        cout<<"please confirm the path of yaml!\n";
        return ;
    }

    double fx = configfile["Camera.fx"];
    double fy = configfile["Camera.fy"];
    double cx = configfile["Camera.cx"];
    double cy = configfile["Camera.cy"];

    double k1 = configfile["Camera.k1"];
    double k2 = configfile["Camera.k2"];
    double p1 = configfile["Camera.p1"];
    double p2 = configfile["Camera.p2"];
    double k3 = configfile["Camera.k3"];

    K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    K_Eigen<<fx,   0,   cx,
              0,  fy,   cy,
              0,   0,    1;

    distCoeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
}

inline void camera::undistort(cv::Mat &frame,cv::Mat &output)
{
    cv::undistort(frame ,output ,camera::K,camera::distCoeffs);
}

Point3f camera::pixel2cam(const Point2d &p, const Mat &K)
{
  return Point3f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1),
              1
    );
}

Point2f camera::cam2pixel(const Point3f &p, const Mat &K)
{
    float d = p.z;
    return Point2f( K.at<double>(0, 0) * p.x / d + K.at<double>(0, 2),
                    K.at<double>(1, 1) * p.x / d + K.at<double>(1, 2));
}

Eigen::Vector3d camera::ToEulerAngles(Eigen::Quaterniond q) {
    ///
    /// roll = atan2( 2(q0*q1+q2*q3),1-2(q1^2 + q2^2))
    /// pitch = asin(2(q0*q2 - q3*q1))
    /// yaw = atan2(2(q0*q3+q1*q2),1-2(q2^2 + q3^2))
    ///
    double x,y,z;//roll pitch yaw

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
    double cosr_cosp = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
    x = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (q.w() * q.y() - q.z() * q.x());
    if (std::abs(sinp) >= 1)
        y = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        y = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
    double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
    z = std::atan2(siny_cosp, cosy_cosp);

    return Eigen::Vector3d(x,y,z);
}

VO_slam::VO_slam():state_(VOstate::INITIALIZING),pose_max_trans(0.2),pose_min_trans(0.003),keyFrame_min_trans(0.006),KeyFrame_max_trans(0.2)
{
    map_ = std::make_shared<Map>();
    visual_ = std::make_shared<visualMap>();
    omp_set_num_threads(15);
}

void VO_slam::tracking(Mat img, Mat depth)
{
    Frame frame_cur(img,depth);
    frame_cur.detectKeyPoints();
    frame_cur.getkeyPoints_3d_cam();
    frame_cur.computeDescriptors();
    if(state_ != VOstate::OK)
    {
        map_->addKeyFrames(frame_cur);
        map_->addKeyPoses(Sophus::SE3());
        map_->add3Dpts_world(frame_cur.pts_3d);
        state_ = VOstate::OK;
        return;
    }
    //ref frame
    Frame frame_ref = map_->keyFrames[map_->keyFrames.size()-1];
    //pose estimate
    Sophus::SE3 pose = pnpSolver(frame_ref,frame_cur);

    if(checkEstimatedPose(pose)== true)
    {
        //updata current pose
        map_->pose_cur_Tcw = pose*map_->pose_cur_Tcw;
        if(checkKeyFrame(pose) == true)
        {
            map_->addKeyFrames(frame_cur);
            map_->addKeyPoses(map_->pose_cur_Tcw);
            map_->add3Dpts_world(frame_cur.pts_3d);
            visual_->keyposes.push_back(Eigen::Isometry3d(map_->pose_cur_Tcw.matrix().inverse()));
        }
    }
    else
    {
        return;
    }

//    map_->map_points.find((unsigned long)(map_->KeyPoses.size()-1));
    visual_->drawing(Eigen::Isometry3d(map_->pose_cur_Tcw.matrix().inverse()));
    vector<cv::Point3f> input_pts = map_->map_points[(unsigned long)(map_->KeyPoses.size()-1)];
    visual_->drawPoints(input_pts);
    pangolin::FinishFrame();
}

void VO_slam::setPts3d_refAndPts2d_cur(Frame frame_ref,Frame frame_cur,
                       vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,
                       vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &pts_2d_eigen)
{
    for(auto m:featureMatches)
    {
        cv::Point3f ref_pts3d = frame_ref.pts_3d[m.queryIdx];
        cv::Point2f cur_pts2d = frame_cur.keypoints[m.trainIdx].pt;
        if(ref_pts3d.z>0)
        {
            pts_3d_eigen.push_back(Eigen::Vector3d(ref_pts3d.x,ref_pts3d.y,ref_pts3d.z));
            pts_2d_eigen.push_back(Eigen::Vector2d(cur_pts2d.x,cur_pts2d.y));
        }
    }
}

void VO_slam::solvePnP_ProjectToPose(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,
                                     vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &pts_2d_eigen,
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
    optimizer.optimize(25);

    pose = Sophus::SE3(vSE3->estimate().rotation(),vSE3->estimate().translation());
}

void VO_slam::solvePnP_BA(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,
                          vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &pts_2d_eigen,
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
        edge->setVertex(1,dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(0)));
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

void VO_slam::solvePnP_opencv(vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts_3d_eigen,
                              vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &pts_2d_eigen,
                              Mat &K,
                              Sophus::SE3 &pose)
{
    Mat r, t, R;
    vector<Point2f> pts_2d_cur;
    vector<Point3f> pts_3d_ref;

//    #pragma omp parallel for
    for (size_t i = 0; i < pts_3d_eigen.size(); ++i) {
        pts_3d_ref.push_back(cv::Point3f(pts_3d_eigen[i][0], pts_3d_eigen[i][1], pts_3d_eigen[i][2]));
        pts_2d_cur.push_back(cv::Point2f(pts_2d_eigen[i][0], pts_2d_eigen[i][1]));
      }

    cv::solvePnPRansac(pts_3d_ref, pts_2d_cur, K, Mat(), r, t, false, 100, 4.0, 0.99);     // 调用OpenCV 的 PnPRansac 求解，可选择EPNP，DLS等方法
    cv::Rodrigues(r, R);                                            // r为旋转向量形式，用Rodrigues公式转换为矩阵
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R,R_eigen);
    cv::cv2eigen(t,t_eigen);
    pose = Sophus::SE3(R_eigen,t_eigen);
}

bool VO_slam::checkEstimatedPose(Sophus::SE3 &pose)
{
    if(pose.translation().norm()<pose_min_trans || pose.translation().norm()>pose_max_trans)
    {
        return false;
    }
    return true;
}

bool VO_slam::checkKeyFrame(Sophus::SE3 &pose)
{
    if(pose.translation().norm()<keyFrame_min_trans || pose.translation().norm()>KeyFrame_max_trans)
    {
        return false;
    }
    return true;
}

void VO_slam::featureMatch(Mat descriptors_ref,Mat descriptors_cur)
{
    vector<DMatch> match;
    featureMatches.clear();
    // BFMatcher matcher ( NORM_HAMMING );
    Frame::matcher.match(descriptors_ref, descriptors_cur, match);

    double min_dist = 10000, max_dist = 0;

    for (int i = 0; i < match.size(); i++)
    {
      double dist = match[i].distance;
      if (dist < min_dist) min_dist = dist;
      if (dist > max_dist) max_dist = dist;
    }

  for (int i = 0; i < match.size(); i++)
  {
    if (match[i].distance <=  max<float> ( min_dist*2, 30.0 ))
    {
      featureMatches.push_back(match[i]);
    }
  }
}

Sophus::SE3 VO_slam::pnpSolver(Frame frame_ref, Frame frame_cur)
{
    featureMatch(frame_ref.descriptors,frame_cur.descriptors);

    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pts_3d_eigen;
    vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts_2d_eigen;
        setPts3d_refAndPts2d_cur(frame_ref,frame_cur,pts_3d_eigen,pts_2d_eigen);

    Sophus::SE3 pose;
    //solvePnP_opencv(pts_3d_eigen,pts_2d_eigen,K,pose);
    //solvePnP_BA(pts_3d_eigen,pts_2d_eigen,K,pose);
    solvePnP_ProjectToPose(pts_3d_eigen,pts_2d_eigen,camera::K,pose);
    return pose;
}

//**********************************visual************************************//
visualMap::visualMap()
{
    {
            //---------------------------------------------------------------map init----------------------------------------------------//
            pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
            glEnable(GL_DEPTH_TEST);//深度测试
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            s_cam = pangolin::OpenGlRenderState(                                                  //摆放一个相机
                    pangolin::ProjectionMatrix(1024, 768, 500, 500, 800, 400, 0.1, 1000),
                    pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, pangolin::AxisZ)
            );
            d_cam = pangolin::CreateDisplay()//创建一个窗口
                    .SetBounds(0.0, 1.0, 0, 1.0, -1024.0f / 768.0f)
                    .SetHandler(new pangolin::Handler3D(s_cam));
        }
}

void visualMap::drawAxis(Eigen::Isometry3d poses)
{
    //画出坐标轴
    Eigen::Vector3d Ow = poses.translation();
    Eigen::Vector3d Xw = poses * (0.1 * Eigen::Vector3d(1, 0, 0));
    Eigen::Vector3d Yw = poses * (0.1 * Eigen::Vector3d(0, 1, 0));
    Eigen::Vector3d Zw = poses * (0.1 * Eigen::Vector3d(0, 0, 1));
    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(Xw[0], Xw[1], Xw[2]);
    glColor3f(0.0, 1.0, 0.0);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(Yw[0], Yw[1], Yw[2]);
    glColor3f(0.0, 0.0, 1.0);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(Zw[0], Zw[1], Zw[2]);
    glEnd();
}

void visualMap::getVelocity(Eigen::Isometry3d &pose_last,Eigen::Isometry3d &pose_next,double &time_used,Eigen::Vector4d &trans_velocity,Eigen::Vector3d &angluar_velocity)
{
    //平移速度 x y z v_r(合速度)
    double dx = pose_next.translation()[0]-pose_last.translation()[0];
    double dy = pose_next.translation()[1]-pose_last.translation()[1];
    double dz = pose_next.translation()[2]-pose_last.translation()[2];
    double distance_ = sqrt(dx*dx+dy*dy+dz*dz);
    trans_velocity <<dx/time_used,dy/time_used,dz/time_used,distance_/time_used;

    //角速度：绕 z y x--->x y z
    Eigen::AngleAxisd rotation_vector_last(pose_last.rotation());
    Eigen::AngleAxisd rotation_vector_next(pose_next.rotation());
    Eigen::Vector3d dtheta_zyx = camera::ToEulerAngles(Eigen::Quaterniond(rotation_vector_next.matrix())) - camera::ToEulerAngles(Eigen::Quaterniond(rotation_vector_last.matrix()));
    Eigen::Vector3d angluar_zyx = dtheta_zyx/time_used;
    angluar_velocity <<angluar_zyx[2],angluar_zyx[1],angluar_zyx[0];
}

void visualMap::drawCurrentFrame(Eigen::Isometry3d poses)
{
    //变换位姿
    glPushMatrix();
    pangolin::OpenGlMatrix Twc_(poses.matrix());
    glMultMatrixd(Twc_.m);

    glColor3f(229/255.f, 113/255.f, 8/255.f);
    glLineWidth(2);
    glBegin(GL_LINES);
    //画相机模型
    glVertex3f(0, 0, 0);
    glVertex3f(w,h,z);
    glVertex3f(0, 0, 0);
    glVertex3f(w,-h,z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w,-h,z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);

    glEnd();
    glPopMatrix();
}

void visualMap::drawLine(size_t i,vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses,bool drawLine)
{
    glLineWidth(2);
    if(drawLine)
    {
      for (size_t j = 1; j < i; j++) {
          glColor3f(1.0, 0.0, 0.0);
          glBegin(GL_LINES);
          Eigen::Isometry3d p1 = poses[j-1], p2 = poses[j];
          glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
          glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
          glEnd();
          }
    }
}

void visualMap::drawKeyFrame(std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> keyframs)
{
    for(auto Twc:keyframs)
    {
        glPushMatrix();
        pangolin::OpenGlMatrix Twc_(Twc.matrix());
        glMultMatrixd(Twc_.m);

        glLineWidth(2);
        glColor3f(20/255.f, 68/255.f, 106/255.f);
        glBegin(GL_LINES);
        //画相机模型
        glVertex3f(0, 0, 0);
        glVertex3f(w,h,z);
        glVertex3f(0, 0, 0);
        glVertex3f(w,-h,z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w,-h,z);
        glVertex3f(0, 0, 0);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);
        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);
        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);
        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);

        glEnd();
        glPopMatrix();
    }
}

void visualMap::drawCoordinate(float scale)
{
    glLineWidth(3);
    glBegin(GL_LINES);
    //x
    glColor3f(1.0, 0.0, 0.0);
    glVertex3d(0,0,0);
    glVertex3d(scale,0,0);
    //y
    glColor3f(0.0, 1.0, 0.0);
    glVertex3d(0,0,0);
    glVertex3d(0,scale,0);
    //z
    glColor3f(0.0, 0.0, 1.0);
    glVertex3d(0,0,0);
    glVertex3d(0,0,scale);
    glEnd();
}


void visualMap::drawing(Eigen::Isometry3d pose)
{

      if(pangolin::ShouldQuit()==false)
      {
          //消除颜色缓冲
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          d_cam.Activate(s_cam);
          glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

          //画相机模型
          drawCurrentFrame(pose);
          //画出动态坐标轴
          drawAxis(pose);
          //画坐标系
          drawCoordinate(0.5);
          //绘制关键帧
          drawKeyFrame(keyposes);
      }
}

void visualMap::drawPoints(vector<Point3f> &points_world)
{
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_POINTS);
//    #pragma omp parallel for
    for(int i=0;i<points_world.size();i++ )
    {
        glVertex3d(points_world[i].x,points_world[i].y,points_world[i].z);
    }
    glEnd(); //加上glEnd()可以加快绘点的速度
}

//*****************************Frame****************************
Frame::Frame(Mat img, Mat depth)
{
    frame=img.clone();
    depth_ = depth.clone();
}

inline void Frame::detectKeyPoints()
{
    detector_ORB->detect(frame,keypoints);
}

inline void Frame::computeDescriptors()
{
    detector_ORB->compute(frame,keypoints,descriptors);
}

inline void Frame::getkeyPoints_3d_cam()
{
    for(auto keypoint:keypoints)
    {

        int x = cvRound(keypoint.pt.x);
        int y = cvRound(keypoint.pt.y);
        double dd=0;
        ushort d = depth_.ptr<ushort>(y)[x];
        if ( d!=0 )
        {
            dd = double(d) / 5000.0;
        }
        else
        {
            // check the nearby points
            int dx[4] = {-1,0,1,0};
            int dy[4] = {0,-1,0,1};
            for ( int i=0; i<4; i++ )
            {
                d = depth_.ptr<ushort>( y+dy[i] )[x+dx[i]];
                if ( d!=0 )
                {
                    dd = double(d) / 5000.0;
                }
            }
        }
        Point3f p1 = camera::pixel2cam(keypoint.pt, camera::K);
        this->pts_3d.push_back(dd*p1);
    }
}

//*****************************Map****************************
void Map::addKeyFrames(Frame frame)
{
    keyFrames.push_back(frame);
}

void Map::addKeyPoses(Sophus::SE3 frame_pose_Tcw)
{
    Sophus::SE3 T_wc = frame_pose_Tcw.inverse();
    KeyPoses.push_back(T_wc); //camera to world
}

void Map::add3Dpts_world(std::vector<Point3f> pts_3d_Pc)
{
    vector<cv::Point3f> contains;
    for(auto Pc:pts_3d_Pc)
    {
        if(Pc.z == 0)
            continue;
        Eigen::Vector4d Pc3d_(Pc.x,Pc.y,Pc.z,1);
        Eigen::Vector4d Pw3d_ = KeyPoses[KeyPoses.size()-1].matrix() * Pc3d_; //T_wc * Pc = Pw
        contains.push_back(cv::Point3f(Pw3d_[0],Pw3d_[1],Pw3d_[2]));
    }
    map_points.insert(make_pair((unsigned long)(KeyPoses.size()-1),contains));

}


