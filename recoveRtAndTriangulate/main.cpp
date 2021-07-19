#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.h>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,vector<Point2f> &pixel_1,
                          std::vector<KeyPoint> &keypoints_2,vector<Point2f> &pixel_2,
                          std::vector<DMatch> &matches);

void triangulation(vector<Point2f> pixel_1,vector<Point2f> pixel_2,Mat R,Mat t,vector< Point3d >& points);

Eigen::Vector3d triangulatedByEigenSVD(Point2f pixel_1, Point2f pixel_2, Mat R, Mat t, Mat &K);

bool checkRt(Mat R, Mat t, vector<Point2f> pixel_1, vector<Point2f> pixel_2);
bool checkRt(Eigen::Matrix3d R_eigen, Eigen::Vector3d t_eigen, vector<Point2f> pixel_1, vector<Point2f> pixel_2);

//相机内参
 Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
 //Mat K = (Mat_<double>(3, 3) << 1145.74, 0, 282.77, 0, 1145.01, 251.35, 0, 0, 1);

inline Point3d pixel2camera( const Mat &K,const Point2d &p,const double depth = 1.0 )
{
  return Point3d
    (
      depth*(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      depth*(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1),
              depth
    );
}

inline Point2d camera2pixel(const Mat &K,const Point3d &p_cam) {
    return Point2d(
        p_cam.x * K.at<double>(0, 0) / p_cam.z + K.at<double>(0, 2),
        p_cam.y * K.at<double>(1, 1) / p_cam.z + K.at<double>(1, 2)
    );
}

int main(int argc,char**argv)
{
    if(argc <3)
    {
        cout<<"输入参数方法：**exe ./1.png ./2.png \n";
        return 0;
    }
    Mat img_1 = imread(argv[1],IMREAD_COLOR);
    Mat img_2 = imread(argv[2],IMREAD_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<Point2f> pixel_1,pixel_2;
    vector<DMatch> matches;

    // 获取特征点的像素坐标 pixel_1,pixel_2
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    find_feature_matches(img_1,img_2,keypoints_1,pixel_1,keypoints_2,pixel_2,matches);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1); //milliseconds 毫秒
    cout<<"匹配耗时:"<<delay_time.count()<<"秒"<<endl;


    //******下面有三种方法求解R t************

    //***********方法一：------------利用OpenCV从本质矩阵中分解R t----------------------------//
    t1 = chrono::steady_clock::now();
    Mat essential_matrix;
    int focalLength = K.at<double>(1,1);
    Point2d principal_point = Point2d(K.at<double>(0, 2),K.at<double>(1, 2));
    essential_matrix = findEssentialMat ( pixel_1, pixel_2, focalLength , principal_point);
    cout<<"本质矩阵：\n"<<essential_matrix<<endl;
    //-------利用OpenCV恢复运动------
    Mat R,t;
    recoverPose ( essential_matrix, pixel_1, pixel_2, R, t, focalLength, principal_point );
    vector<Point3d> points_1,points_2;


    //***********方法二：------------利用eigen SVD本质矩阵分解法求R t--------------------------//
    //    ----A constructed by U*S*VT------

    Eigen::Matrix3d A;
    cv::cv2eigen(essential_matrix,A);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto U = svd.matrixU();
    auto V = svd.matrixV();
    auto S = svd.singularValues();
    cout<<"A = \n"<<A<<endl;
    cout<<"S = \n"<<S<<endl;
    cout<<"U = \n"<<U<<endl;
    cout<<"VT = \n"<<V.transpose()<<endl;
    auto A_souce = U*S.asDiagonal()*V.transpose();
    cout<<"A constructed by U*S*VT is \n"<<A_souce<<endl;
    Eigen::Matrix3d Z,W;
    Z<<0,1,0,
       -1,0,0,
       0,0,0;
    W<<0,-1,0,
       1,0,0,
       0,0,1;
    //四种情况，需要利用深度信息（正负）来判断和筛选。
    Eigen::Matrix3d t_eigen_hat_1 = U*Z*U.transpose();              // 也可利用 U.col(2).transpose()
    Eigen::Matrix3d t_eigen_hat_2 = U*Z.transpose()*U.transpose();  // 也可利用 -U.col(2).transpose()
    Eigen::Matrix3d R_eigen_1 = U*W*V.transpose();
    Eigen::Matrix3d R_eigen_2 = U*W.transpose()*V.transpose();

    //----------转换为3X1的向量------
    Eigen::Vector3d t_eigen_1,t_eigen_2;
        t_eigen_1<<-t_eigen_hat_1(1,2),t_eigen_hat_1(0,2),-t_eigen_hat_1(0,1);
        t_eigen_2<<-t_eigen_hat_2(1,2),t_eigen_hat_2(0,2),-t_eigen_hat_2(0,1);

    //注意最后的输出需要对四种情况进行判断......
    bool index_1 = checkRt(R_eigen_1,t_eigen_1,pixel_1,pixel_2);
    bool index_2 = checkRt(R_eigen_1,t_eigen_2,pixel_1,pixel_2);
    bool index_3 = checkRt(R_eigen_2,t_eigen_1,pixel_1,pixel_2);
    bool index_4 = checkRt(R_eigen_2,t_eigen_2,pixel_1,pixel_2);
    cout<<"The R1 t1 is score:"<<index_1<<endl;
    cout<<"The R1 t2 is score:"<<index_2<<endl;
    cout<<"The R2 t1 is score:"<<index_3<<endl;
    cout<<"The R2 t2 is score:"<<index_4<<endl;

    cout<<"The best Rotation and translation is :\n";
    if(index_1 == 1)
    {
        cout<<"R_eigen_1:\n"<<R_eigen_1<<endl;
        cout<<"t_eigen_1:\n"<<t_eigen_1.transpose()<<endl;
    }
    else if(index_2 == 1)
    {
        cout<<"R_eigen_1:\n"<<R_eigen_1<<endl;
        cout<<"t_eigen_2:\n"<<t_eigen_2.transpose()<<endl;
    }
    else if(index_3 == 1)
    {
        cout<<"R_eigen_2:\n"<<R_eigen_2<<endl;
        cout<<"t_eigen_1:\n"<<t_eigen_1.transpose()<<endl;
    }
    else if(index_4 == 1)
    {
        cout<<"R_eigen_2:\n"<<R_eigen_2<<endl;
        cout<<"t_eigen_2:\n"<<t_eigen_2.transpose()<<endl;
    }


    //***********方法三：------------利用OpenCV从单应矩阵中分解R t-----------------------------//
    Mat homography_matrix;
    homography_matrix = findHomography ( pixel_1, pixel_2, RANSAC, 3 );
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;
    vector<cv::Mat> Rs, Ts;

    //单应性矩阵解出四种情况，需要利用深度信息（正负）来判断和筛选。
    decomposeHomographyMat(homography_matrix,K,Rs,Ts,noArray());

    //评分：...
    vector<bool> index_h;
    for(size_t i =0; i<Rs.size();i++)
    {
       index_h.push_back(checkRt(Rs[i],Ts[i],pixel_1,pixel_2));
       cout<<"The R"<<i<<" t"<<i<<" is score:"<<index_h[i]<<endl;
    }

    for(size_t i= 0; i<index_h.size();i++)
        if(index_h[i] == 1)
            cout<<"The best rotation from homography is:\n"<<Rs[i]<<endl<<"The translation is :\n"<<Ts[i]<<endl;


    //*******下面用两种方法实现三角测量*****
    //第一种利用OpenCV自带的函数得到
    //第二种利用Eigen SVD分解法求得

    //************************************三角测量:start*********************************//
    //opencv三角测量
    triangulation(pixel_1,pixel_2,R,t,points_1);//利用三角测量得到P1
    t2 = chrono::steady_clock::now();
    delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1); //milliseconds 毫秒
    cout<<"三角测量 OpenCV 耗时:"<<delay_time.count()<<"秒"<<endl;

    t2 = chrono::steady_clock::now();
    vector<Eigen::Vector3d> points_1_compare;
    for(auto i=0;i<pixel_1.size();i++)
        points_1_compare.push_back(triangulatedByEigenSVD(pixel_1[i],pixel_2[i],R,t,K));
    delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1); //milliseconds 毫秒
    cout<<"三角测量 Eigen SVD 耗时:"<<delay_time.count()<<"秒"<<endl;

    //--输出比较
    for(size_t i = 0;i<5;i++)
    {
        cout<<"opencv:   "<<points_1[i]<<endl<<"svd Eigen:"<<points_1_compare[i].transpose()<<endl;
        cout<<"----\n";
    }

    //求解相机坐标系 2 下的 路标点 P2
    //P2 = R_21*P1+t_21
    for(size_t i =0;i<points_1.size();i++)
    {
        Mat P1 = (Mat_<double>(3,1)<<points_1[i].x,points_1[i].y,points_1[i].z);
        Mat P2 = R*P1+t;
        points_2.push_back(Point3d(P2.at<double>(0,0),P2.at<double>(1,0),P2.at<double>(2,0)));
    }


    //**********计算欧拉角*************//
    Eigen::Matrix3d Rotation;
    Eigen::Vector3d translation;
    cv::cv2eigen(R,Rotation);
    cv::cv2eigen(t,translation);
    Sophus::SE3 T_21(Rotation,translation);

    cout<<"The T_21 matrix is :\n"<<T_21.matrix()<<endl;

    Eigen::AngleAxisd tf(T_21.rotation_matrix());
    cout<<"Rotation matrix R_21 is :\n"<<tf.matrix()<<endl;
    cout<<"Angles of rotate about axis: z y x (°): \n"<<tf.matrix().eulerAngles(2,1,0).transpose()*180/M_PI<<endl; //tf.matrix().eulerAngles(2,1,0)


    //***********************************输出匹配图3s*************************************//
    Mat img_keypoints;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_keypoints);
    imshow("matches",img_keypoints);
    waitKey(4000);
    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1, vector<Point2f> &pixel_1,
                          std::vector<KeyPoint> &keypoints_2, vector<Point2f> &pixel_2,
                          std::vector<DMatch> &matches) {

    Mat descriptors_1, descriptors_2;

    Ptr<Feature2D> detector_ORB = ORB::create();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    detector_ORB->detectAndCompute(img_1,Mat(),keypoints_1,descriptors_1);
    detector_ORB->detectAndCompute(img_2,Mat(),keypoints_2,descriptors_2);

    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    double min_dist = 10000, max_dist = 0;

    for (int i = 0; i < descriptors_1.rows; i++)
    {
      double dist = match[i].distance;
      if (dist < min_dist) min_dist = dist;
      if (dist > max_dist) max_dist = dist;
    }

  for (int i = 0; i < descriptors_1.rows; i++)
  {
    if (match[i].distance <= 0.4*max_dist)
    {
      matches.push_back(match[i]);
    }
  }


  //good matches
  for(auto m:matches)
  {
     pixel_1.push_back(keypoints_1[m.queryIdx].pt);
     pixel_2.push_back(keypoints_2[m.trainIdx].pt);
  }
}

void triangulation(vector<Point2f> pixel_1,vector<Point2f> pixel_2,Mat R,Mat t,vector< Point3d >& points)
{
    Mat projMatr1 = (Mat_<float> (3,4) <<
            1,0,0,0,
            0,1,0,0,
            0,0,1,0);
    Mat projMatr2 = (Mat_<float> (3,4) <<
            R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
        );

    vector<Point2f> projPoints1,projPoints2;
    for(size_t i =0;i<pixel_1.size();i++)
    {
        Point3d camera_1 = pixel2camera(K,pixel_1[i],1.0);
        Point3d camera_2 = pixel2camera(K,pixel_2[i],1.0);
        projPoints1.push_back(Point2f(camera_1.x,camera_1.y));
        projPoints2.push_back(Point2f(camera_2.x,camera_2.y));
    }

    Mat points4D;
    cv::triangulatePoints( projMatr1, projMatr2, projPoints1, projPoints2, points4D );

    // 转换成非齐次坐标
    for ( int i=0; i<points4D.cols; i++ )
    {
        Mat x = points4D.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3d p (
            x.at<float>(0,0),
            x.at<float>(1,0),
            x.at<float>(2,0)
        );
        points.push_back( p );
    }
}

Eigen::Vector3d triangulatedByEigenSVD(Point2f pixel_1,Point2f pixel_2,Mat R,Mat t,Mat &K)
{
    Eigen::Matrix3d K_eigen;
    cv::cv2eigen(K,K_eigen);//eigen 类型的K内参

    Eigen::Matrix<double,3,4> T_1,T_21;
    T_1<<   1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0;

    T_21<<  R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0);

    Eigen::Matrix<double,3,4> ProjectMatrix_1,ProjectMatrix_2;

    ProjectMatrix_1 = K_eigen * T_1;
    ProjectMatrix_2 = K_eigen * T_21;

    Eigen::Matrix4d A;
    A.row(0) = ProjectMatrix_1.row(2)*pixel_1.x - ProjectMatrix_1.row(0);
    A.row(1) = ProjectMatrix_1.row(2)*pixel_1.y - ProjectMatrix_1.row(1);
    A.row(2) = ProjectMatrix_2.row(2)*pixel_2.x - ProjectMatrix_2.row(0);
    A.row(3) = ProjectMatrix_2.row(2)*pixel_2.y - ProjectMatrix_2.row(1);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix4d U = svd.matrixU();
    Eigen::Matrix4d V = svd.matrixV();
    Eigen::Vector4d S = svd.singularValues();

    Eigen::Vector4d p_w = V.col(3);
    p_w /= p_w(3,0);

    return Eigen::Vector3d(p_w(0),p_w(1),p_w(2));
}

bool checkRt(Mat R,Mat t,vector<Point2f> pixel_1,vector<Point2f> pixel_2)
{
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R,R_eigen);
    cv::cv2eigen(t,t_eigen);
    vector<Eigen::Vector3d> P_1,P_2;
    for(auto i=0;i<pixel_1.size();i++)
    {
        P_1.push_back(triangulatedByEigenSVD(pixel_1[i],pixel_2[i],R,t,K));
        //P2 = RP1 + t
        P_2.push_back(R_eigen * P_1[i] + t_eigen);
    }

    int good_num_p1 = 0,good_num_p2 = 0;
    for(auto ptr_1:P_1)
    {
        if (ptr_1(2,0) > 0)
        {
            good_num_p1++;
        }
    }

    for(auto ptr_2:P_2)
    {
        if (ptr_2(2,0) > 0)
        {
            good_num_p2++;
        }
    }

    //设定阈值
    double posibility = (good_num_p1/(double)P_1.size())*(good_num_p2/(double)P_2.size());
    if(posibility > 0.8)
        return true;
    else
        return false;
}

bool checkRt(Eigen::Matrix3d R_eigen, Eigen::Vector3d t_eigen, vector<Point2f> pixel_1, vector<Point2f> pixel_2)
{
    Mat R,t;
    cv::eigen2cv(R_eigen,R);
    cv::eigen2cv(t_eigen,t);
    vector<Eigen::Vector3d> P_1,P_2;
    for(auto i=0;i<pixel_1.size();i++)
    {
        P_1.push_back(triangulatedByEigenSVD(pixel_1[i],pixel_2[i],R,t,K));
        //P2 = RP1 + t
        P_2.push_back(R_eigen * P_1[i] + t_eigen);
    }

    int good_num_p1 = 0,good_num_p2 = 0;
    for(auto ptr_1:P_1)
    {
        if (ptr_1(2,0) > 0)
        {
            good_num_p1++;
        }
    }

    for(auto ptr_2:P_2)
    {
        if (ptr_2(2,0) > 0)
        {
            good_num_p2++;
        }
    }

    //cout<<good_num_p1<<" "<<good_num_p2;

    //设定阈值
    double posibility = (good_num_p1/(double)P_1.size())*(good_num_p2/(double)P_2.size());
    if(posibility > 0.7)
        return true;
    else
        return false;
}
