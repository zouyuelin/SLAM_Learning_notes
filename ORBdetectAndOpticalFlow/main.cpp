#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video.hpp"

#include <vector>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches_CPU(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches);
Mat OpticalFlow(Mat img_1,Mat img_2);

using namespace std;

int main(int argc,char**argv)
{
    std::string path1 = argv[1];
    std::string path2 = argv[2];
    if(argc !=3)
    {
        cout<<"输入参数方法：**/ORBdetect ./1.png ./2.png\n";
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

    Mat img_keypoints;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_keypoints);
    imshow("matches",img_keypoints);
    waitKey(0);

    //OpticalFlow
    Mat rgb;
    rgb = OpticalFlow(img_1,img_2);
    cv::imshow("opticalFlow",rgb);
    cv::waitKey(0);

    return 0;
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
    medianBlur(img_1,img_1,5);
    medianBlur(img_2,img_2,5);

    //创建ORB检测
    Ptr<ORB> detector_ORB = ORB::create();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置 计算 BRIEF 描述子
    detector_ORB->detectAndCompute(imgGray_1,Mat(),keypoints_1,descriptors_1);

    detector_ORB->detectAndCompute(imgGray_2,Mat(),keypoints_2,descriptors_2);


    //-- 第二步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    //FlannBasedMatcher flannMatcher;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);
    //flannMatcher.match(descriptors_1, descriptors_2, match);


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
    if (match[i].distance <= 0.6*max_dist)
    {
      matches.push_back(match[i]);
    }
  }

}

Mat OpticalFlow(Mat img_1,Mat img_2)
{
    cv::Ptr<DenseOpticalFlow> algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
    Mat flow, flow_uv[2];
    Mat mag, ang, rgb;
    Mat hsv_split[3], hsv;
    Mat imgGray_1,imgGray_2;
    cvtColor(img_1,imgGray_1,COLOR_BGR2GRAY);
    cvtColor(img_2,imgGray_2,COLOR_BGR2GRAY);

    algorithm->calc(imgGray_1,imgGray_2,flow);
    split(flow, flow_uv);
    multiply(flow_uv[1], -1, flow_uv[1]);
    cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    hsv_split[0] = ang;
    hsv_split[1] = mag;
    hsv_split[2] = Mat::ones(ang.size(), ang.type());
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    return rgb;
}
