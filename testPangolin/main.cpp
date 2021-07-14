#include <iostream>
#include <pangolin/pangolin.h>
#include <sophus/se3.h>
#include <sophus/so3.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unistd.h>
#include <cmath>
#include <opencv2/core/eigen.hpp>
#include <chrono>

using namespace std;

/*
//在pangolin中构造矩阵pangolin::OpenGlMatrix方法
//******************************cv::Mat********************************
    头文件 #<opencv2/core/eigen.hpp>
        cv::Mat Twc;
        Eigen::Matrix4d Twc_e;
        cv::cv2eigen(Twc,Twc_e);
        pangolin::OpenGlMatrix Twc(Twc_e);
        glMultMatrixf(Twc);

        或者直接
        cv::Mat Twc;
        glMultMatrixf(Twc.ptr<GLfloat>(0));

//*************Eigen::Isometry3d----->pangolin::OpenGlMatrix**********
        Eigen::Isometry3d Twr(Eigen::Quaterniond(qw, qx, qy, qz).normalized());
            Twr.pretranslate(Eigen::Vector3d(tx, ty, tz));
        pangolin::OpenGlMatrix Twc(Eigen::Isometry3d::matrix());
        glMultMatrixd(Twc.m);

//**************Sophus::SE3----->pangolin::OpenGlMatrix***************
        Sophus::SE3 se3(Eigen::Quaterniond::normalized(),Eigen::Vector3d);
        pangolin::OpenGlMatrix Twc(se3.matrix());
        glMultMatrixd(Twc.m);
*/

std::string trajectory_file = "../testPangolin/trajectory.txt";

const float w = 0.06;
const float h = w*0.75;
const float z = w*0.6;
const bool drawline = true;
const long int drawGaps = 5*100;
std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> keyframs;

//画当前位姿
void drawCurrentFrame(size_t i,vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses);

//画关键帧
void drawKeyFrame(std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> keyframs);

//画动态坐标轴
void drawAxis(size_t i,vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses);

//画原点坐标系
void drawCoordinate(float scale);

//画轨迹线
void drawLine(size_t i, vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses, bool drawLine);

//筛选关键帧
void selectKeyFrame(Eigen::Vector4d velocity, std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> &keyframs,
                    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses,int currentIdx);

//求解移动速度
double getVelocity(Eigen::Isometry3d &pose_last,Eigen::Isometry3d &pose_next,double &time_used,Eigen::Vector4d &trans_velocity,Eigen::Vector3d &angluar_velocity);

int main(int argc,char** argv)
{

    if(argc > 1)
        trajectory_file = argv[1];
    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
    ifstream fin(trajectory_file);
    if (!fin) {
       cout << "cannot find trajectory file at " << trajectory_file << endl;
       return 1;
     }

    vector<double> timeRecords;

     while (!fin.eof()) {
       double time, tx, ty, tz, qx, qy, qz, qw;
       fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

       timeRecords.push_back(time);

       Eigen::Isometry3d Twr(Eigen::Quaterniond(qw, qx, qy, qz).normalized());
       Twr.pretranslate(Eigen::Vector3d(tx, ty, tz));
       poses.push_back(Twr);
     }


     //绘画出轨迹图
       pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
       glEnable(GL_DEPTH_TEST);//深度测试
       glEnable(GL_BLEND);
       glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
       pangolin::OpenGlRenderState s_cam(//摆放一个相机
               pangolin::ProjectionMatrix(1024, 768, 500, 500, 800, 200, 0.1, 1000),
               pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, pangolin::AxisZ)
       );
       pangolin::View &d_cam = pangolin::CreateDisplay()//创建一个窗口
               .SetBounds(0.0, 1.0, 0, 1.0, -1024.0f / 768.0f)
               .SetHandler(new pangolin::Handler3D(s_cam));

       for(size_t i=0; (pangolin::ShouldQuit()==false)&&i<poses.size();i++)
       {
           //计时
           chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

           //消除颜色缓冲
           glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
           d_cam.Activate(s_cam);
           glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

           //画相机模型
           drawCurrentFrame(i,poses);
           //画出动态坐标轴
           drawAxis(i,poses);
           //画坐标系
           drawCoordinate(0.5);
           //画出轨迹
           drawLine(i,poses,drawline);
           //求解速度，利用2帧间隔来判断，逐帧检测速度不稳定
           Eigen::Vector4d velocity;
           Eigen::Vector3d angluar_velocity;
           if(i>2)
           {
                double time_used = timeRecords[i]-timeRecords[i-2];
                getVelocity(poses[i-2],poses[i],time_used,velocity,angluar_velocity);
                cout<<"Current Velocity: "<<velocity[3]<<" m/s , ";
           }

           //利用速度筛选
           selectKeyFrame(velocity,keyframs,poses,i);
           //绘制关键帧
           drawKeyFrame(keyframs);

           pangolin::FinishFrame();

           //时间戳
           if(i>1 )
                usleep((timeRecords[i]-timeRecords[i-1])*1000000);
           else
                usleep(33000);          //延时33毫秒 usleep单位是微秒级

           chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
           chrono::duration<double> delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1); //milliseconds 毫秒
           cout<<"time used:"<<delay_time.count()<<" /seconds"<<endl;
       }

       while(pangolin::ShouldQuit()==false)
       {
           //消除颜色缓冲
           glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
           d_cam.Activate(s_cam);
           glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
           //画相机模型
           drawKeyFrame(keyframs);
           //画坐标系
           drawCoordinate(0.5);
           //画出轨迹
           drawLine(poses.size(),poses,drawline);

           pangolin::FinishFrame();
       }

       return 0;

}

void drawAxis(size_t i,vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses)
{
    //画出坐标轴
    Eigen::Vector3d Ow = poses[i].translation();
    Eigen::Vector3d Xw = poses[i] * (0.1 * Eigen::Vector3d(1, 0, 0));
    Eigen::Vector3d Yw = poses[i] * (0.1 * Eigen::Vector3d(0, 1, 0));
    Eigen::Vector3d Zw = poses[i] * (0.1 * Eigen::Vector3d(0, 0, 1));
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

double getVelocity(Eigen::Isometry3d &pose_last,Eigen::Isometry3d &pose_next,double &time_used,Eigen::Vector4d &trans_velocity,Eigen::Vector3d &angluar_velocity)
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
    Eigen::Vector3d dtheta_zyx = rotation_vector_next.matrix().eulerAngles(2,1,0)-rotation_vector_last.matrix().eulerAngles(2,1,0);
    Eigen::Vector3d angluar_zyx = dtheta_zyx/time_used;
    angluar_velocity <<angluar_zyx[2],angluar_zyx[1],angluar_zyx[0];
}

void drawCurrentFrame(size_t i,vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses)
{
    //变换位姿
    glPushMatrix();
    pangolin::OpenGlMatrix Twc_(poses[i].matrix());
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

void drawLine(size_t i,vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses,bool drawLine)
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

void drawKeyFrame(std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> keyframs)
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

void drawCoordinate(float scale)
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

void selectKeyFrame(Eigen::Vector4d velocity, std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> &keyframs,
                    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses,int currentIdx)
{
    if(currentIdx%5==0)
    {
        int estimateNum = 2*velocity[3];
        if(estimateNum == 0 && velocity[3]*10>1)
            keyframs.push_back(poses[currentIdx]);
        else
        {
            for(size_t i =0;i<estimateNum;i++)
                keyframs.push_back(poses[currentIdx-i*2]);
        }
    }
}
