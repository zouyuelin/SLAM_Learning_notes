#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include "sophus/so3.h"
#include "sophus/se3.h"

#define PI M_PI

using namespace std;

int main()
{
    //eigen表示：
    //旋转矩阵（3X3）:Eigen::Matrix3d
    //旋转向量（3X1）:Eigen::AngleAxisd
    //四元数（4X1）:Eigen::Quaterniond
    //欧拉角（3X1）:Eigen::Vector3d
    //欧式变换矩阵（4X4）:Eigen::Isometry3d
    //仿射变换 (4X4):Eigen::Affine3d
    //射影变换 (4X4):Eigen::Projective3d

    // 初始化旋转向量，绕z轴旋转45°，y轴60°，x轴30°；
    Eigen::AngleAxisd R_z(PI/4, Eigen::Vector3d(0,0,1));
    Eigen::AngleAxisd R_y(PI/3, Eigen::Vector3d(0,1,0));
    Eigen::AngleAxisd R_x(PI/6, Eigen::Vector3d(1,0,0));
    // 转换为旋转矩阵
    Eigen::Matrix3d R_matrix  = R_z.toRotationMatrix()*R_y.toRotationMatrix()*R_x.toRotationMatrix();
    cout<<"The matrix from Eigen::AngleAxisd is :\n"<<R_matrix<<endl<<endl;

    // 旋转向量转换为四元数 [x y z w],但构造时输入Quaterniond顺序为 w x y z;
    Eigen::Quaterniond q(R_matrix);
    cout<<"The quternian is [x y z w]:\n"<<q.coeffs().transpose()<<endl<<endl;


    //******************************************* SO3 **********************************
    //sophus的顺规是 x y z，即先绕x旋转，后绕y，再z；
    // 利用sophus构造：两种方法
    //一、利用旋转矩阵Matrix3d类型构造：
    Sophus::SO3 Rotation_1(R_matrix);                  //利用旋转矩阵***1

    //二、利用四元数来构造
    Sophus::SO3 Rotation_2(q);                          //利用四元数


    // 利用对数映射获得它的李代数，使用log函数
    Eigen::Vector3d so3_1 = Rotation_1.log();
    Eigen::Vector3d so3_2 = Rotation_2.log();
    cout<<"le_so3 from matrix is: "<<so3_1.transpose()<<endl;
    cout<<"le_so3 from quternion is: "<<so3_2.transpose()<<endl;

    // 增量扰动模型的更新SO3
    Eigen::Vector3d update_so3(1e-3, 0, 0);                              // 更新量
    Sophus::SO3 SO3_updated = Sophus::SO3::exp(update_so3)*Rotation_1;   // 左乘更新
    cout<<"SO3 updated = "<<SO3_updated<<endl;


    //****************************************** SE3 *************************************
    //利用sophus构造
    Eigen::Vector3d t(1,1,3);           //平移向量

    Sophus::SE3 SE3_Rt(R_matrix,t);     //利用旋转矩阵 构造SE3
    Sophus::SE3 SE3_qt(q,t);            //利用四元数   构造SE3
    cout<<"SE3 from R,t= "<<endl<<SE3_Rt<<endl;
    cout<<"SE3 from q,t= "<<endl<<SE3_qt<<endl;

    //se3李代数是六维的向量：(t=Jp)(R = exp(SE3x))
    typedef Eigen::Matrix<double,6,1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    Eigen::Matrix4d Tcw = Sophus::SE3::exp(se3).matrix();
    cout<<"se3 = \n"<<se3.transpose()<<endl;
    cout<<"SE3 = \n"<<Tcw<<endl;

    // hat:表示反对称的se3，vee：与hat相反
    cout<<"se3 hat = "<<endl<<Sophus::SE3::hat(se3)<<endl;
    cout<<"se3 hat vee = "<<Sophus::SE3::vee( Sophus::SE3::hat(se3) ).transpose()<<endl;

    // 增量扰动模型的更新SE3
    Vector6d update_se3;
    update_se3<<1e-2, 1e-3, 2e-2,1e-3,0,0;                              // 更新量
    Sophus::SE3 SE3_updated = Sophus::SE3::exp(update_se3)*SE3_Rt;      // 左乘更新
    cout<<"SE3 updated = "<<SE3_updated.matrix()<<endl;

    return 0;
}
