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
    // 初始化旋转向量，绕z轴旋转45°
    Eigen::AngleAxisd R(PI/4, Eigen::Vector3d(0,0,1));
    auto R_matrix = R.toRotationMatrix();
    // 旋转向量转换为四元数
    Eigen::Quaterniond q(R);


    //******************************************* SO3 **********************************
    // 利用sophus构造：
    Sophus::SO3 Rotation_1(R_matrix);   //利用旋转矩阵
    Sophus::SO3 Rotation_2(q);          //利用四元数
    Sophus::SO3 Rotation_3(0,0,PI/4); //利用欧拉角

    cout<<R_matrix<<endl;
    cout<<"SO3 matrix 1 from matrix is:\n"<<Rotation_1;
    cout<<"SO3 matrix 2 from quaternion is:\n"<<Rotation_2;
    cout<<"SO3 matrix 3 from Euler is:\n"<<Rotation_3;

    // 使用对数映射获得它的李代数
    Eigen::Vector3d so3 = Rotation_1.log();
    cout<<"so3 = "<<so3.transpose()<<endl;

    // 增量扰动模型的更新SO3
    Eigen::Vector3d update_so3(1e-4, 0, 0);                              // 更新量
    Sophus::SO3 SO3_updated = Sophus::SO3::exp(update_so3)*Rotation_1;   // 左乘更新
    cout<<"SO3 updated = "<<SO3_updated<<endl;


    //****************************************** SE3 *************************************
    //利用sophus构造
    Eigen::Vector3d t(1,1,3);           //平移
    Sophus::SE3 SE3_Rt(R_matrix,t);   //利用旋转矩阵 构造SE3
    Sophus::SE3 SE3_qt(q,t);     //利用四元数   构造SE3
    cout<<"SE3 from R,t= "<<endl<<SE3_Rt<<endl;
    cout<<"SE3 from q,t= "<<endl<<SE3_qt<<endl;

    //se3李代数是六维的向量：(t=Jp)
    typedef Eigen::Matrix<double,6,1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout<<"se3 = "<<se3.transpose()<<endl;

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
