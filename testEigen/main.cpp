#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <chrono>

using namespace std;


int main()
{
    //***********************************基本运算*******************************
    //********定义*****
    Eigen::Matrix<float,2,3> mat1;
    mat1<<2,3,3,1,2,3;
    cout<<"the Mat1 is:\n"<<mat1<<endl;

    Eigen::Vector3d mat2;
    for (int i=0; i<3; i++)
            mat2(i)=i;
    cout<<"the Mat2 is:\n"<<mat2<<endl;

    //******矩阵相乘*****
    Eigen::Matrix<float,3,2> mat3;
    mat3<<2,3,4,4,5,6;

    Eigen::Matrix<float,2,2> mat13 = mat1*mat3;
    cout<<"Mat13 is :\n"<<mat13<<endl;


    //*****矩阵转置、逆、行列式等*****
    Eigen::Matrix<float,3,3> matrix_33 = Eigen::Matrix<float,3,3>::Zero();
    matrix_33<<1,2,7,4,5,6,7,0,9;
    cout<<"matrix_33 is:\n"<<matrix_33;
    cout<<matrix_33.transpose() << endl;    //转置
    cout<<matrix_33.sum() << endl;          //各元素和
    cout<<matrix_33.trace() << endl;        //迹
    cout<<10*matrix_33 << endl;             //数乘
    cout<<matrix_33.inverse() << endl;      //逆
    cout<<matrix_33.determinant() << endl;  //行列式

    //******矩阵的特征值和特征向量****
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    Eigen::EigenSolver<Eigen::Matrix<float,3,3>> eigen_solver ( matrix_33 );
    cout << "matrix values = \n" << eigen_solver.eigenvalues() << endl;//形式为二维向量(4,0)和(-1,0)。真实值为4,-1。
    cout << "matrix vectors = \n" << eigen_solver.eigenvectors() << endl;

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double delay = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(); //milliseconds 毫秒
    cout<<"耗时:"<<delay<<"微秒"<<endl;

    //**********奇异值分解************
    Eigen::Matrix<double,3,3> J;
    J<< 1,0,1,
        0,1,1,
        0,0,0;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto U = svd.matrixU();
    auto V = svd.matrixV();
    auto A = svd.singularValues();
    cout<<"A = \n"<<A<<endl;
    cout<<"U = \n"<<U<<endl;
    cout<<"VT = \n"<<V.transpose()<<endl;

    auto J_souce = U*A.asDiagonal()*V.transpose();
    cout<<"J constructed by U*A*VT is \n"<<J_souce<<endl;


    //*****************************************机器人学中的运算**************************
    //---------------------------坐标变化---------------------
    //eigen表示：
    //旋转矩阵（3X3）:Eigen::Matrix3d
    //旋转向量（3X1）:Eigen::AngleAxisd
    //四元数（4X1）:Eigen::Quaterniond
    //欧拉角（3X1）:Eigen::Vector3d
    //欧式变换矩阵（4X4）:Eigen::Isometry3d
    //仿射变换 (4X4):Eigen::Affine3d
    //射影变换 (4X4):Eigen::Projective3d

    //***********坐标变换的例子*************
    //位置姿态
    Eigen::Vector3d p_1(0.5,0.1,0.2);

    //变换矩阵,使用四元数要归一化
    Eigen::Quaterniond q_1(0.35,0.2,0.5,0.1);
    q_1.normalize();
    Eigen::Vector3d t_1(0.3,0.1,0.1);

    Eigen::Quaterniond q_2(-0.5,0.4,-0.1,0.2);
    q_2.normalize();
    Eigen::Vector3d t_2(-0.1,0.5,0.2);

    //定义变换矩阵Tw1和Tw2
    Eigen::Isometry3d Tw1(q_1.toRotationMatrix());
    Tw1.pretranslate(t_1);

    Eigen::Isometry3d Tw2(q_2.toRotationMatrix());
    Tw2.pretranslate(t_2);

    Eigen::Isometry3d Tw3 = Tw2*Tw1;
    cout<<"Tw2 = \n"<<Tw2.matrix()<<endl;

    //坐标变换
    Eigen::Vector3d p_2 = Tw1 * p_1;
    Eigen::Vector3d p_3 = Tw2 * p_2;
    Eigen::Vector3d p_3_ = Tw3 * p_1;
    cout<<q_2.normalized().coeffs()<<endl;
    cout<<"The position p2 is:\n"<<p_2.transpose()<<endl;
    cout<<"The position p3 is:\n"<<p_3.transpose()<<endl;
    cout<<"The position p3_ is:\n"<<p_3_.transpose()<<endl;

    //---------------------------旋转变换---------------------
    // 初始化旋转向量，绕z轴旋转45°，y轴60°，x轴30°；
    Eigen::AngleAxisd R_z(M_PI/4, Eigen::Vector3d(0,0,1));
    Eigen::AngleAxisd R_y(M_PI/3, Eigen::Vector3d(0,1,0));
    Eigen::AngleAxisd R_x(M_PI/6, Eigen::Vector3d(1,0,0));
    // 转换为旋转矩阵
    Eigen::Matrix3d R_matrix  = R_z.toRotationMatrix()*R_y.toRotationMatrix()*R_x.toRotationMatrix();
    cout<<"The matrix from Eigen::AngleAxisd is :\n"<<R_matrix<<endl<<endl;

    // 旋转向量转换为四元数 [x y z w],但构造时输入Quaterniond顺序为 w x y z;
    Eigen::Quaterniond q(R_matrix);
    cout<<"The quternian is [x y z w]:\n"<<q.coeffs().transpose()<<endl<<endl;

    Eigen::AngleAxisd tf(R_matrix);
    cout<<"Rotation matrix is :\n"<<tf.matrix()<<endl;
    cout<<"Angle is z x y:\n"<<tf.matrix().eulerAngles(2,1,0)<<endl; //tf.matrix().eulerAngles(2,1,0)

    //设为单位矩阵
    R_matrix.setIdentity(3,3);
    Eigen::AngleAxisd tf_E(R_matrix);
    cout<<"Rotation matrix is :\n"<<R_matrix<<endl;
    cout<<"Angle is z x y:\n"<<tf_E.matrix().eulerAngles(2,1,0)<<endl; //tf.matrix().eulerAngles(2,1,0)

    return 0;
}
