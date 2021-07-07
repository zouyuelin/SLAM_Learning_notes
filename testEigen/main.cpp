#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <chrono>

using namespace std;


int main()
{
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
    cout<<matrix_33.transpose() << endl; //转置
    cout<<matrix_33.sum() << endl; //各元素和
    cout<<matrix_33.trace() << endl; //迹
    cout<<10*matrix_33 << endl; //数乘
    cout<<matrix_33.inverse() << endl; //逆
    cout<<matrix_33.determinant() << endl; //行列式

    //******矩阵的特征值和特征向量****
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    Eigen::EigenSolver<Eigen::Matrix<float,3,3>> eigen_solver ( matrix_33 );
    cout << "matrix values = \n" << eigen_solver.eigenvalues() << endl;//形式为二维向量(4,0)和(-1,0)。真实值为4,-1。
    cout << "matrix vectors = \n" << eigen_solver.eigenvectors() << endl;

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double delay = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(); //milliseconds 毫秒
    cout<<"耗时:"<<delay;

    return 0;
}
