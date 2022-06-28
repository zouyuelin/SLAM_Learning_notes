#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>

class transform{
public:
    static double DegToRad(double angle);
    static double RadToDeg(double angle);

    static Eigen::Vector3d QuaternionToEulerAngles(Eigen::Quaterniond q);
    static Eigen::Matrix3d QuaternionToMatrix(Eigen::Quaterniond q);

    static Eigen::Quaterniond EulerAngle2Quat(double rx, double ry, double rz);
    static Eigen::Quaterniond EulerAngle2Quat(Eigen::Vector3d angle);
    static Eigen::Matrix3d EulerAngle2Matrix(double rx, double ry, double rz);
    static Eigen::Matrix3d EulerAngle2Matrix(Eigen::Vector3d angle);

    static Eigen::Vector3d MatrixToEulerAngles(Eigen::Matrix3d mat);
    static Eigen::Quaterniond MatrixToQuaternion(Eigen::Matrix3d mat);

};

#endif // TRANSFORMER_H
