#include "Transformer.h"

double transform::DegToRad(double angle){
    return angle / 180.0 * M_PI;
}

double transform::RadToDeg(double angle){
    return angle / M_PI * 180;
}

Eigen::Vector3d transform::QuaternionToEulerAngles(Eigen::Quaterniond q) {
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

    //rad
    return Eigen::Vector3d(x,y,z);
}

Eigen::Matrix3d transform::QuaternionToMatrix(Eigen::Quaterniond q){
    return q.matrix();
}



Eigen::Quaterniond transform::EulerAngle2Quat(double rx, double ry, double rz){
    return Eigen::AngleAxisd(rz, ::Eigen::Vector3d::UnitZ())*
           Eigen::AngleAxisd(ry, ::Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(rx, ::Eigen::Vector3d::UnitX());
}

Eigen::Quaterniond transform::EulerAngle2Quat(Eigen::Vector3d angle){
    return EulerAngle2Quat(angle.x(),angle.y(),angle.z());
}

Eigen::Matrix3d transform::EulerAngle2Matrix(double rx, double ry, double rz){
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()));
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix = yawAngle * pitchAngle * rollAngle;
    return rotation_matrix;
}

Eigen::Matrix3d transform::EulerAngle2Matrix(Eigen::Vector3d angle){
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(angle.x(), Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(angle.y(), Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(angle.z(), Eigen::Vector3d::UnitZ()));
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix = yawAngle * pitchAngle * rollAngle;
    return rotation_matrix;
}


Eigen::Vector3d transform::MatrixToEulerAngles(Eigen::Matrix3d mat) {
    return QuaternionToEulerAngles(MatrixToQuaternion(mat));
}

Eigen::Quaterniond transform::MatrixToQuaternion(Eigen::Matrix3d mat){
    return Eigen::Quaterniond(mat);
}

