#include <iostream>
#include "Transformer.h"

using namespace std;

int main()
{
    Eigen::Vector3d angle(0.781,0.785,0.785);
    Eigen::Quaterniond a =transform::EulerAngle2Quat(angle);
    cout<<transform::QuaternionToEulerAngles(a)<<endl;
    cout << transform::EulerAngle2Matrix(angle)<< endl;
    cout<< transform::MatrixToEulerAngles(transform::EulerAngle2Matrix(angle))<<endl;
    return 0;
}
