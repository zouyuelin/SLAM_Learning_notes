#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

int main(int argc,char**argv)
{
    float squareSize = 1.0;    //the length or height of the per broad, unit[mm]
    if(argc<4)
    {
        cout<<"./calibrationCamera [imagepath] [width-1] [height-1] [squareSize = 1.0]\n";
        cout<<"输入 [图像路径] [棋盘宽格数-1] [棋盘横格数-1] [每个格子的边长/mm{如果不知道格子长度可以不用输入，默认1.0}]\n";
        return -1;
    }

    if(argc==5)
        squareSize = atof(argv[4]);

    vector<string> files;
    glob(argv[1],files);

    cv::Size boardSize(atoi(argv[2]),atoi(argv[3]));

    //saver
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    Mat cameraMatrix;  //Camera intrinsic Matrix
    Mat distCoeffs;    //five parameters of distCoeffs，(k1,k2,p1,p2[,k3[,k4,k5,k6]])

    std::vector<cv::Mat> rvecs, tvecs;
    cv::Size imageSize;

    vector<Point2f> imageCorners;
    vector<Point3f> objectCorners;

    //get the Corners' position
    for (int i = 0; i < boardSize.height; i++)
    {
       for (int j = 0; j < boardSize.width; j++)
       {
          objectCorners.push_back(cv::Point3f(i*squareSize, j*squareSize, 0.0f));
       }
    }

    for(size_t i =0;i<files.size();i++)
    {
        Mat image = imread(files[i]);
        Mat gray;
        if(image.empty())
            continue;

        cvtColor(image,gray,COLOR_BGR2GRAY);
        //get Chessboard Corners
        bool found = cv::findChessboardCorners(gray, boardSize,imageCorners);

        if(found)
        {
            TermCriteria  termCriteria;
            termCriteria = TermCriteria(TermCriteria::MAX_ITER +TermCriteria::EPS,
                                                             30,    //maxCount
                                                             0.1);  //epsilon
            cornerSubPix(gray,
                         imageCorners,
                         cv::Size(5, 5), // winSize
                         cv::Size(-1, -1),
                         termCriteria); // epsilon
            if(imageCorners.size() == boardSize.area())
            {
                imagePoints.push_back(imageCorners);
                objectPoints.push_back(objectCorners);
            }

            cv::drawChessboardCorners(image, boardSize, imageCorners, found);
            imshow("Corners on Chessboard", image);
            waitKey(100);

        }
        imageSize = image.size();
    }

    cv::destroyAllWindows();

    cout<<imageSize<<endl;

    calibrateCamera(objectPoints, // 三维点
                    imagePoints, // 图像点
                    imageSize, // 图像尺寸
                    cameraMatrix, // 输出相机矩阵
                    distCoeffs, // 输出畸变矩阵
                    rvecs, tvecs // Rs、Ts（外参）
                    );

    cout<<"camera Matrix 3D to 2D is  = \n" << initCameraMatrix2D(objectPoints,imagePoints,imageSize,0)<<endl<<endl;

    cout<<"K = \n"<<cameraMatrix<<endl;
    cout<<"distCoeffs = \n"<<distCoeffs<<endl<<endl;

    //clear the old .yaml file
    string path = string(argv[1])+"/calib.yaml";
    ofstream fs(path);
    fs.clear();

    fs << "# ------camera Intrinsic--------"<<endl;
    fs << "Camera.fx:  " << cameraMatrix.at<double>(0,0)<<endl;
    fs << "Camera.fy:  " << cameraMatrix.at<double>(1,1)<<endl;
    fs << "Camera.cx:  " << cameraMatrix.at<double>(0,2)<<endl;
    fs << "Camera.cy:  " << cameraMatrix.at<double>(1,2)<<endl<<endl;

    fs << "# ------camera Distortion--------"<<endl;
    fs << "Camera.k1:  " << distCoeffs.at<double>(0,0)<<endl;
    fs << "Camera.k2:  " << distCoeffs.at<double>(0,1)<<endl;
    fs << "Camera.p1:  " << distCoeffs.at<double>(0,2)<<endl;
    fs << "Camera.p2:  " << distCoeffs.at<double>(0,3)<<endl;
    fs << "Camera.k3:  " << distCoeffs.at<double>(0,4)<<endl<<endl;


    Mat newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix,distCoeffs,imageSize,0);//you can use 0 to test the images

    cout<<"after undistort newCameraMatrix = \n"<<newCameraMatrix<<endl<<endl;

    for(size_t i=0;i<files.size();i++)
    {
        Mat outputfile;
        Mat src = imread(files[i]);
        if(src.empty())
            continue;
        cv::undistort(src,outputfile,cameraMatrix,distCoeffs,newCameraMatrix);

        Mat srcAnddst(src.rows,src.cols + outputfile.cols,src.type());
        Mat submat =srcAnddst.colRange(0,src.cols);
           src.copyTo(submat);
           submat = srcAnddst.colRange(src.cols,outputfile.cols*2);
           outputfile.copyTo(submat);

        imshow("sources and undistort",srcAnddst);
        waitKey(300);
    }

    cv::destroyAllWindows();

    fs <<endl;

    fs << "# ------after undistort --new camera matrix--------"<<endl;
    fs << "# if you want use the camera matrix on slam, unditort the image or ORB points,then use the new camera matrix "<<endl;
    fs << "NewCamera.fx:  " << newCameraMatrix.at<double>(0,0)<<endl;
    fs << "NewCamera.fy:  " << newCameraMatrix.at<double>(1,1)<<endl;
    fs << "NewCamera.cx:  " << newCameraMatrix.at<double>(0,2)<<endl;
    fs << "NewCamera.cy:  " << newCameraMatrix.at<double>(1,2)<<endl<<endl;

    fs.close();
    std::string commad("gedit "+path);
    system(commad.c_str());

    cout<<"The last config yaml file is : "<<path<<endl;

    return 0;
}
