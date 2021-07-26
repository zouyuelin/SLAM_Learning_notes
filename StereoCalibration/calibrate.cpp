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

void calibCamera(vector<string> &files, cv::Size boardSize, float squareSize,
                string imagePath,
                cv::Mat &cameraMatrix,
                cv::Mat &distCoeffs,
                std::vector<std::vector<cv::Point2f>> &imagePoints,
                cv::Size &imageSize,
                std::vector<std::vector<cv::Point3f>> &objectPoints,
                std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs
                 );

//去畸变
Point2f undistortmypoints(Point2f xyd , Mat distCoeffs, Mat cameraMatrix);

int main(int argc,char**argv)
{
    float squareSize = 1.0;
    if(argc<5)
    {
        cout<<"./calibrationCamera [leftImagePath] [rightImagePath] [width-1] [height-1]\n";
        cout<<"example: 如果你是10*6（宽10，高6）的棋盘，那么输入的就是 9 5\n";
        return -1;
    }

    if(argc == 6)
    {
        squareSize = atof(argv[5]);
    }

    cv::Size boardSize(atoi(argv[3]),atoi(argv[4]));
    cv::Size imageSize; //empty, get the data when running the calibCamera


    vector<string> leftfiles,rightfiles;
    string leftImagePath(argv[1]);
    string rightImagePath(argv[2]);
    glob(leftImagePath,leftfiles);
    glob(rightImagePath,rightfiles);


    //saver
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePointsleft;
    std::vector<std::vector<cv::Point2f>> imagePointsRight;


    //**********************************************************单目分别标定*********************************************
    //----calibrate--
    //left camera matrix
    cv::Mat cameraMatrixLeft;  //Camera intrinsic Matrix
    cv::Mat distCoeffsLeft;    //five parameters of distCoeffs，(k1,k2,p1,p2[,k3[,k4,k5,k6]])
    std::vector<cv::Mat> rvecsLeft, tvecsLeft;
    cout<<endl<<"-------------------Left Camera intrinsic Matrix--------------------\n";
    calibCamera(leftfiles,boardSize,squareSize,leftImagePath,cameraMatrixLeft,distCoeffsLeft,imagePointsleft,imageSize,objectPoints,rvecsLeft,tvecsLeft);
    objectPoints.clear();

    //right camera matrix
    cv::Mat cameraMatrixRight;  //Camera intrinsic Matrix
    cv::Mat distCoeffsRight;    //five parameters of distCoeffs，(k1,k2,p1,p2[,k3[,k4,k5,k6]])
    std::vector<cv::Mat> rvecsRight, tvecsRight;
    cout<<endl<<"-------------------Right Camera intrinsic Matrix--------------------\n";
    calibCamera(rightfiles,boardSize,squareSize,rightImagePath,cameraMatrixRight,distCoeffsRight,imagePointsRight,imageSize,objectPoints,rvecsRight,tvecsRight);


    //************************************************对重投影进行测试********************************
    //test the eqution
    /*

      s*x                     X                 X
      s*y = K * [r1 r2 r3 t]* Y = K * [r1 r2 t]*Y
       s                      0                 1
                              1

    */
    //**** 验证外参Rve 和 tve的准确性  ****
    // 默认选择 0 1，这里只选择一个点来进行测试，当然，你也可以多试几个
    int k=0,j=1;//k要小于图片的数量，j要小于棋盘纸 高X宽
    Mat Rrw,Rlw;
    cv::Rodrigues(rvecsRight[k],Rrw);
    cv::Rodrigues(rvecsLeft[k],Rlw);

    Mat po = Mat::zeros(3,3,CV_64F);
    po.at<double>(0,0) = Rrw.at<double>(0,0);
    po.at<double>(1,0) = Rrw.at<double>(1,0);
    po.at<double>(2,0) = Rrw.at<double>(2,0);
    po.at<double>(0,1) = Rrw.at<double>(0,1);
    po.at<double>(1,1) = Rrw.at<double>(1,1);
    po.at<double>(2,1) = Rrw.at<double>(2,1);
    po.at<double>(0,2) = tvecsLeft[k].at<double>(0,0);
    po.at<double>(1,2) = tvecsLeft[k].at<double>(0,1);
    po.at<double>(2,2) = tvecsLeft[k].at<double>(0,2);

    Mat obj(3,1,CV_64F);
    obj.at<double>(0,0) =  objectPoints[k][j].x;
    obj.at<double>(1,0) =  objectPoints[k][j].y;
    obj.at<double>(2,0) =  1;

    Mat uv = cameraMatrixLeft*po*obj;

    Point2f xyd = imagePointsleft[k][j];
    //对该点进行畸变矫正
    Point2f xyp = undistortmypoints(xyd,distCoeffsLeft,cameraMatrixLeft);

    cout<<"Test the outer parameters（请查看下面两个数据的差距，理论上是一样的）"<<endl;
    cout<<(uv/uv.at<double>(2,0)).t()<<endl; // [x y] = [x/w y/w]
    cout<<xyp<<endl;

    /*
     * 这里需要说明一点，如果上面两个值输入的差距越小，说明重投影的误差小，那么由公式 Rrl = Rrw*Rlw.inv()计算
     * 得到的，即从第一个相机到第二个相机的变换矩阵R的准确率越高；
    */


    //****************************************************************双目标定*****************************************************
    cout<<endl<<"********************stereo calibrate running******************\n";
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints,
                    imagePointsleft, imagePointsRight,
                    cameraMatrixLeft, distCoeffsLeft,
                    cameraMatrixRight, distCoeffsRight,
                    imageSize, R, T, E, F,
                    CALIB_FIX_INTRINSIC,// CALIB_USE_INTRINSIC_GUESS CALIB_FIX_INTRINSIC CALIB_NINTRINSIC
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );

    cout << "done with RMS error=" << rms << endl;

    /*
        Pl = Rlw * Pw+Tlw;
        Pr = Rrw * Pw+Trw;

        --Pr = Rrl * Pl + Trl

       so =====>
                    Rrl = Rrw * Rlw^T
                    Trl = Trw - Rrl * Tlw

    */

    //**** 验证上述公式是否成立  ****
    //--多试试几组值，如果和最后得到的R差距非常大，很可能标定失败；
    cout<<endl<<"Test the R(=Rrl)（与输出的R进行比较）:"<<endl;
    cout<<endl<<Rrw*Rlw.inv()<<endl;

    //如果上面Rrw*Rlw.inv() 输出结果 与R输出差距较大，可以去除下列注释测试
    //如果最后标定输出的remap()图像效果很差，也可以去除下列注释看看测试结果
    //R = Rrw*Rlw.inv();                      //Rrl = Rrw * Rlw^T
    //T = tvecsRight[k] - R * tvecsLeft[k];   //Trl = Trw - Rrl * Tlw

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    //************************************************双目矫正*******************************************
    stereoRectify(cameraMatrixLeft, distCoeffsLeft,
                  cameraMatrixRight, distCoeffsRight,
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, -1, imageSize, &validRoi[0], &validRoi[1]);

    cout<<"R is :\n"<<R<<endl;
    cout<<"T is :\n"<<T<<endl;
    cout<<"R1 is :\n"<<R1<<endl;
    cout<<"R2 is :\n"<<R2<<endl;
    cout<<"P1 is :\n"<<P1<<endl;
    cout<<"P1 is :\n"<<P2<<endl;
    cout<<"Q is :\n"<<Q<<endl;


    //****************************************双目畸变矫正*************************
    //remap
    Mat rmap[2][2];
    initUndistortRectifyMap(cameraMatrixLeft, distCoeffsLeft, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrixRight, distCoeffsRight, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    for(size_t i=0;i<leftfiles.size();i++)
    {
        Mat il = imread(leftfiles[i],0),ilremap;
        Mat ir = imread(rightfiles[i],0),irremap;

        if(il.empty() || ir.empty())
            continue;

        remap(il, ilremap, rmap[0][0], rmap[0][1], INTER_LINEAR);
        remap(ir, irremap,  rmap[1][0], rmap[1][1], INTER_LINEAR);

        Mat srcAnddst(ilremap.rows,ilremap.cols + irremap.cols,ilremap.type());
        Mat submat =srcAnddst.colRange(0,ilremap.cols);
           ilremap.copyTo(submat);
           submat = srcAnddst.colRange(ilremap.cols,ilremap.cols + irremap.cols);
           irremap.copyTo(submat);

       cvtColor(srcAnddst,srcAnddst,COLOR_GRAY2BGR);

       //draw rectified image
        for (int i = 0; i < srcAnddst.rows;i+=16)
           line(srcAnddst, Point(0, i), Point(srcAnddst.cols, i), Scalar(0, 255, 0), 1, 8);

        imshow("remap",srcAnddst);
        //imshow("ir",irremap);
        waitKey(500);
    }


    //save the config parameters
    //clear the old .yaml file
    String path = "./calib.yaml";
    ofstream fs(path);
    fs.clear();

    fs << "# ------left camera Intrinsic--------"<<endl;
    fs << "Left.Camera.fx:  " << cameraMatrixLeft.at<double>(0,0)<<endl;
    fs << "Left.Camera.fy:  " << cameraMatrixLeft.at<double>(1,1)<<endl;
    fs << "Left.Camera.cx:  " << cameraMatrixLeft.at<double>(0,2)<<endl;
    fs << "Left.Camera.cy:  " << cameraMatrixLeft.at<double>(1,2)<<endl<<endl;

    fs << "# ------left camera Distortion--------"<<endl;
    fs << "Left.Camera.k1:  " << distCoeffsLeft.at<double>(0,0)<<endl;
    fs << "Left.Camera.k2:  " << distCoeffsLeft.at<double>(0,1)<<endl;
    fs << "Left.Camera.p1:  " << distCoeffsLeft.at<double>(0,2)<<endl;
    fs << "Left.Camera.p2:  " << distCoeffsLeft.at<double>(0,3)<<endl;
    fs << "Left.Camera.k3:  " << distCoeffsLeft.at<double>(0,4)<<endl<<endl;

    fs <<endl<<endl;

    fs << "# ------right camera Intrinsic--------"<<endl;
    fs << "Right.Camera.fx:  " << cameraMatrixRight.at<double>(0,0)<<endl;
    fs << "Right.Camera.fy:  " << cameraMatrixRight.at<double>(1,1)<<endl;
    fs << "Right.Camera.cx:  " << cameraMatrixRight.at<double>(0,2)<<endl;
    fs << "Right.Camera.cy:  " << cameraMatrixRight.at<double>(1,2)<<endl<<endl;

    fs << "# ------right camera Distortion--------"<<endl;
    fs << "Right.Camera.k1:  " << distCoeffsRight.at<double>(0,0)<<endl;
    fs << "Right.Camera.k2:  " << distCoeffsRight.at<double>(0,1)<<endl;
    fs << "Right.Camera.p1:  " << distCoeffsRight.at<double>(0,2)<<endl;
    fs << "Right.Camera.p2:  " << distCoeffsRight.at<double>(0,3)<<endl;
    fs << "Right.Camera.k3:  " << distCoeffsRight.at<double>(0,4)<<endl<<endl;

    fs.close();

    //R, T, R1, R2, P1, P2, Q,
    cv::FileStorage f(path,cv::FileStorage::APPEND);
    f.writeComment(" \n------camera parameters saved by yaml data--------",true);
    f<<"R"<<R;
    f<<"T"<<T;
    f<<"R1"<<R1;
    f<<"R2"<<R2;
    f<<"P1"<<P1;
    f<<"P2"<<P2;
    f<<"Q"<<Q;

    f.release();

    //查看保存的数据
    system("cat ./calib.yaml");

    return 0;
}

void calibCamera(vector<string> &files, cv::Size boardSize,
                float squareSize,
                string imagePath, 
                cv::Mat &cameraMatrix,
                cv::Mat &distCoeffs,
                std::vector<std::vector<cv::Point2f>> &imagePoints,
                cv::Size &imageSize,
                std::vector<std::vector<cv::Point3f>> &objectPoints,
                std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs)
{

        //data objectCorners
    vector<Point3f> objectCorners;

    vector<Point2f> imageCorners;

        //get the Corners' position
    for (int i = 0; i < boardSize.height; i++)
    {
       for (int j = 0; j < boardSize.width; j++)
       {
          objectCorners.push_back(cv::Point3f(squareSize*i, squareSize*j, 0.0f));
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
                                                             50,    //maxCount
                                                             0.1);  //epsilon
            cornerSubPix(gray,
                         imageCorners,
                         cv::Size(5, 5), // winSize
                         cv::Size(-1, -1),
                         termCriteria); // epsilon

            imagePoints.push_back(imageCorners);
            objectPoints.push_back(objectCorners);


            cv::drawChessboardCorners(image, boardSize, imageCorners, found);
            imshow("Corners on Chessboard", image);
            waitKey(33);

        }
        imageSize = image.size();
    }

    cv::destroyAllWindows();

    cout<<"图像尺寸:"<<imageSize<<endl;

    double rms = calibrateCamera(objectPoints, // 三维点
                    imagePoints, // 图像点
                    imageSize, // 图像尺寸
                    cameraMatrix, // 输出相机矩阵
                    distCoeffs, // 输出畸变矩阵
                    rvecs, tvecs // Rs、Ts（外参）
                    );

    cout << " RMS error=" << rms << endl;


    cout<<"K = \n"<<cameraMatrix<<endl;
    cout<<"distCoeffs = \n"<<distCoeffs<<endl;
}

//去畸变
Point2f undistortmypoints(Point2f xyd , Mat distCoeffs, Mat cameraMatrix)
{
    double x = (xyd.x-cameraMatrix.at<double>(0,2))/cameraMatrix.at<double>(0,0);
    double y = (xyd.y-cameraMatrix.at<double>(1,2))/cameraMatrix.at<double>(1,1);
    double r = sqrt(x*x+y*y);

    double x_distorted = (1+distCoeffs.at<double>(0,0)*r*r+distCoeffs.at<double>(0,1)*r*r*r*r+distCoeffs.at<double>(0,4)*r*r*r*r*r*r)*x
                                     +2*distCoeffs.at<double>(0,2)*x*y+distCoeffs.at<double>(0,3)*(r*r+2*x*x);
    double y_distorted = (1+distCoeffs.at<double>(0,0)*r*r+distCoeffs.at<double>(0,1)*r*r*r*r+distCoeffs.at<double>(0,4)*r*r*r*r*r*r)*y
                                     +2*distCoeffs.at<double>(0,3)*x*y+distCoeffs.at<double>(0,2)*(r*r+2*y*y);

    double xp = cameraMatrix.at<double>(0,0)*x_distorted+cameraMatrix.at<double>(0,2);
    double yp = cameraMatrix.at<double>(1,1)*y_distorted+cameraMatrix.at<double>(1,2);
    return Point2f(xp,yp);
}
