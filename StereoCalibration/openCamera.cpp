#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc,char ** argv)
{
    if(argc < 3)
    {
        cout<<"please use : openCamera [leftImagePath] [rightImagePath]\n";
        return -1;
    }
    VideoCapture capture(0);
    capture.set(cv::CAP_PROP_FRAME_WIDTH,1280);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT,480);

    if(!capture.isOpened())
    {
        cout<<"can't open camera\n";
        return -1;
    }

    string leftPath(argv[1]);
    string rightPath(argv[2]);

    int index = 0;

    while(capture.isOpened())
    {
        Mat frame,frameRigth,frameLeft;
        capture>>frame;
        frameLeft = frame(cv::Rect(0,0,640,480));
        frameRigth = frame(cv::Rect(640,0,640,480));
        imshow("frameLeft",frameLeft);
        imshow("frameRight",frameRigth);
        int key = waitKey(33);
        if(key == 'q')
            break;
        if(key == 'c')
        {
            index++;
            imwrite(leftPath+"/"+to_string(index)+".jpg",frameLeft);
            imwrite(rightPath+"/"+to_string(index)+".jpg",frameRigth);
        }
    }
    return 0;
}
