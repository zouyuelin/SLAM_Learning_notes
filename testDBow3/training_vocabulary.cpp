#include <iostream>
#include <DBoW3/DBoW3.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv ;

void findDescriptors(vector<cv::Mat> &images,vector<cv::Mat> &descriptors);

int main(int argc, char** argv )
{
    if(argc !=2)
    {
        cout<<"use: ./build/creatDBow3vocabulary data\n";
        return -1;
    }
    //读取 文件夹下的图片
    string path = argv[1];
    string cmd1 = "ls "+path+">"+" ./filelist.txt";
    system(cmd1.c_str());
    ifstream readImages("filelist.txt");

    vector<cv::Mat> images;
    string temp;
    while (getline(readImages,temp))
    {
        images.push_back(cv::imread(path+"/"+temp));
    }

    cout<<"The number of the images is:"<<images.size()<<endl;
    vector<cv::Mat> descriptors;
    findDescriptors(images,descriptors);

    // create vocabulary
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    cout<<vocab<<endl;
    vocab.save( "vocabulary.yml.gz" );

    return 0;
}

void findDescriptors(vector<cv::Mat> &images,vector<cv::Mat> &descriptors)
{
    Ptr<cv::Feature2D> detector = ORB::create();
    for ( Mat image:images )
        {
            vector<KeyPoint> keypoints;
            Mat descriptor;
            detector->detectAndCompute( image, Mat(), keypoints, descriptor );
            descriptors.push_back( descriptor );
        }
}
