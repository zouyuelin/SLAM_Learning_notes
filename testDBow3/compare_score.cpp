#include <iostream>
#include <DBoW3/DBoW3.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <chrono>

using namespace std;
using namespace cv;

void findDescriptors(vector<cv::Mat> &images,vector<cv::Mat> &descriptors);

int main(int argc, char** argv)
{
    //加载词典
    cout<<"loading the vocabulary......\n"<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    DBoW3::Vocabulary vocab("./vocabulary.yml.gz");

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1); //milliseconds 毫秒
    cout<<"加载耗时:"<<delay_time.count()<<"秒"<<endl;
    cout<<vocab<<endl;

    //加载图片进行匹配测试
    if(argc !=2)
    {
        cout<<"use: ./build/creatDBow3vocabulary data\n";
        return -1;
    }
    //读取 文件夹下的图片
    string path = argv[1];
    ifstream readImages("filelist.txt");
    vector<cv::Mat> images;
    string temp;

    //抽取10张
    int i = 0;
    while (i != 10)
    {
        getline(readImages,temp);
        images.push_back(cv::imread(path+"/"+temp));
        i++;
    }
    cout<<"The number of the images is:"<<images.size()<<endl;
    vector<cv::Mat> descriptors;
    findDescriptors(images,descriptors);

    //直接利用词典对两张图片进行相似性比较
    DBoW3::BowVector v1,v2;
    for(size_t i=0;i<descriptors.size();i++)
    {
        vocab.transform(descriptors[i],v1);
        for(size_t j=0;j<descriptors.size();j++)
        {
            vocab.transform(descriptors[j],v2);
            double score = vocab.score(v1,v2);
            cout<<"Image "<<i<<"vs Image "<<j<<":"<<score<<endl;
        }
    }

    //利用字典和实时得到的图像创建数据库
    //在实时slam中，可以不断的向数据库中添加

    cout<<"creating database...\n"<<endl;
    DBoW3::Database db(vocab, false, 0);
    for (size_t i = 0; i < descriptors.size(); i++)
         db.add(descriptors[i]);
    cout << "database info: \n" << db << endl;

    //在建立的数据库中查询检索：
    for (int i = 0; i < descriptors.size(); i++)
    {
            DBoW3::QueryResults ret;

                t1 = chrono::steady_clock::now();
            db.query(descriptors[i], ret, 4);      // 检索出得分最高的四张图片
                t2 = chrono::steady_clock::now();
            delay_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1); //milliseconds 毫秒
                cout<<"检索耗时:"<<delay_time.count()<<"秒"<<endl;
            cout << "searching for image " << i << " returns " << ret << endl << endl;
    }

    //保存数据库
    db.save("database.yml.gz");

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
