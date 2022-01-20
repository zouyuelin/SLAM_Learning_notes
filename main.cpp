#include "vo_slam.hpp"

int main(int argc,char**argv)
{
    if(argc <2)
    {
        cout<<"请输入: ./vo_slam dataset_dir\n";
    }
    std::string dataset_dir = argv[1];
    std::string assPath = dataset_dir+"/associate.txt";
    std::ifstream txt;
    txt.open(assPath.data());
    assert(txt.is_open());
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while (!txt.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        txt>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( txt.good() == false )
            break;
    }


    VO_slam::Ptr slam = std::make_shared<VO_slam>();
    for(size_t i=0;i<rgb_files.size()-2;i++)
    {
    //---------------------------------------------------------------tracing------------------------------------------------------//
        boost::timer timer;
        Mat img = imread(rgb_files[i],IMREAD_COLOR);
        Mat depth = imread(depth_files[i], IMREAD_UNCHANGED);

        slam->tracking(img,depth);

        cout<<"VO cost time is:"<<timer.elapsed()<<endl;
        cv::imshow("frame",img);
        cv::waitKey(10);

     }

    return 0;
}
