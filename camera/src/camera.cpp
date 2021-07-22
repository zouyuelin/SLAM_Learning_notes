#include <ros/ros.h>  
#include <image_transport/image_transport.h>  
#include <opencv2/highgui/highgui.hpp>  
#include <cv_bridge/cv_bridge.h>  
#include <iostream>
#include <sstream> // for converting the command line parameter to integer  

int main(int argc, char** argv)  
{  
    // Check if video source has been passed as a parameter  
    if(argv[1] == NULL)   
    {  
        ROS_INFO("argv[1]=NULL\n");  
        return 1;  
    }  

    ros::init(argc, argv, "camera_stereo");  
    ros::NodeHandle nh;  
    image_transport::ImageTransport it(nh);  
    image_transport::Publisher pub_left = it.advertise("camera/left/image_raw", 1); 
    image_transport::Publisher pub_right = it.advertise("camera/right/image_raw", 1); 


    // Convert the passed as command line parameter index for the video device to an integer  
    std::istringstream video_sourceCmd(argv[1]);  
    int video_source;  
    // Check if it is indeed a number  
    if(!(video_sourceCmd >> video_source))   
    {  
        ROS_INFO("video_sourceCmd is %d\n",video_source);  
        return 1;  
    }  

    std::cout<<"the video index is :"<<video_source<<std::endl;

    cv::VideoCapture cap(video_source);  
    cap.set(cv::CAP_PROP_FRAME_WIDTH,1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,480);
    // Check if video device can be opened with the given index  
    if(!cap.isOpened())   
    {  
        ROS_INFO("can not opencv video device\n");  
        return 1;  
    }  
    cv::Mat frame;  
    sensor_msgs::ImagePtr msg_left; 
    sensor_msgs::ImagePtr msg_right; 

    ros::Rate loop_rate(5);  
    while (nh.ok()) 
    {  
        cap >> frame;  

        cv::Mat frame_left;
        cv::Mat frame_right;

        frame_left = frame(cv::Rect(0,0,640,480));
        frame_right = frame(cv::Rect(640,0,640,480));

        // Check if grabbed frame is actually full with some content  
        if(!frame.empty()) 
        {  
            msg_left = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame_left).toImageMsg(); 
            msg_right = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame_right).toImageMsg();
            pub_left.publish(msg_left); 
            pub_right.publish(msg_right);

            //cv::Wait(1);  
    	}  
    }
    
    ros::spinOnce();  
    loop_rate.sleep();  
}  
