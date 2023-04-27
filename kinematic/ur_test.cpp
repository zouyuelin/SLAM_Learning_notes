#include "ros/ros.h"
// sensor messages
#include <trajectory_msgs/JointTrajectory.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/WrenchStamped.h>
// standard library
#include <iostream>
#include <chrono>
// Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>
// Kinematic solution
#include "follow_drag/ur_kine.h"
#include <netft_utils/SetBias.h>

using namespace std;

#define ROBOT_DOF 6
#define FORCE_THRESHOLD 0.4

//Functions
//forward kinematic
void forward(const double* q, double* T){
  ur_Kinematics::forward(q,T);
  //print the matrix
  printf("The transformation matrix is: \n");
  for(int i=0;i<4;i++) {
    for(int j=i*4;j<(i+1)*4;j++)
      printf("%1.3f ", T[j]);
    printf("\n");
  }
}
//inverse kinematic
void inverse(const double* T, double *sq, double* q_sols){
  int num_sols = ur_Kinematics::inverse(T,q_sols);
  double sum[num_sols]={0};
  for(int i=0;i<num_sols;i++){
    for(int j=0;j<6;j++){
      if( q_sols[i*6+j] > double(3.14159265358979) ) {
          q_sols[i*6+j] -= double(6.28318530717959);
      }
      else if( q_sols[i*6+j]  < double(-3.14159265358979) ) {
          q_sols[i*6+j]  += double(6.28318530717959);
      }
      sum[i] = abs(q_sols[i*6+j] - sq[j]);
    }
  }
  //Print the solutions
  printf("All of the solutions are :\n");
  for(int i=0;i<num_sols;i++){
    printf("%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f\n",
       q_sols[i*6+2], q_sols[i*6+1], q_sols[i*6+0], q_sols[i*6+3], q_sols[i*6+4], q_sols[i*6+5]);
  }
  //constraints: to get the best suitable solution.
  //minimize the energy from start to end posture.
  double* minElement = std::min_element(sum, sum + num_sols);
  int minIndex = std::distance(sum, minElement) + 1;
  //"elbow_joint" "shoulder_lift_joint" "shoulder_pan_joint" "wrist_1_joint" "wrist_2_joint" "wrist_3_joint"
  printf("the best solution is :\n %1.6f %1.6f %1.6f %1.6f %1.6f %1.6f\n",
     q_sols[minIndex*6+2], q_sols[minIndex*6+1], q_sols[minIndex*6+0], q_sols[minIndex*6+3], q_sols[minIndex*6+4], q_sols[minIndex*6+5]);
}

Eigen::VectorXd cur_pos(6);
Eigen::Vector3d cur_force,last_force;
Eigen::Vector3d cur_torque,last_torque;
int flag_force = 0;
int flag_position = 0;
// Callback function
void jointStateCallback (const sensor_msgs::JointState& js_msg);
void forceStateCallback (const geometry_msgs::WrenchStamped& force_msg);

int main(int argc, char **argv)
{
  ros::init(argc, argv, "ur_test");
  ros::NodeHandle nh;

  //Control the real UR robot arm.
  ros::Publisher arm_pub = nh.advertise<trajectory_msgs::JointTrajectory>("/scaled_pos_joint_traj_controller/command", 10);
  ros::Subscriber force_sub = nh.subscribe("/wrench",1,forceStateCallback);
  ros::Subscriber joint_state_sub = nh.subscribe("/joint_states", 1, jointStateCallback);
  ros::ServiceClient set_bias=nh.serviceClient<netft_utils::SetBias>("/bias");

  //parameter initiation.
  trajectory_msgs::JointTrajectory traj_msg;
    traj_msg.header.frame_id = "222";
    traj_msg.joint_names.resize(6);
    traj_msg.points.resize(1);
    traj_msg.points[0].positions.resize(6);
    traj_msg.joint_names[0] = "elbow_joint";          //elbow     2
    traj_msg.joint_names[1] = "shoulder_lift_joint";  //shoulder  1
    traj_msg.joint_names[2] = "shoulder_pan_joint";   //base      0
    traj_msg.joint_names[3] = "wrist_1_joint";        //          3
    traj_msg.joint_names[4] = "wrist_2_joint";        //          4
    traj_msg.joint_names[5] = "wrist_3_joint";        //          5

  netft_utils::SetBias bias;
  bias.request.toBias= true;
  bias.request.forceMax=50;
  bias.request.torqueMax=10;
  set_bias.call(bias);

  double T[16]={0};
  double q[6]={0};
  cur_pos<< 0, 0, 0, 0, 0, 0;

  //initialize the force by spin once.
  while(flag_force == 0 || flag_position == 0) ros::spinOnce();

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    //Count the processing time.
    auto t1 = chrono::system_clock::now();

    printf("Position of joints is :\n %1.6f %1.6f %1.6f %1.6f %1.6f %1.6f \n",cur_pos[0],cur_pos[1],cur_pos[2],cur_pos[3],cur_pos[4],cur_pos[5]);
    printf("Current Tcp-force x y z is :%1.6f %1.6f %1.6f \n",cur_force[0],cur_force[1],cur_force[2]);
    printf("Current Tcp-Torque rx ry rz is :%1.6f %1.6f %1.6f \n",cur_torque[0],cur_torque[1],cur_torque[2]);
    q[2] = traj_msg.points[0].positions[0] = cur_pos[0];
    q[1] = traj_msg.points[0].positions[1] = cur_pos[1];
    q[0] = traj_msg.points[0].positions[2] = cur_pos[2];
    q[3] = traj_msg.points[0].positions[3] = cur_pos[3];
    q[4] = traj_msg.points[0].positions[4] = cur_pos[4];
    q[5] = traj_msg.points[0].positions[5] = cur_pos[5];

    //kinematic forward
    forward(q,T);

    //kinematic inverse calculation
    double q_sols[8*6];
    inverse(T,q,q_sols);

    traj_msg.header.stamp = ros::Time::now();
    traj_msg.points[0].time_from_start = ros::Duration(0.01);
    //Publish the trajectory message.
    arm_pub.publish(traj_msg);

    ros::spinOnce();
    loop_rate.sleep();

    //Finish the time counting.
    auto t2 = chrono::system_clock::now();
    cout<<"time consumed:"
        <<chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()<<"ms"
        <<endl;
  }
  return 0;
}

void jointStateCallback (const sensor_msgs::JointState& js_msg){
  //Copy the message from topic.
    for(int i = 0; i<ROBOT_DOF; i++)
      cur_pos[i] = js_msg.position[i];
    if(cur_pos[0] !=0 && flag_position == 0){
      flag_position = 1;
      return;
    }
}

void forceStateCallback (const geometry_msgs::WrenchStamped& force_msg){
  //Initialize the tcp force.
  cur_force[0] = force_msg.wrench.force.x;
  cur_force[1] = - force_msg.wrench.force.y;
  cur_force[2] = - force_msg.wrench.force.z;
  //Ensure the cur_force is not zero vector.
  if(cur_force[0] !=0 && flag_force == 0){
    flag_force = 1;
    return;
  }
  //Torque
  last_torque = cur_torque;
  cur_torque[0] = force_msg.wrench.torque.x;
  cur_torque[1] = force_msg.wrench.torque.y;
  cur_torque[2] = force_msg.wrench.torque.z;
}
