#ifndef UR_KINE_H
#define UR_KINE_H

namespace ur_Kinematics {
  // @param q       The 6 joint values
  // @param T       The 4x4 end effector pose in row-major ordering
  void forward(const double* q, double* T);

  // @param q       The 6 joint values
  // @param Ti      The 4x4 link i pose in row-major ordering. If NULL, nothing is stored.
  void forward_all(const double* q, double* T1, double* T2, double* T3,
                                    double* T4, double* T5, double* T6);

  // @param T       The 4x4 end effector pose in row-major ordering
  // @param q_sols  An 8x6 array of doubles returned, all angles should be in [0,2*PI)
  // @param q6_des  An optional parameter which designates what the q6 value should take
  //                in case of an infinite solution on that joint.
  // @return        Number of solutions found (maximum of 8)
  int inverse(const double* T, double* q_sols, double q6_des=0.0);
};


#endif // UR_KINE_H
