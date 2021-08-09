/* Trials with CHOMP.
 *
 * Copyright (C) 2014 Roland Philippsen. All rights reserved.
 *
 * BSD license:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of
 *    contributors to this software may be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR THE CONTRIBUTORS TO THIS SOFTWARE BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
   \file pp2d.cpp
   
   Interactive trials with CHOMP for point vehicles moving
   holonomously in the plane.  There is a fixed start and goal
   configuration, and you can drag a circular obstacle around to see
   how the CHOMP algorithm reacts to that.  Some of the computations
   involve guesswork, for instance how best to compute velocities, so
   a simple first-order scheme has been used.  This appears to produce
   some unwanted drift of waypoints from the start configuration to
   the end configuration.  Parameters could also be tuned a bit
   better.  Other than that, it works pretty nicely.
*/
#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include "gfx.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <stdlib.h>
#include <sys/time.h>
#include <err.h>

namespace {
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using Transform = Eigen::Isometry3d;
}  // namespace

//////////////////////////////////////////////////
// trajectory etc

Vector xi;			// the trajectory (q_1, q_2, ...q_n)
Vector qs;			// the start config a.k.a. q_0
Vector qe;			// the end config a.k.a. q_(n+1)
static size_t const nq (20);	// number of q stacked into xi
static size_t const cdim (2);	// dimension of config space
static size_t const xidim (nq * cdim); // dimension of trajectory, xidim = nq * cdim
static double const dt (1.0);	       // time step
static double const eta (100.0); // >= 1, regularization factor for gradient descent
static double const lambda (1.0); // weight of smoothness objective

//////////////////////////////////////////////////
// gradient descent etc

Matrix AA;			// metric
Vector bb;			// acceleration bias for start and end config
Matrix Ainv;			// inverse of AA

////////////////////////////////////////////////////// MASON
//turtlebot variables

static Vector old_obj1_pos;
static Vector old_obj2_pos;
double orientation1;
double orientation2;
static double old_obj1_ore;
static double old_obj2_ore;

static double rotate1;
static double rotate2;
////////////////////////////////////////////////////// MASON
// ros turtlebot variables
static nav_msgs::Odometry obj1;
static nav_msgs::Odometry obj2;
ros::Publisher pub_pid1;
ros::Publisher pub_pid2;


//////////////////////////////////////////////////
// gui stuff

enum { PAUSE, STEP, RUN } state;

struct handle_s {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  handle_s (double px, double py, double radius, double red, double green, double blue, double alpha)
    : point_(2),
      radius_(radius),
      red_(red),
      green_(green),
      blue_(blue),
      alpha_(alpha)
  {
    point_ << px, py;
  }
  
  Vector point_;
  double radius_, red_, green_, blue_, alpha_;
};

static handle_s rep1 (3.0, 0.0,   2.0, 0.0, 0.0, 1.0, 0.2);
static handle_s rep2 (0.0, 3.0,   2.0, 0.0, 0.5, 1.0, 0.2);

static handle_s * handle[] = { &rep1, &rep2, 0 };
static handle_s * grabbed (0);
static Vector grab_offset (3);


//////////////////////////////////////////////////
// robot (one per waypoint)

class Robot
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  Robot ()
    : position_ (Vector::Zero(2))
  {
  }
  
  
  void update (Vector const & position)
  {
    if (position.size() != 2) {
      errx (EXIT_FAILURE, "Robot::update(): position has %zu DOF (but needs 2)",
	    (size_t) position.size());
    }
    position_ = position;
  }
  
  
  void draw () const
  {
    // translucent disk for base
    gfx::set_pen (1.0, 0.7, 0.7, 0.7, 0.5);
    gfx::fill_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);
    
    // thick circle outline for base
    gfx::set_pen (3.0, 0.2, 0.2, 0.2, 1.0);
    gfx::draw_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);
  }
  
  static double const radius_;
  
  Vector position_;
};

double const Robot::radius_ (0.5);

Robot rstart;
Robot rend;
std::vector <Robot> robots;


//////////////////////////////////////////////////

static void update_robots ()
{
  rstart.update (qs);
  rend.update (qe);
  if (nq != robots.size()) {
    robots.resize (nq);
  }
  for (size_t ii (0); ii < nq; ++ii) {
    robots[ii].update (xi.block (ii * cdim, 0, cdim, 1));
  }
}


static void init_chomp ()
{
  qs.resize (cdim);
  qs << -5.0, -5.0;
  qe.resize (cdim);
  qe << 7.0, 7.0;
  
  xi = Vector::Zero (xidim);
  for (size_t ii (0); ii < nq; ++ii) {
    xi.block (cdim * ii, 0, cdim, 1) = qs;
  }
  
  AA = Matrix::Zero (xidim, xidim);
  for (size_t ii(0); ii < nq; ++ii) {
    AA.block (cdim * ii, cdim * ii, cdim , cdim) = 2.0 * Matrix::Identity (cdim, cdim);
    if (ii > 0) {
      AA.block (cdim * (ii-1), cdim * ii, cdim , cdim) = -1.0 * Matrix::Identity (cdim, cdim);
      AA.block (cdim * ii, cdim * (ii-1), cdim , cdim) = -1.0 * Matrix::Identity (cdim, cdim);
    }
  }
  AA /= dt * dt * (nq + 1);
  
  bb = Vector::Zero (xidim);
  bb.block (0,            0, cdim, 1) = qs;
  bb.block (xidim - cdim, 0, cdim, 1) = qe;
  bb /= - dt * dt * (nq + 1);
  
  // not needed anyhow
  // double cc (double (qs.transpose() * qs) + double (qe.transpose() * qe));
  // cc /= dt * dt * (nq + 1);
  
  Ainv = AA.inverse();
  
  // cout << "AA\n" << AA
  //      << "\nAinv\n" << Ainv
  //      << "\nbb\n" << bb << "\n\n";

}


static void cb_step ()
{
  state = STEP;
}


static void cb_run ()
{
  if (RUN == state) {
    state = PAUSE;
  }
  else {
    state = RUN;
  }
}


static void cb_jumble ()
{
  for (size_t ii (0); ii < xidim; ++ii) {
    xi[ii] = double (rand()) / (0.1 * std::numeric_limits<int>::max()) - 5.0;
  }
  update_robots();
}


static void cb_idle ()
{
  if (PAUSE == state) {
    return;
  }
  if (STEP == state) {
    state = PAUSE;
  }
  
  //////////////////////////////////////////////////
  // beginning of "the" CHOMP iteration
  
  // static size_t stepcounter (0);
  // cout << "step " << stepcounter++ << "\n";
  
  Vector nabla_smooth (AA * xi + bb);
  Vector const & xidd (nabla_smooth); // indeed, it is the same in this formulation...
  
  Vector nabla_obs (Vector::Zero (xidim));
  for (size_t iq (0); iq < nq; ++iq) {
    Vector const qq (xi.block (iq * cdim, 0, cdim, 1));
    Vector qd;
    if (0 == iq) {
      qd = 0.5 * (xi.block ((iq+1) * cdim, 0, cdim, 1) - qs);
    }
    else if (iq == nq - 1) {
      qd = 0.5 * (qe - xi.block ((iq-1) * cdim, 0, cdim, 1));
    }
    else {
      qd = 0.5 * (xi.block ((iq+1) * cdim, 0, cdim, 1) - xi.block ((iq-1) * cdim, 0, cdim, 1));;
    }
    
    // In this case, C and W are the same, Jacobian is identity.  We
    // still write more or less the full-fledged CHOMP expressions
    // (but we only use one body point) to make subsequent extension
    // easier.
    //
    Vector const & xx (qq);
    Vector const & xd (qd);
    Matrix const JJ (Matrix::Identity (2, 2)); // a little silly here, as noted above.
    double const vel (xd.norm());
    if (vel < 1.0e-3) {	// avoid div by zero further down
      continue;
    }
    Vector const xdn (xd / vel);
    Vector const xdd (JJ * xidd.block (iq * cdim, 0, cdim , 1));
    Matrix const prj (Matrix::Identity (2, 2) - xdn * xdn.transpose()); // hardcoded planar case
    Vector const kappa (prj * xdd / pow (vel, 2.0));
    
    int c (0); // c indicates which object we are dealing with... 1 = object1 , 1 = object 2
    static int loop;
    static double rotate1_track;
    static double rotate2_track;
    for (handle_s ** hh (handle); *hh != 0; ++hh) {
      ++c;
      Vector delta (xx - (*hh)->point_);
      double const dist (delta.norm());

      if (loop >= 2) {
        if (c==1) {
          if ((*hh)->point_ != old_obj1_pos) {
            double disp_x1 = (*hh)->point_[0] - old_obj1_pos[0];
            double disp_y1 = (*hh)->point_[1] - old_obj1_pos[1];
            double ang1 = atan(disp_y1/disp_x1);


            if (abs(disp_x1) == disp_x1 && abs(disp_y1) == disp_y1) {
              orientation1 = ang1;
            }
            else if (abs(disp_x1) != disp_x1 && abs(disp_y1) == disp_y1) {
              orientation1 = 3.141593 + ang1;
            }
            else if (abs(disp_x1) != disp_x1 && abs(disp_y1) != disp_y1) {
              orientation1 = ang1 + 3.141593;
            }
            else if (abs(disp_x1) == disp_x1 && abs(disp_y1) != disp_y1) {
              orientation1 = 2*3.141593 + ang1;
            }

            // calulate how much robot will need to rotate to point towards end position
            rotate1 = orientation1 - rotate1_track; // how much robot object 1 will have to rotate to point in the direction it needs to go


            rotate1_track = rotate1_track + rotate1;
            if (rotate1_track >= 2*3.141593) {
              rotate1_track = rotate1_track-2*3.141593;
            }

            if (abs(rotate1) != rotate1) {
              rotate1 = 2*3.141593 + rotate1;
            }

            // calculate distance from start goal to end goal
            Vector distance_obj1 = (*hh)->point_ - old_obj1_pos;
            double dist1 = distance_obj1.norm()/8; // hard coded distance reducer.  Reduces maximum step to 3 meters
            ROS_INFO("X distance: %f",disp_x1);
            ROS_INFO("Y distance: %f",disp_y1);
            ROS_INFO("Total Distance: %f",dist1);

            // store data required to be published
            obj1.pose.pose.position.x = dist1;
            obj1.pose.pose.orientation.z = rotate1;

            //publish to the topic
            pub_pid1.publish(obj1);


            //references for next iteration
            old_obj1_pos = (*hh)->point_;
            old_obj1_ore = orientation1;
          }

        }
        if (c==2) {
          if ((*hh)->point_ != old_obj2_pos) {
            //std::cout << "We made it here" << std::endl;
            double disp_x2 = (*hh)->point_[0] - old_obj2_pos[0];
            double disp_y2 = (*hh)->point_[1] - old_obj2_pos[1];
            double ang2 = atan(disp_y2/disp_x2);

            if (abs(disp_x2) == disp_x2 && abs(disp_y2) == disp_y2) {
              orientation2 = ang2;
            }
            else if (abs(disp_x2) != disp_x2 && abs(disp_y2) == disp_y2) {
              orientation2 = 3.141593 + ang2;
            }
            else if (abs(disp_x2) != disp_x2 && abs(disp_y2) != disp_y2) {
              orientation2 = ang2 + 3.141593;
            }
            else if (abs(disp_x2) == disp_x2 && abs(disp_y2) != disp_y2) {
              orientation2 = 2*3.141593 + ang2;
            }

            // calulate how much robot will need to rotate to point towards end position
            rotate2 = orientation2 - rotate2_track; // how much robot object 1 will have to rotate to point in the direction it needs to go


            rotate2_track = rotate2_track + rotate2;
            if (rotate2_track >= 2*3.141593) {
              rotate2_track = rotate2_track-2*3.141593;
            }

            if (abs(rotate2) != rotate2) {
              rotate2 = 2*3.141593 + rotate2;
            }

            // calculate the distance needed to travel from start goal to end goal
            Vector distance_obj2 = (*hh)->point_ - old_obj2_pos;
            double dist2 = distance_obj2.norm()/8;
            ROS_INFO("X distance: %f",disp_x2);
            ROS_INFO("Y distance: %f",disp_y2);
            ROS_INFO("Total Distance: %f",dist2);
            
            // load data to be publised
            obj2.pose.pose.position.x = dist2;
            obj2.pose.pose.orientation.z = rotate2;

            // publish data
            pub_pid2.publish(obj2);

            // provide reference for next iteration
            old_obj2_pos = (*hh)->point_;
            old_obj2_ore = orientation2;
          }

        }
      }
      else {
        if (c==1) {
          //std::cout << "checkpoint1" << std::endl;
          old_obj1_pos = (*hh)->point_;
          old_obj1_ore = 0;
        }
        if (c==2) {
          //std::cout << "checkpoint2" << std::endl;
          old_obj2_pos = (*hh)->point_;
          old_obj2_ore = 0;
        }
      }
      ++loop;

      if ((dist >= (*hh)->radius_) || (dist < 1e-9)) {
	continue;
      }
      static double const gain (10.0); // hardcoded param
      double const cost (gain * (*hh)->radius_ * pow (1.0 - dist / (*hh)->radius_, 3.0) / 3.0); // hardcoded param
      delta *= - gain * pow (1.0 - dist / (*hh)->radius_, 2.0) / dist; // hardcoded param
      nabla_obs.block (iq * cdim, 0, cdim, 1) += JJ.transpose() * vel * (prj * delta - cost * kappa);
    }
  }
  
  Vector dxi (Ainv * (nabla_obs + lambda * nabla_smooth));
  xi -= dxi / eta;
  //std::cout << std::endl << xi << std::endl;
  // end of "the" CHOMP iteration
  //////////////////////////////////////////////////
  
  update_robots ();
}


static void cb_draw ()
{
  //////////////////////////////////////////////////
  // set bounds
  
  Vector bmin (qs);
  Vector bmax (qs);
  for (size_t ii (0); ii < 2; ++ii) {
    if (qe[ii] < bmin[ii]) {
      bmin[ii] = qe[ii];
    }
    if (qe[ii] > bmax[ii]) {
      bmax[ii] = qe[ii];
    }
    for (size_t jj (0); jj < nq; ++jj) {
      if (xi[ii + cdim * jj] < bmin[ii]) {
	bmin[ii] = xi[ii + cdim * jj];
      }
      if (xi[ii + cdim * jj] > bmax[ii]) {
	bmax[ii] = xi[ii + cdim * jj];
      }
    }
  }
  
  gfx::set_view (bmin[0] - 2.0, bmin[1] - 2.0, bmax[0] + 2.0, bmax[1] + 2.0);
  
  //////////////////////////////////////////////////
  // robots
  
  rstart.draw();
  for (size_t ii (0); ii < robots.size(); ++ii) {
    robots[ii].draw();
  }
  rend.draw();
  
  //////////////////////////////////////////////////
  // trj
  
  gfx::set_pen (1.0, 0.2, 0.2, 0.2, 1.0);
  gfx::draw_line (qs[0], qs[1], xi[0], xi[1]);
  for (size_t ii (1); ii < nq; ++ii) {
    gfx::draw_line (xi[(ii-1) * cdim], xi[(ii-1) * cdim + 1], xi[ii * cdim], xi[ii * cdim + 1]);
  }
  gfx::draw_line (xi[(nq-1) * cdim], xi[(nq-1) * cdim + 1], qe[0], qe[1]);
  
  gfx::set_pen (5.0, 0.8, 0.2, 0.2, 1.0);
  gfx::draw_point (qs[0], qs[1]);
  gfx::set_pen (5.0, 0.5, 0.5, 0.5, 1.0);
  for (size_t ii (0); ii < nq; ++ii) {
    gfx::draw_point (xi[ii * cdim], xi[ii * cdim + 1]);
  }
  gfx::set_pen (5.0, 0.2, 0.8, 0.2, 1.0);
  gfx::draw_point (qe[0], qe[1]);
  
  //////////////////////////////////////////////////
  // handles
  
  for (handle_s ** hh (handle); *hh != 0; ++hh) {
    gfx::set_pen (1.0, (*hh)->red_, (*hh)->green_, (*hh)->blue_, (*hh)->alpha_);
    gfx::fill_arc ((*hh)->point_[0], (*hh)->point_[1], (*hh)->radius_, 0.0, 2.0 * M_PI);
  }
}


static void cb_mouse (double px, double py, int flags)
{
  if (flags & gfx::MOUSE_PRESS) {
    for (handle_s ** hh (handle); *hh != 0; ++hh) {
      Vector offset ((*hh)->point_);
      offset[0] -= px;
      offset[1] -= py;
      if (offset.norm() <= (*hh)->radius_) {
    	grab_offset = offset;
    	grabbed = *hh;
    	break;
      }
    }
  }
  else if (flags & gfx::MOUSE_DRAG) {
    if (0 != grabbed) {
      grabbed->point_[0] = px;
      grabbed->point_[1] = py;
      grabbed->point_ += grab_offset;
    }
  }
  else if (flags & gfx::MOUSE_RELEASE) {
    grabbed = 0;
  }
}


int main(int argc, char **argv)
{

  ros::init(argc,argv,"CHOMP");
  ros::NodeHandle n;
  pub_pid1 = n.advertise<nav_msgs::Odometry>("/turtlebot1/set_goal1",1000);
  pub_pid2 = n.advertise<nav_msgs::Odometry>("/turtlebot2/set_goal2",1000);

  ros::Rate loop_rate(10);

  while(ros::ok()) {
    struct timeval tt;
    gettimeofday (&tt, NULL);
    srand (tt.tv_usec);
    
    init_chomp();
    update_robots();  
    state = PAUSE;
    gfx::add_button ("jumble", cb_jumble);
    gfx::add_button ("step", cb_step);
    gfx::add_button ("run", cb_run);
    gfx::main ("chomp", cb_idle, cb_draw, cb_mouse);

    loop_rate.sleep();
  }

}
