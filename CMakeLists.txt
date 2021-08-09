cmake_minimum_required(VERSION 3.0.2)
project(chomp)


## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  geometry_msgs
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(gtk REQUIRED gtk+-2.0)

find_package(GTK2 REQUIRED)
find_package(GTK2 COMPONENTS gtk)


generate_messages(DEPENDENCIES geometry_msgs)

catkin_package(
#  INCLUDE_DIRS include
   LIBRARIES lib
#  CATKIN_DEPENDS roscpp rospy
   DEPENDS system_lib opencv gtk gfx.hpp
)

include_directories ( include
	${catkin_INCLUDE_DIRS}
	${GTK2_INCLUDE_DIRS}
	${GTK2_CFLAGS}
	src/trychomp/library
	src/trychomp/executable
)

add_subdirectory(src/trychomp/library)
add_subdirectory(src/trychomp/executable)
