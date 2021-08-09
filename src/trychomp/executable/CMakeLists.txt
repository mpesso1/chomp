add_executable(exe pp2d.cpp)

target_link_libraries(exe PUBLIC lib ${catkin_LIBRARIES} ${GTK2_LIBRARIES})

