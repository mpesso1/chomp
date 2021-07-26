add_executable(exe pp2d.cpp)

target_link_libraries(exe PUBLIC lib)

target_link_libraries(exe PUBLIC ${catkin_LIBRARIES} ${GTK2_LIBRARIES})
