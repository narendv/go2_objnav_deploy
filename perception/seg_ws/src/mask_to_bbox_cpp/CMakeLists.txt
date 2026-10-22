cmake_minimum_required(VERSION 3.8)
project(mask_to_bbox_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(mask_to_bbox src/mask_to_bbox.cpp)

ament_target_dependencies(
  mask_to_bbox
  rclcpp
  sensor_msgs
  vision_msgs
  cv_bridge
  OpenCV
)

install(TARGETS
  mask_to_bbox
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/bb_to_global.py
  scripts/bb_to_global_pose_graph.py
  scripts/to_json.py
  scripts/new_segmentation/segment_ros2.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
