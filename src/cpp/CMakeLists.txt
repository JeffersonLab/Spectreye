cmake_minimum_required(VERSION 2.8)
project( spectreye )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( spectreye spectreye.cpp )
target_include_directories( spectreye PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" )
target_link_libraries( spectreye ${OpenCV_LIBS} tesseract )
