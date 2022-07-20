cmake_minimum_required(VERSION 2.8)
project( spectreye-cli )

set (CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
find_library(SPECTREYE_LIB libspectreye.so)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( spectreye-cli spectreye_cli.cpp)
target_include_directories( spectreye-cli PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" )
target_link_libraries( spectreye-cli ${OpenCV_LIBS} tesseract ${SPECTREYE_LIB})



