cmake_minimum_required(VERSION 2.8)
project(spectreye)

set (CMAKE_CXX_STANDARD 11)

if(NOT EAST)
	set(EAST "${CMAKE_CURRENT_SOURCE_DIR}/data/east.pb")
endif()
add_definitions(-DEAST_PATH="${EAST}")
message("east path: ${EAST}")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(spectreye SHARED src/spectreye.cpp)

target_include_directories(spectreye PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(spectreye ${OpenCV_LIBS} tesseract)

install(TARGETS spectreye DESTINATION /usr/lib)
install(FILES include/spectreye.h DESTINATION include)

#add_executable( spectreye src/spectreye_cli.cpp src/spectreye.cpp )


