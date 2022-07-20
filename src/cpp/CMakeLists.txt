cmake_minimum_required(VERSION 2.8)
project( spectreye )

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-g -O0")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( spectreye spectreye_cli.cpp spectreye.cpp )
target_include_directories( spectreye PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" )
target_link_libraries( spectreye ${OpenCV_LIBS} tesseract )

SET ( CMAKE_EXPORT_COMPILE_COMMANDS ON )
IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
  EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  )
ENDIF()


