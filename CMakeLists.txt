cmake_minimum_required(VERSION 3.10)
project(opencv_project)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

aux_source_directory(tools TOOLS)
add_executable(shit src/main.cpp)

message(${TOOLS})
# 链接库 静态库，动态库，对象库？
target_link_libraries(shit ${OpenCV_LIBS})

add_library(TOOLS_STATIC STATIC ${TOOLS})
target_link_libraries(shit TOOLS_STATIC)
