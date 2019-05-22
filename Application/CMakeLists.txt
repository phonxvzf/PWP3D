cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra ")
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

file(GLOB SOURCES "src/*.cpp" "src/*/*.cpp" "src/*.c")

find_package(OpenCV REQUIRED core imgproc features2d flann calib3d objdetect
  highgui ml video
)
find_package(PWP3D REQUIRED)
find_package(assimp REQUIRED)

include_directories(${PWP3D_INCLUDE_DIRS})

add_executable(PWP3DAPP ${SOURCES})
target_link_libraries(PWP3DAPP ${PWP3D_LIBRARIES})

