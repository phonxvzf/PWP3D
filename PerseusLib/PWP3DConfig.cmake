# Compute paths
get_filename_component( PROJECT_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH )
SET( PWP3D_INCLUDE_DIRS "/home/phon/Development/Projects/PWP3D/PerseusLib/..;/home/phon/Development/Projects/PWP3D/PerseusLib/..;/opt/cuda/include;/usr/include/eigen3;/usr/include;/usr/include;/usr/include/opencv4" )

# Library dependencies (contains definitions for IMPORTED targets)
if( NOT TARGET pwp3d AND NOT PWP3D_BINARY_DIR )
  include( "${PROJECT_CMAKE_DIR}/PWP3DTargets.cmake" )
endif()

SET( PWP3D_LIBRARIES "pwp3d" )
