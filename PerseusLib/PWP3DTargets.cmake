# Generated by CMake

if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" LESS 2.5)
   message(FATAL_ERROR "CMake >= 2.6.0 required")
endif()
cmake_policy(PUSH)
cmake_policy(VERSION 2.6)
#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Protect against multiple inclusion, which would fail when already imported targets are added once more.
set(_targetsDefined)
set(_targetsNotDefined)
set(_expectedTargets)
foreach(_expectedTarget pwp3d)
  list(APPEND _expectedTargets ${_expectedTarget})
  if(NOT TARGET ${_expectedTarget})
    list(APPEND _targetsNotDefined ${_expectedTarget})
  endif()
  if(TARGET ${_expectedTarget})
    list(APPEND _targetsDefined ${_expectedTarget})
  endif()
endforeach()
if("${_targetsDefined}" STREQUAL "${_expectedTargets}")
  unset(_targetsDefined)
  unset(_targetsNotDefined)
  unset(_expectedTargets)
  set(CMAKE_IMPORT_FILE_VERSION)
  cmake_policy(POP)
  return()
endif()
if(NOT "${_targetsDefined}" STREQUAL "")
  message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_targetsDefined}\nTargets not yet defined: ${_targetsNotDefined}\n")
endif()
unset(_targetsDefined)
unset(_targetsNotDefined)
unset(_expectedTargets)


# Create imported target pwp3d
add_library(pwp3d SHARED IMPORTED)

# Import target "pwp3d" for configuration "Release"
set_property(TARGET pwp3d APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pwp3d PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "/opt/cuda/lib64/libcudart_static.a;-lpthread;dl;/usr/lib/librt.so;/opt/cuda/lib64/libnppc.so;/opt/cuda/lib64/libnppial.so;/opt/cuda/lib64/libnppicc.so;/opt/cuda/lib64/libnppicom.so;/opt/cuda/lib64/libnppidei.so;/opt/cuda/lib64/libnppif.so;/opt/cuda/lib64/libnppig.so;/opt/cuda/lib64/libnppim.so;/opt/cuda/lib64/libnppist.so;/opt/cuda/lib64/libnppisu.so;/opt/cuda/lib64/libnppitc.so;/opt/cuda/lib64/libnpps.so;/usr/lib/libfreeimage.so;/usr/lib/libassimp.so;opencv_core;opencv_imgproc;opencv_features2d;opencv_flann;opencv_calib3d;opencv_objdetect;opencv_highgui;opencv_ml;opencv_video;opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_gapi;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio;opencv_aruco;opencv_bgsegm;opencv_bioinspired;opencv_ccalib;opencv_cvv;opencv_datasets;opencv_dnn_objdetect;opencv_dpm;opencv_face;opencv_freetype;opencv_fuzzy;opencv_hdf;opencv_hfs;opencv_img_hash;opencv_line_descriptor;opencv_optflow;opencv_phase_unwrapping;opencv_plot;opencv_quality;opencv_reg;opencv_rgbd;opencv_saliency;opencv_shape;opencv_stereo;opencv_structured_light;opencv_superres;opencv_surface_matching;opencv_text;opencv_tracking;opencv_videostab;opencv_viz;opencv_xfeatures2d;opencv_ximgproc;opencv_xobjdetect;opencv_xphoto"
  IMPORTED_LOCATION_RELEASE "/home/phon/Development/Projects/PWP3D/PerseusLib/libpwp3d.so"
  IMPORTED_SONAME_RELEASE "libpwp3d.so"
  )

# This file does not depend on other imported targets which have
# been exported from the same project but in a separate export set.

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
cmake_policy(POP)
