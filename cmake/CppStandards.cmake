# Keep the baseline available even when a newer standard is installed.
set(TEXUP_CXX_STANDARD "AUTO" CACHE STRING "C++ standard: AUTO, 17, 20, or 23")
set_property(CACHE TEXUP_CXX_STANDARD PROPERTY STRINGS AUTO 17 20 23)

if(NOT TEXUP_CXX_STANDARD MATCHES "^(AUTO|17|20|23)$")
  message(FATAL_ERROR "TEXUP_CXX_STANDARD must be AUTO, 17, 20, or 23")
endif()

# CMake's compile-feature table distinguishes standards the selected compiler
# supports. The actual CUDA/TensorRT/OpenCV build remains the compatibility test.
if(TEXUP_CXX_STANDARD STREQUAL "AUTO")
  set(_texup_candidate 17)
  foreach(_standard IN ITEMS 23 20 17)
    if("cxx_std_${_standard}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
      set(_texup_candidate "${_standard}")
      break()
    endif()
  endforeach()
else()
  set(_texup_candidate "${TEXUP_CXX_STANDARD}")
  if(NOT "cxx_std_${_texup_candidate}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER_ID} does not support C++${_texup_candidate}")
  endif()
endif()

set(CMAKE_CXX_STANDARD "${_texup_candidate}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "TexUpSR C++ standard: C++${CMAKE_CXX_STANDARD} (requested: ${TEXUP_CXX_STANDARD})")
