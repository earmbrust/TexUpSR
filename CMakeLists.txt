cmake_minimum_required(VERSION 3.20)
project(trt_sr_upscaler LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TensorRT_ROOT "E:/Projects/TensorRT-10.13.3.9" CACHE PATH "TensorRT installation directory")
set(OpenCV_DIR "C:/opencv/build" CACHE PATH "OpenCV CMake package directory")
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_path(TENSORRT_INCLUDE NvInfer.h PATHS "${TensorRT_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
find_library(TENSORRT_INFER NAMES nvinfer_10 nvinfer PATHS "${TensorRT_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
find_library(TENSORRT_PARSER NAMES nvonnxparser_10 nvonnxparser PATHS "${TensorRT_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
add_executable(trt_sr
  main.cpp
  src/Options.cpp
  src/Engine.cpp
  src/Upscaler.cpp
  src/ImageTiles.cpp
)
target_include_directories(trt_sr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(trt_sr PRIVATE "${TENSORRT_INCLUDE}" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trt_sr PRIVATE ${OpenCV_LIBS} CUDA::cudart "${TENSORRT_INFER}" "${TENSORRT_PARSER}")
if(WIN32)
  file(GLOB TRT_DLLS "${TensorRT_ROOT}/lib/*.dll")
  foreach(dll IN LISTS TRT_DLLS)
    add_custom_command(TARGET trt_sr POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:trt_sr>")
  endforeach()
  file(GLOB OPENCV_DLLS "${OpenCV_DIR}/x64/vc16/bin/*.dll")
  foreach(dll IN LISTS OPENCV_DLLS)
    add_custom_command(TARGET trt_sr POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:trt_sr>")
  endforeach()
  if(EXISTS "${CUDAToolkit_BIN_DIR}/cudart64_130.dll")
    add_custom_command(TARGET trt_sr POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDAToolkit_BIN_DIR}/cudart64_130.dll" "$<TARGET_FILE_DIR:trt_sr>")
  endif()
endif()

# TexUpSR TensorRT runtime deployment
# Import libraries live under lib; Windows runtime DLLs live under bin.
foreach(_trt_runtime IN ITEMS nvinfer_10.dll nvonnxparser_10.dll)
    set(_trt_runtime_path "${TensorRT_ROOT}/bin/${_trt_runtime}")
    if(NOT EXISTS "${_trt_runtime_path}")
        message(FATAL_ERROR "TensorRT runtime DLL not found: ${_trt_runtime_path}")
    endif()
    add_custom_command(TARGET trt_sr POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_trt_runtime_path}"
            "$<TARGET_FILE_DIR:trt_sr>"
        VERBATIM
        COMMENT "Deploying TensorRT runtime: ${_trt_runtime}"
    )
endforeach()
