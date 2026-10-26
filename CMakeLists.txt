cmake_minimum_required(VERSION 3.25)
project(trt_sr_upscaler VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TensorRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT installation directory")
if(NOT TensorRT_ROOT)
  message(FATAL_ERROR "Set TensorRT_ROOT to the TensorRT SDK directory")
endif()
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
  src/Sizing.cpp
  src/ImageTiles.cpp
  src/TilePipeline.cpp
)
target_include_directories(trt_sr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(trt_sr PRIVATE "${TENSORRT_INCLUDE}" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trt_sr PRIVATE ${OpenCV_LIBS} CUDA::cudart "${TENSORRT_INFER}" "${TENSORRT_PARSER}")

if(WIN32)
  foreach(_trt_runtime IN ITEMS nvinfer_10.dll nvonnxparser_10.dll)
    set(_trt_runtime_path "${TensorRT_ROOT}/bin/${_trt_runtime}")
    if(NOT EXISTS "${_trt_runtime_path}")
      message(FATAL_ERROR "TensorRT runtime DLL not found: ${_trt_runtime_path}")
    endif()
    add_custom_command(TARGET trt_sr POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_trt_runtime_path}" "$<TARGET_FILE_DIR:trt_sr>"
      VERBATIM)
  endforeach()

  file(GLOB _cuda_runtime_dlls
    "${CUDAToolkit_BIN_DIR}/cudart64_*.dll"
    "${CUDAToolkit_BIN_DIR}/x64/cudart64_*.dll")
  if(NOT _cuda_runtime_dlls)
    message(FATAL_ERROR "Could not find cudart64_*.dll under ${CUDAToolkit_BIN_DIR}")
  endif()
  list(GET _cuda_runtime_dlls 0 _cuda_runtime_dll)
  add_custom_command(TARGET trt_sr POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_cuda_runtime_dll}" "$<TARGET_FILE_DIR:trt_sr>"
    VERBATIM)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(options_tests tests/OptionsTests.cpp src/Options.cpp)
  target_include_directories(options_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME options_parse COMMAND options_tests)
  add_executable(image_tiles_tests tests/ImageTilesTests.cpp src/ImageTiles.cpp)
  target_include_directories(image_tiles_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_include_directories(image_tiles_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(image_tiles_tests PRIVATE ${OpenCV_LIBS})
  add_test(NAME image_tiles COMMAND image_tiles_tests)
  add_executable(tile_pipeline_tests tests/TilePipelineTests.cpp src/TilePipeline.cpp src/ImageTiles.cpp)
  target_include_directories(tile_pipeline_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(tile_pipeline_tests PRIVATE ${OpenCV_LIBS})
  add_test(NAME tile_pipeline COMMAND tile_pipeline_tests)
  add_executable(tile_pipeline_progress_tests tests/TilePipelineProgressTests.cpp src/TilePipeline.cpp src/ImageTiles.cpp)
  target_include_directories(tile_pipeline_progress_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(tile_pipeline_progress_tests PRIVATE ${OpenCV_LIBS})
  add_test(NAME tile_pipeline_progress COMMAND tile_pipeline_progress_tests)
  add_executable(gpu_inference_tests
    tests/GpuInferenceTests.cpp
    src/Engine.cpp
    src/Upscaler.cpp
    src/Sizing.cpp
  src/Sizing.cpp
    src/TilePipeline.cpp
    src/ImageTiles.cpp
  )
  target_include_directories(gpu_inference_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${TENSORRT_INCLUDE}" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(gpu_inference_tests PRIVATE
    ${OpenCV_LIBS} CUDA::cudart "${TENSORRT_INFER}" "${TENSORRT_PARSER}")
  target_compile_definitions(gpu_inference_tests PRIVATE
    TEXUPSR_GPU_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/relu.onnx")
  if(WIN32)
    add_dependencies(gpu_inference_tests trt_sr)
    foreach(_trt_dll IN ITEMS nvinfer_10.dll nvonnxparser_10.dll)
      add_custom_command(TARGET gpu_inference_tests POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${TensorRT_ROOT}/bin/${_trt_dll}" "$<TARGET_FILE_DIR:gpu_inference_tests>"
        VERBATIM)
    endforeach()
    file(GLOB _trt_builder_dlls "${TensorRT_ROOT}/bin/nvinfer_builder_resource_*.dll")
    foreach(_builder_dll IN LISTS _trt_builder_dlls)
      add_custom_command(TARGET gpu_inference_tests POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_builder_dll}" "$<TARGET_FILE_DIR:gpu_inference_tests>"
        VERBATIM)
    endforeach()
    add_custom_command(TARGET gpu_inference_tests POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${_cuda_runtime_dll}" "$<TARGET_FILE_DIR:gpu_inference_tests>"
      VERBATIM)
  endif()
  add_test(NAME gpu_inference COMMAND gpu_inference_tests)
  set_tests_properties(gpu_inference PROPERTIES LABELS "gpu;integration" SKIP_RETURN_CODE 77)

  add_executable(sizing_tests tests/SizingTests.cpp src/Sizing.cpp)
  target_include_directories(sizing_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME sizing COMMAND sizing_tests)
endif()
