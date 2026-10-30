cmake_minimum_required(VERSION 3.25)
project(trt_sr_upscaler VERSION 0.1.0 LANGUAGES CXX)
include(cmake/CppStandards.cmake)

option(TEXUPSR_BUILD_APP "Build the TensorRT upscaler" ON)
option(TEXUPSR_BUILD_GPU_TESTS "Build the separate GPU integration test" OFF)

if(TEXUPSR_BUILD_GPU_TESTS AND NOT TEXUPSR_BUILD_APP)
  message(FATAL_ERROR "GPU integration tests require TEXUPSR_BUILD_APP=ON")
endif()

find_package(OpenCV REQUIRED)

if(TEXUPSR_BUILD_APP)
  find_package(Protobuf QUIET CONFIG)
  if(NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
  endif()
  if(WIN32)
    # The default is a shared Protobuf runtime. Static linking is an explicit opt-in.
    set(TEXUPSR_PROTOBUF_LINKAGE "SHARED" CACHE STRING "Protobuf linkage: SHARED or STATIC")
    set_property(CACHE TEXUPSR_PROTOBUF_LINKAGE PROPERTY STRINGS SHARED STATIC)
    if(NOT TEXUPSR_PROTOBUF_LINKAGE STREQUAL "SHARED" AND
       NOT TEXUPSR_PROTOBUF_LINKAGE STREQUAL "STATIC")
      message(FATAL_ERROR "TEXUPSR_PROTOBUF_LINKAGE must be SHARED or STATIC")
    endif()
    get_target_property(_pb_target_type protobuf::libprotobuf TYPE)
    get_target_property(_pb_location protobuf::libprotobuf IMPORTED_LOCATION_RELEASE)
    if(NOT _pb_location)
      get_target_property(_pb_location protobuf::libprotobuf IMPORTED_LOCATION)
    endif()
    if(TEXUPSR_PROTOBUF_LINKAGE STREQUAL "SHARED")
      if(NOT _pb_location OR NOT _pb_location MATCHES "[.]dll$")
        message(FATAL_ERROR
          "TexUpSR defaults to shared Protobuf, but protobuf::libprotobuf has no DLL location. "
          "Install a dynamically linked Protobuf package (vcpkg x64-windows), "
          "or explicitly select -DTEXUPSR_PROTOBUF_LINKAGE=STATIC with a static package.")
      endif()
    elseif(_pb_location AND _pb_location MATCHES "[.]dll$")
      message(FATAL_ERROR "Static Protobuf requested, but protobuf::libprotobuf points to a DLL")
    endif()
  endif()
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/onnx-generated")
  add_library(model_inspection STATIC src/ModelInspection.cpp)
  set_target_properties(model_inspection PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  target_include_directories(model_inspection PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/onnx-generated")
  target_link_libraries(model_inspection PUBLIC protobuf::libprotobuf)
  if(WIN32 AND TEXUPSR_PROTOBUF_LINKAGE STREQUAL "SHARED")
    target_compile_definitions(model_inspection PUBLIC PROTOBUF_USE_DLLS)
  endif()
  protobuf_generate(TARGET model_inspection LANGUAGE cpp APPEND_PATH
    PROTOS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx/onnx.proto"
    PROTOC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnx-generated")
  add_custom_target(onnx_generated
    DEPENDS
      "${CMAKE_CURRENT_BINARY_DIR}/onnx-generated/onnx.pb.h"
      "${CMAKE_CURRENT_BINARY_DIR}/onnx-generated/onnx.pb.cc")

  set(TensorRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "Optional TensorRT SDK root (empty for system search)")
  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE NAMES NvInfer.h
    HINTS "${TensorRT_ROOT}/include" "${TensorRT_ROOT}/include/x86_64-linux-gnu" "${TensorRT_ROOT}/include/aarch64-linux-gnu"
    REQUIRED)
  find_library(TENSORRT_INFER NAMES nvinfer_10 nvinfer
    HINTS "${TensorRT_ROOT}/lib" "${TensorRT_ROOT}/lib64" "${TensorRT_ROOT}/lib/x86_64-linux-gnu" "${TensorRT_ROOT}/lib/aarch64-linux-gnu"
    REQUIRED)
  find_library(TENSORRT_PARSER NAMES nvonnxparser_10 nvonnxparser
    HINTS "${TensorRT_ROOT}/lib" "${TensorRT_ROOT}/lib64" "${TensorRT_ROOT}/lib/x86_64-linux-gnu" "${TensorRT_ROOT}/lib/aarch64-linux-gnu"
    REQUIRED)
  add_executable(trt_sr
    main.cpp
    src/Options.cpp
    src/Engine.cpp
    src/Upscaler.cpp
    src/TensorSizes.cpp
    src/Sizing.cpp
    src/ImageTiles.cpp
    src/ModelProfile.cpp
    src/TilePipeline.cpp
  )
  target_include_directories(trt_sr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_include_directories(trt_sr PRIVATE "${TENSORRT_INCLUDE}" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(trt_sr PRIVATE model_inspection ${OpenCV_LIBS} CUDA::cudart "${TENSORRT_INFER}" "${TENSORRT_PARSER}")

  if(WIN32)
    if(TEXUPSR_PROTOBUF_LINKAGE STREQUAL "SHARED")
      add_custom_command(TARGET trt_sr POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_pb_location}" "$<TARGET_FILE_DIR:trt_sr>"
        VERBATIM)
    endif()
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
endif()

include(CTest)
if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)

  add_executable(options_tests tests/OptionsTests.cpp src/Options.cpp)
  target_include_directories(options_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(options_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(options_tests DISCOVERY_TIMEOUT 30)
  add_executable(model_profile_tests tests/ModelProfileTests.cpp src/ImageTiles.cpp src/ModelProfile.cpp)
  target_include_directories(model_profile_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(model_profile_tests PRIVATE ${OpenCV_LIBS} GTest::gtest_main)
  gtest_discover_tests(model_profile_tests DISCOVERY_TIMEOUT 30)
  add_executable(image_tiles_tests tests/ImageTilesTests.cpp src/ImageTiles.cpp src/ModelProfile.cpp)
  target_include_directories(image_tiles_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_include_directories(image_tiles_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(image_tiles_tests PRIVATE ${OpenCV_LIBS})
  target_link_libraries(image_tiles_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(image_tiles_tests DISCOVERY_TIMEOUT 30)
  add_executable(tile_pipeline_tests tests/TilePipelineTests.cpp src/TilePipeline.cpp src/ImageTiles.cpp src/ModelProfile.cpp)
  target_include_directories(tile_pipeline_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(tile_pipeline_tests PRIVATE ${OpenCV_LIBS})
  target_link_libraries(tile_pipeline_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(tile_pipeline_tests DISCOVERY_TIMEOUT 30)
  add_executable(tile_pipeline_progress_tests tests/TilePipelineProgressTests.cpp src/TilePipeline.cpp src/ImageTiles.cpp src/ModelProfile.cpp)
  target_include_directories(tile_pipeline_progress_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(tile_pipeline_progress_tests PRIVATE ${OpenCV_LIBS})
  target_link_libraries(tile_pipeline_progress_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(tile_pipeline_progress_tests DISCOVERY_TIMEOUT 30)

  add_executable(sizing_tests tests/SizingTests.cpp src/Sizing.cpp)
  target_include_directories(sizing_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(sizing_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(sizing_tests DISCOVERY_TIMEOUT 30)

  add_executable(tensor_sizes_tests tests/TensorSizesTests.cpp src/TensorSizes.cpp)
  target_include_directories(tensor_sizes_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(tensor_sizes_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(tensor_sizes_tests DISCOVERY_TIMEOUT 30)

  if(TEXUPSR_BUILD_APP)
  add_executable(model_inspection_tests tests/ModelInspectionTests.cpp)
  target_link_libraries(model_inspection_tests PRIVATE model_inspection GTest::gtest_main)
  if(WIN32 AND TEXUPSR_PROTOBUF_LINKAGE STREQUAL "SHARED")
    add_custom_command(TARGET model_inspection_tests POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_pb_location}" "$<TARGET_FILE_DIR:model_inspection_tests>"
      VERBATIM)
  endif()
  gtest_discover_tests(model_inspection_tests DISCOVERY_TIMEOUT 30)
  add_executable(cuda_half_tests tests/CudaHalfTests.cpp)
  target_include_directories(cuda_half_tests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(cuda_half_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(cuda_half_tests DISCOVERY_TIMEOUT 30)
  endif()
endif()

if(TEXUPSR_BUILD_GPU_TESTS)
  add_executable(gpu_inference_tests
    tests/GpuInferenceTests.cpp
    src/Engine.cpp
    src/Upscaler.cpp
    src/TensorSizes.cpp
    src/Sizing.cpp
    src/TilePipeline.cpp
    src/ImageTiles.cpp
    src/ModelProfile.cpp
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
endif()

if(TEXUPSR_BUILD_APP)
  include(cmake/QualityTargets.cmake)
endif()
