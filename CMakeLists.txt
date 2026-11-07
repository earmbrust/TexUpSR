cmake_minimum_required(VERSION 3.25)

# vcpkg supplies dependency discovery, not runtime DLL deployment.
# Disabling app-local copying keeps the installed DLLs outside build outputs.
set(VCPKG_APPLOCAL_DEPS OFF CACHE BOOL "Do not copy vcpkg DLLs into build outputs" FORCE)

# TexUpSR vcpkg auto-discovery: honor an explicit toolchain first.
# Resolve a standard VCPKG_ROOT installation or a sibling checkout without
# embedding any developer-specific paths in the project.
if(CMAKE_HOST_WIN32 AND NOT CMAKE_TOOLCHAIN_FILE)
  set(_texupsr_vcpkg_roots)
  if(DEFINED ENV{VCPKG_ROOT} AND NOT "$ENV{VCPKG_ROOT}" STREQUAL "")
    list(APPEND _texupsr_vcpkg_roots "$ENV{VCPKG_ROOT}")
  endif()
  if(DEFINED ENV{VCPKG_INSTALLATION_ROOT} AND NOT "$ENV{VCPKG_INSTALLATION_ROOT}" STREQUAL "")
    list(APPEND _texupsr_vcpkg_roots "$ENV{VCPKG_INSTALLATION_ROOT}")
  endif()
  list(APPEND _texupsr_vcpkg_roots "${CMAKE_CURRENT_LIST_DIR}/../vcpkg")
  foreach(_texupsr_vcpkg_root IN LISTS _texupsr_vcpkg_roots)
    if(EXISTS "${_texupsr_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
      set(CMAKE_TOOLCHAIN_FILE "${_texupsr_vcpkg_root}/scripts/buildsystems/vcpkg.cmake"
          CACHE FILEPATH "vcpkg toolchain selected by TexUpSR")
      message(STATUS "TexUpSR using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
      break()
    endif()
  endforeach()
  unset(_texupsr_vcpkg_roots)
  unset(_texupsr_vcpkg_root)
endif()

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

  set(TensorRT_ROOT "" CACHE PATH "Optional TensorRT SDK location hint")
  # Use the configured SDK first; otherwise search common environment hints and
  # a sibling dependencies directory. No machine-specific paths are stored here.
  if(NOT TensorRT_ROOT)
    set(_texupsr_trt_candidates)
    foreach(_texupsr_trt_var IN ITEMS TensorRT_ROOT TENSORRT_ROOT TensorRT_DIR)
      if(DEFINED ENV{${_texupsr_trt_var}} AND NOT "$ENV{${_texupsr_trt_var}}" STREQUAL "")
        list(APPEND _texupsr_trt_candidates "$ENV{${_texupsr_trt_var}}")
      endif()
    endforeach()
    file(GLOB _texupsr_trt_siblings LIST_DIRECTORIES true
      "${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/TensorRT-*")
    list(SORT _texupsr_trt_siblings COMPARE NATURAL ORDER DESCENDING)
    list(APPEND _texupsr_trt_candidates ${_texupsr_trt_siblings})
    foreach(_texupsr_trt_candidate IN LISTS _texupsr_trt_candidates)
      if(EXISTS "${_texupsr_trt_candidate}/include/NvInfer.h" AND
         (EXISTS "${_texupsr_trt_candidate}/lib/nvinfer_10.lib" OR
          EXISTS "${_texupsr_trt_candidate}/lib/nvinfer.lib") AND
         (EXISTS "${_texupsr_trt_candidate}/lib/nvonnxparser_10.lib" OR
          EXISTS "${_texupsr_trt_candidate}/lib/nvonnxparser.lib"))
        set(TensorRT_ROOT "${_texupsr_trt_candidate}")
        message(STATUS "TexUpSR using TensorRT SDK: ${TensorRT_ROOT}")
        break()
      endif()
    endforeach()
    unset(_texupsr_trt_candidates)
    unset(_texupsr_trt_siblings)
    unset(_texupsr_trt_candidate)
    unset(_texupsr_trt_var)
  endif()
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

endif()

include(CTest)
if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)

  add_executable(engine_cache_tests tests/EngineCacheTests.cpp)
  target_include_directories(engine_cache_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(engine_cache_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(engine_cache_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)

  add_executable(options_tests tests/OptionsTests.cpp src/Options.cpp)
  target_include_directories(options_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(options_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(options_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)
  add_executable(model_profile_tests tests/ModelProfileTests.cpp src/ImageTiles.cpp src/ModelProfile.cpp)
  target_include_directories(model_profile_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(model_profile_tests PRIVATE ${OpenCV_LIBS} GTest::gtest_main)
  gtest_discover_tests(model_profile_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)
  add_executable(image_tiles_tests tests/ImageTilesTests.cpp src/ImageTiles.cpp src/ModelProfile.cpp)
  target_include_directories(image_tiles_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_include_directories(image_tiles_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(image_tiles_tests PRIVATE ${OpenCV_LIBS})
  target_link_libraries(image_tiles_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(image_tiles_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)
  add_executable(tile_pipeline_tests tests/TilePipelineTests.cpp src/TilePipeline.cpp src/ImageTiles.cpp src/ModelProfile.cpp)
  target_include_directories(tile_pipeline_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(tile_pipeline_tests PRIVATE ${OpenCV_LIBS})
  target_link_libraries(tile_pipeline_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(tile_pipeline_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)
  add_executable(tile_pipeline_progress_tests tests/TilePipelineProgressTests.cpp src/TilePipeline.cpp src/ImageTiles.cpp src/ModelProfile.cpp)
  target_include_directories(tile_pipeline_progress_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(tile_pipeline_progress_tests PRIVATE ${OpenCV_LIBS})
  target_link_libraries(tile_pipeline_progress_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(tile_pipeline_progress_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)

  add_executable(sizing_tests tests/SizingTests.cpp src/Sizing.cpp)
  target_include_directories(sizing_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(sizing_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(sizing_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)

  add_executable(tensor_sizes_tests tests/TensorSizesTests.cpp src/TensorSizes.cpp)
  target_include_directories(tensor_sizes_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(tensor_sizes_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(tensor_sizes_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)

  if(TEXUPSR_BUILD_APP)
  add_executable(model_inspection_tests tests/ModelInspectionTests.cpp)
  target_link_libraries(model_inspection_tests PRIVATE model_inspection GTest::gtest_main)
  gtest_discover_tests(model_inspection_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)
  add_executable(cuda_half_tests tests/CudaHalfTests.cpp)
  target_include_directories(cuda_half_tests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(cuda_half_tests PRIVATE GTest::gtest_main)
  gtest_discover_tests(cuda_half_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)
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
    TEXUPSR_GPU_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/relu.onnx"
    TEXUPSR_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/fixtures")
  add_test(NAME gpu_inference COMMAND gpu_inference_tests)
  set_tests_properties(gpu_inference PROPERTIES SKIP_RETURN_CODE 77 LABELS integration)
endif()

if(TEXUPSR_BUILD_APP)
  include(cmake/QualityTargets.cmake)
endif()
