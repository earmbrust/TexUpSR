# Native CMake quality targets. The Ninja analysis preset supplies compile_commands.json.
option(TEXUPSR_ENABLE_QUALITY "Expose developer static-analysis targets" ON)
if(NOT TEXUPSR_ENABLE_QUALITY)
  return()
endif()

# Query the targets instead of maintaining a second hand-written C++ source list.
get_property(_quality_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
set(_quality_compiled)
foreach(_target IN LISTS _quality_targets)
  get_target_property(_target_type "${_target}" TYPE)
  if(NOT _target_type STREQUAL "EXECUTABLE" AND
     NOT _target_type STREQUAL "STATIC_LIBRARY" AND
     NOT _target_type STREQUAL "SHARED_LIBRARY" AND
     NOT _target_type STREQUAL "OBJECT_LIBRARY")
    continue()
  endif()
  get_target_property(_sources "${_target}" SOURCES)
  foreach(_source IN LISTS _sources)
    if(_source MATCHES "\\.cpp$")
      cmake_path(ABSOLUTE_PATH _source BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)
      list(APPEND _quality_compiled "${_source}")
    endif()
  endforeach()
endforeach()
list(REMOVE_DUPLICATES _quality_compiled)
list(SORT _quality_compiled)

# Catch newly created, unregistered translation units instead of silently skipping them.
file(GLOB_RECURSE _all_cpp
  "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/tests/*.cpp")
if(TEXUPSR_BUILD_APP)
  list(APPEND _all_cpp "${CMAKE_SOURCE_DIR}/main.cpp")
else()
  list(REMOVE_ITEM _all_cpp
    "${CMAKE_SOURCE_DIR}/src/Engine.cpp"
    "${CMAKE_SOURCE_DIR}/src/Upscaler.cpp")
endif()
if(NOT TEXUPSR_BUILD_GPU_TESTS)
  list(REMOVE_ITEM _all_cpp "${CMAKE_SOURCE_DIR}/tests/GpuInferenceTests.cpp")
endif()
foreach(_source IN LISTS _all_cpp)
  if(NOT _source IN_LIST _quality_compiled)
    message(FATAL_ERROR "C++ source has no CMake build target: ${_source}")
  endif()
endforeach()

file(GLOB_RECURSE _headers
  "${CMAKE_SOURCE_DIR}/src/*.hpp" "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/tests/*.hpp" "${CMAKE_SOURCE_DIR}/tests/*.h")
set(_format_files ${_quality_compiled} ${_headers})
list(REMOVE_DUPLICATES _format_files)
list(SORT _format_files)

# A generated include file passes source lists to CMake -P without shell quoting tricks.
set(_manifest "${CMAKE_CURRENT_BINARY_DIR}/quality-manifest.cmake")
file(WRITE "${_manifest}" "set(QUALITY_CPP [==[${_quality_compiled}]==])\nset(QUALITY_FORMAT [==[${_format_files}]==])\n")

foreach(_mode IN ITEMS format-check tidy cppcheck)
  add_custom_target(${_mode}
    COMMAND "${CMAKE_COMMAND}" "-DQUALITY_MODE=${_mode}"
      "-DQUALITY_ROOT=${CMAKE_SOURCE_DIR}" "-DQUALITY_BUILD=${CMAKE_BINARY_DIR}"
      "-DQUALITY_MANIFEST=${_manifest}"
      -P "${CMAKE_SOURCE_DIR}/cmake/RunQuality.cmake"
    COMMENT "TexUpSR ${_mode}" VERBATIM)
endforeach()
add_custom_target(quality DEPENDS format-check tidy cppcheck)
