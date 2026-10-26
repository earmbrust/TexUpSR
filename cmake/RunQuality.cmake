cmake_minimum_required(VERSION 3.25)
if(NOT DEFINED QUALITY_MODE OR NOT DEFINED QUALITY_ROOT OR
   NOT DEFINED QUALITY_BUILD OR NOT DEFINED QUALITY_MANIFEST)
  message(FATAL_ERROR "Use CMake's format-check, tidy, cppcheck or quality targets")
endif()
include("${QUALITY_MANIFEST}")

function(run_checked tool)
  execute_process(COMMAND "${tool}" ${ARGN}
    WORKING_DIRECTORY "${QUALITY_ROOT}"
    RESULT_VARIABLE result)
  if(NOT result STREQUAL "0")
    message(FATAL_ERROR "${tool} failed: ${result}")
  endif()
endfunction()

if(QUALITY_MODE STREQUAL "format-check")
  find_program(FORMAT_EXE NAMES clang-format REQUIRED)
  execute_process(COMMAND "${FORMAT_EXE}" --version OUTPUT_VARIABLE version RESULT_VARIABLE result)
  if(NOT result STREQUAL "0" OR NOT version MATCHES "22\\.1\\.3")
    message(FATAL_ERROR "clang-format 22.1.3 required; found ${version}")
  endif()
  message(STATUS "clang-format: checking CMake project source and headers")
  run_checked("${FORMAT_EXE}" --dry-run --Werror --style=file ${QUALITY_FORMAT})
elseif(QUALITY_MODE STREQUAL "tidy")
  if(NOT EXISTS "${QUALITY_BUILD}/compile_commands.json")
    message(FATAL_ERROR "clang-tidy requires compile_commands.json; configure the Ninja analysis preset")
  endif()
  find_program(TIDY_EXE NAMES clang-tidy REQUIRED)
  message(STATUS "clang-tidy: checking all ${QUALITY_CPP}")
  foreach(source IN LISTS QUALITY_CPP)
    run_checked("${TIDY_EXE}" --quiet --warnings-as-errors=* -p "${QUALITY_BUILD}" "${source}")
  endforeach()
elseif(QUALITY_MODE STREQUAL "cppcheck")
  find_program(CPPCHECK_EXE NAMES cppcheck REQUIRED)
  message(STATUS "Cppcheck: checking source and tests")
  run_checked("${CPPCHECK_EXE}" --enable=warning,performance,portability
    --std=c++17 --language=c++ --error-exitcode=1 --inline-suppr
    --suppress=missingIncludeSystem --suppress=missingInclude
    -Isrc main.cpp src tests)
else()
  message(FATAL_ERROR "Unknown QUALITY_MODE: ${QUALITY_MODE}")
endif()
message(STATUS "${QUALITY_MODE} PASS")
