if(NOT WIN32)
  return()
endif()

if(POLICY CMP0207)
  cmake_policy(SET CMP0207 NEW)
endif()

string(REPLACE "|" ";" _search "${TEXUPSR_SEARCH}")
# Include the runtime loader's configured directories without SDK-specific paths.
# CI and local development can add package runtime directories to PATH.
if(DEFINED ENV{PATH})
  string(REPLACE "\\" "/" _runtime_path "$ENV{PATH}")
  list(APPEND _search ${_runtime_path})
endif()
list(REMOVE_DUPLICATES _search)
set(_dirs)
foreach(_dir IN LISTS _search)
  if(IS_DIRECTORY "${_dir}")
    list(APPEND _dirs "${_dir}")
  endif()
endforeach()

# Windows API sets and system modules are supplied by Windows, not the project.
file(GET_RUNTIME_DEPENDENCIES
  EXECUTABLES "${TEXUPSR_EXE}"
  DIRECTORIES ${_dirs}
  PRE_EXCLUDE_REGEXES
    "^[Aa][Pp][Ii]-[Mm][Ss]-"
    "^[Ee][Xx][Tt]-[Mm][Ss]-"
    "^[Aa]zure[Aa]ttest[Mm]anager\\.dll$"
    "^[Aa]zure[Aa]ttest[Nn]ormal\\.dll$"
    "^[Hh]vsi[Ff]ile[Tt]rust\\.dll$"
    "^[Pp]dm[Uu]tilities\\.dll$"
    "^[Ww]paxholder\\.dll$"
  POST_EXCLUDE_REGEXES
    ".*[/\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\].*"
  RESOLVED_DEPENDENCIES_VAR _resolved
  UNRESOLVED_DEPENDENCIES_VAR _unresolved
  CONFLICTING_DEPENDENCIES_PREFIX _conflicts)

# A previously deployed DLL beside the executable is the one Windows loads.
# Do not silently choose between two different external installations.
foreach(_name IN LISTS _conflicts_FILENAMES)
  set(_deployed "${TEXUPSR_DEST}/${_name}")
  set(_candidates "${_conflicts_${_name}}")
  if(NOT EXISTS "${_deployed}")
    message(FATAL_ERROR "Conflicting runtime dependencies for ${_name}: ${_candidates}")
  endif()
  set(_found FALSE)
  foreach(_candidate IN LISTS _candidates)
    if("${_candidate}" STREQUAL "${_deployed}")
      set(_found TRUE)
      break()
    endif()
  endforeach()
  if(NOT _found)
    message(FATAL_ERROR "Existing ${_deployed} is not a resolved candidate for ${_name}: ${_candidates}")
  endif()
endforeach()

if(_unresolved)
  list(REMOVE_DUPLICATES _unresolved)
  message(FATAL_ERROR
    "Cannot locate third-party runtime dependencies for ${TEXUPSR_EXE}: ${_unresolved}")
endif()

foreach(_dll IN LISTS _resolved)
  if(NOT EXISTS "${_dll}")
    message(FATAL_ERROR "Resolved runtime dependency does not exist: ${_dll}")
  endif()
  file(COPY "${_dll}" DESTINATION "${TEXUPSR_DEST}")
endforeach()
