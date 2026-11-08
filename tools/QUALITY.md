# Quality checks (CMake-native)

TexUpSR exposes CMake targets for clang-format, clang-tidy, and Cppcheck. These checks apply to the project sources and tests; the source lists are derived from CMake targets. An unregistered `.cpp` file in `src/` or `tests/` causes configuration to fail rather than silently escaping analysis.

## Local analysis

Configure the checked-in Ninja `analysis` preset with the required SDKs and dependencies available, then run the quality target:

```powershell
cmake --preset analysis
cmake --build build-analysis --target quality
```

On Windows, start from a developer environment that can locate the compiler, CUDA, TensorRT, OpenCV, Protobuf, and GoogleTest. The project uses its normal dependency discovery; no machine-specific package paths belong in tracked presets. A locally ignored `CMakeUserPresets.json` can provide overrides when automatic discovery is insufficient. See `CMakeUserPresets.json.example` for the optional `windows-analysis-local` preset, which inherits the checked-in `analysis` preset.

You can also run individual targets:

- `format-check`: requires clang-format **22.1.3**, as enforced by `cmake/RunQuality.cmake`; checks project C++ source files and headers.
- `tidy`: checks compiled project translation units using Ninja's `compile_commands.json` and the repository's `.clang-tidy` policy.
- `cppcheck`: checks source and tests for warnings, portability, and performance concerns.
- `quality`: depends on all three targets.

The Ninja analysis preset is separate from the application and integration build presets. Visual Studio generators do not provide the compilation database needed by this clang-tidy target.

## Pre-commit hook and CI

Enable the optional pre-commit hook with:

```powershell
git config core.hooksPath .githooks
```

The hook expects `build-analysis` to have been configured and runs the CMake `quality` target against the working tree. Stage the intended changes before committing, but remember the check is not limited to staged lines.

The main CI workflow independently runs Cppcheck, SDK-backed clang-tidy, and a CMake configuration check alongside Windows Visual Studio 2026 and Linux GCC/Clang builds for C++17, C++20, and C++23. The reusable Windows build workflow also checks formatting and Cppcheck. CI gates are already active; they are not deferred work. GPU integration on an actual NVIDIA GPU remains a separate hardware-dependent check.
