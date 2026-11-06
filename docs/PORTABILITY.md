# C++ standards and platform compatibility

TexUpSR requires an NVIDIA CUDA/TensorRT-capable system. Windows x64 and Linux x64 are validated build targets. Linux ARM64 (SBSA/Jetson) is planned and requires SDK- and hardware-specific validation. macOS is not supported.

## C++ standard

`-DTEXUP_CXX_STANDARD=AUTO` (default) selects the newest language mode advertised by the selected compiler. Explicit values `17`, `20`, and `23` are supported. C++17 remains the minimum language requirement; the application does not require C++20 or C++23 features.

A compiler advertising a standard does not by itself establish compatibility with the installed CUDA, TensorRT, OpenCV, and C++ standard library. The complete build and test matrix validates those combinations used in CI.

## Configure

On Linux with the required NVIDIA SDKs, OpenCV, Protobuf, and GoogleTest installed, configure a local build directory. The repository does not define a `linux-release` preset:

```sh
cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release -DTEXUP_CXX_STANDARD=17 -DBUILD_TESTING=ON
cmake --build build-linux
ctest --test-dir build-linux -L unit --output-on-failure
```

`-DTensorRT_ROOT=/path/to/sdk` can be used with a custom TensorRT installation. Otherwise CMake searches system paths, including multiarch directories. On Windows, choose any compatible generator and compiler. Select the standard with `-DTEXUP_CXX_STANDARD=17`, `20`, or `23`.

## CI verification

The main CI workflow runs three independent checks (Cppcheck, SDK-backed clang-tidy, and SDK-independent CMake configuration) alongside nine platform configurations:

- Windows x64: Visual Studio 2026 with C++17, C++20, and C++23.
- Linux x64: GCC and Clang, each with C++17, C++20, and C++23, using the NVIDIA TensorRT development image.

Each platform job builds the application, runs its configured tests, and checks the command-line entry point. These jobs do not publish releases. The nine platform jobs and three pre-build checks passed for M6; the PR #16 closeout also passed its CodeQL checks. The workflows remain the source of truth for the current matrix.

Successful CI builds and command-line smoke tests do not establish working GPU inference. Hardware-backed inference validation requires a compatible NVIDIA GPU and driver. Linux ARM64, Jetson, Windows Clang, and GPU-backed Linux inference remain outside the current CI compatibility matrix.

# Build and test portability

TexUpSR uses CMake's selected generator and C++ compiler. No particular IDE, compiler
version, or package manager is required by the project configuration. The installed
CUDA and TensorRT SDKs must support the selected compiler, platform, and architecture.

OpenCV, Protobuf, CUDA, TensorRT, and (when tests are enabled) GoogleTest must be
installed and discoverable. Use CMAKE_PREFIX_PATH or package-specific CMake hints
when packages are outside the normal search paths. TensorRT_ROOT is an optional hint,
not a required environment variable. Windows runtime DLLs must be available to the operating system's loader, typically through installed dependencies on `PATH`. CMake does not copy third-party runtime DLLs into build outputs.

Configure and build with any supported generator:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release -L unit --output-on-failure
```

To enable the hardware-backed functional/integration tests, configure a separate
build directory (or reuse one by reconfiguring it):

```sh
cmake -S . -B build-integration -DTEXUPSR_BUILD_GPU_TESTS=ON
cmake --build build-integration --config Release
ctest --test-dir build-integration -C Release -L integration --output-on-failure
```

GPU tests require an NVIDIA GPU, compatible driver, and TensorRT runtime. They are
not part of the normal CPU test phase. The integration executable returns CTest's
skip code (77) when no suitable GPU is present. A missing or broken SDK still
causes configuration/build errors and is not silently treated as a skip.

Static quality analysis is separate from unit and integration testing. Configure
with a generator that exports a compilation database (such as Ninja):

```sh
cmake --preset analysis
cmake --build build-analysis --target quality
```

Existing CI compiler matrices explicitly select their compilers and dependency
locations for reproducibility; those selections are not project requirements.
