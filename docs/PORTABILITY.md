# C++ standards and platform compatibility

TexUpSR requires an NVIDIA CUDA/TensorRT-capable system. Windows x64 and Linux x64 are validated build targets. Linux ARM64 (SBSA/Jetson) is planned and requires SDK- and hardware-specific validation. macOS is not supported.

## C++ standard

`-DTEXUP_CXX_STANDARD=AUTO` (default) selects the newest language mode advertised by the selected compiler. Explicit values `17`, `20`, and `23` are supported. C++17 remains the minimum language requirement; the application does not require C++20 or C++23 features.

A compiler advertising a standard does not by itself establish compatibility with the installed CUDA, TensorRT, OpenCV, and C++ standard library. The complete build and test matrix validates those combinations used in CI.

## Configure

On Linux with the required NVIDIA SDKs, OpenCV, and GoogleTest installed:

```sh
cmake --preset linux-release -DTEXUP_CXX_STANDARD=17
cmake --build --preset linux-release
ctest --test-dir build-linux --output-on-failure
```

`-DTensorRT_ROOT=/path/to/sdk` can be used with a custom TensorRT installation. Otherwise CMake searches system paths, including multiarch directories. On Windows, choose any compatible generator and compiler. Select the standard with `-DTEXUP_CXX_STANDARD=17`, `20`, or `23`.

## CI verification

The main CI workflow has three sequential pre-build gates: Cppcheck, clang-tidy with an SDK-backed compilation database, and an SDK-independent CMake configuration check. Once those pass, all 12 platform configurations can run independently:

- Windows x64: Visual Studio 2022 and Visual Studio 2026, each with C++17, C++20, and C++23.
- Linux x64: GCC and Clang, each with C++17, C++20, and C++23, using the NVIDIA TensorRT development image.

Each platform job configures and builds the actual application, runs its configured CTest suite, and checks the application command-line entry point. The build jobs do not package or publish a release. GitHub Actions run 40628257485 completed successfully for all 15 jobs, including the three pre-build gates and 12 platform configurations.

Successful CI builds and command-line smoke tests do not establish working GPU inference. Hardware-backed inference validation requires a compatible NVIDIA GPU and driver. Linux ARM64, Jetson, Windows Clang, and GPU-backed Linux inference remain outside the current CI compatibility matrix.

# Build and test portability

TexUpSR uses CMake's selected generator and C++ compiler. No particular IDE, compiler
version, or package manager is required by the project configuration. The installed
CUDA and TensorRT SDKs must support the selected compiler, platform, and architecture.

OpenCV, Protobuf, CUDA, TensorRT, and (when tests are enabled) GoogleTest must be
installed and discoverable. Use CMAKE_PREFIX_PATH or package-specific CMake hints
when packages are outside the normal search paths. TensorRT_ROOT is an optional hint,
not a required environment variable. Windows runtime DLLs are resolved using the
located TensorRT libraries and normal CMake search paths.

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
cmake -S . -B build-analysis -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-analysis --target quality
```

Existing CI compiler matrices explicitly select their compilers and dependency
locations for reproducibility; those selections are not project requirements.
