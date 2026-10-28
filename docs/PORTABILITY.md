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

`-DTensorRT_ROOT=/path/to/sdk` can be used with a custom TensorRT installation. Otherwise CMake searches system paths, including multiarch directories. On Windows, use the Visual Studio preset and select the standard with `-DTEXUP_CXX_STANDARD=17`, `20`, or `23`.

## CI verification

The main CI workflow has three sequential pre-build gates: Cppcheck, clang-tidy with an SDK-backed compilation database, and an SDK-independent CMake configuration check. Once those pass, all 12 platform configurations can run independently:

- Windows x64: Visual Studio 2022 and Visual Studio 2026, each with C++17, C++20, and C++23.
- Linux x64: GCC and Clang, each with C++17, C++20, and C++23, using the NVIDIA TensorRT development image.

Each platform job configures and builds the actual application, runs its configured CTest suite, and checks the application command-line entry point. The build jobs do not package or publish a release. GitHub Actions run 40628257485 completed successfully for all 15 jobs, including the three pre-build gates and 12 platform configurations.

Successful CI builds and command-line smoke tests do not establish working GPU inference. Hardware-backed inference validation requires a compatible NVIDIA GPU and driver. Linux ARM64, Jetson, Windows Clang, and GPU-backed Linux inference remain outside the current CI compatibility matrix.
