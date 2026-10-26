# C++ standards and platform compatibility

TexUpSR requires an NVIDIA CUDA/TensorRT-capable system. Supported build targets are Windows x64 and Linux x64; Linux ARM64 (SBSA/Jetson) is a planned target requiring device-specific validation. macOS is not supported.

## C++ standard

`-DTEXUP_CXX_STANDARD=AUTO` (default) selects the newest language mode reported by CMake for the current compiler. `17`, `20`, and `23` are explicit overrides. Selecting a language mode does **not** prove that the CUDA, TensorRT, OpenCV, and standard-library combination works; a successful build and tests provide that evidence. Always validate C++17 explicitly before describing it as supported.

## Configure

On Linux with suitable system dependencies installed:

```sh
cmake --preset linux-release -DTEXUP_CXX_STANDARD=17
cmake --build --preset linux-release
ctest --test-dir build-linux -C Release --output-on-failure
```

Optional `-DTensorRT_ROOT=/path/to/sdk` can point at a non-system SDK. Otherwise CMake searches normal system locations, including multiarch include/library paths. On Windows, continue to use the existing local preset with installed SDK paths and set `-DTEXUP_CXX_STANDARD=20` or `23` as desired.

The GPU test reports a skip (return code 77) on hosts without a usable GPU. A skipped GPU test is **not** evidence that inference works on that host. NVIDIA TensorRT engines need to be built for compatible target environments.

## CI

The existing Windows workflow provides an x64 build baseline. The CI entrypoint runs the existing Windows workflow and a Linux x86-64
compatibility matrix. The Linux jobs use NVIDIA's public
`nvcr.io/nvidia/tensorrt:26.05-py3` development container (CUDA 13.2,
TensorRT 10.16.1), with GCC and Clang-18. Each job configures, builds, and
runs the **six CPU-only CTest tests** separately in C++17, C++20, and C++23.
The test named `gpu_inference` is explicitly excluded on hosted CPU runners;
these jobs do not establish working GPU inference on Linux. The container
must be downloadable on hosted CI for these jobs to work. No Linux CI
success has been demonstrated until Actions completes a run.

Linux ARM64, Jetson, Windows Clang, and GPU-backed Linux inference require
separate hardware / SDK validation. The Linux workflow is an initial x64
compatibility gate, not a claim that every NVIDIA platform is supported.
