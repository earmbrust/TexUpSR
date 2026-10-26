# TexUpSR

TexUpSR is a Windows command-line image upscaler using NVIDIA TensorRT, CUDA and OpenCV. It accepts ONNX super-resolution models that meet the current tensor requirements, builds TensorRT engines, and processes images in overlapping tiles.

**Version:** 0.1.0. See [the development roadmap](docs/ROADMAP.md) for planned model compatibility work.

## Model requirements

- One RGB input and one RGB output, in NCHW layout.
- FP16 or FP32 input/output tensors.
- Integer output scale matching `--scale` (default: 4).
- A graph that TensorRT's ONNX parser and engine builder can handle.

A model's architecture name alone does not establish compatibility. TensorRT conversion, preprocessing assumptions, and output correctness must be verified for each model export. The current preprocessing converts OpenCV BGR pixels to RGB float values in the range 0–1; output is converted back to 8-bit BGR. No model-specific normalization or padding profiles are implemented.

## Build

Requirements: CMake 3.25 or newer, a C++17 compiler, CUDA Toolkit, TensorRT, and an OpenCV CMake package. The repository includes a Visual Studio 2026/vcpkg preset; Visual Studio 2022 is also used by CI.

On Windows PowerShell, set `VCPKG_ROOT`, `TENSORRT_ROOT` and `CUDA_PATH` to installed dependencies, then configure and build:

```powershell
cmake --preset windows-vs2026
cmake --build --preset windows-release
```

Alternatively, configure directly with `-DTensorRT_ROOT=...`, `-DCUDAToolkit_ROOT=...` and an appropriate OpenCV package path or vcpkg toolchain. `trt_sr.exe` is placed in the build tree's `Release` directory for Visual Studio builds.

The project has been compiled and run on Windows with Visual Studio 2026, CUDA 13.0, TensorRT 10.16.1.11 and OpenCV 4.14.0 via vcpkg. CI uses Visual Studio 2022/2026, CUDA 13.2, TensorRT 10.16.1.11 and official OpenCV 4.12.0 Windows binaries; these are compile and CLI smoke checks, not GPU inference tests.

## Usage

```powershell
.\build-vcpkg\Release\trt_sr.exe -i input.png -o output.png -m model.onnx --scale 4
```

| Option | Description | Default |
| --- | --- | --- |
| `-i`, `--input` | Input image path | Required |
| `-o`, `--output` | Output image path | Required |
| `-m`, `--model` | ONNX model path | Required |
| `-e`, `--engine` | TensorRT engine path | Generated under `engines/` |
| `-t`, `--tile` | Input tile size; `0` or `-1` selects auto | Auto |
| `-b`, `--batch` | Batch size; `0` or `-1` selects auto | Auto |
| `-s`, `--scale` | Integer upscaling factor | `4` |
| `-w`, `--workspace` | TensorRT workspace limit in GiB | `4` |
| `--streams` | CUDA stream count | `2` |
| `--overlap` | Tile overlap in source pixels | `16` |
| `-v`, `--verbose` | Verbose diagnostics | Off |
| `-q`, `--quiet` | Suppress progress | Off |
| `-d`, `--debug` | Write `debug_first_tile.png` | Off |
| `-h`, `--help` | Usage | — |

The application requires an NVIDIA GPU with a supported driver for inference. For CPU-only option parsing tests after building:

```powershell
ctest --test-dir build-vcpkg -C Release --output-on-failure
```

## Engine cache

Automatically generated engine names currently include only the model basename, tile size and batch size. They do **not** include the model's contents, scale, TensorRT version, or GPU identity. Delete a stale `.plan` file when changing models or build environments. Serialized engines are not generally portable across arbitrary GPU and TensorRT combinations. Cache identity and recovery are planned work; see the roadmap.

## CI and releases

Windows CI compiles the project, runs CPU-only option tests and basic CLI smoke checks, and packages dependencies. It does not run inference, benchmark throughput, or verify image quality. Private-repository builds are subject to the account's GitHub Actions allowance.

Release publication is triggered by a `vMAJOR.MINOR.PATCH` tag on a commit contained in `main`, matching the version in `CMakeLists.txt`. Release artifact redistribution must comply with CUDA, TensorRT, OpenCV and model licenses.

## License

Copyright (c) 2025 Elden Armbrust.

MIT License; see [LICENSE](LICENSE). Dependencies and ONNX models have their own licenses.
