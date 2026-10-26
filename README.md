# TexUpSR

TexUpSR is a Windows command-line image upscaler built on NVIDIA TensorRT, CUDA, and OpenCV. It loads compatible ONNX super-resolution models, builds TensorRT engines, and processes images in overlapping tiles.

**Version:** 0.1.0.
See [docs/ROADMAP.md](docs/ROADMAP.md) for planned model-compatibility and tooling work.

## Model requirements

Current model support requires all of the following:

- One RGB input tensor and one RGB output tensor.
- NCHW tensor layout.
- FP16 or FP32 input/output tensors.
- Integer output scale matching `--scale` (default: `4`).
- An ONNX graph accepted by TensorRT's ONNX parser and engine builder.

The model name alone does not establish compatibility. TensorRT conversion, preprocessing assumptions, and output correctness must be verified for each ONNX export. The current preprocessing converts OpenCV BGR pixels to RGB float values in the range 0–1; output is converted back to 8-bit BGR. No model-specific normalization or padding profiles are implemented.

## Build

Requirements:

- CMake 3.25 or newer
- A C++17 compiler
- CUDA Toolkit
- TensorRT
- OpenCV with a CMake package configuration

The repository includes a Visual Studio 2026/vcpkg preset. On Windows PowerShell, set `VCPKG_ROOT`, `TENSORRT_ROOT`, and `CUDA_PATH` to your installed dependencies, then configure and build:

```powershell
cmake --preset windows-vs2026
cmake --build --preset windows-release
```

You can also configure directly with `-DTensorRT_ROOT=...`, `-DCUDAToolkit_ROOT=...`, and either an OpenCV package path or a vcpkg toolchain. For Visual Studio builds, `trt_sr.exe` is written to the build tree's `Release` directory.

The project has been built and run on Windows with Visual Studio 2026, CUDA 13.0, TensorRT 10.16.1.11, and OpenCV 4.14.0 via vcpkg.

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

The application requires an NVIDIA GPU with a supported driver for inference.

## Tests

CPU-only option parsing tests can be run after building:

```powershell
ctest --test-dir build-vcpkg -C Release --output-on-failure
```

## Engine cache

Automatically generated engine names currently include only the model basename, tile size, and batch size. They do **not** include the model contents, scale, TensorRT version, or GPU identity. Delete a stale `.plan` file when changing models or build environments. Serialized engines are not generally portable across arbitrary GPU and TensorRT combinations.

Improving cache identity and recovery is planned work; see [docs/ROADMAP.md](docs/ROADMAP.md).

## Releases

Release publication is triggered by a `vMAJOR.MINOR.PATCH` tag on a commit contained in `main` that matches the version in `CMakeLists.txt`. Redistributing release artifacts must comply with CUDA, TensorRT, OpenCV, and model licenses.

## License

Copyright (c) 2025-2026 Elden Armbrust.

MIT License; see [LICENSE](LICENSE). Dependencies and ONNX models have their own licenses.
