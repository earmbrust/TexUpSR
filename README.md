# TexUpSR

**GPU-accelerated image super-resolution with NVIDIA TensorRT and ONNX**

TexUpSR is a C++17 command-line image upscaler for single-model ONNX inference. It loads or builds TensorRT engines, processes images in overlapping tiles across configurable CUDA streams, and blends the results with OpenCV.

## Features

- Single-model super-resolution using TensorRT's ONNX parser and CUDA runtime.
- FP16 and FP32 tensor input/output, including mixed precision.
- Automatic or explicitly configured tile and batch sizes.
- Multiple CUDA streams, persistent device buffers and pinned host memory.
- Feather-blended overlapping tiles and cached TensorRT engines.
- Configurable scale factor, overlap, workspace, streams and debug output.

## Supported environment

The application has been compiled and run on Windows with Visual Studio 2026, CUDA Toolkit 13.0, TensorRT 10.16.1.11 and OpenCV 4.14.0 (vcpkg). Other toolchain combinations may work but have not been verified. Inference requires a supported NVIDIA GPU and driver. A model must provide a single RGB NCHW input and output, and `--scale` must agree with its output dimensions.

The Windows CI workflow uses Visual Studio 2022, CUDA 13.2, TensorRT 10.16.1.11, and official prebuilt OpenCV 4.12.0 Windows binaries for compilation and CLI smoke checks. GitHub-hosted runners do not validate GPU inference or generated image quality.

## Build

Provide installed CUDA, TensorRT and OpenCV dependencies. Configure using vcpkg on Windows:

```powershell
cmake -S . -B build-vcpkg -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE="C:/development/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0" -DTensorRT_ROOT="C:/development/dependencies/TensorRT-10.16.1.11"
cmake --build build-vcpkg --config Release --parallel
```

Update paths for your local installations. The CMake project is also compatible with other supported Visual Studio generators. Release builds place `trt_sr.exe` in the `Release` directory of the build tree.

## Usage

```powershell
.\build-vcpkg\Release\trt_sr.exe -i input.png -o output.png -m model.onnx --scale 2
```

| Option | Description | Default |
| --- | --- | --- |
| `-i`, `--input` | Source image (required) | — |
| `-o`, `--output` | Output image (required) | — |
| `-m`, `--model` | ONNX model path (required) | — |
| `-e`, `--engine` | TensorRT engine path | Automatically named under `engines/` |
| `-t`, `--tile` | Input tile size (`0` or `-1`: auto) | Auto |
| `-b`, `--batch` | Batch size (`0` or `-1`: auto) | Auto |
| `-s`, `--scale` | Integer scaling factor | `4` |
| `-w`, `--workspace` | TensorRT workspace limit (GiB) | `4` |
| `--streams` | CUDA stream count | `2` |
| `--overlap` | Tile overlap in input pixels | `16` |
| `-v`, `--verbose` | Verbose logging | Off |
| `-q`, `--quiet` | Suppress progress output | Off |
| `-d`, `--debug` | Write first processed tile | Off |
| `-h`, `--help` | Show CLI help | — |

## Cached engines

Default engine filenames include the model basename, tile size and batch size. They do not incorporate a model-content hash, GPU architecture or the installed TensorRT version. If any of those change, delete the corresponding cached `.plan` file to force a rebuild. Engines compiled for one hardware/software setup may not be portable to another.

## Continuous integration and releases

GitHub Actions checks Windows compilation and CLI behavior for pushes to `develop` and feature branches, and for pull requests targeting `develop` or `main`. A separate Windows build workflow packages the compiled executable with selected dependencies. These checks do not require a GPU and do not run ONNX inference.

GitHub releases are created only when a `vMAJOR.MINOR.PATCH` tag points to a commit already on `main` and matches the version in `CMakeLists.txt`. Release tags should be published only after the corresponding release has been accepted and merged into `main`.

The CI environment downloads NVIDIA SDK packages and builds OpenCV through vcpkg. Binary packaging may be subject to third-party redistribution terms; verify the applicable licenses before publishing release artifacts.

## License

Copyright (c) 2025 Elden Armbrust.

TexUpSR is licensed under the MIT License. See [LICENSE](LICENSE). NVIDIA TensorRT, CUDA, OpenCV, third-party ONNX models and other dependencies have their own licenses.
