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

The model name alone does not establish compatibility. TensorRT conversion, preprocessing assumptions, and output correctness must be verified for each ONNX export. The current preprocessing converts OpenCV BGR pixels to RGB float values in the range 0-1; output is converted back to 8-bit BGR. No model-specific normalization or padding profiles are implemented.

## Build

Requirements:

- CMake 3.25 or newer
- A C++17 compiler
- CUDA Toolkit
- TensorRT
- OpenCV with a CMake package configuration
- Protobuf C++ library and compiler (protoc)

The repository includes a Visual Studio 2026/vcpkg preset. On Windows PowerShell, set `VCPKG_ROOT`, `TENSORRT_ROOT`, and `CUDA_PATH` to your installed dependencies, then configure and build:

```powershell
cmake --preset windows-vs2026
cmake --build --preset windows-release
```

You can also configure directly with `-DTensorRT_ROOT=...`, `-DCUDAToolkit_ROOT=...`, and either an OpenCV package path or a vcpkg toolchain. For Visual Studio builds, `trt_sr.exe` is written to the build tree's `Release` directory.

The project has been built and run on Windows with Visual Studio 2026, CUDA 13.0, TensorRT 10.16.1.11, and OpenCV 4.14.0 via vcpkg.

## Model inspection

Use `trt_sr --inspect model.onnx` to report ONNX graph input and output names,
shapes, data types, opsets, and a spatial scale when static dimensions establish
one. This operation does not initialize CUDA and does not build an engine.
TensorRT parser acceptance, engine construction, and runtime compatibility are
not determined by CPU-only inspection. The ONNX v1.19.0 schema is used under
its Apache-2.0 license. Protobuf development libraries and `protoc` are required
to build the application; installed applications do not require `protoc`.
Supplying `--engine existing.plan` reports its file size, but does not
deserialize or verify the engine without CUDA.

## Optional model profile

TensorRT parses the ONNX graph; TexUpSR handles pixel packing and unpacking.
Without a profile, existing RGB/NCHW floating-point 0..1 behavior is preserved.
Use `--profile model.json` for models requiring different pixel conventions:

```json
{
  "layout": "nhwc",
  "input_order": "bgr",
  "output_order": "bgr",
  "input_scale": 255.0,
  "input_offset": 0.0,
  "output_scale": 0.00392156862745098,
  "output_offset": 0.0
}
```

Input conversion is `(pixel / 255) * input_scale + input_offset`. Output
conversion is `model_output * output_scale + output_offset` into the
normalized output image. Tensor layout applies to both input and output.
Only single-input, single-output, three-channel FP16/FP32 models with an
integer isotropic scale are supported. Fixed dimensions must match the
requested tile and batch; incompatible models fail clearly. Metadata cannot
reliably establish channel order or normalization, so use an explicit profile.

## Usage

```powershell
.\build-vcpkg\Release\trt_sr.exe -i input.png -o output.png -m model.onnx --scale 4
```

| Option | Description | Default |
| --- | --- | --- |
| `--inspect` | CPU-only ONNX model inspection | Off |
| `--profile` | JSON pixel/layout profile | Legacy RGB/NCHW conventions |
| `-i`, `--input` | Input image path | Required for upscaling |
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
| `-h`, `--help` | Usage | - |

The application requires an NVIDIA GPU with a supported driver for inference.

## Tests

CPU-only option parsing tests can be run after building:

```powershell
ctest --test-dir build-vcpkg -C Release --output-on-failure
```

## Engine cache

Automatically generated engine names currently include only the model basename, tile size, and batch size. They do **not** include the model contents, scale, TensorRT version, or GPU identity. Delete a stale `.plan` file when changing models or build environments. Serialized engines are not generally portable across arbitrary GPU and TensorRT combinations.

Improving cache identity and recovery is planned work; see [docs/ROADMAP.md](docs/ROADMAP.md).

## Windows runtime requirements

Windows release archives contain `trt_sr.exe`, `README.md`, and `LICENSE` only.
Third-party runtime DLLs are not bundled. Install compatible x64 versions of
CUDA (13.2 for CI builds), TensorRT (10.16.1.11), OpenCV (4.12 for CI builds),
Protobuf (shared runtime), and any runtime libraries they require (including
Abseil when required by the Protobuf build). The Microsoft Visual C++ runtime
and an NVIDIA driver compatible with the installed CUDA/TensorRT versions are
also required. Configure the Windows DLL search path (`PATH`) or an equivalent
installation method so the application can locate the installed libraries.

A release package is not standalone. Dependency versions and binary ABI must
match the release build. The SDK and development tools (such as `protoc`) are
required to build TexUpSR, not to run the executable.

## Releases

Release publication is triggered by a `vMAJOR.MINOR.PATCH` tag on a commit contained in `main` that matches the version in `CMakeLists.txt`. Redistributing release artifacts must comply with CUDA, TensorRT, OpenCV, and model licenses.

## License

Copyright (c) 2025-2026 Elden Armbrust.

MIT License; see [LICENSE](LICENSE). Dependencies and ONNX models have their own licenses.
