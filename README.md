# TexUpSR

TexUpSR is a Windows command-line image upscaler built on NVIDIA TensorRT, CUDA, and OpenCV. It loads compatible ONNX super-resolution models, builds TensorRT engines, and processes images in overlapping tiles.

**Version:** 0.1.0.
See [docs/ROADMAP.md](docs/ROADMAP.md) for completed work and planned changes.

## Model requirements

Current TensorRT model support requires:

- One input tensor and one output tensor with three image channels.
- NCHW or NHWC layout, selected through an optional model profile.
- FP16 or FP32 input/output tensors.
- An integer, isotropic output scale matching `--scale` (default: `4`).
- Tile and batch dimensions compatible with the model's fixed dimensions or supported dynamic dimensions.
- An ONNX graph accepted by TensorRT's parser and engine builder.

Without `--profile`, input pixels are converted from BGR to RGB floating-point values in the range 0-1 and output is converted to 8-bit BGR. A JSON profile supports channel order, layout, scale, and offset conversions. General model-specific padding is not implemented. The model name or successful CPU inspection alone does not establish TensorRT or output compatibility; verify the particular ONNX export.

## Build

Requirements:

- CMake 3.25 or newer
- A C++17 compiler
- CUDA Toolkit
- TensorRT
- OpenCV with a CMake package configuration
- Protobuf C++ library and compiler (protoc)

Use the checked-in CMake presets. The `default` preset builds the application and unit tests; `integration` also enables GPU integration tests; `analysis` uses Ninja; and `unit` builds CPU-only tests without the application. The generator for the non-Ninja presets is selected by your CMake environment.

For a Windows application and GPU integration build:

```powershell
cmake --preset integration
cmake --build --preset integration
```

The executable is under `build-integration/Release` for multi-configuration Visual Studio generators, or directly under `build-integration` for single-configuration generators. The `default` build uses `cmake --preset default` followed by `cmake --build --preset build`.

On Windows, CMake first honors an explicit toolchain; otherwise it searches standard vcpkg environment settings and a sibling vcpkg checkout. TensorRT discovery honors an explicit `TensorRT_ROOT` and then supported environment hints and sibling installations. CUDA uses CMake's normal toolkit discovery. Ordinary supported setups do not require passing developer-specific `-D` paths on each invocation; explicit dependency overrides remain available.

TexUpSR has been built and run on Windows with Visual Studio 2026, CUDA 13.0, TensorRT 10.16.1.11, and OpenCV 4.14.0 via vcpkg. The Windows build must not copy third-party runtime DLLs next to its executables.

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
.\build-integration\Release\trt_sr.exe -i input.png -o output.png -m model.onnx --scale 4
```

To generate an engine without processing an image, specify a model, an explicit plan path,
positive tile and batch sizes, and the other required model settings:

```powershell
.\build-integration\Release\trt_sr.exe --build-engine -m model.onnx -e model.plan -t 256 -b 1 -s 2
```

This checks TensorRT model requirements and the resulting output shape before saving.
It refuses an existing engine path unless `--overwrite-engine` is supplied.
Pre-generation does not use `--input` or `--output`. Use matching model, tile,
batch, scale, and layout settings when loading the engine for normal upscaling.

| Option | Description | Default |
| --- | --- | --- |
| `--inspect` | CPU-only ONNX model inspection | Off |
| `--build-engine` | Build an explicit TensorRT engine without an image | Off |
| `--benchmark` | Print CUDA timing, wall time, throughput, and device memory snapshots | Off |
| `--profile` | JSON pixel/layout profile | Legacy RGB/NCHW conventions |
| `-i`, `--input` | Input image path | Required for upscaling |
| `-o`, `--output` | Output image path | Required |
| `-m`, `--model` | ONNX model path | Required |
| `-e`, `--engine` | TensorRT engine path | Generated under `engines/` |
| `--overwrite-engine` | Force rebuild and safely replace engine plan | Off |
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

Use the `unit` preset for the CPU-only suite (OpenCV and GoogleTest are still required):

```powershell
cmake --preset unit
cmake --build --preset unit
ctest --test-dir build-unit -C Release -L unit --output-on-failure
```

For GPU integration testing, build with the `integration` preset and run `ctest --preset integration`. This is separate from ordinary unit testing. See [docs/performance.md](docs/performance.md) for benchmark measurements and commands.

## Engine cache identity and overwrite

Automatic engine cache filenames include a SHA-256 identity derived from ONNX
contents, effective tile and batch, scale, workspace, tensor layout, GPU
name/compute capability, and TensorRT runtime version. Valid plans are reused.
An automatic cache that cannot be read or deserialized is rebuilt. Existing
explicit `--engine` files are never modified during normal loading; invalid
explicit plans cause an error rather than being silently replaced.

Use `--overwrite-engine` to force a fresh TensorRT build and replace either an
automatic cache or an explicitly selected `--engine` file. If building or
validation fails, the old plan is preserved. Engine files are written using
a same-directory temporary file and replacement operation.

Serialized TensorRT engines are not generally portable between GPU architectures,
TensorRT releases, platforms, or incompatible runtime environments. Generate and
use an engine on the intended target GPU and TensorRT installation. Model profiles
control image conversion; pre-generation must use the same tensor layout as
inference. Incompatible plans may fail to deserialize or execute.

## Windows runtime requirements

Windows release archives contain `trt_sr.exe`, `README.md`, and `LICENSE` only.
Third-party runtime DLLs are neither bundled nor copied into application build outputs. The
CMake build disables vcpkg app-local dependency deployment (`VCPKG_APPLOCAL_DEPS=OFF`).
Install compatible x64 versions of
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
