# TexUpSR

TexUpSR is a command-line image upscaler for Windows and Linux built on NVIDIA TensorRT, CUDA, and OpenCV. It loads compatible ONNX super-resolution models, builds TensorRT engines, and processes images in overlapping tiles.

**Version:** 0.2.0.
See [the roadmap](docs/ROADMAP.md) for release scope, [portability](docs/PORTABILITY.md)
for supported build environments, [compatibility fixtures](docs/COMPATIBILITY-FIXTURES.md)
for model test coverage, and [performance](docs/PERFORMANCE.md) for benchmarks.

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

### Simple image upscaling

No worker process, IPC configuration, or network service is needed to upscale an
image. Supply an image, a compatible ONNX model, and an output path:

```powershell
.\build\Release\trt_sr.exe -i .\input.png -o .\output.png -m .\model.onnx --scale 4
```

The paths above are examples; point them to your own input image and model.
For a single-configuration Linux build, use `./build/trt_sr` with the same
arguments. Engine caching and tile processing work automatically.

### Using TexUpSR from another application

A host application may launch a persistent worker and submit raw images or
video frames without temporary image files. The host retains control of its
workflow (video decoding, timestamps, layer selection, and final output).
TexUpSR loads its model and returns upscaled BGR24 pixels. Choose **one**
local transport per worker:

| Transport | Example worker arguments | How the host connects |
| --- | --- | --- |
| Stdio | `--stream` | Launch as a child and connect binary stdin/stdout |
| Named pipe / FIFO | `--pipe editor_01` | Connect to the input and output endpoints by name |
| Shared memory | `--shared-memory editor_01` | Open the mapping and request/response synchronization objects |

All local worker modes require `--width`, `--height`, `--format bgr24`,
and `--model` and accept the normal model, tile, batch, scale, and stream
configuration flags. Dimensions and format are fixed for the worker's lifetime.
These transports are usable with a single GPU; they do **not** enable automatic
multi-GPU scheduling. Networking and distributed workers are not part of 0.2.0.

### Raw-frame streaming

Run a persistent worker for fixed-size, packed BGR24 video frames:

```powershell
.\build-integration\Release\trt_sr.exe --stream --width 720 --height 480 --format bgr24 --scale 2 -m model.onnx
```

Write exactly `width * height * 3` bytes per input frame to stdin, in BGR
channel order, 8 bits per channel, tightly packed with no row padding.
The worker writes exactly `(width * scale) * (height * scale) * 3` bytes per
output frame to stdout in the same BGR24 format. There are no frame headers,
length prefixes, timestamps, or text on stdout. Stderr carries diagnostics.
Read each full output frame before sending another input frame if the parent
uses synchronous pipes; independently buffered or concurrent I/O is also valid.
The model, engine, and inference workers stay initialized across frames.
Normal EOF at a frame boundary exits successfully. A partial input frame,
failed inference, or failed output write exits with an error; callers should
treat the stream as failed rather than guessing frame boundaries. All frames
must use the dimensions provided at startup. Existing `--tile`, `--batch`,
`--streams`, `--profile`, and `--engine` settings work in this mode. `--debug`
and `--benchmark` are not accepted while streaming.

### Local worker transports

The following commands start named local workers, which wait for an external
application to connect. They are not standalone image-upscaling commands.
Use a unique worker name for each concurrent instance.

TexUpSR also accepts `--pipe <name>` and `--shared-memory <name>` instead of
`--stream`. All three modes keep a TensorRT engine loaded and process fixed-size
BGR24 frames; the application remains responsible for its own video encoding,
frame ordering, and any other processing. Select exactly one transport. Names
contain 1-64 ASCII letters, digits, hyphens, or underscores. They identify a
single worker session and cannot be reused while an instance is running.

```text
trt_sr --pipe editor_01 --width 128 --height 96 --format bgr24 --scale 4 -m model.onnx
trt_sr --shared-memory editor_01 --width 128 --height 96 --format bgr24 --scale 4 -m model.onnx
```

For all three transports, the input is tightly packed BGR24 with
`width * height * 3` bytes per frame; output is packed BGR24 with
`(width * scale) * (height * scale) * 3` bytes per frame. The protocol does
not transfer timing information, filenames, or image metadata. The host must
preserve that information itself.

For `--pipe`, TexUpSR creates a local, byte-oriented input/output pair and
waits for one client. The client opens the input for writing, **then** the
output for reading. On Windows the endpoints are
`\\.\pipe\TexUpSR-<name>-input` and `\\.\pipe\TexUpSR-<name>-output`.
On POSIX systems they are owner-only FIFOs at
`/tmp/texupsr-<name>-input` and `/tmp/texupsr-<name>-output`; TexUpSR removes
them when it exits normally. A closed input at a frame boundary ends the
worker; a truncated frame is an error. Clients must read output concurrently
with writing input, or exchange one frame at a time to avoid pipe deadlocks.

For `--shared-memory`, TexUpSR creates an input/output memory region and two
synchronization objects. One request can be outstanding at a time. The first
48 bytes of the mapping are a little-endian header:

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u32 | Magic `0x52535554` (`TUSR`) |
| 4 | u32 | Protocol version (`1`) |
| 8, 12, 16 | u32 | Input width, input height, scale |
| 20 | u32 | Format (`1` = BGR24) |
| 24, 32 | u64 | Input byte count, output byte count |
| 40 | u32 | State (`0` idle, `1` request, `2` response, `3` stop, `4` error) |
| 44 | u32 | Reserved (`0`) |

Immediately after the header are `inputBytes` writable bytes, followed by
`outputBytes` bytes for the result. The consumer waits until the mapping and
both synchronization objects exist, checks the header, copies a frame into
the input region, sets state to `1`, and signals `request`. It waits for
`response`, checks for state `2` (or `4` on failure), and reads the result.
Only after receiving a result may it submit another frame. To close cleanly,
set state `3`, signal `request`, and wait for `response` before closing.
Diagnostics and errors are written to stderr, never into frame buffers.

On Windows the memory mapping is `Local\TexUpSR-<name>` and the auto-reset
events are `Local\TexUpSR-<name>-request` and
`Local\TexUpSR-<name>-response`. The producer opens these objects by name.
On POSIX the shared-memory object is `/texupsr-<name>` and the named
semaphores are `/texupsr-<name>-request` and
`/texupsr-<name>-response`. The local worker creates objects with owner-only
permissions and unlinks them at shutdown. The protocol uses native-endian
integer fields; supported x64 Windows/Linux releases are little-endian.

These transports are intended for local application integration, including
image-editor plugins and frame-by-frame video pipelines. A host exchanging
large frames through stdio or named pipes must read and write concurrently,
or use sequential request/response exchanges, to avoid a full-pipe deadlock.
The shared-memory interface supports exactly one outstanding request per
worker; it does not implement a queue or concurrent clients.

These local transports do not add a background service, GPU scheduler, network
listener, or remote-worker functionality. A host application can launch one
worker per chosen GPU, but multi-GPU coordination is not implemented by this
release. Standalone file-based upscaling requires none of these interfaces.

### File-based upscaling

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
| `--stream` | Persistently process raw frames over binary stdin/stdout | Off |
| `--pipe` | Serve a local named pipe/FIFO pair | Off |
| `--shared-memory` | Serve a local shared-memory frame exchange | Off |
| `--width` | Input width for streaming | Required with `--stream` |
| `--height` | Input height for streaming | Required with `--stream` |
| `--format` | Packed stream pixel format | `bgr24` only |
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

For GPU integration testing, build with the `integration` preset and run `ctest --preset integration`. This is separate from ordinary unit testing. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for benchmark measurements and commands.

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
