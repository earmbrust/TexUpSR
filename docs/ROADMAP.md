# Development roadmap

TexUpSR is a TensorRT-based image upscaler. ONNX is its model input format. Additional inference runtimes are not planned.

The items below are planned features, not implemented features. No release dates are assigned.

## Current baseline - 0.1.0

- Single-input, single-output ONNX inference through TensorRT.
- Three-channel NCHW tensors, FP16 or FP32, with a caller-specified integer scale.
- Tiled image processing with overlap blending, CUDA streams, and serialized TensorRT engine reuse.
- Windows AMD64 builds with Visual Studio 2026. The planned consumer release targets Windows 11; runtime compatibility requires validation on Windows 11.

## Planned features

### F1 - Model inspection

Add `--inspect <model.onnx>` to report ONNX graph inputs and outputs, tensor types and shapes, and any scale that can be established. Distinguish successful inspection, TensorRT parser acceptance, and successful engine construction.

Acceptance:

- Report required tensor information without running GPU inference.
- Explain unsupported signatures and parser failures.
- Cover static-shape and dynamic-shape fixtures.

### F2 - Model input and output contracts

Separate model tensor conventions from image processing. Support additional compatible layouts and shape constraints when they can be handled without replacing TensorRT. Use optional profiles for normalization, channel order, padding, and other properties that cannot be inferred reliably from ONNX.

Acceptance:

- Preserve output parity for existing RGB/NCHW models.
- Reject ambiguous model requirements unless explicitly configured.
- Test all supported conversion paths.

### F3 - Engine cache identity and recovery

Replace basename-only cache naming with identity derived from model contents, engine build parameters, and relevant GPU/TensorRT compatibility information. Rebuild invalid automatic-cache entries safely and avoid overwriting explicitly supplied engine files without permission.

Acceptance:

- Changing the model or build settings cannot silently reuse an unrelated plan.
- Corrupted automatic caches recover predictably.
- Cache hits avoid unnecessary rebuilds.

### F4 - Architecture compatibility suite

Maintain a small, licensed set of representative ONNX fixtures covering CNN super-resolution, ESRGAN-family networks, and selected transformer-based models as TensorRT support permits. Record which exact model exports and TensorRT versions were tested.

Acceptance:

- Report parse, build, and runtime compatibility separately.
- For GPU tests, compare output shape, finite values, and image results against known references where available.
- Do not claim compatibility from architecture names alone.

### F5 - Performance measurement

Benchmark model build time, warm inference latency, throughput, VRAM usage, tiling overhead, and multi-stream scaling on identified hardware. Evaluate optimization changes against the same inputs and models.

Acceptance:

- Provide reproducible benchmark commands.
- Record hardware, driver, TensorRT, model, tile, and batch settings with the results.

### F6 - Command-line engine pre-generation

Add a command-line operation to build and save a TensorRT engine from an ONNX model without processing an image. Support automated workflows, batch preparation, and scripting.

Reuse the existing engine-building implementation rather than maintaining a separate compilation path.

Acceptance:

- Build a serialized TensorRT engine without supplying an input image.
- Accept an explicit output path, tile size, batch size, and relevant engine-building parameters.
- Validate model requirements and optimization profiles before compilation.
- Return meaningful exit codes and diagnostics for automation.
- Never silently overwrite an existing engine when an explicit output path is provided.
- Ensure pre-generated engines can be loaded by the normal upscaling workflow.
- Document TensorRT, GPU, and platform compatibility limitations for serialized engines.
