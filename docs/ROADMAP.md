# Development roadmap

TexUpSR is a TensorRT-based image upscaler. ONNX is its model input format. Additional inference runtimes are not planned.

The work described below is implemented and merged. Its original scope and acceptance criteria are retained as a record. No release dates are assigned.

## Current baseline - 0.1.0

- Single-input, single-output ONNX inference through TensorRT.
- Three-channel NCHW or NHWC tensors, FP16 or FP32, with an integer scale and optional JSON pixel/layout profile.
- Tiled image processing with overlap blending, CUDA streams, and serialized TensorRT engine reuse.
- Windows AMD64 builds with Visual Studio 2026. The planned consumer release targets Windows 11; runtime compatibility requires validation on Windows 11.

## Completed work

### Model inspection (completed)

Add `--inspect <model.onnx>` to report ONNX graph inputs and outputs, tensor types and shapes, and any scale that can be established. Distinguish successful inspection, TensorRT parser acceptance, and successful engine construction.

Acceptance:

- Report required tensor information without running GPU inference.
- Explain unsupported signatures and parser failures.
- Cover static-shape and dynamic-shape fixtures.

### Model input and output contracts (completed)

Separate model tensor conventions from image processing. Support additional compatible layouts and shape constraints when they can be handled without replacing TensorRT. Use optional profiles for normalization, channel order, padding, and other properties that cannot be inferred reliably from ONNX.

Acceptance:

- Preserve output parity for existing RGB/NCHW models.
- Reject ambiguous model requirements unless explicitly configured.
- Test all supported conversion paths.

Implementation note: the current model profile supports layout, channel order, and pixel scale/offset. General model-specific padding is not implemented; the original scope above is retained for historical accuracy.

### Engine cache identity and recovery (completed)

Replace basename-only cache naming with identity derived from model contents, engine build parameters, and relevant GPU/TensorRT compatibility information. Rebuild invalid automatic-cache entries safely and avoid overwriting explicitly supplied engine files without permission.

Acceptance:

- Changing the model or build settings cannot silently reuse an unrelated plan.
- Corrupted automatic caches recover predictably.
- Cache hits avoid unnecessary rebuilds.
- `--overwrite-engine` forces rebuilding either automatic caches or explicit engine paths, preserving prior plans on build failure.

### Architecture compatibility suite (completed)

Maintain a small, licensed set of representative ONNX fixtures covering CNN super-resolution, ESRGAN-family networks, and selected transformer-based models as TensorRT support permits. Record which exact model exports and TensorRT versions were tested.

Acceptance:

- Report parse, build, and runtime compatibility separately.
- For GPU tests, compare output shape, finite values, and image results against known references where available.
- Do not claim compatibility from architecture names alone.

Implementation note: The compatibility suite validates original synthetic CNN-, residual-dense-, and attention-style ONNX graphs against checked-in numeric references. These fixtures do not establish support for trained ESRGAN or transformer exports; see [compatibility-fixtures.md](compatibility-fixtures.md).

### Performance measurement (completed)

Measure engine initialization, CUDA inference event sums, upscaling wall time, throughput, device memory snapshots, and multi-stream behavior on identified hardware. Performance optimization remains deferred.

Add an optional `--benchmark` flag that reports:

- Inference time in microseconds: TensorRT GPU execution measured with CUDA events.
- Upscaling time in microseconds: preprocessing, inference, tile handling, and blending, excluding engine initialization and disk I/O.
- Total time in microseconds: application execution measured inside `main`, including engine loading and image I/O but excluding OS process creation and loader startup.

Acceptance:

- Provide reproducible benchmark commands.
- Record hardware, driver, TensorRT, model, tile, and batch settings with the results.

Benchmark measurement semantics: `--benchmark` reports CUDA-event execution
summed across workers (not wall-clock elapsed time during concurrent streams), upscaling
wall time, complete application time and selected run metadata. A single run is not
a warm-inference distribution or a measurement of peak VRAM. Reproducible runs
should pin model, input, tile, batch, streams, overlap, workspace and driver.
See [performance.md](performance.md) for recorded results, measurement limits, and reproducible commands.

### Command-line engine pre-generation (completed)

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
- Preserve the current build and load diagnostics, `--quiet`/`--verbose` behavior, and `--overwrite-engine` safety rules.
- Preserve the no-runtime-DLL-copy build and distribution policy.

Engine pre-generation uses `--build-engine` with `--model`, `--engine`, explicit positive `--tile` and `--batch`, and relevant build options. It reuses the existing `Engine` build and serialization path without loading an input image or performing image processing. The existing short options remain unchanged (`-s` is scale, `-d` is debug). This work was completed and merged through pull request #15 (merge commit 1a4cc2d).
