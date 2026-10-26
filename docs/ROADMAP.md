# Development roadmap

TexUpSR is a TensorRT-based image upscaler. ONNX is its model input format; adding other inference runtimes is not planned. The items below are proposals, not implemented features. They have no committed release dates.

## Current baseline — 0.1.0

- Single-input, single-output ONNX inference through TensorRT.
- Three-channel NCHW tensors, FP16 or FP32, with a caller-specified integer scale.
- Tiled image processing, overlap blending, CUDA streams, and serialized TensorRT engine reuse.
- Windows builds with Visual Studio 2022 and 2026. CI checks compilation and command-line behavior, not GPU inference or image quality.

## Planned features

### F1 — Model inspection

Add `--inspect <model.onnx>` to report ONNX graph inputs and outputs, tensor types and shapes, and any scale that can be established. Distinguish successful inspection, TensorRT parser acceptance, and successful engine construction; none is an automatic guarantee of correct image output.

Acceptance: report the required tensor information without running GPU inference; explain unsupported signatures and parser failures; cover static and dynamic-shape fixtures.

### F2 — Model input and output contracts

Separate model tensor conventions from image processing. Support additional compatible layouts and shape constraints when they can be handled without replacing TensorRT. Use optional profiles for normalization, channel order, padding, and other properties that cannot be inferred reliably from ONNX.

Acceptance: preserve output parity for existing RGB/NCHW models; reject ambiguous model requirements unless explicitly configured; test all supported conversion paths.

### F3 — Engine cache identity and recovery

Replace basename-only cache naming with identity derived from model contents, engine build parameters, and relevant GPU/TensorRT compatibility information. Rebuild invalid automatic-cache entries safely and avoid overwriting explicitly supplied engine files without permission.

Acceptance: changing the model or build settings cannot silently reuse an unrelated plan; corrupted automatic caches recover predictably; cache hits avoid unnecessary rebuilds.

### F4 — Architecture compatibility suite

Maintain a small, licensed set of representative ONNX fixtures covering CNN super-resolution, ESRGAN-family networks, and selected transformer-based models as TensorRT support permits. Record which exact model exports and TensorRT versions were tested.

Acceptance: tests report parse/build/runtime compatibility separately; GPU tests compare output shape, finite values, and image results against known references where available. Do not claim compatibility from architecture names alone.

### F5 — Performance measurement

Benchmark model build time, warm inference latency, throughput, VRAM usage, tiling overhead, and multi-stream scaling on identified hardware. Evaluate optimization changes against the same inputs and models.

Acceptance: reproducible benchmark commands and results with hardware, driver, TensorRT, model, tile and batch settings recorded.

## Not yet validated

GitHub-hosted Windows CI does not have a GPU inference test. CUDA cache-hit behavior for the latest unified cache configuration also remains unverified because the subsequent CI run was blocked by the account's Actions usage limit.
