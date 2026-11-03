# M5 - Performance measurement

TexUpSR supports `--benchmark` on image-upscaling runs. Benchmarking is measurement only: it does not change tile selection, engine caching, or inference algorithms.

## Measurements and limits

- **Inference CUDA event sum (microseconds):** sum of TensorRT enqueue durations measured using CUDA events for each batch. With multiple streams, event intervals may overlap: **this is not the elapsed inference makespan**. Do not subtract it from upscaling wall time.
- **Upscaling wall time (microseconds):** preprocessing, enqueue, transfers, tile handling, and blending, excluding engine construction, file loading, and output encoding. This is the primary metric for comparing tiling and stream configurations.
- **Engine initialization (microseconds):** engine load or build and deserialization, measured together. On a cache miss, this **is not solely the TensorRT compiler's build time**; TensorRT's own build-completion log supplies its own estimate.
- **Total application (microseconds):** elapsed time within `main` from before argument parsing to after output writing and before benchmark printing. OS process creation/loader startup and benchmark-printing time are excluded. For complete external process duration, use PowerShell `Measure-Command` or a dedicated process timer.
- **Output pixels/second:** output width multiplied by output height divided by upscaling wall time.
- **Device memory used before/after:** two CUDA memory snapshots; their difference is a **net allocation change**, not peak VRAM, nor a measure of TensorRT's peak builder allocations. TensorRT build logs sometimes include its own allocator peak, but that number has a different scope.

For apples-to-apples measurements use the same model bytes, image, tile, batch, overlap, streams, workspace, scale, GPU, TensorRT/driver versions, and background workload. Warm the engine first, then repeat cached runs. Compare **median upscaling wall time**, not single runs or cumulative CUDA event sums. Keep engine-build runs separate from steady-state runs.

## Reproducible Windows benchmark

The commands below use only files tracked in the repository. Configure and build using the checked-in `integration` preset:

```powershell
cmake --preset integration
cmake --build --preset integration --target trt_sr
```

For Visual Studio multi-configuration builds the application is `build-integration\Release\trt_sr.exe`; for Ninja and other single-configuration builds it is directly in `build-integration`. Set the following PowerShell variables to your actual executable, compatible ONNX model, and input image paths:

```powershell
$exe = ".\build-integration\Release\trt_sr.exe"
$model = "C:\path\to\model.onnx"
$input = "C:\path\to\input.png"
$output = Join-Path $env:TEMP "texupsr-benchmark-output.png"
1..4 | ForEach-Object { & $exe -i $input -o $output -m $model --benchmark --scale 2 --tile 256 --batch 1 --workspace 2 --overlap 16 --streams 2 }
```

The first run prepares or loads the engine; compare the following three cached runs using the median **Upscaling wall time**. For a cold engine, the first run includes build time only in the separate engine-initialization metric. Repeat the run with tile/stream pairs `128/2`, `256/2`, `512/2`, and `256/1` to compare the original M5 configurations, using the same model, input, GPU, and dependency versions. The model and input image used for the recorded results are not supplied by these commands; provide your own compatible files. The old `patch/runtime-correction/runtime-env.ps1`, `scripts/m5-benchmark.ps1`, and `scripts/m5-validate.ps1` were local milestone helpers and are not part of a fresh checkout.

For routine tests, build the `unit` preset and run `ctest --test-dir build-unit -C Release -L unit --output-on-failure`. GPU integration tests run with `ctest --preset integration` after building that preset; they require suitable TensorRT/CUDA installations and hardware.

## Clean Windows runtime deployment verification

After M5, a fresh Release configuration was validated with `cmake --preset integration -B build-dll-verification` and `cmake --build build-dll-verification --config Release --target trt_sr`. The newly built Release directory contained no third-party DLL copies, and the executable ran `--help` with installed runtime dependencies available in the Windows DLL search path. This verifies that Windows environment and build; it is not a claim of verification on every platform. Old build directories may retain DLLs from earlier configurations. TexUpSR deliberately disables vcpkg app-local deployment and does not include third-party runtime DLLs in release archives.

## Recorded test measurements: RTX 2080 / TensorRT 10.16.1

Configuration: Adore FP32, busorenkin4.png (853 x 480), output 1706 x 960, batch 1, 2 GiB workspace, overlap 16, CUDA driver 13030. Times in milliseconds; measurements supplied by Windows runtime testing on 2026-11-02.

| Tile | Streams | Tiles | Upscaling | CUDA event sum | Notes |
|---:|---:|---:|---:|---:|---|
| 256 | 1 | 12 | 180.65 | 110.76 | Cached engine, one run |
| 256 | 2 | 12 | 133.46 | 188.13 | Cached engine, one run |
| 128 | 2 | 45 | 131.28 | 125.07 | Initial engine-build run; no steady-state repeats |
| 256 | 2 | 12 | 132.61 | 180.10 | Cached engine, one run |
| 512 | 2 | 2 | 47.71 | 42.99 | Initial engine-build run; not representative of subsequent cached runs |
| 512 | 2 | 2 | 89.62 | 129.42 | Cached, additional run |
| 512 | 2 | 2 | 95.70 | 137.74 | Cached repeat 1 |
| 512 | 2 | 2 | 93.74 | 133.03 | Cached repeat 2 |
| 512 | 2 | 2 | 90.47 | 130.42 | Cached repeat 3 |

Median of the final three cached tile-512 repeats: **93.74 ms**; median output throughput **17.47 million pixels/s**. The three cached repeats varied from 90.47 to 95.70 ms. The single 47.71 ms build-run result is not a valid estimate of warm median performance. The observed 1-versus-2-stream comparison at tile 256 reduced upscaling time by 26.1%, but needs repeated measurements before general conclusions.

A separate automated three-run cached-engine comparison on the same hardware and model recorded median upscaling wall times of **125.51 ms** (tile 128, 2 streams), **118.01 ms** (tile 256, 2 streams), **96.42 ms** (tile 512, 2 streams), and **142.18 ms** (tile 256, 1 stream). These are distinct runs from the individual measurements above, not replacements for them. The helper script and raw local CSV/logs were not committed with M5.

TensorRT reported engine generation of **107.58 s** for tile 128 and **134.575 s** for tile 512; corresponding application-side engine initialization was **110.94 s** and **137.52 s**. These are different timers and should not be interchanged.

Peak VRAM and tiled-preprocessing-only time are **not separately measured** by the current runtime. The benchmark currently reports net memory change and inclusive upscaling wall time. Do not label either as a peak or an isolated tiling overhead measurement. Implementing accurate peak GPU sampling or isolated CPU phase timing would require separate instrumentation and validation; these are not fabricated here.
