# M4 generated compatibility fixtures

These are original, deterministic **synthetic** ONNX graphs authored for TexUpSR. They have
no external model weights or licensing dependencies. They exercise operators and tensor
contracts; **they do not demonstrate compatibility with trained CNN, ESRGAN, or SR transformer
model exports**. Test real exports separately before making family-wide compatibility claims.

| File | Coverage | Limitations |
| --- | --- | --- |
| `cnn_2x.onnx` | 1x1 convolution, ReLU, 2x Resize | Minimal CNN, not a trained SR model |
| `rrdb_style_2x.onnx` | Two residual dense-style stages, Concat, LeakyRelu, Conv, Add, 2x Resize | Not a canonical RRDB architecture or trained ESRGAN |
| `attention_2x.onnx` | Spatial token attention: Transpose, Reshape, MatMul, Softmax, residual Add, 2x Resize | Tiny attention graph, not a SwinIR or other trained SR transformer |

All fixtures accept FP32 NCHW `[1,3,8,8]`, produce FP32 NCHW `[1,3,16,16]`,
and use ONNX opset 13. Weights and constants are fixed. No randomness or Python packages
are required to regenerate them.

From the repository root:

```powershell
python tools/generate_m4_fixtures.py
```

To inspect all three fixtures without TensorRT GPU execution:

```powershell
.\build-integration\Release\trt_sr.exe --inspect .\tests\data\m4\cnn_2x.onnx
.\build-integration\Release\trt_sr.exe --inspect .\tests\data\m4\rrdb_style_2x.onnx
.\build-integration\Release\trt_sr.exe --inspect .\tests\data\m4\attention_2x.onnx
```

The GPU integration test checks the three fixtures against tracked `.f32` numeric references without requiring a user-supplied image. On a supported Windows GPU build:

```powershell
cmake --preset integration
cmake --build --preset integration
ctest --preset integration
```

For manual image upscaling, supply your own 8x8 PNG at `samples/m4_input_8x8.png` before running:

```powershell
.\build-integration\Release\trt_sr.exe -i .\samples\m4_input_8x8.png -o .\samples\m4_cnn_result.png -m .\tests\data\m4\cnn_2x.onnx --tile 8 --batch 1 --scale 2 --overlap 0 --streams 1 --verbose
```

The example paths above assume a multi-configuration Visual Studio build; for a single-configuration generator, use the executable directly under `build-integration`.

Record ONNX inspection, TensorRT parsing/building, and GPU inference as **separate**
results, including TensorRT version and GPU. The generation script by itself does not validate TensorRT parser, build, or runtime compatibility. `gpu_inference_tests` checks these synthetic fixtures and compares results with checked-in reference tensors when a usable GPU is available; otherwise the integration test can skip. The Windows GPU integration acceptance run passed during M6 validation. This does not establish support for trained CNN, ESRGAN, or transformer exports.
