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
.\build-vcpkg\Release\trt_sr.exe --inspect .\tests\data\m4\cnn_2x.onnx
.\build-vcpkg\Release\trt_sr.exe --inspect .\tests\data\m4\rrdb_style_2x.onnx
.\build-vcpkg\Release\trt_sr.exe --inspect .\tests\data\m4\attention_2x.onnx
```

Use a controlled 8x8 PNG input for GPU validation. Example for the CNN fixture:

```powershell
.\build-vcpkg\Release\trt_sr.exe -i .\samples\m4_input_8x8.png -o .\samples\m4_cnn_result.png -m .\tests\data\m4\cnn_2x.onnx --tile 8 --batch 1 --scale 2 --overlap 0 --streams 1 --verbose
```

Record ONNX inspection, TensorRT parsing/building, and GPU inference as **separate**
results, including TensorRT version and GPU. The generation script does not validate
TensorRT parser/build/runtime compatibility; GPU acceptance remains pending until run on
supported hardware. For meaningful image reference comparisons, add independent reference
outputs in a later M4 testing tranche.
