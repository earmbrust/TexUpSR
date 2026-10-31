// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once

#include <array>
#include <string>

enum class TensorLayout
{
    NCHW,
    NHWC
};
enum class ChannelOrder
{
    RGB,
    BGR
};

// Pixel conversions are independent of TensorRT and ONNX parsing.
// Input: (pixel / 255) * inputScale + inputOffset.
// Output: modelValue * outputScale + outputOffset (normalized 0..1 BGR).
struct ModelProfile
{
    TensorLayout layout = TensorLayout::NCHW;
    ChannelOrder inputOrder = ChannelOrder::RGB;
    ChannelOrder outputOrder = ChannelOrder::RGB;
    float inputScale = 1.f;
    float inputOffset = 0.f;
    float outputScale = 1.f;
    float outputOffset = 0.f;
};

ModelProfile loadModelProfile(const std::string& path);
std::array<int, 4> modelShape(int batch, int tile, TensorLayout layout);
void validateModelShape(const std::array<int, 4>& declared, const std::array<int, 4>& actual,
                        const std::string& tensor);
