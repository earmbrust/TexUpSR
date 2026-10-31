// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "ModelProfile.hpp"

#include <opencv2/core.hpp>
#include <cmath>
#include <stdexcept>

std::array<int, 4> modelShape(int batch, int tile, TensorLayout layout)
{
    if (batch <= 0 || tile <= 0)
        throw std::invalid_argument("Invalid model batch or tile");
    return layout == TensorLayout::NCHW ? std::array<int, 4>{batch, 3, tile, tile}
                                        : std::array<int, 4>{batch, tile, tile, 3};
}

void validateModelShape(const std::array<int, 4>& declared, const std::array<int, 4>& actual,
                        const std::string& tensor)
{
    for (size_t i = 0; i < declared.size(); ++i)
    {
        if (declared[i] == 0 || declared[i] < -1 || actual[i] <= 0 ||
            (declared[i] != -1 && declared[i] != actual[i]))
            throw std::runtime_error(tensor + " dimension " + std::to_string(i) +
                                     " is incompatible with requested batch/tile/layout");
    }
}

ModelProfile loadModelProfile(const std::string& path)
{
    ModelProfile profile;
    if (path.empty())
        return profile;
    cv::FileStorage file(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!file.isOpened())
        throw std::runtime_error("Cannot read model profile: " + path);

    // Unknown fields are rejected to prevent silent typos and incorrect colors.
    for (const auto& node : file.root())
    {
        const std::string key = node.name();
        if (key != "layout" && key != "input_order" && key != "output_order" && key != "input_scale" &&
            key != "input_offset" && key != "output_scale" && key != "output_offset")
            throw std::runtime_error("Unknown model profile property: " + key);
    }
    const auto choose = [&](const char* key, const char* a, const char* b, bool& second)
    {
        const cv::FileNode node = file[key];
        if (node.empty())
            return;
        if (!node.isString())
            throw std::runtime_error(std::string(key) + " must be a string");
        const std::string value = (std::string)node;
        if (value != a && value != b)
            throw std::runtime_error(std::string("Invalid ") + key + ": " + value);
        second = value == b;
    };
    bool selected = false;
    choose("layout", "nchw", "nhwc", selected);
    profile.layout = selected ? TensorLayout::NHWC : TensorLayout::NCHW;
    selected = false;
    choose("input_order", "rgb", "bgr", selected);
    profile.inputOrder = selected ? ChannelOrder::BGR : ChannelOrder::RGB;
    selected = false;
    choose("output_order", "rgb", "bgr", selected);
    profile.outputOrder = selected ? ChannelOrder::BGR : ChannelOrder::RGB;

    const auto numeric = [&](const char* key, float& field, bool positive)
    {
        const cv::FileNode node = file[key];
        if (node.empty())
            return;
        if (!node.isReal() && !node.isInt())
            throw std::runtime_error(std::string(key) + " must be numeric");
        const double value = (double)node;
        if (!std::isfinite(value) || (positive && value <= 0.0) || std::abs(value) > 1.0e20)
            throw std::runtime_error(std::string("Invalid ") + key);
        field = static_cast<float>(value);
    };
    numeric("input_scale", profile.inputScale, true);
    numeric("input_offset", profile.inputOffset, false);
    numeric("output_scale", profile.outputScale, true);
    numeric("output_offset", profile.outputOffset, false);
    return profile;
}
