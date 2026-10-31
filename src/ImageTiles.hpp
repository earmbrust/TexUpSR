// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once
#include "ModelProfile.hpp"
#include <opencv2/core.hpp>
#include <vector>
struct Tile
{
    int x, y;
};
struct BatchResult
{
    std::vector<Tile> tiles;
    std::vector<cv::Mat> images;
};
void fillInput(const cv::Mat& src, Tile tilePos, int tile, int overlap, float* dst);
void fillInput(const cv::Mat& src, Tile tilePos, int tile, int overlap, float* dst,
               const ModelProfile& profile);
void blendTile(cv::Mat& accum, cv::Mat& weights, const cv::Mat& sr, Tile pos, int tile, int scale,
               int overlap, int srcW, int srcH);

template <typename Sample> cv::Mat tensorToImage(int edge, Sample sample, const ModelProfile& profile)
{
    cv::Mat image(edge, edge, CV_32FC3);
    const size_t plane = static_cast<size_t>(edge) * edge;

    for (int y = 0; y < edge; ++y)
    {
        auto* row = image.ptr<cv::Vec3f>(y);
        for (int x = 0; x < edge; ++x)
        {
            const size_t k = static_cast<size_t>(y) * edge + x;
            const auto at = [&](int channel)
            {
                const size_t index = profile.layout == TensorLayout::NCHW
                                         ? static_cast<size_t>(channel) * plane + k
                                         : k * 3 + channel;
                return sample(index) * profile.outputScale + profile.outputOffset;
            };
            if (profile.outputOrder == ChannelOrder::RGB)
                row[x] = cv::Vec3f(at(2), at(1), at(0));
            else
                row[x] = cv::Vec3f(at(0), at(1), at(2));
        }
    }

    return image;
}

template <typename Sample> cv::Mat tensorToImage(int edge, Sample sample)
{
    return tensorToImage(edge, sample, ModelProfile{});
}
