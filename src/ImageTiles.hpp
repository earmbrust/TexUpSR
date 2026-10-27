// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once
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
void blendTile(cv::Mat& accum, cv::Mat& weights, const cv::Mat& sr, Tile pos, int tile, int scale,
               int overlap, int srcW, int srcH);

template <typename Sample> cv::Mat tensorToImage(int edge, Sample sample)
{
    cv::Mat image(edge, edge, CV_32FC3);
    const size_t plane = static_cast<size_t>(edge) * edge;

    for (int y = 0; y < edge; ++y)
    {
        auto* row = image.ptr<cv::Vec3f>(y);
        for (int x = 0; x < edge; ++x)
        {
            const size_t k = static_cast<size_t>(y) * edge + x;
            row[x] = cv::Vec3f(sample(2 * plane + k), sample(plane + k), sample(k));
        }
    }

    return image;
}
