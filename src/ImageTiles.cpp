// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "ImageTiles.hpp"
#include <algorithm>
void fillInput(const cv::Mat& src, Tile tilePos, int tile, int overlap, float* dst)
{
    const size_t plane = static_cast<size_t>(tile) * tile;
    for (int y = 0; y < tile; ++y)
    {
        const int sy = std::clamp(tilePos.y + y - overlap, 0, src.rows - 1);
        const auto* row = src.ptr<cv::Vec3b>(sy);
        for (int x = 0; x < tile; ++x)
        {
            const auto& p = row[std::clamp(tilePos.x + x - overlap, 0, src.cols - 1)];
            const size_t k = static_cast<size_t>(y) * tile + x;
            dst[k] = p[2] / 255.f;
            dst[plane + k] = p[1] / 255.f;
            dst[2 * plane + k] = p[0] / 255.f;
        }
    }
}
// Weight tapers only at interior tile edges; the image's outside edges remain fully weighted.
void blendTile(cv::Mat& accum, cv::Mat& weights, const cv::Mat& sr, Tile pos, int tile, int scale,
               int overlap, int srcW, int srcH)
{
    const int x0 = pos.x - overlap, y0 = pos.y - overlap;
    const int feather = std::max(1, overlap * scale);
    for (int y = 0; y < sr.rows; ++y)
    {
        const int gy = y0 * scale + y;
        if (gy < 0 || gy >= accum.rows)
            continue;
        auto* a = accum.ptr<cv::Vec3f>(gy);
        auto* w = weights.ptr<float>(gy);
        const auto* src = sr.ptr<cv::Vec3f>(y);
        float wy = 1.f;
        if (y0 > 0)
            wy = std::min(wy, static_cast<float>(y + 1) / feather);
        if (y0 + tile < srcH)
            wy = std::min(wy, static_cast<float>(sr.rows - y) / feather);
        for (int x = 0; x < sr.cols; ++x)
        {
            const int gx = x0 * scale + x;
            if (gx < 0 || gx >= accum.cols)
                continue;
            float wx = 1.f;
            if (x0 > 0)
                wx = std::min(wx, static_cast<float>(x + 1) / feather);
            if (x0 + tile < srcW)
                wx = std::min(wx, static_cast<float>(sr.cols - x) / feather);
            const float weight = std::max(0.00001f, wx * wy);
            a[gx] += src[x] * weight;
            w[gx] += weight;
        }
    }
}
