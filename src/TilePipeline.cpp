// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "TilePipeline.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <deque>
#include <future>
#include <stdexcept>
#include <utility>

TilePipeline::TilePipeline(int tile, int batch, int scale, int overlap,
                           std::vector<Inference*> workers)
    : tile_(tile), batch_(batch), scale_(scale), overlap_(overlap),
      workers_(std::move(workers))
{
    if (tile_ <= 0 || batch_ <= 0 || scale_ <= 0 || overlap_ < 0 ||
        static_cast<int64_t>(overlap_) * 2 >= tile_ || workers_.empty())
    {
        throw std::invalid_argument("Invalid pipeline configuration");
    }

    for (const auto* worker : workers_)
    {
        if (!worker)
        {
            throw std::invalid_argument("Null inference worker");
        }
    }

    for (size_t i = 0; i < workers_.size(); ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            if (workers_[i] == workers_[j])
            {
                throw std::invalid_argument("Duplicate inference worker");
            }
        }
    }
}

cv::Mat TilePipeline::Run(const cv::Mat& image, const Progress& progress) const
{
    if (image.empty() || image.type() != CV_8UC3)
    {
        throw std::invalid_argument("Expected a nonempty BGR image");
    }

    const int64_t width = static_cast<int64_t>(image.cols) * scale_;
    const int64_t height = static_cast<int64_t>(image.rows) * scale_;
    const int64_t outTile = static_cast<int64_t>(tile_) * scale_;

    if (width > INT_MAX || height > INT_MAX || outTile > INT_MAX)
    {
        throw std::overflow_error("Upscaled dimensions overflow int");
    }

    cv::Mat accum(static_cast<int>(height), static_cast<int>(width),
                  CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(static_cast<int>(height), static_cast<int>(width),
                    CV_32FC1, cv::Scalar(0));

    const int stride = tile_ - 2 * overlap_;
    std::vector<Tile> tiles;

    for (int y = 0; y < image.rows; y += stride)
    {
        for (int x = 0; x < image.cols; x += stride)
        {
            tiles.push_back({x, y});
        }
    }

    std::deque<std::future<BatchResult>> jobs;

    size_t processed = 0;
    auto consume = [&](std::future<BatchResult>& job, const std::vector<Tile>& expected)
    {
        auto result = job.get();
        if (result.tiles.size() != expected.size() || result.images.size() != expected.size())
        {
            throw std::runtime_error("Inference returned an invalid batch size");
        }

        for (size_t i = 0; i < expected.size(); ++i)
        {
            if (result.tiles[i].x != expected[i].x || result.tiles[i].y != expected[i].y ||
                result.images[i].type() != CV_32FC3 ||
                result.images[i].cols != outTile || result.images[i].rows != outTile)
            {
                throw std::runtime_error("Inference returned an invalid tile");
            }

            blendTile(accum, weights, result.images[i], expected[i], tile_, scale_,
                      overlap_, image.cols, image.rows);
        }

        processed += result.tiles.size();
        if (progress)
        {
            progress(result, processed, tiles.size());
        }
    };

    std::deque<std::vector<Tile>> pending;
    size_t group = 0;

    for (size_t start = 0; start < tiles.size(); start += static_cast<size_t>(batch_), ++group)
    {
        if (jobs.size() == workers_.size())
        {
            consume(jobs.front(), pending.front());
            jobs.pop_front();
            pending.pop_front();
        }

        const size_t end = std::min(start + static_cast<size_t>(batch_), tiles.size());
        pending.emplace_back(tiles.begin() + start, tiles.begin() + end);
        auto* worker = workers_[group % workers_.size()];
        const auto batchTiles = pending.back();
        jobs.emplace_back(std::async(std::launch::async,
            [worker, &image, batchTiles]() { return worker->Run(image, batchTiles); }));
    }

    while (!jobs.empty())
    {
        consume(jobs.front(), pending.front());
        jobs.pop_front();
        pending.pop_front();
    }

    for (int y = 0; y < accum.rows; ++y)
    {
        auto* pixels = accum.ptr<cv::Vec3f>(y);
        const auto* rowWeights = weights.ptr<float>(y);

        for (int x = 0; x < accum.cols; ++x)
        {
            if (rowWeights[x] <= 0.f)
            {
                throw std::runtime_error("Uncovered output pixel during tile blending");
            }

            pixels[x] /= rowWeights[x];
        }
    }

    return accum;
}
