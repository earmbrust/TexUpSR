// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once

#include "Inference.hpp"

#include <opencv2/core.hpp>
#include <functional>
#include <cstddef>
#include <vector>

class TilePipeline
{
public:
    TilePipeline(int tile, int batch, int scale, int overlap,
                 std::vector<Inference*> workers);

    using Progress = std::function<void(const BatchResult&, size_t, size_t)>;

    cv::Mat Run(const cv::Mat& image, const Progress& progress = {}) const;

private:
    int tile_;
    int batch_;
    int scale_;
    int overlap_;
    std::vector<Inference*> workers_;
};
