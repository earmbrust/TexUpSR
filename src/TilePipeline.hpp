// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once

#include "Inference.hpp"

#include <opencv2/core.hpp>
#include <vector>

class TilePipeline
{
public:
    TilePipeline(int tile, int batch, int scale, int overlap,
                 std::vector<Inference*> workers);

    cv::Mat Run(const cv::Mat& image) const;

private:
    int tile_;
    int batch_;
    int scale_;
    int overlap_;
    std::vector<Inference*> workers_;
};
