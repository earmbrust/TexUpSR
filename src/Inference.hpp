// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once

#include "ImageTiles.hpp"

class Inference
{
public:
    virtual ~Inference() = default;

    virtual BatchResult Run(const cv::Mat& image, const std::vector<Tile>& tiles) = 0;
};
