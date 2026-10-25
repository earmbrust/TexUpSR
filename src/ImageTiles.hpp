// Copyright (c) 2025 Elden Armbrust. MIT License.
#pragma once
#include <opencv2/core.hpp>
#include <vector>
struct Tile { int x, y; };
struct BatchResult { std::vector<Tile> tiles; std::vector<cv::Mat> images; };
void fillInput(const cv::Mat& src, Tile tilePos, int tile, int overlap, float* dst);
void blendTile(cv::Mat& accum, cv::Mat& weights, const cv::Mat& sr,
               Tile pos, int tile, int scale, int overlap, int srcW, int srcH);
