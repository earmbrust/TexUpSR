// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "ImageTiles.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

void near(float actual, float expected, const std::string& context, float tolerance = 0.00001f) {
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error(context + ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
}

void fillInputConvertsBgrToRgbPlanes() {
    cv::Mat source(2, 2, CV_8UC3);
    source.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 64, 255);
    source.at<cv::Vec3b>(0, 1) = cv::Vec3b(255, 0, 128);
    source.at<cv::Vec3b>(1, 0) = cv::Vec3b(20, 60, 100);
    source.at<cv::Vec3b>(1, 1) = cv::Vec3b(12, 34, 56);
    std::vector<float> result(12, -1.f);
    fillInput(source, {0, 0}, 2, 0, result.data());
    const int plane = 4;
    const cv::Vec3b pixels[] = {source.at<cv::Vec3b>(0, 0), source.at<cv::Vec3b>(0, 1),
                                source.at<cv::Vec3b>(1, 0), source.at<cv::Vec3b>(1, 1)};
    for (int i = 0; i < plane; ++i) {
        near(result[i], pixels[i][2] / 255.f, "R plane");
        near(result[plane + i], pixels[i][1] / 255.f, "G plane");
        near(result[2 * plane + i], pixels[i][0] / 255.f, "B plane");
    }
}

void fillInputClampsImageEdges() {
    cv::Mat source(2, 2, CV_8UC3);
    source.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    source.at<cv::Vec3b>(0, 1) = cv::Vec3b(40, 50, 60);
    source.at<cv::Vec3b>(1, 0) = cv::Vec3b(70, 80, 90);
    source.at<cv::Vec3b>(1, 1) = cv::Vec3b(100, 110, 120);
    std::vector<float> result(3 * 16, -1.f);
    fillInput(source, {0, 0}, 4, 1, result.data());
    const int plane = 16;
    auto at = [&](int c, int x, int y) { return result[c * plane + y * 4 + x]; };
    near(at(0, 0, 0), 30 / 255.f, "Top-left clamped R");
    near(at(2, 0, 0), 10 / 255.f, "Top-left clamped B");
    near(at(0, 2, 1), 60 / 255.f, "Right source pixel");
    near(at(0, 3, 3), 120 / 255.f, "Bottom-right clamped pixel");
    near(at(1, 0, 3), 80 / 255.f, "Bottom-left clamped G");

    std::fill(result.begin(), result.end(), -1.f);
    fillInput(source, {1, 1}, 4, 1, result.data());
    near(at(0, 3, 3), 120 / 255.f, "Offset bottom-right clamp");
    near(at(0, 0, 0), 30 / 255.f, "Offset tile origin");
}

void blendSingleTileRetainsAllPixels() {
    const int width = 2, height = 2, scale = 2, tile = 4, overlap = 1;
    cv::Mat accum(height * scale, width * scale, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(height * scale, width * scale, CV_32FC1, cv::Scalar(0));
    cv::Mat sr(tile * scale, tile * scale, CV_32FC3, cv::Scalar(0.1, 0.5, 0.9));
    blendTile(accum, weights, sr, {0, 0}, tile, scale, overlap, width, height);
    for (int y = 0; y < accum.rows; ++y) {
        for (int x = 0; x < accum.cols; ++x) {
            const float weight = weights.at<float>(y, x);
            require(weight > 0.f, "Single tile leaves an uncovered pixel");
            const cv::Vec3f pixel = accum.at<cv::Vec3f>(y, x) / weight;
            near(pixel[0], 0.1f, "Single tile B");
            near(pixel[1], 0.5f, "Single tile G");
            near(pixel[2], 0.9f, "Single tile R");
        }
    }
}

void blendOverlappingTilesCoversUnevenImage() {
    const int width = 5, height = 3, scale = 2, tile = 4, overlap = 1;
    const int stride = tile - 2 * overlap;
    cv::Mat accum(height * scale, width * scale, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(height * scale, width * scale, CV_32FC1, cv::Scalar(0));
    cv::Mat sr(tile * scale, tile * scale, CV_32FC3, cv::Scalar(0.2, 0.4, 0.6));
    for (int y = 0; y < height; y += stride)
        for (int x = 0; x < width; x += stride)
            blendTile(accum, weights, sr, {x, y}, tile, scale, overlap, width, height);
    for (int y = 0; y < accum.rows; ++y) {
        for (int x = 0; x < accum.cols; ++x) {
            const float weight = weights.at<float>(y, x);
            require(weight > 0.f, "Overlapping tiles leave an uncovered pixel");
            const auto color = accum.at<cv::Vec3f>(y, x) / weight;
            near(color[0], 0.2f, "Overlapping B");
            near(color[1], 0.4f, "Overlapping G");
            near(color[2], 0.6f, "Overlapping R");
        }
    }
}

void blendOverlappingTilesUsesWeightedAverage() {
    const int width = 4, height = 2, scale = 1, tile = 4, overlap = 1;
    cv::Mat accum(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(height, width, CV_32FC1, cv::Scalar(0));
    cv::Mat left(tile, tile, CV_32FC3, cv::Scalar(0.2, 0.2, 0.2));
    cv::Mat right(tile, tile, CV_32FC3, cv::Scalar(0.8, 0.8, 0.8));
    blendTile(accum, weights, left, {0, 0}, tile, scale, overlap, width, height);
    blendTile(accum, weights, right, {2, 0}, tile, scale, overlap, width, height);
    for (int y = 0; y < height; ++y) {
        near(accum.at<cv::Vec3f>(y, 0)[0] / weights.at<float>(y, 0), 0.2f, "Left edge");
        near(accum.at<cv::Vec3f>(y, 1)[0] / weights.at<float>(y, 1), 0.5f, "Overlap left");
        near(accum.at<cv::Vec3f>(y, 2)[0] / weights.at<float>(y, 2), 0.5f, "Overlap right");
        near(accum.at<cv::Vec3f>(y, 3)[0] / weights.at<float>(y, 3), 0.8f, "Right edge");
    }
}
}

int main() {
    try {
        fillInputConvertsBgrToRgbPlanes();
        fillInputClampsImageEdges();
        blendSingleTileRetainsAllPixels();
        blendOverlappingTilesCoversUnevenImage();
        blendOverlappingTilesUsesWeightedAverage();
        std::cout << "Image tile tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Image tile tests failed: " << e.what() << '\n';
        return 1;
    }
}
