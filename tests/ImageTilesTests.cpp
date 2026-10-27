// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include <gtest/gtest.h>
#include "ImageTiles.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}

void near(float actual, float expected, const std::string& context, float tolerance = 0.00001f)
{
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error(context + ": expected " + std::to_string(expected) + ", got " +
                                 std::to_string(actual));
}

TEST(ImageTiles, fillInputConvertsBgrToRgbPlanes)
{
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
    for (int i = 0; i < plane; ++i)
    {
        near(result[i], pixels[i][2] / 255.f, "R plane");
        near(result[plane + i], pixels[i][1] / 255.f, "G plane");
        near(result[2 * plane + i], pixels[i][0] / 255.f, "B plane");
    }
}

TEST(ImageTiles, fillInputClampsImageEdges)
{
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

TEST(ImageTiles, blendSingleTileRetainsAllPixels)
{
    const int width = 2, height = 2, scale = 2, tile = 4, overlap = 1;
    cv::Mat accum(height * scale, width * scale, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(height * scale, width * scale, CV_32FC1, cv::Scalar(0));
    cv::Mat sr(tile * scale, tile * scale, CV_32FC3, cv::Scalar(0.1, 0.5, 0.9));
    blendTile(accum, weights, sr, {0, 0}, tile, scale, overlap, width, height);
    for (int y = 0; y < accum.rows; ++y)
    {
        for (int x = 0; x < accum.cols; ++x)
        {
            const float weight = weights.at<float>(y, x);
            require(weight > 0.f, "Single tile leaves an uncovered pixel");
            const cv::Vec3f pixel = accum.at<cv::Vec3f>(y, x) / weight;
            near(pixel[0], 0.1f, "Single tile B");
            near(pixel[1], 0.5f, "Single tile G");
            near(pixel[2], 0.9f, "Single tile R");
        }
    }
}

TEST(ImageTiles, blendOverlappingTilesCoversUnevenImage)
{
    const int width = 5, height = 3, scale = 2, tile = 4, overlap = 1;
    const int stride = tile - 2 * overlap;
    cv::Mat accum(height * scale, width * scale, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(height * scale, width * scale, CV_32FC1, cv::Scalar(0));
    cv::Mat sr(tile * scale, tile * scale, CV_32FC3, cv::Scalar(0.2, 0.4, 0.6));
    for (int y = 0; y < height; y += stride)
        for (int x = 0; x < width; x += stride)
            blendTile(accum, weights, sr, {x, y}, tile, scale, overlap, width, height);
    for (int y = 0; y < accum.rows; ++y)
    {
        for (int x = 0; x < accum.cols; ++x)
        {
            const float weight = weights.at<float>(y, x);
            require(weight > 0.f, "Overlapping tiles leave an uncovered pixel");
            const auto color = accum.at<cv::Vec3f>(y, x) / weight;
            near(color[0], 0.2f, "Overlapping B");
            near(color[1], 0.4f, "Overlapping G");
            near(color[2], 0.6f, "Overlapping R");
        }
    }
}

TEST(ImageTiles, blendOverlappingTilesUsesWeightedAverage)
{
    const int width = 4, height = 2, scale = 1, tile = 4, overlap = 1;
    cv::Mat accum(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(height, width, CV_32FC1, cv::Scalar(0));
    cv::Mat left(tile, tile, CV_32FC3, cv::Scalar(0.2, 0.2, 0.2));
    cv::Mat right(tile, tile, CV_32FC3, cv::Scalar(0.8, 0.8, 0.8));
    blendTile(accum, weights, left, {0, 0}, tile, scale, overlap, width, height);
    blendTile(accum, weights, right, {2, 0}, tile, scale, overlap, width, height);
    for (int y = 0; y < height; ++y)
    {
        near(accum.at<cv::Vec3f>(y, 0)[0] / weights.at<float>(y, 0), 0.2f, "Left edge");
        near(accum.at<cv::Vec3f>(y, 1)[0] / weights.at<float>(y, 1), 0.5f, "Overlap left");
        near(accum.at<cv::Vec3f>(y, 2)[0] / weights.at<float>(y, 2), 0.5f, "Overlap right");
        near(accum.at<cv::Vec3f>(y, 3)[0] / weights.at<float>(y, 3), 0.8f, "Right edge");
    }
}

TEST(ImageTiles, fillInputSupportsRoiAndPreservesBounds)
{
    cv::Mat image(4, 5, CV_8UC3, cv::Scalar(1, 2, 3));
    cv::Mat roi = image(cv::Rect(1, 1, 2, 2));
    roi.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    roi.at<cv::Vec3b>(0, 1) = cv::Vec3b(40, 50, 60);
    roi.at<cv::Vec3b>(1, 0) = cv::Vec3b(70, 80, 90);
    roi.at<cv::Vec3b>(1, 1) = cv::Vec3b(100, 110, 120);
    require(!roi.isContinuous(), "ROI must exercise non-contiguous row strides");

    std::vector<float> buffer(16, -7.f);
    fillInput(roi, {0, 0}, 2, 0, buffer.data() + 2);

    near(buffer[0], -7.f, "Prefix guard 0");
    near(buffer[1], -7.f, "Prefix guard 1");
    near(buffer[14], -7.f, "Suffix guard 0");
    near(buffer[15], -7.f, "Suffix guard 1");

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            const cv::Vec3b pixel = roi.at<cv::Vec3b>(y, x);
            const int index = y * 2 + x;
            near(buffer[2 + index], pixel[2] / 255.f, "ROI red");
            near(buffer[6 + index], pixel[1] / 255.f, "ROI green");
            near(buffer[10 + index], pixel[0] / 255.f, "ROI blue");
        }
    }
}

TEST(ImageTiles, blendTilePreservesSpatialOrientationAtScaleTwo)
{
    const int width = 2;
    const int height = 2;
    const int tile = 3;
    const int scale = 2;
    const int overlap = 1;
    cv::Mat accum(height * scale, width * scale, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(height * scale, width * scale, CV_32FC1, cv::Scalar(0));
    cv::Mat sr(tile * scale, tile * scale, CV_32FC3);

    for (int y = 0; y < sr.rows; ++y)
    {
        for (int x = 0; x < sr.cols; ++x)
        {
            sr.at<cv::Vec3f>(y, x) = cv::Vec3f(x / 10.f, y / 10.f, (x + y) / 20.f);
        }
    }

    blendTile(accum, weights, sr, {0, 0}, tile, scale, overlap, width, height);
    for (int y = 0; y < accum.rows; ++y)
    {
        for (int x = 0; x < accum.cols; ++x)
        {
            const float weight = weights.at<float>(y, x);
            require(weight > 0.f, "Scaled tile leaves uncovered output pixel");
            const auto actual = accum.at<cv::Vec3f>(y, x) / weight;
            const auto expected = sr.at<cv::Vec3f>(y + overlap * scale, x + overlap * scale);
            for (int channel = 0; channel < 3; ++channel)
            {
                near(actual[channel], expected[channel], "Scaled tile spatial orientation");
            }
        }
    }
}

TEST(ImageTiles, blendTileWithoutOverlapHasUnitWeights)
{
    cv::Mat accum(2, 2, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(2, 2, CV_32FC1, cv::Scalar(0));
    cv::Mat sr(2, 2, CV_32FC3, cv::Scalar(0.3, 0.5, 0.7));

    blendTile(accum, weights, sr, {0, 0}, 2, 1, 0, 2, 2);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            near(weights.at<float>(y, x), 1.f, "No overlap weight");
            for (int channel = 0; channel < 3; ++channel)
            {
                near(accum.at<cv::Vec3f>(y, x)[channel], sr.at<cv::Vec3f>(y, x)[channel], "No overlap pixel");
            }
        }
    }
}
} // namespace

TEST(ImageTiles, FillInputClampsAsymmetricBoundaries)
{
    cv::Mat source(2, 3, CV_8UC3);
    for (int y = 0; y < source.rows; ++y)
    {
        for (int x = 0; x < source.cols; ++x)
        {
            source.at<cv::Vec3b>(y, x) = cv::Vec3b(10 + x + 10 * y, 40 + 2 * x + 10 * y, 80 + 3 * x + 10 * y);
        }
    }

    constexpr int tile = 4;
    constexpr int overlap = 1;
    constexpr int plane = tile * tile;
    for (const Tile position : {Tile{0, 0}, Tile{2, 1}})
    {
        std::vector<float> output(3 * plane, -1.f);
        fillInput(source, position, tile, overlap, output.data());
        for (int y = 0; y < tile; ++y)
        {
            for (int x = 0; x < tile; ++x)
            {
                const int sx = std::clamp(position.x + x - overlap, 0, source.cols - 1);
                const int sy = std::clamp(position.y + y - overlap, 0, source.rows - 1);
                const auto pixel = source.at<cv::Vec3b>(sy, sx);
                const int index = y * tile + x;
                EXPECT_NEAR(output[index], pixel[2] / 255.f, 0.00001f);
                EXPECT_NEAR(output[plane + index], pixel[1] / 255.f, 0.00001f);
                EXPECT_NEAR(output[2 * plane + index], pixel[0] / 255.f, 0.00001f);
            }
        }
    }
}

TEST(ImageTiles, BlendTileClipsAtImageBoundaries)
{
    constexpr int width = 3;
    constexpr int height = 2;
    constexpr int tile = 4;
    constexpr int scale = 2;
    constexpr int overlap = 1;
    cv::Mat accum(height * scale, width * scale, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(height * scale, width * scale, CV_32FC1, cv::Scalar(0));
    cv::Mat sr(tile * scale, tile * scale, CV_32FC3);

    for (int y = 0; y < sr.rows; ++y)
    {
        for (int x = 0; x < sr.cols; ++x)
        {
            sr.at<cv::Vec3f>(y, x) = cv::Vec3f(x / 10.f, y / 10.f, (x + y) / 20.f);
        }
    }

    blendTile(accum, weights, sr, {0, 0}, tile, scale, overlap, width, height);
    for (int y = 0; y < accum.rows; ++y)
    {
        for (int x = 0; x < accum.cols; ++x)
        {
            const float weight = weights.at<float>(y, x);
            ASSERT_GT(weight, 0.f);
            const auto actual = accum.at<cv::Vec3f>(y, x) / weight;
            const auto expected = sr.at<cv::Vec3f>(y + overlap * scale, x + overlap * scale);
            for (int channel = 0; channel < 3; ++channel)
            {
                EXPECT_NEAR(actual[channel], expected[channel], 0.00001f);
            }
        }
    }
}

TEST(ImageTiles, BlendUnevenImageAtScaleThree)
{
    constexpr int width = 5;
    constexpr int height = 3;
    constexpr int tile = 4;
    constexpr int scale = 3;
    constexpr int overlap = 1;
    constexpr int stride = tile - 2 * overlap;
    cv::Mat accum(height * scale, width * scale, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(height * scale, width * scale, CV_32FC1, cv::Scalar(0));
    cv::Mat sr(tile * scale, tile * scale, CV_32FC3, cv::Scalar(0.2f, 0.4f, 0.6f));

    for (int y = 0; y < height; y += stride)
    {
        for (int x = 0; x < width; x += stride)
        {
            blendTile(accum, weights, sr, {x, y}, tile, scale, overlap, width, height);
        }
    }

    for (int y = 0; y < accum.rows; ++y)
    {
        for (int x = 0; x < accum.cols; ++x)
        {
            const float weight = weights.at<float>(y, x);
            ASSERT_GT(weight, 0.f);
            const auto color = accum.at<cv::Vec3f>(y, x) / weight;
            EXPECT_NEAR(color[0], 0.2f, 0.00001f);
            EXPECT_NEAR(color[1], 0.4f, 0.00001f);
            EXPECT_NEAR(color[2], 0.6f, 0.00001f);
        }
    }
}

TEST(ImageTiles, TensorOutputMapsRgbPlanesToBgrPixels)
{
    const float planes[] = {1.f, 2.f, 3.f, 4.f, 11.f, 12.f, 13.f, 14.f, 21.f, 22.f, 23.f, 24.f};
    const auto image = tensorToImage(2, [&](size_t index) { return planes[index]; });

    ASSERT_EQ(image.type(), CV_32FC3);
    ASSERT_EQ(image.rows, 2);
    ASSERT_EQ(image.cols, 2);
    EXPECT_EQ(image.at<cv::Vec3f>(0, 0), cv::Vec3f(21.f, 11.f, 1.f));
    EXPECT_EQ(image.at<cv::Vec3f>(0, 1), cv::Vec3f(22.f, 12.f, 2.f));
    EXPECT_EQ(image.at<cv::Vec3f>(1, 0), cv::Vec3f(23.f, 13.f, 3.f));
    EXPECT_EQ(image.at<cv::Vec3f>(1, 1), cv::Vec3f(24.f, 14.f, 4.f));
}

TEST(ImageTiles, TensorOutputHandlesHalfPrecisionSamples)
{
    const cv::float16_t planes[] = {cv::float16_t(0.25f), cv::float16_t(0.5f), cv::float16_t(0.75f)};
    const auto image = tensorToImage(1, [&](size_t index) { return static_cast<float>(planes[index]); });

    ASSERT_EQ(image.type(), CV_32FC3);
    EXPECT_EQ(image.at<cv::Vec3f>(0, 0), cv::Vec3f(0.75f, 0.5f, 0.25f));
}
