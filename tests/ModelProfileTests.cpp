// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "ModelProfile.hpp"
#include "ImageTiles.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <cstdio>
#include <vector>

TEST(ModelProfile, DefaultShapeAndConstraints)
{
    EXPECT_EQ(modelShape(2, 8, TensorLayout::NCHW), (std::array<int, 4>{2, 3, 8, 8}));
    EXPECT_EQ(modelShape(2, 8, TensorLayout::NHWC), (std::array<int, 4>{2, 8, 8, 3}));
    EXPECT_NO_THROW(validateModelShape({-1, 3, -1, -1}, {2, 3, 8, 8}, "Input"));
    EXPECT_THROW(validateModelShape({1, 3, 64, 64}, {2, 3, 8, 8}, "Input"), std::runtime_error);
    EXPECT_THROW(validateModelShape({1, 0, 8, 8}, {1, 3, 8, 8}, "Input"), std::runtime_error);
}

TEST(ModelProfile, NHWCAndBgrConversions)
{
    ModelProfile p;
    p.layout = TensorLayout::NHWC;
    p.inputOrder = ChannelOrder::BGR;
    p.outputOrder = ChannelOrder::BGR;
    p.inputScale = 255.f;
    p.outputScale = 1.f / 255.f;
    cv::Mat source(1, 2, CV_8UC3);
    source.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    source.at<cv::Vec3b>(0, 1) = cv::Vec3b(40, 50, 60);
    std::vector<float> data(12);
    fillInput(source, {0, 0}, 2, 0, data.data(), p);
    EXPECT_FLOAT_EQ(data[0], 10.f);
    EXPECT_FLOAT_EQ(data[1], 20.f);
    EXPECT_FLOAT_EQ(data[2], 30.f);
    EXPECT_FLOAT_EQ(data[3], 40.f);
    const auto output = tensorToImage(2, [&](size_t k) { return data[k]; }, p);
    EXPECT_NEAR(output.at<cv::Vec3f>(0, 0)[0], 10.f / 255.f, 0.00001f);
}

TEST(ModelProfile, RejectUnknownProfileProperties)
{
    const std::string path = "texupsr-model-profile-test.json";
    {
        std::ofstream out(path);
        out << "{\"unrecognized\": 1}";
    }
    EXPECT_THROW(loadModelProfile(path), std::runtime_error);
    std::remove(path.c_str());
}
