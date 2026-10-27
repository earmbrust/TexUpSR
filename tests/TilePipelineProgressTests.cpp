// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include <gtest/gtest.h>
#include "TilePipeline.hpp"

#include <opencv2/core.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class FakeInference final : public Inference
{
public:
    explicit FakeInference(float value) : value_(value)
    {
    }

    BatchResult Run(const cv::Mat&, const std::vector<Tile>& tiles) override
    {
        BatchResult result;
        result.tiles = tiles;
        for (size_t i = 0; i < tiles.size(); ++i)
        {
            result.images.emplace_back(8, 8, CV_32FC3, cv::Scalar(value_, value_, value_));
        }
        return result;
    }

private:
    float value_;
};

TEST(TilePipelineProgress, ReportsCompletedBatches)
{
    FakeInference first(0.25f);
    FakeInference second(0.5f);
    TilePipeline pipeline(4, 4, 2, 1, {&first, &second});
    const cv::Mat input(3, 5, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<size_t> completed;
    size_t expected = 0;
    const cv::Mat output = pipeline.Run(input,
                                        [&](const BatchResult& batch, size_t count, size_t total)
                                        {
                                            if (batch.images.size() != batch.tiles.size())
                                            {
                                                throw std::runtime_error("Bad progress batch");
                                            }
                                            completed.push_back(count);
                                            expected = total;
                                        });

    if (completed != std::vector<size_t>({4, 6}) || expected != 6 || output.rows != 6 || output.cols != 10)
    {
        throw std::runtime_error("Incorrect batch progress");
    }
}

TEST(TilePipelineProgress, PropagatesCallbackFailure)
{
    FakeInference worker(0.25f);
    TilePipeline pipeline(4, 2, 2, 1, {&worker});
    const cv::Mat input(2, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    size_t callbacks = 0;

    EXPECT_THROW(pipeline.Run(input,
                              [&](const BatchResult&, size_t, size_t)
                              {
                                  ++callbacks;
                                  throw std::runtime_error("Progress callback failed");
                              }),
                 std::runtime_error);
    EXPECT_EQ(callbacks, 1u);
}

TEST(TilePipelineProgress, ReportsPartialFinalBatch)
{
    FakeInference worker(0.25f);
    TilePipeline pipeline(4, 2, 2, 1, {&worker});
    const cv::Mat input(3, 5, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<size_t> completed;
    std::vector<size_t> batchSizes;
    std::vector<size_t> totals;

    const auto output = pipeline.Run(input,
                                     [&](const BatchResult& batch, size_t count, size_t total)
                                     {
                                         completed.push_back(count);
                                         batchSizes.push_back(batch.tiles.size());
                                         totals.push_back(total);
                                     });

    EXPECT_EQ(completed, (std::vector<size_t>{2, 4, 6}));
    EXPECT_EQ(batchSizes, (std::vector<size_t>{2, 2, 2}));
    EXPECT_EQ(totals, (std::vector<size_t>{6, 6, 6}));
    EXPECT_EQ(output.cols, 10);
    EXPECT_EQ(output.rows, 6);
}

TEST(TilePipelineProgress, RunsWithoutCallback)
{
    FakeInference worker(0.25f);
    TilePipeline pipeline(4, 2, 2, 1, {&worker});
    const cv::Mat input(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));

    const auto output = pipeline.Run(input);
    ASSERT_EQ(output.rows, 4);
    ASSERT_EQ(output.cols, 4);
    EXPECT_EQ(output.type(), CV_32FC3);
}
