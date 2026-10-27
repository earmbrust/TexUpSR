// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include <gtest/gtest.h>
#include "TilePipeline.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <thread>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void Require(bool value, const std::string& message)
{
    if (!value)
    {
        throw std::runtime_error(message);
    }
}

class FakeInference final : public Inference
{
public:
    explicit FakeInference(float color) : color_(color)
    {
    }

    BatchResult Run(const cv::Mat&, const std::vector<Tile>& tiles) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        calls_.push_back(tiles);

        if (fail_)
        {
            throw std::runtime_error("Simulated inference failure");
        }

        BatchResult result;
        result.tiles = tiles;
        for (size_t i = 0; i < tiles.size(); ++i)
        {
            result.images.emplace_back(8, 8, CV_32FC3, cv::Scalar(color_, color_, color_));
        }
        return result;
    }

    void Fail()
    {
        fail_ = true;
    }
    size_t Calls() const
    {
        return calls_.size();
    }
    const std::vector<std::vector<Tile>>& Batches() const
    {
        return calls_;
    }

private:
    float color_;
    bool fail_ = false;
    std::mutex mutex_;
    std::vector<std::vector<Tile>> calls_;
};

TEST(TilePipeline, BatchesAndWorkers)
{
    FakeInference first(0.25f);
    FakeInference second(0.25f);
    TilePipeline pipeline(4, 4, 2, 1, {&first, &second});
    cv::Mat source(3, 5, CV_8UC3, cv::Scalar(0, 0, 0));

    const auto output = pipeline.Run(source);
    Require(output.rows == 6 && output.cols == 10, "Output dimensions");
    Require(first.Calls() == 1 && second.Calls() == 1, "Worker scheduling");
    Require(first.Batches()[0].size() == 4, "First batch length");
    Require(second.Batches()[0].size() == 2, "Final batch length");

    for (int y = 0; y < output.rows; ++y)
    {
        for (int x = 0; x < output.cols; ++x)
        {
            const auto pixel = output.at<cv::Vec3f>(y, x);
            for (float value : pixel.val)
            {
                Require(std::abs(value - 0.25f) < 0.00001f, "Incomplete pixel coverage");
            }
        }
    }
}

TEST(TilePipeline, RejectsBadInputs)
{
    FakeInference fake(0.25f);
    for (const auto& workers : {std::vector<Inference*>{}, std::vector<Inference*>{nullptr}})
    {
        try
        {
            TilePipeline pipeline(4, 2, 2, 1, workers);
            throw std::runtime_error("Invalid workers accepted");
        }
        catch (const std::invalid_argument&)
        {
        }
    }

    TilePipeline pipeline(4, 2, 2, 1, {&fake});
    try
    {
        pipeline.Run(cv::Mat());
        throw std::runtime_error("Empty input accepted");
    }
    catch (const std::invalid_argument&)
    {
    }
}

TEST(TilePipeline, RejectsDuplicateWorkers)
{
    FakeInference fake(0.25f);

    try
    {
        TilePipeline pipeline(4, 2, 2, 1, {&fake, &fake});
        throw std::runtime_error("Duplicate worker accepted");
    }
    catch (const std::invalid_argument&)
    {
    }
}

class FailingWorker final : public Inference
{
public:
    explicit FailingWorker(std::shared_future<void> started) : started_(std::move(started))
    {
    }

    BatchResult Run(const cv::Mat&, const std::vector<Tile>&) override
    {
        started_.wait();
        throw std::runtime_error("Expected worker failure");
    }

private:
    std::shared_future<void> started_;
};

class PendingWorker final : public Inference
{
public:
    explicit PendingWorker(std::promise<void>& started) : started_(started)
    {
    }

    BatchResult Run(const cv::Mat&, const std::vector<Tile>& tiles) override
    {
        started_.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        completed_.store(true);
        BatchResult result;
        result.tiles = tiles;
        for (size_t i = 0; i < tiles.size(); ++i)
        {
            result.images.emplace_back(8, 8, CV_32FC3, cv::Scalar(0.25f, 0.25f, 0.25f));
        }
        return result;
    }

    bool Completed() const
    {
        return completed_.load();
    }

private:
    std::promise<void>& started_;
    std::atomic<bool> completed_{false};
};

TEST(TilePipeline, WaitsForPendingWorkerOnFailure)
{
    std::promise<void> started;
    FailingWorker failing(started.get_future().share());
    PendingWorker pending(started);
    TilePipeline pipeline(4, 1, 2, 1, {&failing, &pending});
    const cv::Mat image(2, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    bool failed = false;

    try
    {
        pipeline.Run(image);
    }
    catch (const std::runtime_error& error)
    {
        failed = std::string(error.what()) == "Expected worker failure";
    }

    Require(failed, "Worker failure was not propagated");
    Require(pending.Completed(), "Pipeline returned before pending worker completed");
}

TEST(TilePipeline, PropagatesFailures)
{
    FakeInference fake(0.25f);
    fake.Fail();
    TilePipeline pipeline(4, 2, 2, 1, {&fake});

    try
    {
        pipeline.Run(cv::Mat(2, 2, CV_8UC3, cv::Scalar(0, 0, 0)));
        throw std::runtime_error("Inference failure ignored");
    }
    catch (const std::runtime_error& error)
    {
        Require(std::string(error.what()) == "Simulated inference failure", "Unexpected failure message");
    }
}
class InvalidResultInference final : public Inference
{
public:
    enum class Error
    {
        TileCount,
        ImageCount,
        Position,
        Type,
        Dimensions
    };

    explicit InvalidResultInference(Error error) : error_(error)
    {
    }

    BatchResult Run(const cv::Mat&, const std::vector<Tile>& tiles) override
    {
        BatchResult result;
        result.tiles = tiles;
        for (size_t i = 0; i < tiles.size(); ++i)
        {
            result.images.emplace_back(8, 8, CV_32FC3, cv::Scalar(0.25f, 0.25f, 0.25f));
        }

        switch (error_)
        {
            case Error::TileCount:
                result.tiles.pop_back();
                break;
            case Error::ImageCount:
                result.images.pop_back();
                break;
            case Error::Position:
                ++result.tiles.front().x;
                break;
            case Error::Type:
                result.images.front() = cv::Mat(8, 8, CV_8UC3);
                break;
            case Error::Dimensions:
                result.images.front() = cv::Mat(7, 8, CV_32FC3);
                break;
        }

        return result;
    }

private:
    Error error_;
};

TEST(TilePipeline, RejectsMalformedInferenceResults)
{
    const cv::Mat image(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    for (const auto error :
         {InvalidResultInference::Error::TileCount, InvalidResultInference::Error::ImageCount,
          InvalidResultInference::Error::Position, InvalidResultInference::Error::Type,
          InvalidResultInference::Error::Dimensions})
    {
        InvalidResultInference worker(error);
        TilePipeline pipeline(4, 2, 2, 1, {&worker});
        EXPECT_THROW(pipeline.Run(image), std::runtime_error);
    }
}

TEST(TilePipeline, RejectsInvalidConfiguration)
{
    FakeInference worker(0.25f);
    EXPECT_THROW(TilePipeline(0, 1, 2, 1, {&worker}), std::invalid_argument);
    EXPECT_THROW(TilePipeline(4, 0, 2, 1, {&worker}), std::invalid_argument);
    EXPECT_THROW(TilePipeline(4, 1, 0, 1, {&worker}), std::invalid_argument);
    EXPECT_THROW(TilePipeline(4, 1, 2, -1, {&worker}), std::invalid_argument);
    EXPECT_THROW(TilePipeline(4, 1, 2, 2, {&worker}), std::invalid_argument);
}

TEST(TilePipeline, RejectsUnexpectedInputFormat)
{
    FakeInference worker(0.25f);
    TilePipeline pipeline(4, 1, 2, 1, {&worker});
    EXPECT_THROW(pipeline.Run(cv::Mat(2, 2, CV_8UC1)), std::invalid_argument);
    EXPECT_THROW(pipeline.Run(cv::Mat(2, 2, CV_8UC4)), std::invalid_argument);
    EXPECT_THROW(pipeline.Run(cv::Mat(2, 2, CV_32FC3)), std::invalid_argument);
}

} // namespace
