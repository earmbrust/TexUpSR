// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "TilePipeline.hpp"

#include <cmath>
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
    explicit FakeInference(float color) : color_(color) {}

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
            result.images.emplace_back(8, 8, CV_32FC3,
                                       cv::Scalar(color_, color_, color_));
        }
        return result;
    }

    void Fail() { fail_ = true; }
    size_t Calls() const { return calls_.size(); }
    const std::vector<std::vector<Tile>>& Batches() const { return calls_; }

private:
    float color_;
    bool fail_ = false;
    std::mutex mutex_;
    std::vector<std::vector<Tile>> calls_;
};

void BatchesAndWorkers()
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

void RejectsBadInputs()
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

void PropagatesFailures()
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
        Require(std::string(error.what()) == "Simulated inference failure",
                "Unexpected failure message");
    }
}
}

int main()
{
    try
    {
        BatchesAndWorkers();
        RejectsBadInputs();
        PropagatesFailures();
        std::cout << "Tile pipeline tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Tile pipeline tests failed: " << error.what() << '\n';
        return 1;
    }
}
