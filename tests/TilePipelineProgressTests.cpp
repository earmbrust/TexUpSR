// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
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

int main()
{
    try
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

        if (completed != std::vector<size_t>({4, 6}) || expected != 6 || output.rows != 6 ||
            output.cols != 10)
        {
            throw std::runtime_error("Incorrect batch progress");
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Tile pipeline progress tests failed: " << error.what() << '\n';
        return 1;
    }
}
