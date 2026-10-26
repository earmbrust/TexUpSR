// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Engine.hpp"
#include "TilePipeline.hpp"
#include "Upscaler.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <windows.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
constexpr int Skip = 77;

void Require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

bool SkipRequested()
{
    const char* value = std::getenv("TEXUPSR_SKIP_GPU_TESTS");
    return value && std::string(value) == "1";
}

int CheckGpu()
{
    if (SkipRequested())
    {
        std::cout << "GPU test skipped: TEXUPSR_SKIP_GPU_TESTS=1\n";
        return Skip;
    }

    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
    {
        std::cout << "GPU test skipped: no usable CUDA device or driver\n";
        return Skip;
    }
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA device discovery failed: ") + cudaGetErrorString(status));
    }
    if (count == 0)
    {
        std::cout << "GPU test skipped: no CUDA devices\n";
        return Skip;
    }

    // A visible device is not necessarily usable by the CUDA runtime.
    const cudaError_t initialize = cudaSetDevice(0);
    if (initialize != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA device initialization failed: ") + cudaGetErrorString(initialize));
    }
    return 0;
}

void VerifyPixels(const cv::Mat& actual, const cv::Mat& source)
{
    Require(actual.rows == source.rows && actual.cols == source.cols, "Unexpected image dimensions");
    Require(actual.type() == CV_32FC3, "Unexpected output image type");

    for (int y = 0; y < source.rows; ++y)
    {
        for (int x = 0; x < source.cols; ++x)
        {
            const auto expected = source.at<cv::Vec3b>(y, x);
            const auto pixel = actual.at<cv::Vec3f>(y, x);
            for (int channel = 0; channel < 3; ++channel)
            {
                const float value = expected[channel] / 255.f;
                if (std::abs(pixel[channel] - value) > 0.002f)
                {
                    throw std::runtime_error("GPU output differs from expected pixel at " +
                                             std::to_string(x) + "," + std::to_string(y) +
                                             " channel " + std::to_string(channel));
                }
            }
        }
    }
}

void RunInference(const fs::path& modelPath, const fs::path& cachePath)
{
    Options options;
    options.model = modelPath.string();
    options.engine = cachePath.string();
    options.workspace = 1.0;
    options.scale = 1;
    options.streams = 2;
    options.overlap = 1;
    options.batch = 2;
    options.tile = 4;

    Logger logger;
    cv::Mat input(5, 7, CV_8UC3);
    for (int y = 0; y < input.rows; ++y)
    {
        for (int x = 0; x < input.cols; ++x)
        {
            input.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<unsigned char>(20 + x * 9),
                static_cast<unsigned char>(30 + y * 13),
                static_cast<unsigned char>(40 + x * 7 + y * 5));
        }
    }

    for (int pass = 0; pass < 2; ++pass)
    {
        Engine engine(options, options.tile, options.batch, logger);
        const size_t elements = static_cast<size_t>(options.batch) * 3 * options.tile * options.tile;
        std::vector<std::unique_ptr<Worker>> workers;
        std::vector<Inference*> inference;
        for (int i = 0; i < options.streams; ++i)
        {
            workers.emplace_back(std::make_unique<Worker>(
                engine.Get(), elements, elements, engine.InputType(), engine.OutputType(),
                engine.InputName(), engine.OutputName(), options.tile, options.batch,
                options.scale, options.overlap));
            inference.push_back(workers.back().get());
        }
        TilePipeline pipeline(options.tile, options.batch, options.scale, options.overlap, inference);
        VerifyPixels(pipeline.Run(input), input);
        Require(fs::exists(cachePath) && fs::file_size(cachePath) > 0, "Engine cache missing or empty");
    }

    {
        std::ofstream corrupt(cachePath, std::ios::binary | std::ios::trunc);
        Require(static_cast<bool>(corrupt << "invalid serialized engine"), "Could not corrupt engine cache");
    }
    bool rejected = false;
    try
    {
        Engine invalid(options, options.tile, options.batch, logger);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Require(rejected, "Corrupt engine cache was accepted");
}
}

int main()
{
    try
    {
        const int gpu = CheckGpu();
        if (gpu == Skip)
        {
            return Skip;
        }

        const fs::path modelPath(TEXUPSR_GPU_MODEL_PATH);
        Require(fs::exists(modelPath), "GPU test ONNX fixture is missing");
        const fs::path directory = fs::temp_directory_path() /
            ("texupsr-gpu-test-" + std::to_string(GetCurrentProcessId()));
        fs::create_directories(directory);
        const fs::path cachePath = directory / "relu.plan";
        try
        {
            fs::remove(cachePath);
            RunInference(modelPath, cachePath);
            fs::remove_all(directory);
        }
        catch (...)
        {
            fs::remove_all(directory);
            throw;
        }
        std::cout << "GPU inference and cache tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "GPU inference test failed: " << error.what() << '\n';
        return 1;
    }
}
