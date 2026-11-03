// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Engine.hpp"
#include "EngineCache.hpp"
#include "TilePipeline.hpp"
#include "Upscaler.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

static_assert(!std::is_copy_constructible_v<Stream>);
static_assert(!std::is_copy_assignable_v<Stream>);
static_assert(!std::is_copy_constructible_v<Worker>);

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
        throw std::runtime_error(std::string("CUDA device initialization failed: ") +
                                 cudaGetErrorString(initialize));
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
                                             std::to_string(x) + "," + std::to_string(y) + " channel " +
                                             std::to_string(channel));
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
            input.at<cv::Vec3b>(y, x) =
                cv::Vec3b(static_cast<unsigned char>(20 + x * 9), static_cast<unsigned char>(30 + y * 13),
                          static_cast<unsigned char>(40 + x * 7 + y * 5));
        }
    }

    options.buildEngine = true;
    {
        Engine generated(options, options.tile, options.batch, logger, ModelProfile{});
    }
    Require(fs::exists(cachePath) && fs::file_size(cachePath) > 0, "Pre-generated engine missing or empty");
    bool alreadyExists = false;
    try
    {
        Engine duplicate(options, options.tile, options.batch, logger, ModelProfile{});
    }
    catch (const std::runtime_error&)
    {
        alreadyExists = true;
    }
    Require(alreadyExists, "Pre-generation silently reused an existing engine");
    options.buildEngine = false;

    for (int pass = 0; pass < 2; ++pass)
    {
        Engine engine(options, options.tile, options.batch, logger, ModelProfile{});
        const size_t elements = static_cast<size_t>(options.batch) * 3 * options.tile * options.tile;
        std::vector<std::unique_ptr<Worker>> workers;
        std::vector<Inference*> inference;
        for (int i = 0; i < options.streams; ++i)
        {
            workers.emplace_back(std::make_unique<Worker>(
                engine.Get(), elements, elements, engine.InputType(), engine.OutputType(), engine.InputName(),
                engine.OutputName(), options.tile, options.batch, options.scale, options.overlap));
            inference.push_back(workers.back().get());
        }
        if (pass == 0)
        {
            Worker invalid(engine.Get(), elements, elements, engine.InputType(), engine.OutputType(),
                           engine.InputName(), "missing_output", options.tile, options.batch, options.scale,
                           options.overlap);
            bool rejected = false;
            try
            {
                invalid.Run(input, {{0, 0}});
            }
            catch (const std::runtime_error&)
            {
                rejected = true;
            }
            Require(rejected, "Invalid tensor output name was accepted");
        }
        TilePipeline pipeline(options.tile, options.batch, options.scale, options.overlap, inference);
        VerifyPixels(pipeline.Run(input), input);
        Require(fs::exists(cachePath) && fs::file_size(cachePath) > 0, "Engine cache missing or empty");
    }

    {
        const auto beforeDigest = cacheFileDigest(cachePath);
        Options failure = options;
        failure.buildEngine = true;
        failure.overwriteEngine = true;
        failure.model = (cachePath.parent_path() / "missing-model.onnx").string();
        bool failedBuild = false;
        try
        {
            Engine invalid(failure, failure.tile, failure.batch, logger, ModelProfile{});
        }
        catch (const std::runtime_error&)
        {
            failedBuild = true;
        }
        Require(failedBuild && cacheFileDigest(cachePath) == beforeDigest,
                "Failed engine regeneration did not preserve the existing plan");
    }

    {
        std::ofstream corrupt(cachePath, std::ios::binary | std::ios::trunc);
        Require(static_cast<bool>(corrupt << "invalid serialized engine"), "Could not corrupt engine cache");
    }
    bool rejected = false;
    try
    {
        Engine invalid(options, options.tile, options.batch, logger, ModelProfile{});
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Require(rejected, "Corrupt engine cache was accepted");
}

void RunM4Reference(const fs::path& fixtureDirectory, const fs::path& directory, const std::string& name)
{
    Options options;
    options.model = (fixtureDirectory / (name + ".onnx")).string();
    options.engine = (directory / (name + ".plan")).string();
    options.workspace = 1.0;
    options.scale = 2;
    options.streams = 2;
    options.overlap = 0;
    options.batch = 1;
    options.tile = 8;
    Logger logger;

    cv::Mat input(8, 8, CV_8UC3);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            input.at<cv::Vec3b>(y, x) = cv::Vec3b((x * 19 + y * 7) % 256,
                                                     (30 + x * 11 + y * 9) % 256,
                                                     (80 + x * 3 + y * 13) % 256);

    Engine engine(options, options.tile, options.batch, logger, ModelProfile{});
    const size_t inCount = 3 * 8 * 8;
    const size_t outCount = 3 * 16 * 16;
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<Inference*> inference;
    for (int i = 0; i < options.streams; ++i)
    {
        workers.emplace_back(std::make_unique<Worker>(engine.Get(), inCount, outCount,
            engine.InputType(), engine.OutputType(), engine.InputName(), engine.OutputName(),
            options.tile, options.batch, options.scale, options.overlap));
        inference.push_back(workers.back().get());
    }
    TilePipeline pipeline(options.tile, options.batch, options.scale, options.overlap, inference);
    const cv::Mat actual = pipeline.Run(input);
    Require(actual.rows == 16 && actual.cols == 16 && actual.type() == CV_32FC3,
            name + ": unexpected output dimensions or type");
    const fs::path referencePath = fixtureDirectory / (name + ".f32");
    std::ifstream reference(referencePath, std::ios::binary | std::ios::ate);
    Require(static_cast<bool>(reference), name + ": missing numerical reference");
    Require(reference.tellg() == static_cast<std::streamoff>(outCount * sizeof(float)),
            name + ": unexpected reference size");
    reference.seekg(0);
    std::vector<float> expected(outCount);
    reference.read(reinterpret_cast<char*>(expected.data()),
                   static_cast<std::streamsize>(outCount * sizeof(float)));
    Require(static_cast<bool>(reference), name + ": reference read failed");
    float maxDifference = 0.f;
    for (int y = 0; y < 16; ++y)
    {
        for (int x = 0; x < 16; ++x)
        {
            const auto pixel = actual.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c)
            {
                const float got = pixel[c];
                const float want = expected[(y * 16 + x) * 3 + c];
                Require(std::isfinite(got), name + ": non-finite output at " +
                    std::to_string(x) + "," + std::to_string(y));
                const float difference = std::abs(got - want);
                maxDifference = std::max(maxDifference, difference);
                Require(difference <= 0.003f + 0.003f * std::abs(want),
                    name + ": numerical mismatch at " + std::to_string(x) + "," +
                    std::to_string(y) + " channel " + std::to_string(c) +
                    ": actual " + std::to_string(got) + ", expected " +
                    std::to_string(want));
            }
        }
    }
    std::cout << "M4 " << name << " reference PASS; maximum absolute difference "
              << maxDifference << '\n';
}

} // namespace

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
        const fs::path directory =
            fs::temp_directory_path() / ("texupsr-gpu-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(directory);
        const fs::path cachePath = directory / "relu.plan";
        try
        {
            fs::remove(cachePath);
            RunInference(modelPath, cachePath);
            const fs::path m4Fixtures(TEXUPSR_M4_FIXTURE_DIR);
            RunM4Reference(m4Fixtures, directory, "cnn_2x");
            RunM4Reference(m4Fixtures, directory, "rrdb_style_2x");
            RunM4Reference(m4Fixtures, directory, "attention_2x");
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
