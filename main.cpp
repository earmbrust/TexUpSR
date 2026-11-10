// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Engine.hpp"
#include "LocalTransport.hpp"
#include "ModelInspection.hpp"
#include "Options.hpp"
#include "Sizing.hpp"
#include "TensorSizes.hpp"
#include "TilePipeline.hpp"
#include "Upscaler.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <climits>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#endif

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    try
    {
        const auto applicationStart = std::chrono::steady_clock::now();
        const Options options = parse(argc, argv);
        if (!options.inspect.empty())
        {
            inspectModel(options.inspect, std::cout);
            if (!options.engine.empty())
            {
                const fs::path enginePath(options.engine);
                if (!fs::is_regular_file(enginePath))
                    throw std::runtime_error("Engine file not found: " + options.engine);
                std::cout << "Engine file: " << options.engine << " (" << fs::file_size(enginePath)
                          << " bytes)\n";
                std::cout << "Engine contents: not inspected without CUDA deserialization\n";
            }
            return 0;
        }

        Logger logger;
        logger.verbose = options.verbose;
        logger.quiet = options.quiet;

        if (options.buildEngine)
        {
            if (fs::exists(options.engine) && !options.overwriteEngine)
                throw std::runtime_error("Engine already exists: " + options.engine +
                                         " (use --overwrite-engine to replace)");
            if (static_cast<int64_t>(options.tile) * options.scale > INT_MAX ||
                static_cast<int64_t>(options.overlap) * 2 >= options.tile)
                throw std::runtime_error("Invalid engine tile, scale, or overlap combination");
            TensorElements(options.tile, options.batch, options.scale);
            const ModelProfile profile = loadModelProfile(options.profile);
            Engine engine(options, options.tile, options.batch, logger, profile);
            return 0;
        }

        cv::Mat image;
        if (options.stream)
        {
            const uint64_t width = static_cast<uint64_t>(options.width) * options.scale;
            const uint64_t height = static_cast<uint64_t>(options.height) * options.scale;
            if (width > INT_MAX || height > INT_MAX)
                throw std::runtime_error("Upscaled frame dimensions overflow int");
            const uint64_t inputBytes = static_cast<uint64_t>(options.width) * options.height * 3;
            const uint64_t outputBytes = width * height * 3;
            const auto limit = static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max());
            if (inputBytes > limit || outputBytes > limit ||
                inputBytes > std::numeric_limits<size_t>::max() ||
                outputBytes > std::numeric_limits<size_t>::max())
                throw std::runtime_error("Frame byte count is too large");
            image.create(options.height, options.width, CV_8UC3);
        }
        else
        {
            image = cv::imread(options.input, cv::IMREAD_COLOR);
        }
        if (image.empty())
            throw std::runtime_error("Cannot load or allocate input image");

        size_t freeBytes = 0;
        size_t totalBytes = 0;
        cudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo");

        const int tile =
            options.tile > 0 ? options.tile : autoTile(freeBytes, options.streams, options.overlap);
        const int batch = options.batch > 0
                              ? options.batch
                              : autoBatch(freeBytes, tile, options.scale, options.streams, options.workspace);

        if (!options.quiet)
        {
            std::ostream& messages = options.stream ? std::cerr : std::cout;
            messages << "Tile " << tile << ", batch " << batch << ", streams " << options.streams << '\n';
        }

        if (tile <= 0 || batch <= 0 || static_cast<int64_t>(options.overlap) * 2 >= tile)
        {
            throw std::runtime_error("Invalid tile/overlap/batch combination");
        }

        if (static_cast<int64_t>(image.cols) * options.scale > INT_MAX ||
            static_cast<int64_t>(image.rows) * options.scale > INT_MAX ||
            static_cast<int64_t>(tile) * options.scale > INT_MAX)
        {
            throw std::runtime_error("Upscaled dimensions overflow int");
        }

        const ModelProfile profile = loadModelProfile(options.profile);
        const auto engineStart = std::chrono::steady_clock::now();
        Engine engine(options, tile, batch, logger, profile);
        const auto engineEnd = std::chrono::steady_clock::now();
        const auto tensors = TensorElements(tile, batch, options.scale);
        const size_t inputCount = tensors.input;
        const size_t outputCount = tensors.output;

        std::vector<std::unique_ptr<Worker>> workers;
        std::vector<Inference*> inference;
        workers.reserve(options.streams);
        inference.reserve(options.streams);

        for (int i = 0; i < options.streams; ++i)
        {
            workers.emplace_back(
                std::make_unique<Worker>(engine.Get(), inputCount, outputCount, engine.InputType(),
                                         engine.OutputType(), engine.InputName(), engine.OutputName(), tile,
                                         batch, options.scale, options.overlap, profile, options.benchmark));
            inference.push_back(workers.back().get());
        }

        TilePipeline pipeline(tile, batch, options.scale, options.overlap, inference);
        if (options.stream)
        {
            const auto mode = !options.pipeName.empty() ? LocalTransport::Mode::Pipe :
                              !options.sharedMemoryName.empty() ? LocalTransport::Mode::SharedMemory :
                                                                  LocalTransport::Mode::Stdio;
            const std::string& name = !options.pipeName.empty() ? options.pipeName : options.sharedMemoryName;
            LocalTransport transport(mode, name, options.width, options.height, options.scale);
            const auto inputBytes = image.total() * image.elemSize();
            const auto outputBytes = static_cast<size_t>(options.width * options.scale) *
                                     static_cast<size_t>(options.height * options.scale) * 3;
            try
            {
                while (transport.ReadFrame(image.data, inputBytes))
                {
                    cv::Mat output;
                    pipeline.Run(image).convertTo(output, CV_8UC3, 255.0);
                    if (!output.isContinuous() || output.total() * output.elemSize() != outputBytes)
                        throw std::runtime_error("Invalid BGR24 output frame");
                    transport.WriteFrame(output.data, outputBytes);
                }
            }
            catch (...)
            {
                transport.Fail();
                throw;
            }
            return 0;
        }
        bool debugSaved = false;

        const auto upscaleStart = std::chrono::steady_clock::now();
        const cv::Mat result = pipeline.Run(image,
                                            [&](const BatchResult& completed, size_t processed, size_t total)
                                            {
                                                if (options.debug && !debugSaved && !completed.images.empty())
                                                {
                                                    cv::Mat debug;
                                                    completed.images.front().convertTo(debug, CV_8UC3, 255.0);
                                                    if (!cv::imwrite("debug_first_tile.png", debug))
                                                    {
                                                        throw std::runtime_error("Cannot save debug tile");
                                                    }
                                                    debugSaved = true;
                                                }

                                                if (!options.quiet)
                                                {
                                                    std::cout << "Processed " << processed << "/" << total
                                                              << " tiles\r" << std::flush;
                                                }
                                            });
        const auto upscaleEnd = std::chrono::steady_clock::now();

        cv::Mat output;
        result.convertTo(output, CV_8UC3, 255.0);
        const fs::path path(options.output);
        if (path.has_parent_path())
        {
            fs::create_directories(path.parent_path());
        }
        if (!cv::imwrite(options.output, output))
        {
            throw std::runtime_error("Could not write output image");
        }
        if (!options.quiet)
        {
            std::cout << "\nWrote " << options.output << '\n';
        }
        if (options.benchmark)
        {
            const auto totalEnd = std::chrono::steady_clock::now();
            const auto elapsed = [](auto start, auto end) -> double
            { return std::chrono::duration<double, std::micro>(end - start).count(); };
            double inferenceUs = 0.0;
            size_t inferenceBatches = 0;
            for (const auto& worker : workers)
            {
                inferenceUs += worker->InferenceMicroseconds();
                inferenceBatches += worker->InferenceBatches();
            }
            int device = 0;
            cudaCheck(cudaGetDevice(&device), "cudaGetDevice benchmark");
            cudaDeviceProp gpu{};
            cudaCheck(cudaGetDeviceProperties(&gpu, device), "cudaGetDeviceProperties benchmark");
            int driver = 0;
            cudaCheck(cudaDriverGetVersion(&driver), "cudaDriverGetVersion benchmark");
            size_t freeAfter = 0;
            size_t totalAfter = 0;
            cudaCheck(cudaMemGetInfo(&freeAfter, &totalAfter), "cudaMemGetInfo benchmark");
            const double upscaleUs = elapsed(upscaleStart, upscaleEnd);
            const double totalUs = elapsed(applicationStart, totalEnd);
            std::cout << std::fixed << std::setprecision(2) << "\nBenchmark (microseconds):\n"
                      << "  Inference CUDA event sum: " << inferenceUs << " (" << inferenceBatches
                      << " batches across " << options.streams << " streams)\n"
                      << "  Upscaling wall time: " << upscaleUs << "\n"
                      << "  Engine initialization: " << elapsed(engineStart, engineEnd) << "\n"
                      << "  Total application: " << totalUs << "\n"
                      << "  Output pixels/second (upscaling): "
                      << (upscaleUs > 0 ? static_cast<double>(output.total()) * 1000000.0 / upscaleUs : 0.0)
                      << "\n  GPU: " << gpu.name << " (compute " << gpu.major << '.' << gpu.minor
                      << ", CUDA driver " << driver << ")\n"
                      << "  TensorRT: " << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.'
                      << NV_TENSORRT_PATCH << "\n"
                      << "  Model: " << options.model << "\n"
                      << "  Input: " << options.input << " (" << image.cols << 'x' << image.rows << ")\n"
                      << "  Tile/batch/streams/overlap: " << tile << '/' << batch << '/' << options.streams
                      << '/' << options.overlap << "\n"
                      << "  Device memory used before/after (bytes): " << (totalBytes - freeBytes) << '/'
                      << (totalAfter - freeAfter) << " (not peak VRAM)\n";
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
