// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Engine.hpp"
#include "Options.hpp"
#include "Sizing.hpp"
#include "TensorSizes.hpp"
#include "TilePipeline.hpp"
#include "Upscaler.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <cstdint>
#include <climits>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    try
    {
        const Options options = parse(argc, argv);
        Logger logger;
        logger.verbose = options.verbose;

        const cv::Mat image = cv::imread(options.input, cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw std::runtime_error("Cannot load input image");
        }

        size_t freeBytes = 0;
        size_t totalBytes = 0;
        cudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo");

        const int tile = options.tile > 0 ? options.tile
            : autoTile(freeBytes, options.streams, options.overlap);
        const int batch = options.batch > 0 ? options.batch
            : autoBatch(freeBytes, tile, options.scale, options.streams, options.workspace);

        if (!options.quiet)
        {
            std::cout << "Tile " << tile << ", batch " << batch
                      << ", streams " << options.streams << '\n';
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

        Engine engine(options, tile, batch, logger);
        const auto tensors = TensorElements(tile, batch, options.scale);
        const size_t inputCount = tensors.input;
        const size_t outputCount = tensors.output;

        std::vector<std::unique_ptr<Worker>> workers;
        std::vector<Inference*> inference;
        workers.reserve(options.streams);
        inference.reserve(options.streams);

        for (int i = 0; i < options.streams; ++i)
        {
            workers.emplace_back(std::make_unique<Worker>(
                engine.Get(), inputCount, outputCount,
                engine.InputType(), engine.OutputType(),
                engine.InputName(), engine.OutputName(),
                tile, batch, options.scale, options.overlap));
            inference.push_back(workers.back().get());
        }

        TilePipeline pipeline(tile, batch, options.scale, options.overlap, inference);
        bool debugSaved = false;

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
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
