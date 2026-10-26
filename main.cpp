// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Options.hpp"
#include "Engine.hpp"
#include "Upscaler.hpp"
#include "ImageTiles.hpp"
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <climits>
#include <deque>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    try {
        Options o = parse(argc, argv);
        Logger logger; logger.verbose = o.verbose;
        cv::Mat image = cv::imread(o.input, cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Cannot load input image");
        size_t freeBytes = 0, totalBytes = 0;
        cudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo");
        const int tile = o.tile > 0 ? o.tile : autoTile(freeBytes, o.streams, o.overlap);
        const int batch = o.batch > 0 ? o.batch : autoBatch(freeBytes, tile, o.scale, o.streams, o.workspace);
        if (tile <= 2 * o.overlap || tile <= 0 || batch <= 0) throw std::runtime_error("Invalid tile/overlap/batch combination");
        if (static_cast<int64_t>(image.cols) * o.scale > INT_MAX ||
            static_cast<int64_t>(image.rows) * o.scale > INT_MAX ||
            static_cast<int64_t>(tile) * o.scale > INT_MAX)
            throw std::runtime_error("Upscaled dimensions overflow int");
        if (!o.quiet) std::cout << "Tile " << tile << ", batch " << batch << ", streams " << o.streams << "\n";
        Engine engine(o, tile, batch, logger);
        const auto& inName = engine.InputName();
        const auto& outName = engine.OutputName();
        const auto inType = engine.InputType();
        const auto outType = engine.OutputType();
        const size_t inCount = static_cast<size_t>(batch) * 3 * tile * tile;
        const size_t outCount = inCount * static_cast<size_t>(o.scale) * o.scale;
        std::vector<std::unique_ptr<Worker>> workers;
        for (int i = 0; i < o.streams; ++i)
            workers.emplace_back(std::make_unique<Worker>(engine.Get(), inCount, outCount, inType, outType));
        const int width = image.cols * o.scale, height = image.rows * o.scale;
        cv::Mat accum(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
        cv::Mat weights(height, width, CV_32FC1, cv::Scalar(0));
        const int stride = tile - 2 * o.overlap;
        std::vector<Tile> all;
        for (int y = 0; y < image.rows; y += stride)
            for (int x = 0; x < image.cols; x += stride) all.push_back({x, y});
        std::deque<std::future<BatchResult>> jobs;
        bool debugSaved = false;
        size_t done = 0;
        auto consume = [&](std::future<BatchResult>& f) {
            BatchResult result = f.get();
            for (size_t j = 0; j < result.tiles.size(); ++j) {
                if (o.debug && !debugSaved) {
                    cv::Mat tmp; result.images[j].convertTo(tmp, CV_8UC3, 255.0);
                    if (!cv::imwrite("debug_first_tile.png", tmp)) throw std::runtime_error("Cannot save debug tile");
                    debugSaved = true;
                }
                blendTile(accum, weights, result.images[j], result.tiles[j], tile, o.scale,
                          o.overlap, image.cols, image.rows);
                ++done;
            }
            if (!o.quiet) std::cout << "Processed " << done << "/" << all.size() << " tiles\r" << std::flush;
        };
        // Each worker owns an independent TensorRT execution context, CUDA stream,
        // persistent device buffers and page-locked staging buffers.
        size_t group = 0;
        for (size_t start = 0; start < all.size(); start += static_cast<size_t>(batch), ++group) {
            const size_t end = std::min(start + static_cast<size_t>(batch), all.size());
            if (jobs.size() == workers.size()) { consume(jobs.front()); jobs.pop_front(); }
            std::vector<Tile> tiles(all.begin() + start, all.begin() + end);
            Worker* worker = workers[group % workers.size()].get();
            jobs.push_back(std::async(std::launch::async, [worker, &image, tiles = std::move(tiles), tile, batch, &o, &inName, &outName, inType, outType]() {
                return runBatch(*worker, image, tiles, tile, batch, o.scale, o.overlap, inName, outName, inType, outType);
            }));
        }
        while (!jobs.empty()) { consume(jobs.front()); jobs.pop_front(); }
        for (int y = 0; y < height; ++y) {
            auto* row = accum.ptr<cv::Vec3f>(y);
            const auto* w = weights.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                if (w[x] <= 0.f) throw std::runtime_error("Uncovered output pixel during tile blending");
                row[x] /= w[x];
            }
        }
        cv::Mat output; accum.convertTo(output, CV_8UC3, 255.0);
        if (fs::path(o.output).has_parent_path()) fs::create_directories(fs::path(o.output).parent_path());
        if (!cv::imwrite(o.output, output)) throw std::runtime_error("Could not write output image");
        if (!o.quiet) std::cout << "\nWrote " << o.output << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
