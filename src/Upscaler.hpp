// Copyright (c) 2025 Elden Armbrust. MIT License.
#pragma once
#include "Resources.hpp"
#include "ImageTiles.hpp"
#include <cuda_fp16.h>
#include <cstddef>
#include <stdexcept>
#include <string>
inline size_t tensorElementBytes(nvinfer1::DataType type) {
    if (type == nvinfer1::DataType::kHALF) return sizeof(__half);
    if (type == nvinfer1::DataType::kFLOAT) return sizeof(float);
    throw std::runtime_error("Unsupported tensor datatype (expected FP16 or FP32)");
}
struct PinnedBuffer {
    void* ptr = nullptr;
    explicit PinnedBuffer(size_t bytes) { cudaCheck(cudaMallocHost(&ptr, bytes), "cudaMallocHost"); }
    ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};
struct Worker {
    Stream stream;
    TrtPtr<nvinfer1::IExecutionContext> ctx;
    DeviceBuffer input, output;
    PinnedBuffer hostIn, hostOut;
    Worker(nvinfer1::ICudaEngine& engine, size_t inCount, size_t outCount,
           nvinfer1::DataType inType, nvinfer1::DataType outType)
        : ctx(engine.createExecutionContext()), input(inCount * tensorElementBytes(inType)),
          output(outCount * tensorElementBytes(outType)),
          hostIn(inCount * tensorElementBytes(inType)),
          hostOut(outCount * tensorElementBytes(outType)) {
        if (!ctx) throw std::runtime_error("Cannot create TensorRT execution context");
    }
};
BatchResult runBatch(Worker& w, const cv::Mat& image, const std::vector<Tile>& tiles,
                     int tile, int batch, int scale, int overlap,
                     const std::string& inName, const std::string& outName,
                     nvinfer1::DataType inType, nvinfer1::DataType outType);
int autoTile(size_t freeBytes, int streams, int overlap);
int autoBatch(size_t freeBytes, int tile, int scale, int streams, double workspaceGB);
