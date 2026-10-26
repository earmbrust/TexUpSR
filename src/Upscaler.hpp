// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once

#include "Inference.hpp"
#include "Resources.hpp"

#include <cuda_fp16.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

inline size_t tensorElementBytes(nvinfer1::DataType type)
{
    if (type == nvinfer1::DataType::kHALF)
    {
        return sizeof(__half);
    }

    if (type == nvinfer1::DataType::kFLOAT)
    {
        return sizeof(float);
    }

    throw std::runtime_error("Unsupported tensor datatype (expected FP16 or FP32)");
}

class PinnedBuffer
{
public:
    explicit PinnedBuffer(size_t bytes);
    ~PinnedBuffer();

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void* Data() const;

private:
    void* data_ = nullptr;
};

class Worker final : public Inference
{
public:
    Worker(nvinfer1::ICudaEngine& engine, size_t inCount, size_t outCount,
           nvinfer1::DataType inType, nvinfer1::DataType outType,
           std::string inputName, std::string outputName,
           int tile, int batch, int scale, int overlap);

    BatchResult Run(const cv::Mat& image, const std::vector<Tile>& tiles) override;

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

private:
    Stream stream_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    DeviceBuffer input_;
    DeviceBuffer output_;
    PinnedBuffer hostInput_;
    PinnedBuffer hostOutput_;
    std::string inputName_;
    std::string outputName_;
    nvinfer1::DataType inputType_;
    nvinfer1::DataType outputType_;
    int tile_;
    int batch_;
    int scale_;
    int overlap_;
};

int autoTile(size_t freeBytes, int streams, int overlap);
int autoBatch(size_t freeBytes, int tile, int scale, int streams, double workspaceGB);
