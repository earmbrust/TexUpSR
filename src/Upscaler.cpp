// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Upscaler.hpp"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

PinnedBuffer::PinnedBuffer(size_t bytes)
{
    cudaCheck(cudaMallocHost(&data_, bytes), "cudaMallocHost");
}

PinnedBuffer::~PinnedBuffer()
{
    if (data_)
    {
        cudaFreeHost(data_);
    }
}

void* PinnedBuffer::Data() const
{
    return data_;
}

Worker::Worker(nvinfer1::ICudaEngine& engine, size_t inCount, size_t outCount,
               nvinfer1::DataType inType, nvinfer1::DataType outType,
               std::string inputName, std::string outputName)
    : context_(engine.createExecutionContext()),
      input_(inCount * tensorElementBytes(inType)),
      output_(outCount * tensorElementBytes(outType)),
      hostInput_(inCount * tensorElementBytes(inType)),
      hostOutput_(outCount * tensorElementBytes(outType)),
      inputName_(std::move(inputName)),
      outputName_(std::move(outputName)),
      inputType_(inType),
      outputType_(outType)
{
    if (!context_)
    {
        throw std::runtime_error("Cannot create TensorRT execution context");
    }
}

BatchResult Worker::Run(const cv::Mat& image, const std::vector<Tile>& tiles,
                        int tile, int batch, int scale, int overlap)
{
    const size_t inPlane = static_cast<size_t>(tile) * tile;
    const size_t inPer = 3 * inPlane;
    const int outTile = tile * scale;
    const size_t outPlane = static_cast<size_t>(outTile) * outTile;
    const size_t outPer = 3 * outPlane;

    // The final batch is padded because the TensorRT engine uses a fixed batch shape.
    std::vector<float> prepared(inPer * batch, 0.f);

    for (size_t i = 0; i < tiles.size(); ++i)
    {
        fillInput(image, tiles[i], tile, overlap, prepared.data() + i * inPer);
    }

    if (inputType_ == nvinfer1::DataType::kFLOAT)
    {
        std::memcpy(hostInput_.Data(), prepared.data(), prepared.size() * sizeof(float));
    }
    else
    {
        auto* packed = static_cast<__half*>(hostInput_.Data());

        for (size_t i = 0; i < prepared.size(); ++i)
        {
            packed[i] = __float2half(prepared[i]);
        }
    }

    if (!context_->setInputShape(inputName_.c_str(), nvinfer1::Dims4(batch, 3, tile, tile)))
    {
        throw std::runtime_error("TensorRT setInputShape failed");
    }

    const auto dims = context_->getTensorShape(outputName_.c_str());

    if (dims.nbDims != 4 || dims.d[0] != batch || dims.d[1] != 3 ||
        dims.d[2] != outTile || dims.d[3] != outTile)
    {
        throw std::runtime_error("Model output shape does not match requested scale");
    }

    cudaCheck(cudaMemcpyAsync(input_.p, hostInput_.Data(), inPer * batch * tensorElementBytes(inputType_),
                              cudaMemcpyHostToDevice, stream_.value), "H2D batch");

    if (!context_->setTensorAddress(inputName_.c_str(), input_.p) ||
        !context_->setTensorAddress(outputName_.c_str(), output_.p))
    {
        throw std::runtime_error("Cannot bind TensorRT tensor addresses");
    }

    if (!context_->enqueueV3(stream_.value))
    {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }

    cudaCheck(cudaMemcpyAsync(hostOutput_.Data(), output_.p, outPer * batch * tensorElementBytes(outputType_),
                              cudaMemcpyDeviceToHost, stream_.value), "D2H batch");
    cudaCheck(cudaStreamSynchronize(stream_.value), "Batch stream synchronization");

    BatchResult result;
    result.tiles = tiles;

    for (size_t i = 0; i < tiles.size(); ++i)
    {
        cv::Mat sr(outTile, outTile, CV_32FC3);
        const float* data32 = outputType_ == nvinfer1::DataType::kFLOAT
            ? static_cast<const float*>(hostOutput_.Data()) + i * outPer : nullptr;
        const __half* data16 = outputType_ == nvinfer1::DataType::kHALF
            ? static_cast<const __half*>(hostOutput_.Data()) + i * outPer : nullptr;

        auto sample = [&](size_t index) -> float
        {
            return data32 ? data32[index] : __half2float(data16[index]);
        };

        for (int y = 0; y < outTile; ++y)
        {
            auto* row = sr.ptr<cv::Vec3f>(y);

            for (int x = 0; x < outTile; ++x)
            {
                const size_t k = static_cast<size_t>(y) * outTile + x;
                row[x] = cv::Vec3f(sample(2 * outPlane + k), sample(outPlane + k), sample(k));
            }
        }

        result.images.push_back(std::move(sr));
    }

    return result;
}

int autoTile(size_t freeBytes, int streams, int overlap)
{
    // Conservative choices leave memory for TensorRT weights and workspace.
    const size_t perStream = freeBytes / static_cast<size_t>(std::max(1, streams));
    const int candidate = perStream >= (size_t(2) << 30) ? 512 : perStream >= (size_t(768) << 20) ? 256 : 128;
    return std::max(candidate, overlap * 2 + 32);
}

int autoBatch(size_t freeBytes, int tile, int scale, int streams, double workspaceGB)
{
    // Activation memory is model-dependent; reserve most VRAM for TensorRT's internals.
    const size_t reserve = static_cast<size_t>(workspaceGB * (1ull << 30));
    const size_t available = freeBytes > reserve ? freeBytes - reserve : freeBytes / 4;
    const double floatsPerTile = 3.0 * tile * tile * (1.0 + scale * scale);
    const double conservative = floatsPerTile * sizeof(float) * 8.0 * std::max(1, streams);
    return std::clamp(static_cast<int>(available / std::max(1.0, conservative)), 1, 8);
}
