// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Upscaler.hpp"
#include "TensorSizes.hpp"

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

Worker::Worker(nvinfer1::ICudaEngine& engine, size_t inCount, size_t outCount, nvinfer1::DataType inType,
               nvinfer1::DataType outType, std::string inputName, std::string outputName, int tile, int batch,
               int scale, int overlap, const ModelProfile& profile)
    : context_(engine.createExecutionContext()), input_(TensorBytes(inCount, tensorElementBytes(inType))),
      output_(TensorBytes(outCount, tensorElementBytes(outType))),
      hostInput_(TensorBytes(inCount, tensorElementBytes(inType))),
      hostOutput_(TensorBytes(outCount, tensorElementBytes(outType))), inputName_(std::move(inputName)),
      outputName_(std::move(outputName)), inputType_(inType), outputType_(outType), tile_(tile),
      batch_(batch), scale_(scale), overlap_(overlap), profile_(profile)
{
    if (!context_)
    {
        throw std::runtime_error("Cannot create TensorRT execution context");
    }
}

Worker::~Worker()
{
    cudaStreamSynchronize(stream_.value);
}

BatchResult Worker::Run(const cv::Mat& image, const std::vector<Tile>& tiles)
{
    const int tile = tile_;
    const int batch = batch_;
    const int scale = scale_;
    const int overlap = overlap_;
    const size_t inPlane = static_cast<size_t>(tile) * tile;
    const size_t inPer = 3 * inPlane;
    const int outTile = tile * scale;
    const size_t outPlane = static_cast<size_t>(outTile) * outTile;
    const size_t outPer = 3 * outPlane;

    // The final batch is padded because the TensorRT engine uses a fixed batch shape.
    std::vector<float> prepared(inPer * batch, 0.f);

    for (size_t i = 0; i < tiles.size(); ++i)
    {
        fillInput(image, tiles[i], tile, overlap, prepared.data() + i * inPer, profile_);
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

    const auto inShape = modelShape(batch, tile, profile_.layout);
    if (!context_->setInputShape(inputName_.c_str(),
                                 nvinfer1::Dims4(inShape[0], inShape[1], inShape[2], inShape[3])))
    {
        throw std::runtime_error("TensorRT setInputShape failed");
    }

    const auto dims = context_->getTensorShape(outputName_.c_str());

    const auto outShape = modelShape(batch, outTile, profile_.layout);
    if (dims.nbDims != 4 || dims.d[0] != outShape[0] || dims.d[1] != outShape[1] ||
        dims.d[2] != outShape[2] || dims.d[3] != outShape[3])
    {
        throw std::runtime_error("Model output shape does not match requested scale");
    }

    try
    {
        cudaCheck(cudaMemcpyAsync(input_.p, hostInput_.Data(), inPer * batch * tensorElementBytes(inputType_),
                                  cudaMemcpyHostToDevice, stream_.value),
                  "H2D batch");

        if (!context_->setTensorAddress(inputName_.c_str(), input_.p) ||
            !context_->setTensorAddress(outputName_.c_str(), output_.p))
        {
            throw std::runtime_error("Cannot bind TensorRT tensor addresses");
        }

        if (!context_->enqueueV3(stream_.value))
        {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }

        cudaCheck(cudaMemcpyAsync(hostOutput_.Data(), output_.p,
                                  outPer * batch * tensorElementBytes(outputType_), cudaMemcpyDeviceToHost,
                                  stream_.value),
                  "D2H batch");
        cudaCheck(cudaStreamSynchronize(stream_.value), "Batch stream synchronization");
    }
    catch (...)
    {
        cudaStreamSynchronize(stream_.value);
        throw;
    }

    BatchResult result;
    result.tiles = tiles;

    for (size_t i = 0; i < tiles.size(); ++i)
    {

        const auto sample = [&](size_t index) -> float
        {
            const size_t offset = i * outPer + index;
            if (outputType_ == nvinfer1::DataType::kFLOAT)
            {
                return static_cast<const float*>(hostOutput_.Data())[offset];
            }
            return __half2float(static_cast<const __half*>(hostOutput_.Data())[offset]);
        };

        result.images.push_back(tensorToImage(outTile, sample, profile_));
    }

    return result;
}
