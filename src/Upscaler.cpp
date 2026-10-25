// Copyright (c) 2025 Elden Armbrust. MIT License.
#include "Upscaler.hpp"
#include <algorithm>
#include <cstring>
#include <vector>
BatchResult runBatch(Worker& w, const cv::Mat& image, const std::vector<Tile>& tiles,
                            int tile, int batch, int scale, int overlap,
                            const std::string& inName, const std::string& outName,
                            nvinfer1::DataType inType, nvinfer1::DataType outType) {
    const size_t inPlane = static_cast<size_t>(tile) * tile;
    const size_t inPer = 3 * inPlane;
    const int outTile = tile * scale;
    const size_t outPlane = static_cast<size_t>(outTile) * outTile;
    const size_t outPer = 3 * outPlane;
    // A fixed-shape engine always receives the complete configured batch; pad the final batch.
    // Image preprocessing stays FP32; pack into the exact input tensor datatype.
    // FP16 conversion is performed on the CPU once per tile, avoiding a GPU conversion kernel.
    std::vector<float> prepared(inPer * batch, 0.f);
    for (size_t i = 0; i < tiles.size(); ++i)
        fillInput(image, tiles[i], tile, overlap, prepared.data() + i * inPer);
    if (inType == nvinfer1::DataType::kFLOAT) {
        std::memcpy(w.hostIn.ptr, prepared.data(), prepared.size() * sizeof(float));
    } else {
        auto* packed = static_cast<__half*>(w.hostIn.ptr);
        for (size_t i = 0; i < prepared.size(); ++i) packed[i] = __float2half(prepared[i]);
    }
    if (!w.ctx->setInputShape(inName.c_str(), nvinfer1::Dims4(batch, 3, tile, tile)))
        throw std::runtime_error("TensorRT setInputShape failed");
    auto dims = w.ctx->getTensorShape(outName.c_str());
    if (dims.nbDims != 4 || dims.d[0] != batch || dims.d[1] != 3 ||
        dims.d[2] != outTile || dims.d[3] != outTile)
        throw std::runtime_error("Model output shape does not match requested scale");
    cudaCheck(cudaMemcpyAsync(w.input.p, w.hostIn.ptr, inPer * batch * tensorElementBytes(inType),
                              cudaMemcpyHostToDevice, w.stream.value), "H2D batch");
    if (!w.ctx->setTensorAddress(inName.c_str(), w.input.p) ||
        !w.ctx->setTensorAddress(outName.c_str(), w.output.p))
        throw std::runtime_error("Cannot bind TensorRT tensor addresses");
    if (!w.ctx->enqueueV3(w.stream.value)) throw std::runtime_error("TensorRT enqueueV3 failed");
    cudaCheck(cudaMemcpyAsync(w.hostOut.ptr, w.output.p, outPer * batch * tensorElementBytes(outType),
                              cudaMemcpyDeviceToHost, w.stream.value), "D2H batch");
    cudaCheck(cudaStreamSynchronize(w.stream.value), "Batch stream synchronization");
    BatchResult result; result.tiles = tiles;
    for (size_t i = 0; i < tiles.size(); ++i) {
        cv::Mat sr(outTile, outTile, CV_32FC3);
        const float* data32 = outType == nvinfer1::DataType::kFLOAT
            ? static_cast<const float*>(w.hostOut.ptr) + i * outPer : nullptr;
        const __half* data16 = outType == nvinfer1::DataType::kHALF
            ? static_cast<const __half*>(w.hostOut.ptr) + i * outPer : nullptr;
        auto sample = [&](size_t index) -> float {
            return data32 ? data32[index] : __half2float(data16[index]);
        };
        for (int y = 0; y < outTile; ++y) {
            auto* row = sr.ptr<cv::Vec3f>(y);
            for (int x = 0; x < outTile; ++x) {
                const size_t k = static_cast<size_t>(y) * outTile + x;
                row[x] = cv::Vec3f(sample(2 * outPlane + k), sample(outPlane + k), sample(k));
            }
        }
        result.images.push_back(std::move(sr));
    }
    return result;
}
int autoTile(size_t freeBytes, int streams, int overlap) {
    // Conservative choices leave memory for TensorRT weights and workspace.
    const size_t perStream = freeBytes / static_cast<size_t>(std::max(1, streams));
    const int candidate = perStream >= (size_t(2) << 30) ? 512 : perStream >= (size_t(768) << 20) ? 256 : 128;
    return std::max(candidate, overlap * 2 + 32);
}
int autoBatch(size_t freeBytes, int tile, int scale, int streams, double workspaceGB) {
    // Activation memory is model-dependent; reserve most VRAM for TensorRT's internals.
    const size_t reserve = static_cast<size_t>(workspaceGB * (1ull << 30));
    const size_t available = freeBytes > reserve ? freeBytes - reserve : freeBytes / 4;
    const double floatsPerTile = 3.0 * tile * tile * (1.0 + scale * scale);
    const double conservative = floatsPerTile * sizeof(float) * 8.0 * std::max(1, streams);
    return std::clamp(static_cast<int>(available / std::max(1.0, conservative)), 1, 8);
}
