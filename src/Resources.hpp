// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
struct Logger final : nvinfer1::ILogger
{
    bool verbose = false;
    void log(Severity s, const char* msg) noexcept override
    {
        if (s <= (verbose ? Severity::kINFO : Severity::kWARNING))
            std::cerr << "TensorRT: " << msg << '\n';
    }
};
inline void cudaCheck(cudaError_t result, const char* where)
{
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(result));
}
template <typename T> struct TrtDeleter
{
    void operator()(T* p) const
    {
        delete p;
    }
};
template <typename T> using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;
struct DeviceBuffer
{
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes)
    {
        cudaCheck(cudaMalloc(&p, bytes), "cudaMalloc");
    }
    ~DeviceBuffer()
    {
        if (p)
            cudaFree(p);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Stream
{
    cudaStream_t value = nullptr;
    Stream()
    {
        cudaCheck(cudaStreamCreate(&value), "cudaStreamCreate");
    }
    ~Stream()
    {
        if (value)
            cudaStreamDestroy(value);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
