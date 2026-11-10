// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Fixed BGR24 local IPC contract, version 1. Shared mappings use native-endian
// integers; all supported release platforms use little-endian integers.
struct SharedFrameHeader
{
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t scale;
    std::uint32_t format;
    std::uint64_t inputBytes;
    std::uint64_t outputBytes;
    std::uint32_t state;
    std::uint32_t reserved;
};
static_assert(sizeof(SharedFrameHeader) == 48, "Unexpected shared frame header layout");

class LocalTransport final
{
public:
    enum class Mode { Stdio, Pipe, SharedMemory };

    LocalTransport(Mode mode, const std::string& name, int width, int height, int scale);
    ~LocalTransport();

    LocalTransport(const LocalTransport&) = delete;
    LocalTransport& operator=(const LocalTransport&) = delete;

    bool ReadFrame(unsigned char* pixels, std::size_t bytes);
    void WriteFrame(const unsigned char* pixels, std::size_t bytes);
    void Fail() noexcept;

private:
    void Close() noexcept;
    void OpenPipe(const std::string& name);
    void OpenSharedMemory(const std::string& name, int width, int height, int scale);

    Mode mode_;
    int width_, height_, scale_;
    std::size_t inputBytes_ = 0;
    std::size_t outputBytes_ = 0;
    void* memory_ = nullptr;
    std::size_t mappingBytes_ = 0;
    void* mapping_ = nullptr;
    void* request_ = nullptr;
    void* response_ = nullptr;
    std::string inputName_, outputName_, mappingName_, requestName_, responseName_;
    bool inputOwned_ = false, outputOwned_ = false;
    bool mappingOwned_ = false, requestOwned_ = false, responseOwned_ = false;
#ifdef _WIN32
    void* inputPipe_ = nullptr;
    void* outputPipe_ = nullptr;
#else
    int inputPipe_ = -1;
    int outputPipe_ = -1;
    int mappingFd_ = -1;
#endif
};
