// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "LocalTransport.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace
{
constexpr std::uint32_t Magic = 0x52535554; // "TUSR" in little-endian byte order.
constexpr std::uint32_t Version = 1;
constexpr std::uint32_t Bgr24 = 1;
constexpr std::uint32_t Request = 1;
constexpr std::uint32_t Response = 2;
constexpr std::uint32_t Stop = 3;
constexpr std::uint32_t Error = 4;

void requireName(const std::string& name)
{
    if (name.empty() || name.size() > 64)
        throw std::invalid_argument("Local transport name must be 1-64 characters");
    for (const unsigned char c : name)
    {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '-' || c == '_'))
            throw std::invalid_argument("Local transport name may contain only letters, digits, '-' and '_'");
    }
}

std::size_t frameBytes(int width, int height)
{
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("Frame dimensions must be positive");
    const auto bytes = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * 3;
    if (bytes > std::numeric_limits<std::size_t>::max() ||
        bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::overflow_error("Frame size exceeds supported limits");
    return static_cast<std::size_t>(bytes);
}

#ifdef _WIN32
HANDLE asHandle(void* handle) { return static_cast<HANDLE>(handle); }
void connectPipe(HANDLE handle)
{
    if (!ConnectNamedPipe(handle, nullptr) && GetLastError() != ERROR_PIPE_CONNECTED)
        throw std::runtime_error("Cannot connect local named pipe");
}

bool readPart(HANDLE handle, unsigned char* data, std::size_t& count, std::size_t remaining)
{
    DWORD received = 0;
    const auto length = static_cast<DWORD>(std::min<std::size_t>(remaining, 1024 * 1024));
    if (!ReadFile(handle, data, length, &received, nullptr))
    {
        if (GetLastError() == ERROR_BROKEN_PIPE)
            return false;
        throw std::runtime_error("Cannot read local named pipe");
    }
    count = received;
    return received != 0;
}

void writePart(HANDLE handle, const unsigned char* data, std::size_t remaining, std::size_t& written)
{
    DWORD count = 0;
    const auto length = static_cast<DWORD>(std::min<std::size_t>(remaining, 1024 * 1024));
    if (!WriteFile(handle, data, length, &count, nullptr) || !count)
        throw std::runtime_error("Cannot write local named pipe");
    written = count;
}
#else
bool readPart(int fd, unsigned char* data, std::size_t& count, std::size_t remaining)
{
    ssize_t received = -1;
    do
    {
        received = read(fd, data, std::min<std::size_t>(remaining, 1024 * 1024));
    } while (received == -1 && errno == EINTR);
    if (received < 0)
        throw std::runtime_error("Cannot read local FIFO");
    count = static_cast<std::size_t>(received);
    return received != 0;
}

void writePart(int fd, const unsigned char* data, std::size_t remaining, std::size_t& written)
{
    ssize_t count = -1;
    do
    {
        count = write(fd, data, std::min<std::size_t>(remaining, 1024 * 1024));
    } while (count == -1 && errno == EINTR);
    if (count <= 0)
        throw std::runtime_error("Cannot write local FIFO");
    written = static_cast<std::size_t>(count);
}

void waitSemaphore(void* semaphore)
{
    int result;
    do
    {
        result = sem_wait(static_cast<sem_t*>(semaphore));
    } while (result == -1 && errno == EINTR);
    if (result != 0)
        throw std::runtime_error("Cannot wait for shared-memory request");
}
#endif
}

LocalTransport::LocalTransport(Mode mode, const std::string& name, int width, int height, int scale)
    : mode_(mode), width_(width), height_(height), scale_(scale)
{
    if (scale <= 0 || static_cast<std::uint64_t>(width) * scale > INT_MAX ||
        static_cast<std::uint64_t>(height) * scale > INT_MAX)
        throw std::invalid_argument("Invalid output frame dimensions");
    inputBytes_ = frameBytes(width, height);
    outputBytes_ = frameBytes(width * scale, height * scale);

    try
    {
        if (mode == Mode::Stdio)
        {
#ifdef _WIN32
            if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
                _setmode(_fileno(stdout), _O_BINARY) == -1)
                throw std::runtime_error("Cannot configure binary stdin/stdout");
#endif
        }
        else
        {
            requireName(name);
            if (mode == Mode::Pipe)
                OpenPipe(name);
            else
                OpenSharedMemory(name, width, height, scale);
        }
    }
    catch (...)
    {
        Close();
        throw;
    }
}

LocalTransport::~LocalTransport() { Close(); }

void LocalTransport::OpenPipe(const std::string& name)
{
#ifdef _WIN32
    inputName_ = "\\\\.\\pipe\\TexUpSR-" + name + "-input";
    outputName_ = "\\\\.\\pipe\\TexUpSR-" + name + "-output";
    const DWORD mode = PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS;
    auto in = CreateNamedPipeA(inputName_.c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_FIRST_PIPE_INSTANCE, mode, 1, 1 << 20, 1 << 20, 0, nullptr);
    if (in == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Cannot create input named pipe");
    inputPipe_ = in;
    auto out = CreateNamedPipeA(outputName_.c_str(), PIPE_ACCESS_OUTBOUND | FILE_FLAG_FIRST_PIPE_INSTANCE, mode, 1, 1 << 20, 1 << 20, 0, nullptr);
    if (out == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Cannot create output named pipe");
    outputPipe_ = out;
    std::cerr << "Local pipes ready: " << inputName_ << " / " << outputName_ << '\n';
    connectPipe(in);
    connectPipe(out);
#else
    inputName_ = "/tmp/texupsr-" + name + "-input";
    outputName_ = "/tmp/texupsr-" + name + "-output";
    if (mkfifo(inputName_.c_str(), 0600) != 0)
        throw std::runtime_error("Cannot create input FIFO");
    inputOwned_ = true;
    if (mkfifo(outputName_.c_str(), 0600) != 0)
        throw std::runtime_error("Cannot create output FIFO");
    outputOwned_ = true;
    std::cerr << "Local pipes ready: " << inputName_ << " / " << outputName_ << '\n';
    inputPipe_ = open(inputName_.c_str(), O_RDONLY | O_NOFOLLOW);
    if (inputPipe_ < 0)
        throw std::runtime_error("Cannot open input FIFO");
    outputPipe_ = open(outputName_.c_str(), O_WRONLY | O_NOFOLLOW);
    if (outputPipe_ < 0)
        throw std::runtime_error("Cannot open output FIFO");
    signal(SIGPIPE, SIG_IGN);
#endif
}

void LocalTransport::OpenSharedMemory(const std::string& name, int width, int height, int scale)
{
    const std::size_t headerBytes = sizeof(SharedFrameHeader);
    if (inputBytes_ > std::numeric_limits<std::size_t>::max() - headerBytes ||
        outputBytes_ > std::numeric_limits<std::size_t>::max() - headerBytes - inputBytes_)
        throw std::overflow_error("Shared-memory mapping size overflow");
    mappingBytes_ = headerBytes + inputBytes_ + outputBytes_;
    const auto initialize = [&]
    {
        auto* header = static_cast<SharedFrameHeader*>(memory_);
        *header = {Magic, Version, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                   static_cast<std::uint32_t>(scale), Bgr24, inputBytes_, outputBytes_, 0, 0};
    };
#ifdef _WIN32
    mappingName_ = "Local\\TexUpSR-" + name;
    requestName_ = mappingName_ + "-request";
    responseName_ = mappingName_ + "-response";
    const auto bytes = static_cast<std::uint64_t>(mappingBytes_);
    mapping_ = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                 static_cast<DWORD>(bytes >> 32), static_cast<DWORD>(bytes),
                                 mappingName_.c_str());
    if (!mapping_)
        throw std::runtime_error("Cannot create shared-memory mapping");
    if (GetLastError() == ERROR_ALREADY_EXISTS)
        throw std::runtime_error("Shared-memory name already in use");
    memory_ = MapViewOfFile(asHandle(mapping_), FILE_MAP_ALL_ACCESS, 0, 0, mappingBytes_);
    if (!memory_)
        throw std::runtime_error("Cannot map shared-memory buffers");
    initialize();
    request_ = CreateEventA(nullptr, FALSE, FALSE, requestName_.c_str());
    if (!request_)
        throw std::runtime_error("Cannot create shared-memory request event");
    if (GetLastError() == ERROR_ALREADY_EXISTS)
        throw std::runtime_error("Request event already in use");
    response_ = CreateEventA(nullptr, FALSE, FALSE, responseName_.c_str());
    if (!response_)
        throw std::runtime_error("Cannot create shared-memory response event");
    if (GetLastError() == ERROR_ALREADY_EXISTS)
        throw std::runtime_error("Response event already in use");
#else
    mappingName_ = "/texupsr-" + name;
    requestName_ = mappingName_ + "-request";
    responseName_ = mappingName_ + "-response";
    if (mappingBytes_ > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
        throw std::overflow_error("Shared-memory mapping exceeds supported file size");
    mappingFd_ = shm_open(mappingName_.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    if (mappingFd_ == -1)
        throw std::runtime_error("Cannot create shared-memory object");
    mappingOwned_ = true;
    if (ftruncate(mappingFd_, static_cast<off_t>(mappingBytes_)) != 0)
        throw std::runtime_error("Cannot size shared-memory object");
    memory_ = mmap(nullptr, mappingBytes_, PROT_READ | PROT_WRITE, MAP_SHARED, mappingFd_, 0);
    if (memory_ == MAP_FAILED)
    {
        memory_ = nullptr;
        throw std::runtime_error("Cannot map shared-memory object");
    }
    initialize();
    request_ = sem_open(requestName_.c_str(), O_CREAT | O_EXCL, 0600, 0);
    if (request_ == SEM_FAILED)
    {
        request_ = nullptr;
        throw std::runtime_error("Cannot create shared-memory request semaphore");
    }
    requestOwned_ = true;
    response_ = sem_open(responseName_.c_str(), O_CREAT | O_EXCL, 0600, 0);
    if (response_ == SEM_FAILED)
    {
        response_ = nullptr;
        throw std::runtime_error("Cannot create shared-memory response semaphore");
    }
    responseOwned_ = true;
#endif
    std::cerr << "Shared memory ready: " << mappingName_ << '\n';
}

bool LocalTransport::ReadFrame(unsigned char* pixels, std::size_t bytes)
{
    if (bytes != inputBytes_ || !pixels)
        throw std::invalid_argument("Invalid input buffer size");
    if (mode_ == Mode::SharedMemory)
    {
#ifdef _WIN32
        if (WaitForSingleObject(asHandle(request_), INFINITE) != WAIT_OBJECT_0)
            throw std::runtime_error("Cannot wait for shared-memory request");
#else
        waitSemaphore(request_);
#endif
        const auto* header = static_cast<const SharedFrameHeader*>(memory_);
        if (header->state == Stop)
        {
#ifdef _WIN32
            SetEvent(asHandle(response_));
#else
            sem_post(static_cast<sem_t*>(response_));
#endif
            return false;
        }
        if (header->magic != Magic || header->version != Version || header->state != Request ||
            header->format != Bgr24 || header->width != static_cast<std::uint32_t>(width_) ||
            header->height != static_cast<std::uint32_t>(height_) ||
            header->scale != static_cast<std::uint32_t>(scale_) ||
            header->inputBytes != inputBytes_ || header->outputBytes != outputBytes_)
            throw std::runtime_error("Invalid shared-memory request");
        std::memcpy(pixels, static_cast<const unsigned char*>(memory_) + sizeof(SharedFrameHeader), bytes);
        return true;
    }
    if (mode_ == Mode::Stdio)
    {
        std::cin.read(reinterpret_cast<char*>(pixels), static_cast<std::streamsize>(bytes));
        const auto count = std::cin.gcount();
        if (!count && std::cin.eof())
            return false;
        if (count != static_cast<std::streamsize>(bytes))
            throw std::runtime_error("Incomplete BGR24 input frame");
        return true;
    }

    std::size_t total = 0;
    while (total < bytes)
    {
        std::size_t received = 0;
#ifdef _WIN32
        const bool available = readPart(asHandle(inputPipe_), pixels + total, received, bytes - total);
#else
        const bool available = readPart(inputPipe_, pixels + total, received, bytes - total);
#endif
        if (!available)
        {
            if (!total)
                return false;
            throw std::runtime_error("Incomplete BGR24 input frame");
        }
        total += received;
    }
    return true;
}

void LocalTransport::WriteFrame(const unsigned char* pixels, std::size_t bytes)
{
    if (bytes != outputBytes_ || !pixels)
        throw std::invalid_argument("Invalid output buffer size");
    if (mode_ == Mode::SharedMemory)
    {
        auto* header = static_cast<SharedFrameHeader*>(memory_);
        if (header->state != Request)
            throw std::runtime_error("Shared-memory request state changed unexpectedly");
        std::memcpy(static_cast<unsigned char*>(memory_) + sizeof(SharedFrameHeader) + inputBytes_, pixels, bytes);
        header->state = Response;
#ifdef _WIN32
        if (!SetEvent(asHandle(response_)))
            throw std::runtime_error("Cannot signal shared-memory response");
#else
        if (sem_post(static_cast<sem_t*>(response_)) != 0)
            throw std::runtime_error("Cannot signal shared-memory response");
#endif
        return;
    }
    if (mode_ == Mode::Stdio)
    {
        std::cout.write(reinterpret_cast<const char*>(pixels), static_cast<std::streamsize>(bytes));
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("Cannot write BGR24 output frame");
        return;
    }
    std::size_t total = 0;
    while (total < bytes)
    {
        std::size_t written = 0;
#ifdef _WIN32
        writePart(asHandle(outputPipe_), pixels + total, bytes - total, written);
#else
        writePart(outputPipe_, pixels + total, bytes - total, written);
#endif
        total += written;
    }
}

void LocalTransport::Fail() noexcept
{
    if (mode_ != Mode::SharedMemory || !memory_ || !response_)
        return;
    static_cast<SharedFrameHeader*>(memory_)->state = Error;
#ifdef _WIN32
    SetEvent(asHandle(response_));
#else
    sem_post(static_cast<sem_t*>(response_));
#endif
}

void LocalTransport::Close() noexcept
{
#ifdef _WIN32
    if (memory_)
        UnmapViewOfFile(memory_);
    if (response_)
        CloseHandle(asHandle(response_));
    if (request_)
        CloseHandle(asHandle(request_));
    if (mapping_)
        CloseHandle(asHandle(mapping_));
    if (inputPipe_)
    {
        DisconnectNamedPipe(asHandle(inputPipe_));
        CloseHandle(asHandle(inputPipe_));
    }
    if (outputPipe_)
    {
        DisconnectNamedPipe(asHandle(outputPipe_));
        CloseHandle(asHandle(outputPipe_));
    }
#else
    if (memory_)
        munmap(memory_, mappingBytes_);
    if (mappingFd_ != -1)
        close(mappingFd_);
    if (response_)
        sem_close(static_cast<sem_t*>(response_));
    if (request_)
        sem_close(static_cast<sem_t*>(request_));
    if (outputPipe_ != -1)
        close(outputPipe_);
    if (inputPipe_ != -1)
        close(inputPipe_);
    if (inputOwned_)
        unlink(inputName_.c_str());
    if (outputOwned_)
        unlink(outputName_.c_str());
    if (responseOwned_)
        sem_unlink(responseName_.c_str());
    if (requestOwned_)
        sem_unlink(requestName_.c_str());
    if (mappingOwned_)
        shm_unlink(mappingName_.c_str());
#endif
}
