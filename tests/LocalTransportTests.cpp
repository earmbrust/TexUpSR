// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "LocalTransport.hpp"

#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <climits>
#include <stdexcept>
#include <cstring>
#include <exception>
#include <string>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace
{
constexpr std::size_t InBytes = 4 * 3 * 3;
constexpr std::size_t OutBytes = 8 * 6 * 3;
using Input = std::array<unsigned char, InBytes>;
using Output = std::array<unsigned char, OutBytes>;

std::string uniqueName(const char* label)
{
#ifdef _WIN32
    return std::string(label) + std::to_string(GetCurrentProcessId());
#else
    return std::string(label) + std::to_string(getpid());
#endif
}

void produce(LocalTransport& worker)
{
    Input input{};
    Output output{};
    for (int frame = 0; frame < 2; ++frame)
    {
        if (!worker.ReadFrame(input.data(), input.size()))
            throw std::runtime_error("Missing test frame");
        for (std::size_t j = 0; j < output.size(); ++j)
            output[j] = input[j % input.size()];
        worker.WriteFrame(output.data(), output.size());
    }
    if (worker.ReadFrame(input.data(), input.size()))
        throw std::runtime_error("Unexpected extra frame");
}

#ifdef _WIN32
using ClientPipe = HANDLE;
ClientPipe openPipe(const std::string& name, DWORD access)
{
    for (int attempt = 0; attempt < 100; ++attempt)
    {
        HANDLE pipe = CreateFileA(name.c_str(), access, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (pipe != INVALID_HANDLE_VALUE)
            return pipe;
        Sleep(20);
    }
    throw std::runtime_error("Cannot connect to test pipe");
}
void closePipe(ClientPipe pipe) { CloseHandle(pipe); }
void pipeWrite(ClientPipe pipe, const unsigned char* bytes, std::size_t size)
{
    DWORD sent = 0;
    if (!WriteFile(pipe, bytes, static_cast<DWORD>(size), &sent, nullptr) || sent != size)
        throw std::runtime_error("Test pipe write failed");
}
void pipeRead(ClientPipe pipe, unsigned char* bytes, std::size_t size)
{
    std::size_t total = 0;
    while (total < size)
    {
        DWORD readBytes = 0;
        if (!ReadFile(pipe, bytes + total, static_cast<DWORD>(size - total), &readBytes, nullptr) || !readBytes)
            throw std::runtime_error("Test pipe read failed");
        total += readBytes;
    }
}
#else
using ClientPipe = int;
ClientPipe openPipe(const std::string& name, int flags)
{
    for (int attempt = 0; attempt < 100; ++attempt)
    {
        const int fd = open(name.c_str(), flags);
        if (fd >= 0)
            return fd;
        if (errno != ENOENT)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error("Cannot connect to test FIFO");
}
void closePipe(ClientPipe pipe) { close(pipe); }
void pipeWrite(ClientPipe pipe, const unsigned char* bytes, std::size_t size)
{
    std::size_t total = 0;
    while (total < size)
    {
        const auto sent = write(pipe, bytes + total, size - total);
        if (sent <= 0)
            throw std::runtime_error("Test pipe write failed");
        total += static_cast<std::size_t>(sent);
    }
}
void pipeRead(ClientPipe pipe, unsigned char* bytes, std::size_t size)
{
    std::size_t total = 0;
    while (total < size)
    {
        const auto got = read(pipe, bytes + total, size - total);
        if (got <= 0)
            throw std::runtime_error("Test pipe read failed");
        total += static_cast<std::size_t>(got);
    }
}
#endif

void roundTripPipe(const std::string& name)
{
#ifdef _WIN32
    const std::string prefix = "\\\\.\\pipe\\TexUpSR-" + name;
    ClientPipe input = openPipe(prefix + "-input", GENERIC_WRITE);
    ClientPipe output = openPipe(prefix + "-output", GENERIC_READ);
#else
    const std::string prefix = "/tmp/texupsr-" + name;
    ClientPipe input = openPipe(prefix + "-input", O_WRONLY);
    ClientPipe output = openPipe(prefix + "-output", O_RDONLY);
    if (input < 0 || output < 0)
        throw std::runtime_error("Cannot open test FIFOs");
#endif
    for (int i = 0; i < 2; ++i)
    {
        Input bytes{};
        Output returned{};
        bytes.fill(static_cast<unsigned char>(30 + i));
        pipeWrite(input, bytes.data(), bytes.size() / 2);
        pipeWrite(input, bytes.data() + bytes.size() / 2, bytes.size() - bytes.size() / 2);
        pipeRead(output, returned.data(), returned.size());
        for (const auto value : returned)
            EXPECT_EQ(value, 30 + i);
    }
    closePipe(input);
    closePipe(output);
}

TEST(LocalTransport, NamedPipeRoundTrip)
{
    const auto name = uniqueName("pipe");
    std::exception_ptr failure;
    std::thread worker([&]
                       {
                           try
                           {
                               LocalTransport transport(LocalTransport::Mode::Pipe, name, 4, 3, 2);
                               produce(transport);
                           }
                           catch (...)
                           {
                               failure = std::current_exception();
                           }
                       });
    roundTripPipe(name);
    worker.join();
    if (failure)
        std::rethrow_exception(failure);
}

#ifdef _WIN32
struct SharedClient
{
    HANDLE mapping = nullptr;
    HANDLE request = nullptr;
    HANDLE response = nullptr;
    void* data = nullptr;
    ~SharedClient()
    {
        if (data) UnmapViewOfFile(data);
        if (response) CloseHandle(response);
        if (request) CloseHandle(request);
        if (mapping) CloseHandle(mapping);
    }
    void open(const std::string& name)
    {
        const std::string root = "Local\\TexUpSR-" + name;
        for (int i = 0; i < 100; ++i)
        {
            mapping = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, root.c_str());
            if (mapping) break;
            Sleep(20);
        }
        if (!mapping) throw std::runtime_error("Shared mapping missing");
        data = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedFrameHeader) + InBytes + OutBytes);
        if (!data) throw std::runtime_error("Cannot map shared memory");
        for (int i = 0; i < 100; ++i)
        {
            request = OpenEventA(EVENT_MODIFY_STATE, FALSE, (root + "-request").c_str());
            response = OpenEventA(SYNCHRONIZE, FALSE, (root + "-response").c_str());
            if (request && response) break;
            if (request) { CloseHandle(request); request = nullptr; }
            if (response) { CloseHandle(response); response = nullptr; }
            Sleep(20);
        }
        if (!request || !response) throw std::runtime_error("Shared events missing");
    }
    void signal() { if (!SetEvent(request)) throw std::runtime_error("Request signal failed"); }
    void wait() { if (WaitForSingleObject(response, 10000) != WAIT_OBJECT_0) throw std::runtime_error("Response wait failed"); }
};
#else
struct SharedClient
{
    int fd = -1;
    sem_t* request = SEM_FAILED;
    sem_t* response = SEM_FAILED;
    void* data = nullptr;
    ~SharedClient()
    {
        if (data) munmap(data, sizeof(SharedFrameHeader) + InBytes + OutBytes);
        if (response != SEM_FAILED) sem_close(response);
        if (request != SEM_FAILED) sem_close(request);
        if (fd >= 0) close(fd);
    }
    void open(const std::string& name)
    {
        const std::string root = "/texupsr-" + name;
        for (int i = 0; i < 100; ++i)
        {
            fd = shm_open(root.c_str(), O_RDWR, 0);
            if (fd >= 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (fd < 0) throw std::runtime_error("Shared mapping missing");
        data = mmap(nullptr, sizeof(SharedFrameHeader) + InBytes + OutBytes,
                    PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED) { data = nullptr; throw std::runtime_error("Cannot map shared memory"); }
        for (int i = 0; i < 100; ++i)
        {
            request = sem_open((root + "-request").c_str(), 0);
            response = sem_open((root + "-response").c_str(), 0);
            if (request != SEM_FAILED && response != SEM_FAILED) break;
            if (request != SEM_FAILED) { sem_close(request); request = SEM_FAILED; }
            if (response != SEM_FAILED) { sem_close(response); response = SEM_FAILED; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (request == SEM_FAILED || response == SEM_FAILED) throw std::runtime_error("Shared semaphores missing");
    }
    void signal() { if (sem_post(request)) throw std::runtime_error("Request signal failed"); }
    void wait()
    {
        for (int i = 0; i < 500; ++i)
        {
            if (sem_trywait(response) == 0)
                return;
            if (errno != EAGAIN && errno != EINTR)
                throw std::runtime_error("Response wait failed");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        throw std::runtime_error("Response wait timed out");
    }
};
#endif

TEST(LocalTransport, SharedMemoryRoundTrip)
{
    const auto name = uniqueName("shared");
    std::exception_ptr failure;
    std::thread worker([&]
                       {
                           try
                           {
                               LocalTransport transport(LocalTransport::Mode::SharedMemory, name, 4, 3, 2);
                               produce(transport);
                           }
                           catch (...)
                           {
                               failure = std::current_exception();
                           }
                       });
    SharedClient client;
    client.open(name);
    auto* header = static_cast<SharedFrameHeader*>(client.data);
    EXPECT_EQ(header->magic, 0x52535554U);
    EXPECT_EQ(header->version, 1U);
    EXPECT_EQ(header->width, 4U);
    EXPECT_EQ(header->height, 3U);
    EXPECT_EQ(header->scale, 2U);
    EXPECT_EQ(header->inputBytes, InBytes);
    EXPECT_EQ(header->outputBytes, OutBytes);
    auto* data = static_cast<unsigned char*>(client.data) + sizeof(SharedFrameHeader);
    for (int i = 0; i < 2; ++i)
    {
        std::memset(data, 30 + i, InBytes);
        header->state = 1;
        client.signal();
        client.wait();
        ASSERT_EQ(header->state, 2U);
        for (std::size_t j = 0; j < OutBytes; ++j)
            EXPECT_EQ(data[InBytes + j], 30 + i);
    }
    header->state = 3;
    client.signal();
    client.wait();
    EXPECT_EQ(header->state, 3U);
    worker.join();
    if (failure)
        std::rethrow_exception(failure);
}

TEST(LocalTransport, RejectsInvalidNamesAndDimensions)
{
    EXPECT_THROW(LocalTransport(LocalTransport::Mode::Pipe, "../bad", 4, 3, 2), std::invalid_argument);
    EXPECT_THROW(LocalTransport(LocalTransport::Mode::SharedMemory, "", 4, 3, 2), std::invalid_argument);
    EXPECT_THROW(LocalTransport(LocalTransport::Mode::Stdio, "", 0, 3, 2), std::invalid_argument);
    EXPECT_THROW(LocalTransport(LocalTransport::Mode::Stdio, "", INT_MAX, INT_MAX, 2), std::invalid_argument);
}
}
