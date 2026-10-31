// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Self-contained SHA-256 for cache identity; does not add a crypto runtime dependency.
class CacheDigest
{
public:
    void Update(const void* source, size_t count)
    {
        const auto* bytes = static_cast<const uint8_t*>(source);
        for (size_t i = 0; i < count; ++i)
        {
            buffer_[used_++] = bytes[i];
            if (used_ == buffer_.size())
            {
                Transform();
                used_ = 0;
            }
        }
        length_ += static_cast<uint64_t>(count);
    }

    void Update(const std::string& text)
    {
        Update(text.data(), text.size());
    }

    std::string Finish()
    {
        const uint64_t bits = length_ * 8;
        buffer_[used_++] = 0x80;
        if (used_ > 56)
        {
            while (used_ < 64)
                buffer_[used_++] = 0;
            Transform();
            used_ = 0;
        }
        while (used_ < 56)
            buffer_[used_++] = 0;
        for (int i = 7; i >= 0; --i)
            buffer_[used_++] = static_cast<uint8_t>(bits >> (i * 8));
        Transform();
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (auto word : state_)
            out << std::setw(8) << word;
        return out.str();
    }

private:
    static uint32_t Rotate(uint32_t word, unsigned amount)
    {
        return (word >> amount) | (word << (32 - amount));
    }
    void Transform()
    {
        static constexpr uint32_t constants[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64] = {};
        for (size_t i = 0; i < 16; ++i)
        {
            const size_t p = i * 4;
            w[i] = (uint32_t(buffer_[p]) << 24) | (uint32_t(buffer_[p + 1]) << 16) |
                   (uint32_t(buffer_[p + 2]) << 8) | uint32_t(buffer_[p + 3]);
        }
        for (size_t i = 16; i < 64; ++i)
        {
            const uint32_t a = Rotate(w[i - 15], 7) ^ Rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t b = Rotate(w[i - 2], 17) ^ Rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + a + w[i - 7] + b;
        }
        auto a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        auto e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t i = 0; i < 64; ++i)
        {
            const uint32_t x = h + (Rotate(e, 6) ^ Rotate(e, 11) ^ Rotate(e, 25)) + ((e & f) ^ (~e & g)) +
                               constants[i] + w[i];
            const uint32_t y = (Rotate(a, 2) ^ Rotate(a, 13) ^ Rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + x;
            d = c;
            c = b;
            b = a;
            a = x + y;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }
    std::array<uint8_t, 64> buffer_{};
    size_t used_ = 0;
    uint64_t length_ = 0;
    std::array<uint32_t, 8> state_{
        {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
};

inline std::string cacheFileDigest(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot hash model: " + path.string());
    CacheDigest digest;
    std::array<char, 65536> buffer{};
    while (input)
    {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0)
            digest.Update(buffer.data(), static_cast<size_t>(count));
    }
    if (!input.eof())
        throw std::runtime_error("Cannot finish hashing model: " + path.string());
    return digest.Finish();
}

inline std::string engineCacheKey(const std::string& modelDigest, int tile, int batch, int scale,
                                  double workspace, bool nhwc, const std::string& runtimeIdentity)
{
    std::ostringstream key;
    key << "texupsr-engine-cache-v1\n"
        << modelDigest << '\n'
        << tile << '\n'
        << batch << '\n'
        << scale << '\n'
        << std::hexfloat << workspace << '\n'
        << (nhwc ? "nhwc" : "nchw") << '\n'
        << runtimeIdentity;
    CacheDigest digest;
    digest.Update(key.str());
    return digest.Finish();
}
