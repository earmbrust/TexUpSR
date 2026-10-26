// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Sizing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

size_t workspaceBytes(double gigabytes)
{
    if (!std::isfinite(gigabytes) || gigabytes <= 0.0)
    {
        throw std::invalid_argument("Invalid TensorRT workspace size");
    }

    const long double bytes = static_cast<long double>(gigabytes) * (1ULL << 30);
    const long double limit = std::ldexp(1.0L, std::numeric_limits<size_t>::digits);

    if (bytes >= limit)
    {
        throw std::overflow_error("TensorRT workspace size exceeds size_t range");
    }

    if (bytes < 1.0L)
    {
        throw std::invalid_argument("TensorRT workspace must be at least one byte");
    }

    return static_cast<size_t>(bytes);
}

int autoTile(size_t freeBytes, int streams, int overlap)
{
    if (streams <= 0 || overlap < 0)
    {
        throw std::invalid_argument("Invalid automatic tile configuration");
    }

    const int64_t required = static_cast<int64_t>(overlap) * 2 + 32;
    if (required > std::numeric_limits<int>::max())
    {
        throw std::overflow_error("Automatic tile size exceeds int range");
    }

    const size_t perStream = freeBytes / static_cast<size_t>(streams);
    const int candidate = perStream >= (size_t(2) << 30) ? 512 : perStream >= (size_t(768) << 20) ? 256 : 128;
    return std::max(candidate, static_cast<int>(required));
}

int autoBatch(size_t freeBytes, int tile, int scale, int streams, double workspaceGB)
{
    if (tile <= 0 || scale <= 0 || streams <= 0 || !std::isfinite(workspaceGB) || workspaceGB <= 0.0)
    {
        throw std::invalid_argument("Invalid automatic batch configuration");
    }

    const long double reserveBytes = static_cast<long double>(workspaceGB) * (1ULL << 30);
    if (reserveBytes >= static_cast<long double>(std::numeric_limits<size_t>::max()))
    {
        return 1;
    }
    const size_t reserve = static_cast<size_t>(reserveBytes);
    const size_t available = freeBytes > reserve ? freeBytes - reserve : freeBytes / 4;
    const long double tilePixels = static_cast<long double>(tile) * tile;
    const long double scalePixels = static_cast<long double>(scale) * scale;
    const long double bytesPerBatch =
        3.0L * tilePixels * (1.0L + scalePixels) * sizeof(float) * 8.0L * streams;
    const long double estimate = static_cast<long double>(available) / bytesPerBatch;

    if (estimate < 1.0L)
    {
        return 1;
    }
    if (estimate >= 8.0L)
    {
        return 8;
    }
    return static_cast<int>(estimate);
}
