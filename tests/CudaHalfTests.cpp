// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include <gtest/gtest.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace
{
uint16_t HalfBits(float value)
{
    const __half half = __float2half(value);
    uint16_t bits = 0;
    static_assert(sizeof(half) == sizeof(bits));
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}
} // namespace

TEST(CudaHalf, SignedZero)
{
    EXPECT_EQ(HalfBits(0.f), 0x0000u);
    EXPECT_EQ(HalfBits(-0.f), 0x8000u);
    EXPECT_EQ(__half2float(__float2half(0.f)), 0.f);
    EXPECT_TRUE(std::signbit(__half2float(__float2half(-0.f))));
}

TEST(CudaHalf, NormalAndSubnormal)
{
    EXPECT_EQ(HalfBits(1.f), 0x3c00u);
    EXPECT_EQ(HalfBits(-2.f), 0xc000u);
    EXPECT_EQ(HalfBits(65504.f), 0x7bffu);
    EXPECT_EQ(HalfBits(std::ldexp(1.f, -14)), 0x0400u);
    EXPECT_EQ(HalfBits(std::ldexp(1.f, -24)), 0x0001u);
    EXPECT_FLOAT_EQ(__half2float(__float2half(std::ldexp(1.f, -24))), std::ldexp(1.f, -24));
}

TEST(CudaHalf, RoundsToNearestEven)
{
    EXPECT_EQ(HalfBits(1.f + std::ldexp(1.f, -11)), 0x3c00u);
    EXPECT_EQ(HalfBits(1.f + 3.f * std::ldexp(1.f, -11)), 0x3c02u);
    EXPECT_EQ(HalfBits(std::ldexp(1.f, -25)), 0x0000u);
    EXPECT_EQ(HalfBits(3.f * std::ldexp(1.f, -25)), 0x0002u);
}

TEST(CudaHalf, OverflowsAndNonfiniteValues)
{
    EXPECT_EQ(HalfBits(1e10f), 0x7c00u);
    EXPECT_EQ(HalfBits(-1e10f), 0xfc00u);
    EXPECT_EQ(HalfBits(std::numeric_limits<float>::infinity()), 0x7c00u);
    EXPECT_EQ(HalfBits(-std::numeric_limits<float>::infinity()), 0xfc00u);
    EXPECT_TRUE(std::isinf(__half2float(__float2half(1e10f))));
    EXPECT_TRUE(std::isnan(__half2float(__float2half(std::numeric_limits<float>::quiet_NaN()))));
}
