// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "TensorSizes.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

TEST(TensorSizes, ElementCounts)
{
    const auto normal = TensorElements(128, 2, 4);
    EXPECT_EQ(normal.input, 98304);
    EXPECT_EQ(normal.output, 1572864);
}

TEST(TensorSizes, AllocationBytes)
{
    const auto normal = TensorElements(128, 2, 4);
    EXPECT_EQ(TensorBytes(normal.input, 4), 393216);
    EXPECT_EQ(TensorBytes(normal.output, 2), 3145728);
}

TEST(TensorSizes, RejectsInvalidDimensions)
{
    EXPECT_ANY_THROW(TensorElements(0, 1, 1));
    EXPECT_ANY_THROW(TensorElements(4, -1, 1));
    EXPECT_ANY_THROW(TensorElements(4, 1, 0));
    EXPECT_THROW(TensorElements(std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 4),
                 std::overflow_error);
}

TEST(TensorSizes, RejectsInvalidByteCounts)
{
    EXPECT_ANY_THROW(TensorBytes(std::numeric_limits<size_t>::max(), 2));
    EXPECT_ANY_THROW(TensorBytes(1, 0));
}
