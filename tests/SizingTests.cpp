// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Sizing.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

TEST(Sizing, Workspaces)
{
    const size_t gib = size_t(1) << 30;
    EXPECT_EQ(workspaceBytes(1.0), gib);
    EXPECT_EQ(workspaceBytes(4.0), 4 * gib);
    EXPECT_EQ(workspaceBytes(0.5), gib / 2);
    EXPECT_EQ(workspaceBytes(1.0 / (1ULL << 30)), 1);

    EXPECT_ANY_THROW(workspaceBytes(0.0));
    EXPECT_ANY_THROW(workspaceBytes(-1.0));
    EXPECT_ANY_THROW(workspaceBytes(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_ANY_THROW(workspaceBytes(std::numeric_limits<double>::infinity()));
    EXPECT_ANY_THROW(workspaceBytes(1e300));
    EXPECT_ANY_THROW(workspaceBytes(1e-300));

    const double boundary = std::ldexp(1.0, std::numeric_limits<size_t>::digits - 30);
    EXPECT_ANY_THROW(workspaceBytes(boundary));
}

TEST(Sizing, Tiles)
{
    const size_t gib = size_t(1) << 30;
    EXPECT_EQ(autoTile(512 * (size_t(1) << 20), 1, 16), 128);
    EXPECT_EQ(autoTile(gib, 1, 16), 256);
    EXPECT_EQ(autoTile(4 * gib, 1, 16), 512);
    EXPECT_EQ(autoTile(4 * gib, 4, 16), 256);
    EXPECT_EQ(autoTile(gib, 1, 200), 432);
    EXPECT_ANY_THROW(autoTile(1024, 0, 1));
    EXPECT_ANY_THROW(autoTile(1024, 1, -1));
    EXPECT_ANY_THROW(autoTile(1024, 1, std::numeric_limits<int>::max()));
}

TEST(Sizing, Batches)
{
    const size_t gib = size_t(1) << 30;
    EXPECT_EQ(autoBatch(8 * gib, 128, 2, 1, 1.0), 8);
    EXPECT_EQ(autoBatch(0, 512, 4, 2, 4.0), 1);
    EXPECT_EQ(autoBatch(gib, 512, 4, 2, 4.0), 1);
    EXPECT_EQ(autoBatch(gib, 128, 2, 1, 1e300), 1);
    EXPECT_EQ(autoBatch(std::numeric_limits<size_t>::max(), std::numeric_limits<int>::max(),
                        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 1.0),
              1);
    EXPECT_ANY_THROW(autoBatch(1024, 0, 2, 1, 1));
    EXPECT_ANY_THROW(autoBatch(1024, 4, -1, 1, 1));
    EXPECT_ANY_THROW(autoBatch(1024, 4, 2, 0, 1));
    EXPECT_ANY_THROW(autoBatch(1024, 4, 2, 1, std::numeric_limits<double>::infinity()));
}
