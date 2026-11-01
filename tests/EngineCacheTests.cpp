// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "EngineCache.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

TEST(EngineCache, sha256)
{
    CacheDigest digest;
    digest.Update(std::string("abc"));
    EXPECT_EQ(digest.Finish(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}
TEST(EngineCache, identityChanges)
{
    const auto key = engineCacheKey("abc", 128, 2, 4, 4., false, "gpu-A trt-A");
    EXPECT_EQ(key, engineCacheKey("abc", 128, 2, 4, 4., false, "gpu-A trt-A"));
    EXPECT_NE(key, engineCacheKey("def", 128, 2, 4, 4., false, "gpu-A trt-A"));
    EXPECT_NE(key, engineCacheKey("abc", 256, 2, 4, 4., false, "gpu-A trt-A"));
    EXPECT_NE(key, engineCacheKey("abc", 128, 3, 4, 4., false, "gpu-A trt-A"));
    EXPECT_NE(key, engineCacheKey("abc", 128, 2, 2, 4., false, "gpu-A trt-A"));
    EXPECT_NE(key, engineCacheKey("abc", 128, 2, 4, 8., false, "gpu-A trt-A"));
    EXPECT_NE(key, engineCacheKey("abc", 128, 2, 4, 4., true, "gpu-A trt-A"));
    EXPECT_NE(key, engineCacheKey("abc", 128, 2, 4, 4., false, "gpu-B trt-A"));
    EXPECT_NE(key, engineCacheKey("abc", 128, 2, 4, 4., false, "gpu-A trt-B"));
}
