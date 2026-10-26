// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Sizing.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
void Require(bool value, const char* message)
{
    if (!value)
    {
        throw std::runtime_error(message);
    }
}

template <typename F>
void Reject(F&& action, const char* message)
{
    try
    {
        action();
    }
    catch (const std::invalid_argument&)
    {
        return;
    }
    catch (const std::overflow_error&)
    {
        return;
    }
    throw std::runtime_error(message);
}

void Workspaces()
{
    const size_t gib = size_t(1) << 30;
    Require(workspaceBytes(1.0) == gib, "One GiB workspace");
    Require(workspaceBytes(4.0) == 4 * gib, "Default workspace");
    Require(workspaceBytes(0.5) == gib / 2, "Fractional workspace");
    Require(workspaceBytes(1.0 / (1ULL << 30)) == 1, "One byte workspace");

    Reject([] { workspaceBytes(0.0); }, "Zero workspace accepted");
    Reject([] { workspaceBytes(-1.0); }, "Negative workspace accepted");
    Reject([] { workspaceBytes(std::numeric_limits<double>::quiet_NaN()); }, "NaN workspace accepted");
    Reject([] { workspaceBytes(std::numeric_limits<double>::infinity()); }, "Infinite workspace accepted");
    Reject([] { workspaceBytes(1e300); }, "Huge workspace accepted");
    Reject([] { workspaceBytes(1e-300); }, "Sub-byte workspace accepted");

    const double boundary = std::ldexp(1.0, std::numeric_limits<size_t>::digits - 30);
    Reject([&] { workspaceBytes(boundary); }, "Size_t boundary accepted");
}

void Tiles()
{
    const size_t gib = size_t(1) << 30;
    Require(autoTile(512 * (size_t(1) << 20), 1, 16) == 128, "Small memory tile");
    Require(autoTile(gib, 1, 16) == 256, "Medium memory tile");
    Require(autoTile(4 * gib, 1, 16) == 512, "Large memory tile");
    Require(autoTile(4 * gib, 4, 16) == 256, "Per-stream memory tile");
    Require(autoTile(gib, 1, 200) == 432, "Overlap constraint");
    Reject([] { autoTile(1024, 0, 1); }, "Zero streams accepted");
    Reject([] { autoTile(1024, 1, -1); }, "Negative overlap accepted");
    Reject([] { autoTile(1024, 1, std::numeric_limits<int>::max()); }, "Tile overflow accepted");
}

void Batches()
{
    const size_t gib = size_t(1) << 30;
    Require(autoBatch(8 * gib, 128, 2, 1, 1.0) == 8, "High memory batch");
    Require(autoBatch(0, 512, 4, 2, 4.0) == 1, "Zero memory batch");
    Require(autoBatch(gib, 512, 4, 2, 4.0) == 1, "Workspace reserve batch");
    Require(autoBatch(gib, 128, 2, 1, 1e300) == 1, "Huge workspace batch");
    Require(autoBatch(std::numeric_limits<size_t>::max(),
                      std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
                      std::numeric_limits<int>::max(), 1.0) == 1, "Huge tile and scale batch");
    Reject([] { autoBatch(1024, 0, 2, 1, 1); }, "Zero tile accepted");
    Reject([] { autoBatch(1024, 4, -1, 1, 1); }, "Negative scale accepted");
    Reject([] { autoBatch(1024, 4, 2, 0, 1); }, "Zero streams accepted");
    Reject([] { autoBatch(1024, 4, 2, 1, std::numeric_limits<double>::infinity()); }, "Infinity workspace accepted");
}
}

int main()
{
    try
    {
        Workspaces();
        Tiles();
        Batches();
        std::cout << "Sizing tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Sizing tests failed: " << error.what() << '\n';
        return 1;
    }
}
