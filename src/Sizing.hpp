// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once
#include <cstddef>

size_t workspaceBytes(double gigabytes);
int autoTile(size_t freeBytes, int streams, int overlap);
int autoBatch(size_t freeBytes, int tile, int scale, int streams, double workspaceGB);
