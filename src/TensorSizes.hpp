// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once

#include <cstddef>

struct TensorSizes
{
    size_t input;
    size_t output;
};

TensorSizes TensorElements(int tile, int batch, int scale);
size_t TensorBytes(size_t elements, size_t elementBytes);
