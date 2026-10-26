// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "TensorSizes.hpp"

#include <limits>
#include <stdexcept>

namespace
{
size_t Multiply(size_t left, size_t right)
{
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
    {
        throw std::overflow_error("Tensor size exceeds addressable memory");
    }
    return left * right;
}
}

TensorSizes TensorElements(int tile, int batch, int scale)
{
    if (tile <= 0 || batch <= 0 || scale <= 0)
    {
        throw std::invalid_argument("Invalid tensor dimensions");
    }

    const auto edge = static_cast<size_t>(tile);
    const auto factor = static_cast<size_t>(scale);
    const auto pixels = Multiply(edge, edge);
    const auto input = Multiply(Multiply(static_cast<size_t>(batch), 3), pixels);
    return {input, Multiply(Multiply(input, factor), factor)};
}

size_t TensorBytes(size_t elements, size_t elementBytes)
{
    if (elements == 0 || elementBytes == 0)
    {
        throw std::invalid_argument("Invalid tensor allocation size");
    }
    return Multiply(elements, elementBytes);
}
