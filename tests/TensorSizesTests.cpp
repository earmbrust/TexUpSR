// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "TensorSizes.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
void Require(bool value, const char* message)
{
    if (!value)
    {
        throw std::runtime_error(message);
    }
}

template <typename F> void Reject(F&& action)
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
    throw std::runtime_error("Invalid tensor size was accepted");
}
} // namespace

int main()
{
    try
    {
        const auto normal = TensorElements(128, 2, 4);
        Require(normal.input == 98304, "Input element count");
        Require(normal.output == 1572864, "Output element count");
        Require(TensorBytes(normal.input, 4) == 393216, "FP32 input allocation");
        Require(TensorBytes(normal.output, 2) == 3145728, "FP16 output allocation");
        Reject([] { TensorElements(0, 1, 1); });
        Reject([] { TensorElements(4, -1, 1); });
        Reject([] { TensorElements(4, 1, 0); });
        Reject([] { TensorElements(std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 4); });
        Reject([] { TensorBytes(std::numeric_limits<size_t>::max(), 2); });
        Reject([] { TensorBytes(1, 0); });
        std::cout << "Tensor sizing tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Tensor sizing tests failed: " << error.what() << '\n';
        return 1;
    }
}
