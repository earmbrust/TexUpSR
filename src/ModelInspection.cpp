// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "ModelInspection.hpp"

#include "onnx.pb.h"
#include <cstdint>

#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace
{
const char* dataType(int type)
{
    switch (type)
    {
        case onnx::TensorProto::FLOAT:
            return "FP32";
        case onnx::TensorProto::FLOAT16:
            return "FP16";
        case onnx::TensorProto::DOUBLE:
            return "FP64";
        case onnx::TensorProto::UINT8:
            return "UINT8";
        case onnx::TensorProto::INT8:
            return "INT8";
        case onnx::TensorProto::INT16:
            return "INT16";
        case onnx::TensorProto::INT32:
            return "INT32";
        case onnx::TensorProto::INT64:
            return "INT64";
        case onnx::TensorProto::BOOL:
            return "BOOL";
        case onnx::TensorProto::BFLOAT16:
            return "BF16";
        default:
            return "Other";
    }
}

void printTensor(const onnx::ValueInfoProto& value, std::ostream& output)
{
    output << "  " << value.name() << ": ";
    if (!value.has_type() || !value.type().has_tensor_type())
    {
        output << "non-tensor or unspecified type\n";
        return;
    }

    const auto& tensor = value.type().tensor_type();
    output << dataType(tensor.elem_type()) << " (ONNX type " << tensor.elem_type() << ")";
    if (!tensor.has_shape())
    {
        output << ", shape unknown\n";
        return;
    }

    output << " [";
    for (int i = 0; i < tensor.shape().dim_size(); ++i)
    {
        if (i)
            output << ", ";
        const auto& dimension = tensor.shape().dim(i);
        if (dimension.has_dim_value())
            output << dimension.dim_value();
        else if (dimension.has_dim_param())
            output << dimension.dim_param();
        else
            output << '?';
    }
    output << "]\n";
}

bool staticScale(const onnx::ValueInfoProto& input, const onnx::ValueInfoProto& output, int64_t& scale)
{
    if (!input.has_type() || !output.has_type() || !input.type().has_tensor_type() ||
        !output.type().has_tensor_type())
        return false;
    const auto& a = input.type().tensor_type();
    const auto& b = output.type().tensor_type();
    if (!a.has_shape() || !b.has_shape() || a.shape().dim_size() != 4 || b.shape().dim_size() != 4)
        return false;
    if (!a.shape().dim(2).has_dim_value() || !a.shape().dim(3).has_dim_value() ||
        !b.shape().dim(2).has_dim_value() || !b.shape().dim(3).has_dim_value())
        return false;
    const auto h = a.shape().dim(2).dim_value();
    const auto w = a.shape().dim(3).dim_value();
    const auto oh = b.shape().dim(2).dim_value();
    const auto ow = b.shape().dim(3).dim_value();
    if (h <= 0 || w <= 0 || oh <= 0 || ow <= 0 || oh % h || ow % w || oh / h != ow / w)
        return false;
    scale = oh / h;
    return true;
}
} // namespace

void inspectModel(const std::string& path, std::ostream& output)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open ONNX model: " + path);

    onnx::ModelProto model;
    if (!model.ParseFromIstream(&file) || !model.has_graph())
        throw std::runtime_error("Cannot read ONNX model metadata: " + path);

    const auto& graph = model.graph();
    output << "ONNX model: " << path << '\n';
    output << "IR version: " << model.ir_version() << '\n';
    output << "Opsets:\n";
    for (const auto& opset : model.opset_import())
        output << "  " << (opset.domain().empty() ? "ai.onnx" : opset.domain()) << ": " << opset.version()
               << '\n';

    output << "Inputs (" << graph.input_size() << "):\n";
    for (const auto& input : graph.input())
        printTensor(input, output);
    output << "Outputs (" << graph.output_size() << "):\n";
    for (const auto& tensor : graph.output())
        printTensor(tensor, output);

    int64_t scale = 0;
    if (graph.input_size() == 1 && graph.output_size() == 1 &&
        staticScale(graph.input(0), graph.output(0), scale))
        output << "Static spatial scale (assuming NCHW): " << scale << "x\n";
    else
        output << "Static spatial scale: unknown\n";

    output << "TensorRT parser: not attempted (CPU-only inspection)\n";
    output << "Engine build: not attempted\n";
    output << "GPU compatibility: not established\n";
}
