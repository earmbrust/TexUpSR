// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "ModelInspection.hpp"
#include "onnx.pb.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{
void tensor(onnx::ValueInfoProto* value, const char* name, bool dynamic, int spatial)
{
    value->set_name(name);
    auto* type = value->mutable_type()->mutable_tensor_type();
    type->set_elem_type(onnx::TensorProto::FLOAT);
    auto* shape = type->mutable_shape();
    shape->add_dim()->set_dim_value(1);
    shape->add_dim()->set_dim_value(3);
    if (dynamic)
    {
        shape->add_dim()->set_dim_param("height");
        shape->add_dim()->set_dim_param("width");
    }
    else
    {
        shape->add_dim()->set_dim_value(spatial);
        shape->add_dim()->set_dim_value(spatial);
    }
}

std::string inspect(bool dynamic)
{
    onnx::ModelProto model;
    model.set_ir_version(9);
    model.mutable_opset_import()->Add()->set_version(18);
    tensor(model.mutable_graph()->add_input(), "image", dynamic, 16);
    tensor(model.mutable_graph()->add_output(), "result", dynamic, 32);
    const auto path = std::filesystem::temp_directory_path() /
                      (dynamic ? "texupsr-inspection-dynamic.onnx" : "texupsr-inspection-static.onnx");
    {
        std::ofstream file(path, std::ios::binary);
        EXPECT_TRUE(model.SerializeToOstream(&file));
    }
    std::ostringstream report;
    inspectModel(path.string(), report);
    std::filesystem::remove(path);
    return report.str();
}
} // namespace

TEST(ModelInspection, StaticShapes)
{
    const auto report = inspect(false);
    EXPECT_NE(report.find("image: FP32"), std::string::npos);
    EXPECT_NE(report.find("[1, 3, 16, 16]"), std::string::npos);
    EXPECT_NE(report.find("Static spatial scale (assuming NCHW): 2x"), std::string::npos);
    EXPECT_NE(report.find("TensorRT parser: not attempted"), std::string::npos);
}

TEST(ModelInspection, DynamicShapes)
{
    const auto report = inspect(true);
    EXPECT_NE(report.find("[1, 3, height, width]"), std::string::npos);
    EXPECT_NE(report.find("Static spatial scale: unknown"), std::string::npos);
}

TEST(ModelInspection, MissingFile)
{
    std::ostringstream report;
    EXPECT_THROW(inspectModel("missing-texupsr-model.onnx", report), std::runtime_error);
}
