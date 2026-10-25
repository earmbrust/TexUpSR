// Copyright (c) 2025 Elden Armbrust. MIT License.
#pragma once
#include "Options.hpp"
#include "Resources.hpp"
#include <string>
#include <vector>
struct EngineState {
    TrtPtr<nvinfer1::IRuntime> runtime;
    TrtPtr<nvinfer1::ICudaEngine> engine;
    std::string inName, outName;
    nvinfer1::DataType inType, outType;
};
EngineState loadEngine(const Options& o, int tile, int batch, Logger& logger);
