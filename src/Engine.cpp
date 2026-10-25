// Copyright (c) 2025 Elden Armbrust. MIT License.
#include "Engine.hpp"
#include "Upscaler.hpp"
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
namespace fs = std::filesystem;
static std::vector<char> readBinary(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open " + path.string());
    const auto len = in.tellg();
    if (len <= 0) throw std::runtime_error("Empty file " + path.string());
    std::vector<char> data(static_cast<size_t>(len));
    in.seekg(0); in.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!in) throw std::runtime_error("Cannot read " + path.string());
    return data;
}
static void writeBinary(const fs::path& path, const void* data, size_t size) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size))) throw std::runtime_error("Cannot write " + path.string());
}
static std::vector<char> buildEngine(const Options& o, int tile, int batch, Logger& logger) {
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("Cannot create TensorRT builder");
    auto flags = 1U << static_cast<unsigned>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser || !parser->parseFromFile(o.model.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        throw std::runtime_error("ONNX parse failed: " + o.model);
    if (network->getNbInputs() != 1 || network->getNbOutputs() != 1) throw std::runtime_error("Model must have exactly one input and output");
    auto* input = network->getInput(0);
    if (input->getType() != nvinfer1::DataType::kFLOAT && input->getType() != nvinfer1::DataType::kHALF)
        throw std::runtime_error("Model input must be FP16 or FP32");
    auto dims = input->getDimensions();
    if (dims.nbDims != 4 || (dims.d[1] != 3 && dims.d[1] != -1)) throw std::runtime_error("Expected NCHW RGB input");
    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<size_t>(o.workspace * 1024.0 * 1024 * 1024));
    if (builder->platformHasFastFp16()) config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto* profile = builder->createOptimizationProfile();
    nvinfer1::Dims4 shape(batch, 3, tile, tile);
    if (!profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, shape) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, shape) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, shape))
        throw std::runtime_error("Cannot set model input dimensions");
    if (config->addOptimizationProfile(profile) < 0) throw std::runtime_error("Cannot add optimization profile");
    TrtPtr<nvinfer1::IHostMemory> blob(builder->buildSerializedNetwork(*network, *config));
    if (!blob) throw std::runtime_error("TensorRT engine build failed");
    const auto* first = static_cast<const char*>(blob->data());
    return std::vector<char>(first, first + blob->size());
}

EngineState loadEngine(const Options& o, int tile, int batch, Logger& logger) {
    const fs::path enginePath = o.engine.empty()
        ? fs::path("engines") / (fs::path(o.model).stem().string() + "_t" + std::to_string(tile) + "_b" + std::to_string(batch) + ".plan")
        : fs::path(o.engine);
    std::vector<char> bytes;
    if (fs::exists(enginePath)) bytes = readBinary(enginePath);
    else { bytes = buildEngine(o, tile, batch, logger); writeBinary(enginePath, bytes.data(), bytes.size()); }
    TrtPtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime) throw std::runtime_error("Cannot create TensorRT runtime");
    TrtPtr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine) throw std::runtime_error("Engine deserialization failed; remove stale .plan file");
    if (engine->getNbIOTensors() != 2) throw std::runtime_error("Expected one input and one output tensor");
    std::string inName, outName;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) inName = name;
        else outName = name;
    }
    if (inName.empty() || outName.empty())
        throw std::runtime_error("Could not identify engine input/output tensors");
    const auto inType = engine->getTensorDataType(inName.c_str());
    const auto outType = engine->getTensorDataType(outName.c_str());
    tensorElementBytes(inType);
    tensorElementBytes(outType);
    if (o.verbose) {
        std::cout << "Tensor formats: input " << (inType == nvinfer1::DataType::kHALF ? "FP16" : "FP32")
                  << ", output " << (outType == nvinfer1::DataType::kHALF ? "FP16" : "FP32") << '\\n';
    }
    return EngineState{std::move(runtime), std::move(engine), std::move(inName), std::move(outName), inType, outType};
}
