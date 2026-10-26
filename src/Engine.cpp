// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Engine.hpp"
#include "Sizing.hpp"
#include "Upscaler.hpp"

#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

std::vector<char> Engine::Read(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);

    if (!input)
    {
        throw std::runtime_error("Cannot open " + path.string());
    }

    const auto size = input.tellg();

    if (size <= 0)
    {
        throw std::runtime_error("Empty file " + path.string());
    }

    std::vector<char> data(static_cast<size_t>(size));
    input.seekg(0);
    input.read(data.data(), static_cast<std::streamsize>(data.size()));

    if (!input)
    {
        throw std::runtime_error("Cannot read " + path.string());
    }

    return data;
}

void Engine::Write(const fs::path& path, const std::vector<char>& data)
{
    if (path.has_parent_path())
    {
        fs::create_directories(path.parent_path());
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);

    if (!output.write(data.data(), static_cast<std::streamsize>(data.size())))
    {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

std::vector<char> Engine::Build(const Options& options, int tile, int batch, Logger& logger)
{
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));

    if (!builder)
    {
        throw std::runtime_error("Cannot create TensorRT builder");
    }

    const auto flags = 1U << static_cast<unsigned>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));

    if (!parser || !parser->parseFromFile(options.model.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        throw std::runtime_error("ONNX parse failed: " + options.model);
    }

    if (network->getNbInputs() != 1 || network->getNbOutputs() != 1)
    {
        throw std::runtime_error("Model must have exactly one input and output");
    }

    auto* input = network->getInput(0);

    if (input->getType() != nvinfer1::DataType::kFLOAT && input->getType() != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error("Model input must be FP16 or FP32");
    }

    const auto dims = input->getDimensions();

    if (dims.nbDims != 4 || (dims.d[1] != 3 && dims.d[1] != -1))
    {
        throw std::runtime_error("Expected NCHW RGB input");
    }

    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               workspaceBytes(options.workspace));

    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    auto* profile = builder->createOptimizationProfile();
    const nvinfer1::Dims4 shape(batch, 3, tile, tile);

    if (!profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, shape) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, shape) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, shape))
    {
        throw std::runtime_error("Cannot set model input dimensions");
    }

    if (config->addOptimizationProfile(profile) < 0)
    {
        throw std::runtime_error("Cannot add optimization profile");
    }

    TrtPtr<nvinfer1::IHostMemory> blob(builder->buildSerializedNetwork(*network, *config));

    if (!blob)
    {
        throw std::runtime_error("TensorRT engine build failed");
    }

    const auto* first = static_cast<const char*>(blob->data());
    return {first, first + blob->size()};
}

Engine::Engine(const Options& options, int tile, int batch, Logger& logger)
{
    const fs::path path = options.engine.empty()
        ? fs::path("engines") / (fs::path(options.model).stem().string() + "_t" + std::to_string(tile) +
                                 "_b" + std::to_string(batch) + ".plan")
        : fs::path(options.engine);

    std::vector<char> bytes;

    if (fs::exists(path))
    {
        bytes = Read(path);
    }
    else
    {
        bytes = Build(options, tile, batch, logger);
        Write(path, bytes);
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger));

    if (!runtime_)
    {
        throw std::runtime_error("Cannot create TensorRT runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));

    if (!engine_)
    {
        throw std::runtime_error("Engine deserialization failed; remove stale .plan file");
    }

    if (engine_->getNbIOTensors() != 2)
    {
        throw std::runtime_error("Expected one input and one output tensor");
    }

    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char* name = engine_->getIOTensorName(i);

        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            inputName_ = name;
        }
        else
        {
            outputName_ = name;
        }
    }

    if (inputName_.empty() || outputName_.empty())
    {
        throw std::runtime_error("Could not identify engine input/output tensors");
    }

    inputType_ = engine_->getTensorDataType(inputName_.c_str());
    outputType_ = engine_->getTensorDataType(outputName_.c_str());
    tensorElementBytes(inputType_);
    tensorElementBytes(outputType_);

    if (options.verbose)
    {
        std::cout << "Tensor formats: input "
                  << (inputType_ == nvinfer1::DataType::kHALF ? "FP16" : "FP32")
                  << ", output " << (outputType_ == nvinfer1::DataType::kHALF ? "FP16" : "FP32")
                  << '\n';
    }
}

nvinfer1::ICudaEngine& Engine::Get() const
{
    return *engine_;
}

const std::string& Engine::InputName() const
{
    return inputName_;
}

const std::string& Engine::OutputName() const
{
    return outputName_;
}

nvinfer1::DataType Engine::InputType() const
{
    return inputType_;
}

nvinfer1::DataType Engine::OutputType() const
{
    return outputType_;
}
