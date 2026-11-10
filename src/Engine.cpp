// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Engine.hpp"
#include "EngineCache.hpp"
#include "Sizing.hpp"
#include "Upscaler.hpp"

#include <NvOnnxParser.h>
#include <climits>
#include <chrono>
#include <system_error>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
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
        fs::create_directories(path.parent_path());
    fs::path temporary = path;
    temporary += ".tmp." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    try
    {
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output || !output.write(data.data(), static_cast<std::streamsize>(data.size())))
                throw std::runtime_error("Cannot write temporary engine: " + temporary.string());
            output.close();
            if (!output)
                throw std::runtime_error("Cannot finish writing temporary engine");
        }
#ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("Cannot replace engine: " + path.string());
#else
        fs::rename(temporary, path);
#endif
    }
    catch (...)
    {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
}

std::vector<char> Engine::Build(const Options& options, int tile, int batch, Logger& logger,
                                const ModelProfile& profile)
{
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));

    if (!builder)
    {
        throw std::runtime_error("Cannot create TensorRT builder");
    }

    const auto flags = 1U << static_cast<unsigned>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));

    if (!parser || !parser->parseFromFile(options.model.c_str(),
                                          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
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

    if (dims.nbDims != 4)
        throw std::runtime_error("Model input must have four dimensions");
    const auto requested = modelShape(batch, tile, profile.layout);
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < -1 || dims.d[i] > INT_MAX)
            throw std::runtime_error("Input model dimension is out of range");
    }
    validateModelShape({static_cast<int>(dims.d[0]), static_cast<int>(dims.d[1]), static_cast<int>(dims.d[2]),
                        static_cast<int>(dims.d[3])},
                       requested, "Input");

    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceBytes(options.workspace));

    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    const nvinfer1::Dims4 shape(requested[0], requested[1], requested[2], requested[3]);

    bool dynamic = false;
    for (int i = 0; i < dims.nbDims; ++i)
        dynamic |= dims.d[i] == -1;
    if (dynamic)
    {
        auto* optimizationProfile = builder->createOptimizationProfile();
        if (!optimizationProfile)
            throw std::runtime_error("Cannot create optimization profile");
        if (!optimizationProfile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
                                                shape) ||
            !optimizationProfile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
                                                shape) ||
            !optimizationProfile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
                                                shape) ||
            config->addOptimizationProfile(optimizationProfile) < 0)
            throw std::runtime_error("Cannot configure dynamic model input profile");
    }

    TrtPtr<nvinfer1::IHostMemory> blob(builder->buildSerializedNetwork(*network, *config));

    if (!blob)
    {
        throw std::runtime_error("TensorRT engine build failed");
    }

    const auto* first = static_cast<const char*>(blob->data());
    return {first, first + blob->size()};
}

Engine::Engine(const Options& options, int tile, int batch, Logger& logger, const ModelProfile& profile)
{
    std::ostream& messages = options.stream ? std::cerr : std::cout;
    // An engine plan is specific to the ONNX content, build settings, TensorRT, and GPU.
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        throw std::runtime_error("Cannot query CUDA device for engine cache identity");
    cudaDeviceProp gpu{};
    if (cudaGetDeviceProperties(&gpu, device) != cudaSuccess)
        throw std::runtime_error("Cannot query CUDA device properties for engine cache identity");
    const std::string runtimeIdentity = std::string(gpu.name) + ":" + std::to_string(gpu.major) + ":" +
                                        std::to_string(gpu.minor) + ":" + std::to_string(NV_TENSORRT_MAJOR) +
                                        ":" + std::to_string(NV_TENSORRT_MINOR) + ":" +
                                        std::to_string(NV_TENSORRT_PATCH);
    const bool automatic = options.engine.empty();
    const auto identity =
        automatic ? engineCacheKey(cacheFileDigest(options.model), tile, batch, options.scale,
                                   options.workspace, profile.layout == TensorLayout::NHWC, runtimeIdentity)
                  : std::string();
    const fs::path path =
        automatic ? fs::path("engines") / (fs::path(options.model).stem().string() + "_" + identity + ".plan")
                  : fs::path(options.engine);
    if (options.buildEngine && fs::exists(path) && !options.overwriteEngine)
        throw std::runtime_error("Engine already exists: " + path.string() +
                                 " (use --overwrite-engine to replace)");

    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_)
        throw std::runtime_error("Cannot create TensorRT runtime");

    auto deserialize = [&](const std::vector<char>& bytes) -> bool
    {
        if (bytes.empty())
            return false;
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        return engine_ && engine_->getNbIOTensors() == 2;
    };

    bool rebuild = options.overwriteEngine || !fs::exists(path);
    if (!rebuild)
    {
        if (!options.quiet)
            messages << "Loading TensorRT engine: " << path.string() << '\n';
        try
        {
            if (!deserialize(Read(path)))
            {
                if (!automatic)
                    throw std::runtime_error(
                        "Explicit engine cannot be deserialized; use --overwrite-engine to rebuild");
                rebuild = true;
            }
        }
        catch (const std::exception&)
        {
            if (!automatic)
                throw;
            rebuild = true;
        }
    }

    std::vector<char> pendingBytes;
    if (rebuild)
    {
        if (!options.quiet)
            messages << "Building TensorRT engine: " << path.string() << std::endl;
        logger.building = !options.quiet;
        try
        {
            pendingBytes = Build(options, tile, batch, logger, profile);
        }
        catch (...)
        {
            logger.building = false;
            throw;
        }
        logger.building = false;
        if (!deserialize(pendingBytes))
            throw std::runtime_error("New TensorRT engine failed deserialization; existing engine preserved");
    }

    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            inputName_ = name;
        else
            outputName_ = name;
    }
    if (inputName_.empty() || outputName_.empty())
        throw std::runtime_error("Could not identify engine input/output tensors");
    inputType_ = engine_->getTensorDataType(inputName_.c_str());
    outputType_ = engine_->getTensorDataType(outputName_.c_str());
    tensorElementBytes(inputType_);
    tensorElementBytes(outputType_);
    if (options.buildEngine)
    {
        TrtPtr<nvinfer1::IExecutionContext> context(engine_->createExecutionContext());
        if (!context)
            throw std::runtime_error("Cannot create TensorRT context for engine validation");
        const auto inputShape = modelShape(batch, tile, profile.layout);
        if (!context->setInputShape(inputName_.c_str(), nvinfer1::Dims4(inputShape[0], inputShape[1],
                                                                        inputShape[2], inputShape[3])))
            throw std::runtime_error("Cannot set engine input shape");
        const int64_t outputTile = static_cast<int64_t>(tile) * options.scale;
        if (outputTile > INT_MAX)
            throw std::runtime_error("Upscaled tile dimensions overflow int");
        const auto expected = modelShape(batch, static_cast<int>(outputTile), profile.layout);
        const auto actual = context->getTensorShape(outputName_.c_str());
        if (actual.nbDims != 4)
            throw std::runtime_error("Engine output must have four dimensions");
        for (int i = 0; i < 4; ++i)
        {
            if (actual.d[i] != expected[i])
                throw std::runtime_error("Model output shape does not match requested scale");
        }
    }
    if (rebuild)
    {
        Write(path, pendingBytes);
        if (!options.quiet)
            messages << "TensorRT engine saved: " << path.string() << '\n';
    }
    else if (!options.quiet)
    {
        messages << "TensorRT engine loaded.\n";
    }
    if (options.verbose && !options.quiet)
    {
        messages << "Tensor formats: input " << (inputType_ == nvinfer1::DataType::kHALF ? "FP16" : "FP32")
                  << ", output " << (outputType_ == nvinfer1::DataType::kHALF ? "FP16" : "FP32") << '\n';
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
