// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once

#include "Options.hpp"
#include "ModelProfile.hpp"
#include "Resources.hpp"

#include <filesystem>
#include <string>
#include <vector>

class Engine
{
public:
    Engine(const Options& options, int tile, int batch, Logger& logger, const ModelProfile& profile);

    nvinfer1::ICudaEngine& Get() const;
    const std::string& InputName() const;
    const std::string& OutputName() const;
    nvinfer1::DataType InputType() const;
    nvinfer1::DataType OutputType() const;

private:
    static std::vector<char> Read(const std::filesystem::path& path);
    static void Write(const std::filesystem::path& path, const std::vector<char>& data);
    static std::vector<char> Build(const Options& options, int tile, int batch, Logger& logger,
                                   const ModelProfile& profile);

    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    std::string inputName_;
    std::string outputName_;
    nvinfer1::DataType inputType_{};
    nvinfer1::DataType outputType_{};
};
