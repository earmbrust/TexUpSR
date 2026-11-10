// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include <gtest/gtest.h>
#include "Options.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using Args = std::vector<std::string>;

Options parseArgs(Args args)
{
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args)
        argv.push_back(arg.data());
    return parse(static_cast<int>(argv.size()), argv.data());
}

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void rejects(const Args& args, const std::string& context)
{
    try
    {
        parseArgs(args);
    }
    catch (const std::runtime_error&)
    {
        return;
    }
    throw std::runtime_error("Expected rejection: " + context);
}

Args base()
{
    return {"trt_sr", "-i", "input.png", "-o", "out.png", "-m", "model.onnx"};
}

void rejectsValue(const std::string& option, const std::string& value)
{
    auto args = base();
    args.insert(args.end(), {option, value});
    rejects(args, option + "=" + value);
}

TEST(Options, defaults)
{
    const auto opts = parseArgs(base());
    require(opts.input == "input.png" && opts.output == "out.png" && opts.model == "model.onnx",
            "Required paths");
    require(opts.engine.empty(), "Default engine path");
    require(opts.tile == 0 && opts.batch == 0, "Automatic defaults");
    require(opts.scale == 4 && opts.streams == 2 && opts.overlap == 16, "Integer defaults");
    require(opts.workspace == 4.0, "Workspace default");
    require(!opts.verbose && !opts.quiet && !opts.debug && !opts.overwriteEngine && !opts.benchmark &&
                !opts.buildEngine && !opts.stream,
            "Boolean defaults");
    require(opts.width == 0 && opts.height == 0 && opts.format.empty(), "Streaming defaults");
}

TEST(Options, allOptions)
{
    auto args = base();
    args.insert(args.end(), {"--engine", "engine.plan", "--tile", "256", "--batch", "3", "--scale", "2",
                             "--workspace", "1.25", "--streams", "4", "--overlap", "8", "--verbose",
                             "--quiet", "--debug", "--overwrite-engine", "--benchmark"});
    const auto o = parseArgs(args);
    require(o.engine == "engine.plan", "Engine path");
    require(o.tile == 256 && o.batch == 3 && o.scale == 2 && o.streams == 4 && o.overlap == 8,
            "Numeric options");
    require(o.workspace == 1.25 && o.verbose && o.quiet && o.debug && o.overwriteEngine && o.benchmark,
            "Workspace and flags");

    auto aliases = Args{"trt_sr", "--input", "a", "--output", "b", "--model", "c",    "-e", "d",  "-t",
                        "-1",     "-b",      "0", "-s",       "1", "-w",      "0.25", "-v", "-q", "-d"};
    const auto a = parseArgs(aliases);
    require(a.input == "a" && a.output == "b" && a.model == "c" && a.engine == "d", "Long path names");
    require(a.tile == -1 && a.batch == 0 && a.scale == 1 && a.workspace == 0.25, "Short option names");
    require(a.verbose && a.quiet && a.debug, "Short flags");
}

TEST(Options, requiredArguments)
{
    rejects({"trt_sr"}, "No arguments");
    rejects({"trt_sr", "-o", "b", "-m", "c"}, "Missing input");
    rejects({"trt_sr", "-i", "a", "-m", "c"}, "Missing output");
    rejects({"trt_sr", "-i", "a", "-o", "b"}, "Missing model");
    rejects({"trt_sr", "-i", "", "-o", "b", "-m", "c"}, "Empty input");
    rejects({"trt_sr", "-i", "a", "-o", "", "-m", "c"}, "Empty output");
    rejects({"trt_sr", "-i", "a", "-o", "b", "-m", ""}, "Empty model");
    rejects({"trt_sr", "--unknown"}, "Unknown option");
    rejects({"trt_sr", "-i"}, "Missing option value");
    for (const auto& opt :
         {"--engine", "--tile", "--batch", "--scale", "--workspace", "--streams", "--overlap"})
    {
        auto args = base();
        args.emplace_back(opt);
        rejects(args, std::string("Missing value: ") + opt);
    }
}

TEST(Options, inspection)
{
    const auto options = parseArgs({"trt_sr", "--inspect", "model.onnx"});
    require(options.inspect == "model.onnx", "Inspection path");
    rejects({"trt_sr", "--inspect", "model.onnx", "-i", "input.png"}, "Mixed modes");
    rejects({"trt_sr", "--inspect"}, "Missing inspection path");
    rejects({"trt_sr", "--inspect", "model.onnx", "--overwrite-engine"}, "Overwrite in inspection");
    rejects({"trt_sr", "--inspect", "model.onnx", "--benchmark"}, "Benchmark in inspection");
}

TEST(Options, numericArguments)
{
    for (const auto& opt : {"--tile", "--batch", "--scale", "--streams", "--overlap"})
    {
        for (const auto& val : {"1junk", "junk", "2147483648", "-2147483649", " 1", "1 ", "", "1.2"})
        {
            rejectsValue(opt, val);
        }
    }
    for (const auto& opt : {"--scale", "--streams"})
    {
        rejectsValue(opt, "0");
        rejectsValue(opt, "-1");
    }
    for (const auto& opt : {"--tile", "--batch"})
        rejectsValue(opt, "-2");
    rejectsValue("--overlap", "-1");
    for (const auto& val : {"nan", "inf", "-inf", "1.0garbage", "0", "-2", " 1", "1 ", "", "1e9999"})
    {
        rejectsValue("--workspace", val);
    }
    auto args = base();
    args.insert(args.end(), {"--overlap", "0", "--workspace", "1e-3", "--scale", "1"});
    auto opts = parseArgs(args);
    require(opts.overlap == 0 && std::abs(opts.workspace - 0.001) < 1e-15 && opts.scale == 1,
            "Zero overlap and finite fractional workspace");
}
TEST(Options, enginePreGeneration)
{
    const auto opts =
        parseArgs({"trt_sr", "--build-engine", "-m", "model.onnx", "-e", "out.plan", "-t", "256", "-b", "2",
                   "-s", "2", "-w", "1", "--profile", "model.json", "--overwrite-engine", "-q"});
    require(opts.buildEngine && opts.model == "model.onnx" && opts.engine == "out.plan",
            "Engine generation paths");
    require(opts.tile == 256 && opts.batch == 2 && opts.scale == 2 && opts.workspace == 1.0,
            "Engine generation settings");
    require(opts.overwriteEngine && opts.quiet && opts.profile == "model.json", "Engine generation flags");
    require(opts.input.empty() && opts.output.empty(), "Engine generation does not require images");

    rejects({"trt_sr", "--build-engine", "-m", "model.onnx", "-t", "256", "-b", "1"}, "Missing engine path");
    rejects({"trt_sr", "--build-engine", "-e", "out.plan", "-t", "256", "-b", "1"}, "Missing model");
    rejects({"trt_sr", "--build-engine", "-m", "model.onnx", "-e", "out.plan"}, "Automatic tile/batch");
    rejects({"trt_sr", "--build-engine", "-m", "model.onnx", "-e", "out.plan", "-t", "256", "-b", "0"},
            "Automatic batch");
    rejects({"trt_sr", "--build-engine", "-m", "model.onnx", "-e", "out.plan", "-t", "256", "-b", "1", "-i",
             "image.png"},
            "Image input");
    rejects({"trt_sr", "--build-engine", "-m", "model.onnx", "-e", "out.plan", "-t", "256", "-b", "1", "-o",
             "image.png"},
            "Image output");
    rejects({"trt_sr", "--build-engine", "-m", "model.onnx", "-e", "out.plan", "-t", "256", "-b", "1",
             "--benchmark"},
            "Benchmark mode");
    rejects(
        {"trt_sr", "--build-engine", "-m", "model.onnx", "-e", "out.plan", "-t", "256", "-b", "1", "--debug"},
        "Debug tile");
    rejects({"trt_sr", "--build-engine", "--inspect", "model.onnx"}, "Conflicting operations");
}
TEST(Options, streamMode)
{
    const auto options = parseArgs({"trt_sr", "--stream", "--width", "720", "--height", "480",
                                    "--format", "bgr24", "--scale", "2", "--model", "model.onnx",
                                    "--engine", "model.plan", "--profile", "profile.json", "-q"});
    require(options.stream && options.width == 720 && options.height == 480 &&
                options.format == "bgr24" && options.scale == 2, "Streaming configuration");
    require(options.model == "model.onnx" && options.engine == "model.plan" &&
                options.profile == "profile.json" && options.quiet, "Streaming engine configuration");
    require(options.input.empty() && options.output.empty(), "Streaming does not use image paths");

    const Args start = {"trt_sr", "--stream", "--width", "720", "--height", "480",
                        "--format", "bgr24", "-m", "model.onnx"};
    for (const auto& argument : {"--debug", "--benchmark", "--build-engine"})
    {
        auto args = start;
        args.push_back(argument);
        rejects(args, argument);
    }
    for (const auto& extra : {Args{"-i", "image.png"}, Args{"-o", "image.png"},
                              Args{"--inspect", "model.onnx"}})
    {
        auto args = start;
        args.insert(args.end(), extra.begin(), extra.end());
        rejects(args, "Conflicting streaming mode");
    }
    rejects({"trt_sr", "--stream", "--width", "720", "--height", "480", "-m", "model.onnx"},
            "Missing format");
    rejects({"trt_sr", "--stream", "--height", "480", "--format", "bgr24", "-m", "model.onnx"},
            "Missing width");
    rejects({"trt_sr", "--stream", "--width", "720", "--format", "bgr24", "-m", "model.onnx"},
            "Missing height");
    rejects({"trt_sr", "--stream", "--width", "720", "--height", "480", "--format", "rgb24",
             "-m", "model.onnx"}, "Unsupported pixel format");
    rejects({"trt_sr", "--stream", "--width", "0", "--height", "480", "--format", "bgr24",
             "-m", "model.onnx"}, "Zero width");
    rejects({"trt_sr", "--stream", "--width", "720", "--height", "-1", "--format", "bgr24",
             "-m", "model.onnx"}, "Negative height");
    rejects({"trt_sr", "--width", "720", "-i", "a", "-o", "b", "-m", "c"},
            "Stream dimension in image mode");
    rejects({"trt_sr", "--format", "bgr24", "-i", "a", "-o", "b", "-m", "c"},
            "Stream format in image mode");
}

TEST(Options, localWorkerModes)
{
    const auto pipe = parseArgs({"trt_sr", "--pipe", "test_01", "--width", "128", "--height", "96",
                                 "--format", "bgr24", "-m", "model.onnx", "-s", "4"});
    require(pipe.stream && pipe.pipeName == "test_01" && pipe.sharedMemoryName.empty(), "Pipe mode");
    const auto shared = parseArgs({"trt_sr", "--shared-memory", "test_02", "--width", "128",
                                   "--height", "96", "--format", "bgr24", "-m", "model.onnx"});
    require(shared.stream && shared.sharedMemoryName == "test_02", "Shared-memory mode");
    rejects({"trt_sr", "--stream", "--pipe", "name", "--width", "128", "--height", "96",
             "--format", "bgr24", "-m", "model.onnx"}, "Conflicting stream modes");
    rejects({"trt_sr", "--pipe", "pipe", "--shared-memory", "shared", "--width", "128",
             "--height", "96", "--format", "bgr24", "-m", "model.onnx"}, "Conflicting local modes");
    rejects({"trt_sr", "--pipe", "", "--width", "128", "--height", "96",
             "--format", "bgr24", "-m", "model.onnx"}, "Empty pipe name");
    rejects({"trt_sr", "--shared-memory", "bad/name", "--width", "128", "--height", "96",
             "--format", "bgr24", "-m", "model.onnx"}, "Invalid shared-memory name");
    rejects({"trt_sr", "--pipe", "valid", "-m", "model.onnx"}, "Missing local dimensions");
}

} // namespace
