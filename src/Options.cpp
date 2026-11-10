// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#include "Options.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
int integerValue(const std::string& text, const std::string& option)
{
    size_t consumed = 0;
    try
    {
        int value = std::stoi(text, &consumed);
        if (consumed != text.size() || text.empty() || text.front() == ' ' || text.front() == '\t')
            throw std::runtime_error("Invalid integer for " + option + ": " + text);
        return value;
    }
    catch (const std::invalid_argument&)
    {
        throw std::runtime_error("Invalid integer for " + option + ": " + text);
    }
    catch (const std::out_of_range&)
    {
        throw std::runtime_error("Integer out of range for " + option + ": " + text);
    }
}

double realValue(const std::string& text, const std::string& option)
{
    size_t consumed = 0;
    try
    {
        double value = std::stod(text, &consumed);
        if (consumed != text.size() || text.empty() || text.front() == ' ' || text.front() == '\t' ||
            !std::isfinite(value))
            throw std::runtime_error("Invalid finite number for " + option + ": " + text);
        return value;
    }
    catch (const std::invalid_argument&)
    {
        throw std::runtime_error("Invalid number for " + option + ": " + text);
    }
    catch (const std::out_of_range&)
    {
        throw std::runtime_error("Number out of range for " + option + ": " + text);
    }
}
} // namespace

void usage()
{
    std::cout << "Usage: trt_sr -i input.png -o output.png -m model.onnx [OPTIONS]\n"
              << "  --inspect <path>        CPU-only ONNX inspection\n"
              << "  --build-engine          Generate a TensorRT engine without an input image\n"
              << "  --stream                Process raw BGR24 frames over stdin/stdout\n"
              << "  --pipe <name>           Serve local named pipes/FIFOs\n"
              << "  --shared-memory <name>  Serve local shared-memory frames\n"
              << "  --width <pixels>        Input width for local worker modes\n"
              << "  --height <pixels>       Input height for local worker modes\n"
              << "  --format bgr24          BGR24 pixel format for local workers\n"
              << "  --profile <path>        Optional JSON model pixel/layout profile\n"
              << "  -i, --input <path>       Input image\n"
              << "  -o, --output <path>      Output image\n"
              << "  -m, --model <path>       ONNX model\n"
              << "  -e, --engine <path>      TensorRT engine path\n"
              << "  --overwrite-engine      Rebuild and safely replace the engine\n"
              << "  -t, --tile <size>        Tile size (0 = auto)\n"
              << "  -b, --batch <size>       Batch size (0 = auto)\n"
              << "  -s, --scale <factor>     Upscale factor (default 4)\n"
              << "  -w, --workspace <GB>    TensorRT workspace (default 4)\n"
              << "  --streams <count>        CUDA streams (default 2)\n"
              << "  --overlap <size>         Tile overlap (default 16)\n"
              << "  -v, --verbose            Verbose logging\n"
              << "  -q, --quiet              Quiet logging\n"
              << "  -d, --debug              Save first upscaled tile\n"
              << "  --benchmark              Report performance measurements\n"
              << "  -h, --help               Show help\n";
}

Options parse(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "-h" || a == "--help")
        {
            usage();
            std::exit(0);
        }
        auto val = [&]() -> std::string
        {
            if (++i >= argc)
                throw std::runtime_error("Missing value for " + a);
            return argv[i];
        };
        if (a == "--inspect")
            o.inspect = val();
        else if (a == "--build-engine")
            o.buildEngine = true;
        else if (a == "--stream")
            o.stream = true;
        else if (a == "--pipe")
        {
            o.pipeName = val();
            if (o.pipeName.empty()) throw std::runtime_error("Empty local pipe name");
        }
        else if (a == "--shared-memory")
        {
            o.sharedMemoryName = val();
            if (o.sharedMemoryName.empty()) throw std::runtime_error("Empty shared-memory name");
        }
        else if (a == "--width")
            o.width = integerValue(val(), a);
        else if (a == "--height")
            o.height = integerValue(val(), a);
        else if (a == "--format")
            o.format = val();
        else if (a == "--profile")
            o.profile = val();
        else if (a == "-i" || a == "--input")
            o.input = val();
        else if (a == "-o" || a == "--output")
            o.output = val();
        else if (a == "-m" || a == "--model")
            o.model = val();
        else if (a == "-e" || a == "--engine")
            o.engine = val();
        else if (a == "--overwrite-engine")
            o.overwriteEngine = true;
        else if (a == "-t" || a == "--tile")
            o.tile = integerValue(val(), a);
        else if (a == "-b" || a == "--batch")
            o.batch = integerValue(val(), a);
        else if (a == "-s" || a == "--scale")
            o.scale = integerValue(val(), a);
        else if (a == "-w" || a == "--workspace")
            o.workspace = realValue(val(), a);
        else if (a == "--streams")
            o.streams = integerValue(val(), a);
        else if (a == "--overlap")
            o.overlap = integerValue(val(), a);
        else if (a == "-v" || a == "--verbose")
            o.verbose = true;
        else if (a == "-q" || a == "--quiet")
            o.quiet = true;
        else if (a == "-d" || a == "--debug")
            o.debug = true;
        else if (a == "--benchmark")
            o.benchmark = true;
        else
            throw std::runtime_error("Unknown argument: " + a);
    }
    const bool hasPipe = !o.pipeName.empty();
    const bool hasSharedMemory = !o.sharedMemoryName.empty();
    if ((o.stream ? 1 : 0) + (hasPipe ? 1 : 0) + (hasSharedMemory ? 1 : 0) > 1)
        throw std::runtime_error("Only one worker transport may be selected");
    if (hasPipe || hasSharedMemory)
        o.stream = true;
    if (o.stream)
    {
        const std::string& name = hasPipe ? o.pipeName : o.sharedMemoryName;
        if (hasPipe || hasSharedMemory)
        {
            if (name.size() > 64)
                throw std::runtime_error("Local worker name is too long");
            for (const unsigned char c : name)
            {
                if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '-' || c == '_'))
                    throw std::runtime_error("Invalid local worker name");
            }
        }
        if (!o.inspect.empty() || o.buildEngine || !o.input.empty() || !o.output.empty())
            throw std::runtime_error("--stream cannot be combined with other input/output modes");
        if (o.model.empty() || o.width <= 0 || o.height <= 0 || o.format != "bgr24")
            throw std::runtime_error("--stream requires --model, positive --width/--height and --format bgr24");
        if (o.debug || o.benchmark)
            throw std::runtime_error("--stream does not accept --debug or --benchmark");
    }
    else
    {
        if (o.width != 0 || o.height != 0 || !o.format.empty())
            throw std::runtime_error("--width, --height and --format require --stream");
        if (!o.inspect.empty())
        {
            if (o.buildEngine || !o.input.empty() || !o.output.empty() || !o.model.empty() ||
                !o.profile.empty() || o.overwriteEngine || o.benchmark)
                throw std::runtime_error("--inspect cannot be combined with image processing arguments");
        }
        else if (o.buildEngine)
        {
            if (!o.input.empty() || !o.output.empty())
                throw std::runtime_error("--build-engine does not accept image paths");
            if (o.model.empty() || o.engine.empty())
                throw std::runtime_error("--build-engine requires --model and --engine");
            if (o.tile <= 0 || o.batch <= 0)
                throw std::runtime_error("--build-engine requires positive --tile and --batch values");
            if (o.benchmark || o.debug)
                throw std::runtime_error("--build-engine cannot be combined with --benchmark or --debug");
        }
        else if (o.input.empty() || o.output.empty() || o.model.empty())
            throw std::runtime_error("Input, output, and model are required");
    }
    if (o.scale <= 0 || o.streams <= 0 || o.workspace <= 0 || o.overlap < 0 || o.tile < -1 || o.batch < -1)
        throw std::runtime_error("Invalid numeric argument");
    return o;
}
