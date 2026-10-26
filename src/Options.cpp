// Copyright (c) 2025 Elden Armbrust. MIT License.
#include "Options.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
int integerValue(const std::string& text, const std::string& option) {
    size_t consumed = 0;
    try {
        int value = std::stoi(text, &consumed);
        if (consumed != text.size() || text.empty() || text.front() == ' ' || text.front() == '\t')
            throw std::runtime_error("Invalid integer for " + option + ": " + text);
        return value;
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Invalid integer for " + option + ": " + text);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Integer out of range for " + option + ": " + text);
    }
}

double realValue(const std::string& text, const std::string& option) {
    size_t consumed = 0;
    try {
        double value = std::stod(text, &consumed);
        if (consumed != text.size() || text.empty() || text.front() == ' ' || text.front() == '\t' || !std::isfinite(value))
            throw std::runtime_error("Invalid finite number for " + option + ": " + text);
        return value;
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Invalid number for " + option + ": " + text);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Number out of range for " + option + ": " + text);
    }
}
}

void usage() {
    std::cout << "Usage: trt_sr.exe -i input.png -o output.png -m model.onnx [OPTIONS]\n"
              << "  -i, --input <path>       Input image\n"
              << "  -o, --output <path>      Output image\n"
              << "  -m, --model <path>       ONNX model\n"
              << "  -e, --engine <path>      TensorRT engine path\n"
              << "  -t, --tile <size>        Tile size (0 = auto)\n"
              << "  -b, --batch <size>       Batch size (0 = auto)\n"
              << "  -s, --scale <factor>     Upscale factor (default 4)\n"
              << "  -w, --workspace <GB>    TensorRT workspace (default 4)\n"
              << "  --streams <count>        CUDA streams (default 2)\n"
              << "  --overlap <size>         Tile overlap (default 16)\n"
              << "  -v, --verbose            Verbose logging\n"
              << "  -q, --quiet              Quiet logging\n"
              << "  -d, --debug              Save first upscaled tile\n"
              << "  -h, --help               Show help\n";
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        auto val = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("Missing value for " + a); return argv[i]; };
        if (a == "-i" || a == "--input") o.input = val();
        else if (a == "-o" || a == "--output") o.output = val();
        else if (a == "-m" || a == "--model") o.model = val();
        else if (a == "-e" || a == "--engine") o.engine = val();
        else if (a == "-t" || a == "--tile") o.tile = integerValue(val(), a);
        else if (a == "-b" || a == "--batch") o.batch = integerValue(val(), a);
        else if (a == "-s" || a == "--scale") o.scale = integerValue(val(), a);
        else if (a == "-w" || a == "--workspace") o.workspace = realValue(val(), a);
        else if (a == "--streams") o.streams = integerValue(val(), a);
        else if (a == "--overlap") o.overlap = integerValue(val(), a);
        else if (a == "-v" || a == "--verbose") o.verbose = true;
        else if (a == "-q" || a == "--quiet") o.quiet = true;
        else if (a == "-d" || a == "--debug") o.debug = true;
        else throw std::runtime_error("Unknown argument: " + a);
    }
    if (o.input.empty() || o.output.empty() || o.model.empty()) throw std::runtime_error("Input, output, and model are required");
    if (o.scale <= 0 || o.streams <= 0 || o.workspace <= 0 || o.overlap < 0 || o.tile < -1 || o.batch < -1)
        throw std::runtime_error("Invalid numeric argument");
    return o;
}
