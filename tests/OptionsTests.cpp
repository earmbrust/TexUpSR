// Copyright (c) 2025 Elden Armbrust. MIT License.
#include "Options.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
Options parseArgs(std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args) argv.push_back(arg.data());
    return parse(static_cast<int>(argv.size()), argv.data());
}

void expectRejected(std::vector<std::string> args) {
    try {
        parseArgs(std::move(args));
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Unexpectedly accepted invalid command-line arguments");
}
}

int main() {
    try {
        const std::vector<std::string> base{"trt_sr", "-i", "input.png", "-o", "out.png", "-m", "model.onnx"};
        auto valid = base;
        valid.insert(valid.end(), {"--tile", "256", "--batch", "2", "--scale", "4", "--workspace", "1.5", "--streams", "3", "--overlap", "16"});
        const Options options = parseArgs(valid);
        if (options.tile != 256 || options.batch != 2 || options.scale != 4 || options.workspace != 1.5 || options.streams != 3)
            throw std::runtime_error("Valid arguments were parsed incorrectly");

        for (const auto& value : {"2junk", "abc", "2147483648", " 2", ""}) {
            auto bad = base;
            bad.insert(bad.end(), {"--scale", value});
            expectRejected(bad);
        }
        for (const auto& value : {"nan", "inf", "-inf", "1.0garbage", "0", "-2"}) {
            auto bad = base;
            bad.insert(bad.end(), {"--workspace", value});
            expectRejected(bad);
        }
        for (const auto& option : {"--streams", "--scale"}) {
            auto bad = base;
            bad.insert(bad.end(), {option, "0"});
            expectRejected(bad);
        }
        auto missing = base;
        missing.push_back("--batch");
        expectRejected(missing);
        std::cout << "Option parsing tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Option parsing test failed: " << e.what() << '\n';
        return 1;
    }
}
