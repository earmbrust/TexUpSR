// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
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

void defaults()
{
    const auto opts = parseArgs(base());
    require(opts.input == "input.png" && opts.output == "out.png" && opts.model == "model.onnx",
            "Required paths");
    require(opts.engine.empty(), "Default engine path");
    require(opts.tile == 0 && opts.batch == 0, "Automatic defaults");
    require(opts.scale == 4 && opts.streams == 2 && opts.overlap == 16, "Integer defaults");
    require(opts.workspace == 4.0, "Workspace default");
    require(!opts.verbose && !opts.quiet && !opts.debug, "Boolean defaults");
}

void allOptions()
{
    auto args = base();
    args.insert(args.end(),
                {"--engine", "engine.plan", "--tile", "256", "--batch", "3", "--scale", "2", "--workspace",
                 "1.25", "--streams", "4", "--overlap", "8", "--verbose", "--quiet", "--debug"});
    const auto o = parseArgs(args);
    require(o.engine == "engine.plan", "Engine path");
    require(o.tile == 256 && o.batch == 3 && o.scale == 2 && o.streams == 4 && o.overlap == 8,
            "Numeric options");
    require(o.workspace == 1.25 && o.verbose && o.quiet && o.debug, "Workspace and flags");

    auto aliases = Args{"trt_sr", "--input", "a", "--output", "b", "--model", "c",    "-e", "d",  "-t",
                        "-1",     "-b",      "0", "-s",       "1", "-w",      "0.25", "-v", "-q", "-d"};
    const auto a = parseArgs(aliases);
    require(a.input == "a" && a.output == "b" && a.model == "c" && a.engine == "d", "Long path names");
    require(a.tile == -1 && a.batch == 0 && a.scale == 1 && a.workspace == 0.25, "Short option names");
    require(a.verbose && a.quiet && a.debug, "Short flags");
}

void requiredArguments()
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

void numericArguments()
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
} // namespace

int main()
{
    try
    {
        defaults();
        allOptions();
        requiredArguments();
        numericArguments();
        std::cout << "Options tests passed\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Options tests failed: " << e.what() << '\n';
        return 1;
    }
}
