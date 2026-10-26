// Copyright (c) 2025-2026 Elden Armbrust. MIT License.
#pragma once
#include <string>
struct Options {
    std::string input, output, model, engine;
    int tile = 0, batch = 0, scale = 4, streams = 2, overlap = 16;
    double workspace = 4.0;
    bool verbose = false, quiet = false, debug = false;
};

void usage();
Options parse(int argc, char** argv);
