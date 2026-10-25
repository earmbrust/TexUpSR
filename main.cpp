// Copyright (c) 2025 Elden Armbrust. MIT License.
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <future>
#include <deque>
#include <limits>
#include <cstring>
#include <numeric>
namespace fs = std::filesystem;

struct Options {
    std::string input, output, model, engine;
    int tile = 0, batch = 0, scale = 4, streams = 2, overlap = 16;
    double workspace = 4.0;
    bool verbose = false, quiet = false, debug = false;
};
static void usage() {
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
static Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        auto val = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("Missing value for " + a); return argv[i]; };
        if (a == "-i" || a == "--input") o.input = val();
        else if (a == "-o" || a == "--output") o.output = val();
        else if (a == "-m" || a == "--model") o.model = val();
        else if (a == "-e" || a == "--engine") o.engine = val();
        else if (a == "-t" || a == "--tile") o.tile = std::stoi(val());
        else if (a == "-b" || a == "--batch") o.batch = std::stoi(val());
        else if (a == "-s" || a == "--scale") o.scale = std::stoi(val());
        else if (a == "-w" || a == "--workspace") o.workspace = std::stod(val());
        else if (a == "--streams") o.streams = std::stoi(val());
        else if (a == "--overlap") o.overlap = std::stoi(val());
        else if (a == "-v" || a == "--verbose") o.verbose = true;
        else if (a == "-q" || a == "--quiet") o.quiet = true;
        else if (a == "-d" || a == "--debug") o.debug = true;
        else throw std::runtime_error("Unknown argument: " + a);
    }
    if (o.input.empty() || o.output.empty() || o.model.empty()) throw std::runtime_error("Input, output, and model are required");
    if (o.scale <= 0 || o.streams <= 0 || o.workspace <= 0 || o.overlap < 0 || o.tile < -1 || o.batch < -1) throw std::runtime_error("Invalid numeric argument");
    return o;
}
struct Logger final : nvinfer1::ILogger {
    bool verbose = false;
    void log(Severity s, const char* msg) noexcept override {
        if (s <= (verbose ? Severity::kINFO : Severity::kWARNING)) std::cerr << "TensorRT: " << msg << '\n';
    }
};
static void cudaCheck(cudaError_t result, const char* where) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(result));
}
template <typename T> struct TrtDeleter { void operator()(T* p) const { delete p; } };
template <typename T> using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;
struct DeviceBuffer {
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { cudaCheck(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~DeviceBuffer() { if (p) cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cudaCheck(cudaStreamCreate(&value), "cudaStreamCreate"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
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

struct Tile { int x, y; };
struct BatchResult { std::vector<Tile> tiles; std::vector<cv::Mat> images; };
static size_t tensorElementBytes(nvinfer1::DataType type) {
    if (type == nvinfer1::DataType::kHALF) return sizeof(__half);
    if (type == nvinfer1::DataType::kFLOAT) return sizeof(float);
    throw std::runtime_error("Unsupported tensor datatype (expected FP16 or FP32)");
}
struct PinnedBuffer {
    void* ptr = nullptr;
    explicit PinnedBuffer(size_t bytes) { cudaCheck(cudaMallocHost(&ptr, bytes), "cudaMallocHost"); }
    ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};
struct Worker {
    Stream stream;
    TrtPtr<nvinfer1::IExecutionContext> ctx;
    DeviceBuffer input, output;
    PinnedBuffer hostIn, hostOut;
    Worker(nvinfer1::ICudaEngine& engine, size_t inCount, size_t outCount,
           nvinfer1::DataType inType, nvinfer1::DataType outType)
        : ctx(engine.createExecutionContext()), input(inCount * tensorElementBytes(inType)),
          output(outCount * tensorElementBytes(outType)),
          hostIn(inCount * tensorElementBytes(inType)),
          hostOut(outCount * tensorElementBytes(outType)) {
        if (!ctx) throw std::runtime_error("Cannot create TensorRT execution context");
    }
};
static void fillInput(const cv::Mat& src, Tile tilePos, int tile, int overlap, float* dst) {
    const size_t plane = static_cast<size_t>(tile) * tile;
    for (int y = 0; y < tile; ++y) {
        const int sy = std::clamp(tilePos.y + y - overlap, 0, src.rows - 1);
        const auto* row = src.ptr<cv::Vec3b>(sy);
        for (int x = 0; x < tile; ++x) {
            const auto& p = row[std::clamp(tilePos.x + x - overlap, 0, src.cols - 1)];
            const size_t k = static_cast<size_t>(y) * tile + x;
            dst[k] = p[2] / 255.f;
            dst[plane + k] = p[1] / 255.f;
            dst[2 * plane + k] = p[0] / 255.f;
        }
    }
}
static BatchResult runBatch(Worker& w, const cv::Mat& image, const std::vector<Tile>& tiles,
                            int tile, int batch, int scale, int overlap,
                            const std::string& inName, const std::string& outName,
                            nvinfer1::DataType inType, nvinfer1::DataType outType) {
    const size_t inPlane = static_cast<size_t>(tile) * tile;
    const size_t inPer = 3 * inPlane;
    const int outTile = tile * scale;
    const size_t outPlane = static_cast<size_t>(outTile) * outTile;
    const size_t outPer = 3 * outPlane;
    // A fixed-shape engine always receives the complete configured batch; pad the final batch.
    // Image preprocessing stays FP32; pack into the exact input tensor datatype.
    // FP16 conversion is performed on the CPU once per tile, avoiding a GPU conversion kernel.
    std::vector<float> prepared(inPer * batch, 0.f);
    for (size_t i = 0; i < tiles.size(); ++i)
        fillInput(image, tiles[i], tile, overlap, prepared.data() + i * inPer);
    if (inType == nvinfer1::DataType::kFLOAT) {
        std::memcpy(w.hostIn.ptr, prepared.data(), prepared.size() * sizeof(float));
    } else {
        auto* packed = static_cast<__half*>(w.hostIn.ptr);
        for (size_t i = 0; i < prepared.size(); ++i) packed[i] = __float2half(prepared[i]);
    }
    if (!w.ctx->setInputShape(inName.c_str(), nvinfer1::Dims4(batch, 3, tile, tile)))
        throw std::runtime_error("TensorRT setInputShape failed");
    auto dims = w.ctx->getTensorShape(outName.c_str());
    if (dims.nbDims != 4 || dims.d[0] != batch || dims.d[1] != 3 ||
        dims.d[2] != outTile || dims.d[3] != outTile)
        throw std::runtime_error("Model output shape does not match requested scale");
    cudaCheck(cudaMemcpyAsync(w.input.p, w.hostIn.ptr, inPer * batch * tensorElementBytes(inType),
                              cudaMemcpyHostToDevice, w.stream.value), "H2D batch");
    if (!w.ctx->setTensorAddress(inName.c_str(), w.input.p) ||
        !w.ctx->setTensorAddress(outName.c_str(), w.output.p))
        throw std::runtime_error("Cannot bind TensorRT tensor addresses");
    if (!w.ctx->enqueueV3(w.stream.value)) throw std::runtime_error("TensorRT enqueueV3 failed");
    cudaCheck(cudaMemcpyAsync(w.hostOut.ptr, w.output.p, outPer * batch * tensorElementBytes(outType),
                              cudaMemcpyDeviceToHost, w.stream.value), "D2H batch");
    cudaCheck(cudaStreamSynchronize(w.stream.value), "Batch stream synchronization");
    BatchResult result; result.tiles = tiles;
    for (size_t i = 0; i < tiles.size(); ++i) {
        cv::Mat sr(outTile, outTile, CV_32FC3);
        const float* data32 = outType == nvinfer1::DataType::kFLOAT
            ? static_cast<const float*>(w.hostOut.ptr) + i * outPer : nullptr;
        const __half* data16 = outType == nvinfer1::DataType::kHALF
            ? static_cast<const __half*>(w.hostOut.ptr) + i * outPer : nullptr;
        auto sample = [&](size_t index) -> float {
            return data32 ? data32[index] : __half2float(data16[index]);
        };
        for (int y = 0; y < outTile; ++y) {
            auto* row = sr.ptr<cv::Vec3f>(y);
            for (int x = 0; x < outTile; ++x) {
                const size_t k = static_cast<size_t>(y) * outTile + x;
                row[x] = cv::Vec3f(sample(2 * outPlane + k), sample(outPlane + k), sample(k));
            }
        }
        result.images.push_back(std::move(sr));
    }
    return result;
}
// Smooth feather weights: adjacent overlapping tiles blend instead of leaving hard seams.
// Weight tapers only at interior tile edges; the image's outside edges remain fully weighted.
static void blendTile(cv::Mat& accum, cv::Mat& weights, const cv::Mat& sr,
                      Tile pos, int tile, int scale, int overlap, int srcW, int srcH) {
    const int x0 = pos.x - overlap, y0 = pos.y - overlap;
    const int feather = std::max(1, overlap * scale);
    for (int y = 0; y < sr.rows; ++y) {
        const int gy = y0 * scale + y;
        if (gy < 0 || gy >= accum.rows) continue;
        auto* a = accum.ptr<cv::Vec3f>(gy);
        auto* w = weights.ptr<float>(gy);
        const auto* src = sr.ptr<cv::Vec3f>(y);
        float wy = 1.f;
        if (y0 > 0) wy = std::min(wy, static_cast<float>(y + 1) / feather);
        if (y0 + tile < srcH) wy = std::min(wy, static_cast<float>(sr.rows - y) / feather);
        for (int x = 0; x < sr.cols; ++x) {
            const int gx = x0 * scale + x;
            if (gx < 0 || gx >= accum.cols) continue;
            float wx = 1.f;
            if (x0 > 0) wx = std::min(wx, static_cast<float>(x + 1) / feather);
            if (x0 + tile < srcW) wx = std::min(wx, static_cast<float>(sr.cols - x) / feather);
            const float weight = std::max(0.00001f, wx * wy);
            a[gx] += src[x] * weight;
            w[gx] += weight;
        }
    }
}
static int autoTile(size_t freeBytes, int streams, int overlap) {
    // Conservative choices leave memory for TensorRT weights and workspace.
    const size_t perStream = freeBytes / static_cast<size_t>(std::max(1, streams));
    const int candidate = perStream >= (size_t(2) << 30) ? 512 : perStream >= (size_t(768) << 20) ? 256 : 128;
    return std::max(candidate, overlap * 2 + 32);
}
static int autoBatch(size_t freeBytes, int tile, int scale, int streams, double workspaceGB) {
    // Activation memory is model-dependent; reserve most VRAM for TensorRT's internals.
    const size_t reserve = static_cast<size_t>(workspaceGB * (1ull << 30));
    const size_t available = freeBytes > reserve ? freeBytes - reserve : freeBytes / 4;
    const double floatsPerTile = 3.0 * tile * tile * (1.0 + scale * scale);
    const double conservative = floatsPerTile * sizeof(float) * 8.0 * std::max(1, streams);
    return std::clamp(static_cast<int>(available / std::max(1.0, conservative)), 1, 8);
}
int main(int argc, char** argv) {
    try {
        Options o = parse(argc, argv);
        Logger logger; logger.verbose = o.verbose;
        cv::Mat image = cv::imread(o.input, cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Cannot load input image");
        size_t freeBytes = 0, totalBytes = 0;
        cudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo");
        const int tile = o.tile > 0 ? o.tile : autoTile(freeBytes, o.streams, o.overlap);
        const int batch = o.batch > 0 ? o.batch : autoBatch(freeBytes, tile, o.scale, o.streams, o.workspace);
        if (tile <= 2 * o.overlap || tile <= 0 || batch <= 0) throw std::runtime_error("Invalid tile/overlap/batch combination");
        if (static_cast<int64_t>(image.cols) * o.scale > INT_MAX ||
            static_cast<int64_t>(image.rows) * o.scale > INT_MAX ||
            static_cast<int64_t>(tile) * o.scale > INT_MAX)
            throw std::runtime_error("Upscaled dimensions overflow int");
        if (!o.quiet) std::cout << "Tile " << tile << ", batch " << batch << ", streams " << o.streams << "\n";
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
        const size_t inCount = static_cast<size_t>(batch) * 3 * tile * tile;
        const size_t outCount = inCount * static_cast<size_t>(o.scale) * o.scale;
        std::vector<std::unique_ptr<Worker>> workers;
        for (int i = 0; i < o.streams; ++i)
            workers.emplace_back(std::make_unique<Worker>(*engine, inCount, outCount, inType, outType));
        const int width = image.cols * o.scale, height = image.rows * o.scale;
        cv::Mat accum(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
        cv::Mat weights(height, width, CV_32FC1, cv::Scalar(0));
        const int stride = tile - 2 * o.overlap;
        std::vector<Tile> all;
        for (int y = 0; y < image.rows; y += stride)
            for (int x = 0; x < image.cols; x += stride) all.push_back({x, y});
        std::deque<std::future<BatchResult>> jobs;
        bool debugSaved = false;
        size_t done = 0;
        auto consume = [&](std::future<BatchResult>& f) {
            BatchResult result = f.get();
            for (size_t j = 0; j < result.tiles.size(); ++j) {
                if (o.debug && !debugSaved) {
                    cv::Mat tmp; result.images[j].convertTo(tmp, CV_8UC3, 255.0);
                    if (!cv::imwrite("debug_first_tile.png", tmp)) throw std::runtime_error("Cannot save debug tile");
                    debugSaved = true;
                }
                blendTile(accum, weights, result.images[j], result.tiles[j], tile, o.scale,
                          o.overlap, image.cols, image.rows);
                ++done;
            }
            if (!o.quiet) std::cout << "Processed " << done << "/" << all.size() << " tiles\r" << std::flush;
        };
        // Each worker owns an independent TensorRT execution context, CUDA stream,
        // persistent device buffers and page-locked staging buffers.
        size_t group = 0;
        for (size_t start = 0; start < all.size(); start += static_cast<size_t>(batch), ++group) {
            const size_t end = std::min(start + static_cast<size_t>(batch), all.size());
            if (jobs.size() == workers.size()) { consume(jobs.front()); jobs.pop_front(); }
            std::vector<Tile> tiles(all.begin() + start, all.begin() + end);
            Worker* worker = workers[group % workers.size()].get();
            jobs.push_back(std::async(std::launch::async, [worker, &image, tiles = std::move(tiles), tile, batch, &o, &inName, &outName, inType, outType]() {
                return runBatch(*worker, image, tiles, tile, batch, o.scale, o.overlap, inName, outName, inType, outType);
            }));
        }
        while (!jobs.empty()) { consume(jobs.front()); jobs.pop_front(); }
        for (int y = 0; y < height; ++y) {
            auto* row = accum.ptr<cv::Vec3f>(y);
            const auto* w = weights.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                if (w[x] <= 0.f) throw std::runtime_error("Uncovered output pixel during tile blending");
                row[x] /= w[x];
            }
        }
        cv::Mat output; accum.convertTo(output, CV_8UC3, 255.0);
        if (fs::path(o.output).has_parent_path()) fs::create_directories(fs::path(o.output).parent_path());
        if (!cv::imwrite(o.output, output)) throw std::runtime_error("Could not write output image");
        if (!o.quiet) std::cout << "\nWrote " << o.output << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
