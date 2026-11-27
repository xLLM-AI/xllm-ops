/**
 * Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Correctness + latency check for SigmoidMul against the unfused
// aclnnSigmoid + aclnnMul pair (the kernel-level equivalent of
// torch.sigmoid(x1) * x2 in an eager framework).
//
// Build (inside a CANN container, after installing the custom op package):
//   g++ -O2 -std=c++17 sigmoid_mul_bench.cpp -o sigmoid_mul_bench \
//       -I$CUSTOM_OPP/include -I$ASCEND_HOME_PATH/include \
//       -L$ASCEND_HOME_PATH/lib64 -L$CUSTOM_OPP/lib \
//       -laclnn_sigmoid_mul -lopapi_math -lopapi_nn -lascendcl -lpthread
// Run:
//   ./sigmoid_mul_bench --device 0 --iters 50

#include <acl/acl.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_sigmoid_mul.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void Check(aclError err, const char* what)
{
    if (err != ACL_SUCCESS) {
        std::fprintf(stderr, "[FATAL] %s failed, aclError=%d\n", what, static_cast<int>(err));
        const char* msg = aclGetRecentErrMsg();
        if (msg != nullptr) {
            std::fprintf(stderr, "[FATAL] recent error: %s\n", msg);
        }
        std::exit(1);
    }
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb; // round to nearest even
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

struct Shape {
    int64_t m;
    int64_t n;
    // x1 is [M, 1] instead of [M, N]: out = sigmoid(x1) broadcast * x2.
    bool bcast = false;
};

struct Timing {
    double meanUs;
    double minUs;
};

class EventTimer {
public:
    EventTimer()
    {
        Check(aclrtCreateEvent(&start_), "aclrtCreateEvent");
        Check(aclrtCreateEvent(&end_), "aclrtCreateEvent");
    }
    ~EventTimer()
    {
        aclrtDestroyEvent(start_);
        aclrtDestroyEvent(end_);
    }
    void Begin() { Check(aclrtRecordEvent(start_, stream_), "aclrtRecordEvent"); }
    double End()
    {
        Check(aclrtRecordEvent(end_, stream_), "aclrtRecordEvent");
        Check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        float ms = 0.0f;
        Check(aclrtEventElapsedTime(&ms, start_, end_), "aclrtEventElapsedTime");
        return static_cast<double>(ms) * 1000.0;
    }
    aclrtStream stream_ = nullptr;

private:
    aclrtEvent start_ = nullptr;
    aclrtEvent end_ = nullptr;
};

struct OpCall {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
};

class Tensor {
public:
    Tensor(int64_t m, int64_t n) : bytes_(static_cast<size_t>(m) * n * sizeof(uint16_t))
    {
        dims_ = {m, n};
        strides_ = {n, 1};
        Check(aclrtMalloc(&devPtr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        desc_ = aclCreateTensor(dims_.data(), 2, ACL_BF16, strides_.data(), 0, ACL_FORMAT_ND,
                                dims_.data(), 2, devPtr_);
        if (desc_ == nullptr) {
            std::fprintf(stderr, "[FATAL] aclCreateTensor failed\n");
            std::exit(1);
        }
    }
    ~Tensor()
    {
        if (desc_ != nullptr) {
            aclDestroyTensor(desc_);
        }
        if (devPtr_ != nullptr) {
            aclrtFree(devPtr_);
        }
    }
    void Upload(const std::vector<uint16_t>& host)
    {
        Check(aclrtMemcpy(devPtr_, bytes_, host.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy H2D");
    }
    void Download(std::vector<uint16_t>& host)
    {
        host.resize(bytes_ / sizeof(uint16_t));
        Check(aclrtMemcpy(host.data(), bytes_, devPtr_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy D2H");
    }
    aclTensor* desc() const { return desc_; }
    size_t elements() const { return bytes_ / sizeof(uint16_t); }

private:
    size_t bytes_ = 0;
    void* devPtr_ = nullptr;
    aclTensor* desc_ = nullptr;
    std::vector<int64_t> dims_;
    std::vector<int64_t> strides_;
};

struct Result {
    double maxAbs = 0.0;
    double maxRel = 0.0;
    double meanAbs = 0.0;
    size_t bad = 0;
    size_t total = 0;
};

Result Compare(const std::vector<uint16_t>& got, const std::vector<double>& golden, double rtol,
               double atol)
{
    Result r;
    r.total = got.size();
    double sumAbs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double g = golden[i];
        double a = static_cast<double>(Bf16ToFloat(got[i]));
        double absErr = std::fabs(a - g);
        double relErr = absErr / std::max(std::fabs(g), 1e-12);
        sumAbs += absErr;
        r.maxAbs = std::max(r.maxAbs, absErr);
        r.maxRel = std::max(r.maxRel, relErr);
        if (absErr > atol + rtol * std::fabs(g)) {
            ++r.bad;
        }
    }
    r.meanAbs = got.empty() ? 0.0 : sumAbs / static_cast<double>(got.size());
    return r;
}

struct BenchRow {
    Shape shape;
    double fusedUs;
    double unfusedUs;
    double fusedGBps;
    double unfusedGBps;
};

class Runner {
public:
    Runner(int32_t device, aclrtStream stream) : stream_(stream) { (void)device; }

    // Each aclnn call needs a fresh executor (they are single-use), which matches
    // what a framework pays per eager op call.
    double RunFused(const Tensor& x1, const Tensor& x2, Tensor& out, void* workspace, int iters)
    {
        auto launch = [&]() {
            uint64_t ws = 0;
            aclOpExecutor* exec = nullptr;
            Check(aclnnSigmoidMulGetWorkspaceSize(x1.desc(), x2.desc(), out.desc(), &ws, &exec),
                  "aclnnSigmoidMulGetWorkspaceSize");
            if (ws > 0 && workspace == nullptr) {
                std::fprintf(stderr, "[FATAL] fused op needs %zu workspace bytes\n",
                             static_cast<size_t>(ws));
                std::exit(1);
            }
            Check(aclnnSigmoidMul(workspace, ws, exec, stream_), "aclnnSigmoidMul");
        };
        launch();
        EventTimer timer;
        timer.stream_ = stream_;
        timer.Begin();
        for (int i = 0; i < iters; ++i) {
            launch();
        }
        return timer.End() / iters;
    }

    double RunUnfused(const Tensor& x1, const Tensor& x2, Tensor& tmp, Tensor& out, void* wsSig,
                      void* wsMul, int iters)
    {
        auto launch = [&]() {
            uint64_t wsS = 0;
            uint64_t wsM = 0;
            aclOpExecutor* execS = nullptr;
            aclOpExecutor* execM = nullptr;
            Check(aclnnSigmoidGetWorkspaceSize(x1.desc(), tmp.desc(), &wsS, &execS),
                  "aclnnSigmoidGetWorkspaceSize");
            Check(aclnnMulGetWorkspaceSize(tmp.desc(), x2.desc(), out.desc(), &wsM, &execM),
                  "aclnnMulGetWorkspaceSize");
            if (wsS > 0 && wsSig == nullptr) {
                std::fprintf(stderr, "[FATAL] sigmoid needs %zu workspace bytes\n",
                             static_cast<size_t>(wsS));
                std::exit(1);
            }
            if (wsM > 0 && wsMul == nullptr) {
                std::fprintf(stderr, "[FATAL] mul needs %zu workspace bytes\n",
                             static_cast<size_t>(wsM));
                std::exit(1);
            }
            Check(aclnnSigmoid(wsSig, wsS, execS, stream_), "aclnnSigmoid");
            Check(aclnnMul(wsMul, wsM, execM, stream_), "aclnnMul");
        };
        launch();
        EventTimer timer;
        timer.stream_ = stream_;
        timer.Begin();
        for (int i = 0; i < iters; ++i) {
            launch();
        }
        return timer.End() / iters;
    }

private:
    aclrtStream stream_ = nullptr;
};

std::vector<Shape> DefaultShapes()
{
    std::vector<Shape> shapes;
    for (int64_t m : {1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096}) {
        for (int64_t n : {1024, 2048, 4096}) {
            shapes.push_back({m, n, false});
        }
    }
    for (int64_t m : {1, 16, 32, 4096}) {
        shapes.push_back({m, 2048, true});
    }
    for (int64_t n : {1024, 4096}) {
        shapes.push_back({16, n, true});
    }
    return shapes;
}

std::vector<Shape> ParseShapes(const std::string& spec)
{
    std::vector<Shape> shapes;
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t comma = spec.find(',', pos);
        std::string token = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        bool bcast = false;
        if (!token.empty() && token.back() == 'b') {
            bcast = true;
            token.pop_back();
        }
        size_t x = token.find('x');
        if (x != std::string::npos) {
            shapes.push_back({std::stoll(token.substr(0, x)), std::stoll(token.substr(x + 1)), bcast});
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return shapes;
}

int IntIters(size_t elems, int baseIters)
{
    // Keep total traffic per case roughly constant so large shapes stay cheap.
    const double budget = 2.0e9; // element-visits
    int byBudget = static_cast<int>(budget / static_cast<double>(std::max<size_t>(elems, 1)));
    return std::max(5, std::min(baseIters, byBudget));
}

} // namespace

int main(int argc, char** argv)
{
    int32_t device = 0;
    int iters = 50;
    int warmup = 10;
    int repeats = 3;
    std::string shapeSpec;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (arg == "--shapes" && i + 1 < argc) {
            shapeSpec = argv[++i];
        } else if (arg == "--help") {
            std::printf("usage: %s [--device N] [--iters N] [--warmup N] [--repeat N] "
                        "[--shapes MxN,...]\n",
                        argv[0]);
            return 0;
        }
    }

    std::vector<Shape> shapes = shapeSpec.empty() ? DefaultShapes() : ParseShapes(shapeSpec);
    if (shapes.empty()) {
        std::fprintf(stderr, "[FATAL] no shapes\n");
        return 1;
    }

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::printf("device=%d iters=%d warmup=%d dtype=bf16 op=out=sigmoid(x1)*x2\n", device, iters,
                warmup);
    std::printf("%-24s %10s %10s %10s %10s %10s %10s\n", "inputs(x1*x2)", "fused_us", "unfused_us",
                "speedup", "fused_GB/s", "unfused_GB/s", "max_rel");

    std::vector<BenchRow> rows;
    double worstRel = 0.0;
    size_t worstBad = 0;
    double worstAbs = 0.0;

    for (const Shape& shape : shapes) {
        const size_t elems = static_cast<size_t>(shape.m) * static_cast<size_t>(shape.n);
        const int64_t x1Cols = shape.bcast ? 1 : shape.n;
        const size_t x1Elems = static_cast<size_t>(shape.m) * static_cast<size_t>(x1Cols);
        const std::string label = "[" + std::to_string(shape.m) + "," + std::to_string(x1Cols) +
                                  "]*[" + std::to_string(shape.m) + "," +
                                  std::to_string(shape.n) + "]";
        Tensor x1(shape.m, x1Cols);
        Tensor x2(shape.m, shape.n);
        Tensor out(shape.m, shape.n);
        Tensor tmp(shape.m, x1Cols);

        std::vector<uint16_t> hostX1(x1Elems);
        std::vector<uint16_t> hostX2(elems);
        std::vector<double> golden(elems);
        uint32_t seed = 12345u;
        for (size_t i = 0; i < x1Elems; ++i) {
            seed = seed * 1664525u + 1013904223u;
            float a = (static_cast<float>(seed >> 8) / 16777216.0f) * 12.0f - 6.0f;
            hostX1[i] = FloatToBf16(a);
        }
        for (size_t i = 0; i < elems; ++i) {
            seed = seed * 1664525u + 1013904223u;
            float b = (static_cast<float>(seed >> 8) / 16777216.0f) * 4.0f - 2.0f;
            hostX2[i] = FloatToBf16(b);
        }
        for (size_t i = 0; i < elems; ++i) {
            double av = static_cast<double>(Bf16ToFloat(hostX1[shape.bcast ? i / shape.n : i]));
            double bv = static_cast<double>(Bf16ToFloat(hostX2[i]));
            golden[i] = (1.0 / (1.0 + std::exp(-av))) * bv;
        }
        x1.Upload(hostX1);
        x2.Upload(hostX2);

        uint64_t fusedWs = 0;
        aclOpExecutor* probeExec = nullptr;
        Check(aclnnSigmoidMulGetWorkspaceSize(x1.desc(), x2.desc(), out.desc(), &fusedWs,
                                              &probeExec),
              "aclnnSigmoidMulGetWorkspaceSize(probe)");
        uint64_t sigWs = 0;
        uint64_t mulWs = 0;
        aclOpExecutor* probeSig = nullptr;
        aclOpExecutor* probeMul = nullptr;
        Check(aclnnSigmoidGetWorkspaceSize(x1.desc(), tmp.desc(), &sigWs, &probeSig),
              "aclnnSigmoidGetWorkspaceSize(probe)");
        Check(aclnnMulGetWorkspaceSize(tmp.desc(), x2.desc(), out.desc(), &mulWs, &probeMul),
              "aclnnMulGetWorkspaceSize(probe)");

        void* wsFused = nullptr;
        void* wsSig = nullptr;
        void* wsMul = nullptr;
        if (fusedWs > 0) {
            Check(aclrtMalloc(&wsFused, fusedWs, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc ws");
        }
        if (sigWs > 0) {
            Check(aclrtMalloc(&wsSig, sigWs, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc ws");
        }
        if (mulWs > 0) {
            Check(aclrtMalloc(&wsMul, mulWs, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc ws");
        }

        Runner runner(device, stream);
        const int caseIters = IntIters(elems, iters);
        for (int i = 0; i < warmup; ++i) {
            runner.RunFused(x1, x2, out, wsFused, 1);
            runner.RunUnfused(x1, x2, tmp, out, wsSig, wsMul, 1);
        }
        // A shared NPU gives noisy per-batch times; report the best batch.
        double fusedUs = 1e30;
        double unfusedUs = 1e30;
        for (int r = 0; r < repeats; ++r) {
            fusedUs = std::min(fusedUs, runner.RunFused(x1, x2, out, wsFused, caseIters));
            unfusedUs = std::min(unfusedUs, runner.RunUnfused(x1, x2, tmp, out, wsSig, wsMul, caseIters));
        }
        // Re-run the fused op once more so `out` holds the fused result for the check.
        runner.RunFused(x1, x2, out, wsFused, 1);

        std::vector<uint16_t> hostOut;
        out.Download(hostOut);
        Result cmp = Compare(hostOut, golden, 4.0 / 256.0, 1.0 / 256.0);
        worstRel = std::max(worstRel, cmp.maxRel);
        worstAbs = std::max(worstAbs, cmp.maxAbs);
        worstBad = std::max(worstBad, cmp.bad);
        if (cmp.bad != 0) {
            std::printf("  [warn] %s: %zu/%zu elements exceed bf16 tolerance (max_abs=%.6g)\n",
                        label.c_str(), cmp.bad, cmp.total, cmp.maxAbs);
        }

        const double bytes = static_cast<double>(elems) * sizeof(uint16_t);
        double fusedGBps = 3.0 * bytes / 1e9 / (fusedUs * 1e-6);
        double unfusedGBps = 5.0 * bytes / 1e9 / (unfusedUs * 1e-6);
        rows.push_back({shape, fusedUs, unfusedUs, fusedGBps, unfusedGBps});
        std::printf("%-24s %10.2f %10.2f %10.2fx %10.1f %10.1f %10.2e\n", label.c_str(), fusedUs,
                    unfusedUs, unfusedUs / fusedUs, fusedGBps, unfusedGBps, cmp.maxRel);

        if (wsFused != nullptr) {
            aclrtFree(wsFused);
        }
        if (wsSig != nullptr) {
            aclrtFree(wsSig);
        }
        if (wsMul != nullptr) {
            aclrtFree(wsMul);
        }
    }

    std::printf("\nprecision: max_rel=%.3e max_abs=%.3e max_out_of_tol=%zu (bf16 rtol=1/64, atol=1/256)\n",
                worstRel, worstAbs, worstBad);

    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclrtResetDevice(device), "aclrtResetDevice");
    Check(aclFinalize(), "aclFinalize");
    return 0;
}
