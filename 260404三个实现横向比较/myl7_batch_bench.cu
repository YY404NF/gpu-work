#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>
#include <fss/dcf.cuh>
#include <fss/dpf.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/prg/chacha.cuh>

namespace {

constexpr int kInBits = 20;
constexpr int kThreadsPerBlock = 256;

using In = uint32_t;
using Group = fss::group::Uint<uint64_t>;
using CpuDpfPrg = fss::prg::Aes128Mmo<2>;
using CpuDcfPrg = fss::prg::Aes128Mmo<4>;
using GpuDpfPrg = fss::prg::ChaCha<2>;
using GpuDcfPrg = fss::prg::ChaCha<4>;
using Dpf = fss::Dpf<kInBits, Group, CpuDpfPrg, In>;
using Dcf = fss::Dcf<kInBits, Group, CpuDcfPrg, In>;
using GpuDpf = fss::Dpf<kInBits, Group, GpuDpfPrg, In>;
using GpuDcf = fss::Dcf<kInBits, Group, GpuDcfPrg, In>;

__constant__ int kNonce[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

#define CUDA_CHECK(x)                                                                          \
    do {                                                                                       \
        cudaError_t err__ = (x);                                                               \
        if (err__ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                cudaGetErrorString(err__));                                                    \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

struct Inputs {
    std::vector<int4> seeds;
    std::vector<int4> seeds0;
    std::vector<int4> seeds1;
    std::vector<In> alphas;
    std::vector<int4> betas;
    std::vector<In> xs;
};

struct Timing {
    unsigned long long keygen_us;
    unsigned long long eval_p0_us;
    unsigned long long eval_p1_us;
    unsigned long long transfer_in_us;
    unsigned long long transfer_out_us;
    unsigned long long total_us;
};

unsigned long long microsBetween(const std::chrono::high_resolution_clock::time_point &start,
    const std::chrono::high_resolution_clock::time_point &end) {
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int4 makeSeed(uint64_t idx, uint64_t salt) {
    uint64_t a = splitmix64(idx ^ salt);
    uint64_t b = splitmix64(idx + salt);
    int4 out = {static_cast<int>(a), static_cast<int>(a >> 32), static_cast<int>(b),
        static_cast<int>((b >> 32) & ~1ULL)};
    out.w &= ~1;
    return out;
}

std::vector<In> buildDpfAlphas(std::size_t n) {
    std::vector<In> rin(n);
    constexpr In limit = In{1} << kInBits;
    constexpr In mask = limit - 1;
    constexpr uint64_t kStride = 104729;
    for (std::size_t i = 0; i < n; ++i) {
        rin[i] = static_cast<In>((In{10} + static_cast<In>(i * kStride)) & mask);
    }
    return rin;
}

std::vector<In> buildDpfQueries(const std::vector<In> &alphas) {
    std::vector<In> xs(alphas.size());
    constexpr In limit = In{1} << kInBits;
    for (std::size_t i = 0; i < alphas.size(); ++i) {
        xs[i] = (i % 3 == 0 || alphas[i] + 1 >= limit) ? alphas[i] : static_cast<In>(alphas[i] + 1);
    }
    return xs;
}

std::vector<In> buildDcfAlphas(std::size_t n) {
    std::vector<In> rin(n);
    constexpr In limit = In{1} << kInBits;
    constexpr In span = limit - 1;
    constexpr uint64_t kStride = 104729;
    for (std::size_t i = 0; i < n; ++i) {
        rin[i] = static_cast<In>(1 + ((19 + i * kStride) % span));
    }
    return rin;
}

std::vector<In> buildDcfQueries(const std::vector<In> &alphas) {
    std::vector<In> xs(alphas.size());
    constexpr In limit = In{1} << kInBits;
    for (std::size_t i = 0; i < alphas.size(); ++i) {
        if (i % 4 == 0) {
            xs[i] = alphas[i];
        } else if (i % 4 == 1) {
            xs[i] = static_cast<In>(alphas[i] - 1);
        } else {
            xs[i] = (alphas[i] + 1 < limit) ? static_cast<In>(alphas[i] + 1) : alphas[i];
        }
    }
    return xs;
}

Inputs buildInputs(std::string_view primitive, std::size_t n) {
    Inputs in;
    in.seeds.resize(n * 2);
    in.seeds0.resize(n);
    in.seeds1.resize(n);
    in.alphas = (primitive == "dpf") ? buildDpfAlphas(n) : buildDcfAlphas(n);
    in.xs = (primitive == "dpf") ? buildDpfQueries(in.alphas) : buildDcfQueries(in.alphas);
    in.betas.assign(n, int4{1, 0, 0, 0});

    for (std::size_t i = 0; i < n; ++i) {
        in.seeds[i * 2] = makeSeed(i, 0x1111111111111111ULL);
        in.seeds[i * 2 + 1] = makeSeed(i, 0x2222222222222222ULL);
        in.seeds0[i] = in.seeds[i * 2];
        in.seeds1[i] = in.seeds[i * 2 + 1];
    }
    return in;
}

bool expectDpfHit(std::size_t i, const Inputs &in) {
    return in.xs[i] == in.alphas[i];
}

bool expectDcfHit(std::size_t i, const Inputs &in) {
    return in.xs[i] < in.alphas[i];
}

bool isOne(const int4 &v) {
    return v.x == 1 && v.y == 0 && v.z == 0 && v.w == 0;
}

bool isZero(const int4 &v) {
    return v.x == 0 && v.y == 0 && v.z == 0 && v.w == 0;
}

void verifyOutputs(std::string_view primitive, const Inputs &in, const std::vector<int4> &ys0,
    const std::vector<int4> &ys1) {
    const std::size_t check_n = ys0.size() < 16 ? ys0.size() : 16;
    for (std::size_t i = 0; i < check_n; ++i) {
        int4 sum = (Group::From(ys0[i]) + Group::From(ys1[i])).Into();
        bool hit = primitive == "dpf" ? expectDpfHit(i, in) : expectDcfHit(i, in);
        if ((hit && !isOne(sum)) || (!hit && !isZero(sum))) {
            std::fprintf(stderr, "Verification failed at idx=%zu for %.*s\n", i,
                static_cast<int>(primitive.size()), primitive.data());
            std::exit(2);
        }
    }
}

template <typename T>
T *toDevice(const std::vector<T> &host) {
    T *dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, sizeof(T) * host.size()));
    CUDA_CHECK(cudaMemcpy(dev, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice));
    return dev;
}

template <typename T>
std::vector<T> toHost(const T *dev, std::size_t n) {
    std::vector<T> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), dev, sizeof(T) * n, cudaMemcpyDeviceToHost));
    return host;
}

template <typename DpfLike>
__global__ void DpfGenKernel(typename DpfLike::Cw *cws, const int4 *seeds, const In *alphas,
    const int4 *betas, std::size_t n) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    GpuDpfPrg prg(kNonce);
    DpfLike dpf{prg};
    int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
    dpf.Gen(cws + tid * (kInBits + 1), s, alphas[tid], betas[tid]);
}

template <typename DpfLike>
__global__ void DpfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename DpfLike::Cw *cws, const In *xs, std::size_t n) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    GpuDpfPrg prg(kNonce);
    DpfLike dpf{prg};
    ys[tid] = dpf.Eval(party, seeds[tid], cws + tid * (kInBits + 1), xs[tid]);
}

template <typename DcfLike>
__global__ void DcfGenKernel(typename DcfLike::Cw *cws, const int4 *seeds, const In *alphas,
    const int4 *betas, std::size_t n) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    GpuDcfPrg prg(kNonce);
    DcfLike dcf{prg};
    int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
    dcf.Gen(cws + tid * (kInBits + 1), s, alphas[tid], betas[tid]);
}

template <typename DcfLike>
__global__ void DcfEvalKernel(int4 *ys, bool party, const int4 *seeds,
    const typename DcfLike::Cw *cws, const In *xs, std::size_t n) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    GpuDcfPrg prg(kNonce);
    DcfLike dcf{prg};
    ys[tid] = dcf.Eval(party, seeds[tid], cws + tid * (kInBits + 1), xs[tid]);
}

Timing runCpuDpf(const Inputs &in) {
    Timing t{};
    const std::size_t n = in.alphas.size();
    std::vector<Dpf::Cw> cws(n * (kInBits + 1));
    std::vector<int4> ys0(n);
    std::vector<int4> ys1(n);

    unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    const unsigned char *keys[2] = {key0, key1};
    auto ctxs = CpuDpfPrg::CreateCtxs(keys);
    CpuDpfPrg prg(ctxs);
    Dpf dpf{prg};

    const auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        int4 s[2] = {in.seeds[i * 2], in.seeds[i * 2 + 1]};
        dpf.Gen(cws.data() + i * (kInBits + 1), s, in.alphas[i], in.betas[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    t.keygen_us = microsBetween(start, end);

    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        ys0[i] = dpf.Eval(false, in.seeds0[i], cws.data() + i * (kInBits + 1), in.xs[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    t.eval_p0_us = microsBetween(start, end);

    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        ys1[i] = dpf.Eval(true, in.seeds1[i], cws.data() + i * (kInBits + 1), in.xs[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    t.eval_p1_us = microsBetween(start, end);
    t.total_us = microsBetween(total_start, std::chrono::high_resolution_clock::now());

    CpuDpfPrg::FreeCtxs(ctxs);
    verifyOutputs("dpf", in, ys0, ys1);
    return t;
}

Timing runCpuDcf(const Inputs &in) {
    Timing t{};
    const std::size_t n = in.alphas.size();
    std::vector<Dcf::Cw> cws(n * (kInBits + 1));
    std::vector<int4> ys0(n);
    std::vector<int4> ys1(n);

    unsigned char key0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    unsigned char key1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    unsigned char key2[16] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8};
    unsigned char key3[16] = {8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1};
    const unsigned char *keys[4] = {key0, key1, key2, key3};
    auto ctxs = CpuDcfPrg::CreateCtxs(keys);
    CpuDcfPrg prg(ctxs);
    Dcf dcf{prg};

    const auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        int4 s[2] = {in.seeds[i * 2], in.seeds[i * 2 + 1]};
        dcf.Gen(cws.data() + i * (kInBits + 1), s, in.alphas[i], in.betas[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    t.keygen_us = microsBetween(start, end);

    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        ys0[i] = dcf.Eval(false, in.seeds0[i], cws.data() + i * (kInBits + 1), in.xs[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    t.eval_p0_us = microsBetween(start, end);

    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        ys1[i] = dcf.Eval(true, in.seeds1[i], cws.data() + i * (kInBits + 1), in.xs[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    t.eval_p1_us = microsBetween(start, end);
    t.total_us = microsBetween(total_start, std::chrono::high_resolution_clock::now());

    CpuDcfPrg::FreeCtxs(ctxs);
    verifyOutputs("dcf", in, ys0, ys1);
    return t;
}

Timing runGpuDpf(const Inputs &in) {
    Timing t{};
    const std::size_t n = in.alphas.size();
    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);

    CUDA_CHECK(cudaFree(0));
    const auto total_start = std::chrono::high_resolution_clock::now();

    auto start = std::chrono::high_resolution_clock::now();
    int4 *d_seeds = toDevice(in.seeds);
    int4 *d_seeds0 = toDevice(in.seeds0);
    int4 *d_seeds1 = toDevice(in.seeds1);
    In *d_alphas = toDevice(in.alphas);
    int4 *d_betas = toDevice(in.betas);
    In *d_xs = toDevice(in.xs);
    auto end = std::chrono::high_resolution_clock::now();
    t.transfer_in_us = microsBetween(start, end);

    typename GpuDpf::Cw *d_cws = nullptr;
    int4 *d_ys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename GpuDpf::Cw) * (kInBits + 1) * n));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * n));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(ev_start));
    DpfGenKernel<GpuDpf><<<blocks, kThreadsPerBlock>>>(d_cws, d_seeds, d_alphas, d_betas, n);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    t.keygen_us = static_cast<unsigned long long>(ms * 1000.0f);

    CUDA_CHECK(cudaEventRecord(ev_start));
    DpfEvalKernel<GpuDpf><<<blocks, kThreadsPerBlock>>>(d_ys, false, d_seeds0, d_cws, d_xs, n);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    t.eval_p0_us = static_cast<unsigned long long>(ms * 1000.0f);
    auto copy0_start = std::chrono::high_resolution_clock::now();
    auto ys0 = toHost(d_ys, n);
    auto copy0_end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventRecord(ev_start));
    DpfEvalKernel<GpuDpf><<<blocks, kThreadsPerBlock>>>(d_ys, true, d_seeds1, d_cws, d_xs, n);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    t.eval_p1_us = static_cast<unsigned long long>(ms * 1000.0f);
    auto copy1_start = std::chrono::high_resolution_clock::now();
    auto ys1 = toHost(d_ys, n);
    auto copy1_end = std::chrono::high_resolution_clock::now();

    t.transfer_out_us = microsBetween(copy0_start, copy0_end) + microsBetween(copy1_start, copy1_end);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_ys);
    cudaFree(d_cws);
    cudaFree(d_xs);
    cudaFree(d_betas);
    cudaFree(d_alphas);
    cudaFree(d_seeds1);
    cudaFree(d_seeds0);
    cudaFree(d_seeds);

    t.total_us = microsBetween(total_start, std::chrono::high_resolution_clock::now());
    verifyOutputs("dpf", in, ys0, ys1);
    return t;
}

Timing runGpuDcf(const Inputs &in) {
    Timing t{};
    const std::size_t n = in.alphas.size();
    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);

    CUDA_CHECK(cudaFree(0));
    const auto total_start = std::chrono::high_resolution_clock::now();

    auto start = std::chrono::high_resolution_clock::now();
    int4 *d_seeds = toDevice(in.seeds);
    int4 *d_seeds0 = toDevice(in.seeds0);
    int4 *d_seeds1 = toDevice(in.seeds1);
    In *d_alphas = toDevice(in.alphas);
    int4 *d_betas = toDevice(in.betas);
    In *d_xs = toDevice(in.xs);
    auto copy_in_end = std::chrono::high_resolution_clock::now();
    t.transfer_in_us = microsBetween(start, copy_in_end);

    typename GpuDcf::Cw *d_cws = nullptr;
    int4 *d_ys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(typename GpuDcf::Cw) * (kInBits + 1) * n));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * n));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(ev_start));
    DcfGenKernel<GpuDcf><<<blocks, kThreadsPerBlock>>>(d_cws, d_seeds, d_alphas, d_betas, n);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    t.keygen_us = static_cast<unsigned long long>(ms * 1000.0f);

    CUDA_CHECK(cudaEventRecord(ev_start));
    DcfEvalKernel<GpuDcf><<<blocks, kThreadsPerBlock>>>(d_ys, false, d_seeds0, d_cws, d_xs, n);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    t.eval_p0_us = static_cast<unsigned long long>(ms * 1000.0f);
    auto copy0_start = std::chrono::high_resolution_clock::now();
    auto ys0 = toHost(d_ys, n);
    auto copy0_end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventRecord(ev_start));
    DcfEvalKernel<GpuDcf><<<blocks, kThreadsPerBlock>>>(d_ys, true, d_seeds1, d_cws, d_xs, n);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    t.eval_p1_us = static_cast<unsigned long long>(ms * 1000.0f);
    auto copy1_start = std::chrono::high_resolution_clock::now();
    auto ys1 = toHost(d_ys, n);
    auto copy1_end = std::chrono::high_resolution_clock::now();

    t.transfer_out_us = microsBetween(copy0_start, copy0_end) + microsBetween(copy1_start, copy1_end);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_ys);
    cudaFree(d_cws);
    cudaFree(d_xs);
    cudaFree(d_betas);
    cudaFree(d_alphas);
    cudaFree(d_seeds1);
    cudaFree(d_seeds0);
    cudaFree(d_seeds);

    t.total_us = microsBetween(total_start, std::chrono::high_resolution_clock::now());
    verifyOutputs("dcf", in, ys0, ys1);
    return t;
}

void printTiming(std::string_view primitive, std::string_view impl, std::size_t n, const Timing &t) {
    std::printf(
        "primitive=%.*s impl=%.*s bin=%d n=%zu keygen_us=%llu eval_p0_us=%llu eval_p1_us=%llu "
        "transfer_in_us=%llu transfer_out_us=%llu total_us=%llu\n",
        static_cast<int>(primitive.size()),
        primitive.data(),
        static_cast<int>(impl.size()),
        impl.data(),
        kInBits,
        n,
        t.keygen_us,
        t.eval_p0_us,
        t.eval_p1_us,
        t.transfer_in_us,
        t.transfer_out_us,
        t.total_us);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <dpf|dcf> <n>\n", argv[0]);
        return 1;
    }

    const std::string_view primitive = argv[1];
    const std::size_t n = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    if ((primitive != "dpf" && primitive != "dcf") || n == 0) {
        std::fprintf(stderr, "Usage: %s <dpf|dcf> <n>\n", argv[0]);
        return 1;
    }

    const Inputs in = buildInputs(primitive, n);
    if (primitive == "dpf") {
        printTiming(primitive, "myl7_cpu", n, runCpuDpf(in));
        printTiming(primitive, "myl7_gpu", n, runGpuDpf(in));
    } else {
        printTiming(primitive, "myl7_cpu", n, runCpuDcf(in));
        printTiming(primitive, "myl7_gpu", n, runGpuDcf(in));
    }
    return 0;
}
