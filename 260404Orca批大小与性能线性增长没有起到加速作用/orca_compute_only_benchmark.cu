#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "runtime/standalone_runtime.h"
#include "fss/gpu_dpf.h"
#include "fss/gpu_dcf.h"
#include "gpu/helper_cuda.h"

using T = u64;

namespace
{

struct TimingPair
{
    double p0_us = 0.0;
    double p1_us = 0.0;
};

struct DpfDeviceBuffers
{
    T *d_rin = nullptr;
    T *d_x = nullptr;
    AESBlock *d_s0 = nullptr;
    AESBlock *d_s1 = nullptr;
    AESBlock *d_scw = nullptr;
    AESBlock *d_l0 = nullptr;
    AESBlock *d_l1 = nullptr;
    u32 *d_tR = nullptr;
    u32 *d_out = nullptr;
    int thread_blocks = 0;
    u64 out_words = 0;
};

struct DcfDeviceBuffers
{
    T *d_rin = nullptr;
    T *d_x = nullptr;
    AESBlock *d_s0 = nullptr;
    AESBlock *d_s1 = nullptr;
    AESBlock *d_scw = nullptr;
    AESBlock *d_leaves = nullptr;
    u32 *d_vcw = nullptr;
    u32 *d_out = nullptr;
    int thread_blocks = 0;
    u64 out_words = 0;
};

static void print_usage(const char *prog)
{
    std::fprintf(
        stderr,
        "Usage: %s <mode:dpf|dcf> <bin> <chunk_n> <chunks> [bout]\n",
        prog);
}

static std::vector<T> build_dpf_rin(int bin, int n)
{
    std::vector<T> rin(n);
    if (bin == 64)
    {
        for (int i = 0; i < n; ++i)
            rin[i] = T(10) + T(2) * i;
        return rin;
    }

    const T limit = T(1) << bin;
    const T mask = limit - 1;
    constexpr T k_stride = 104729;
    for (int i = 0; i < n; ++i)
        rin[i] = (T(10) + T(i) * k_stride) & mask;
    return rin;
}

static std::vector<T> build_dpf_queries(int bin, const std::vector<T> &rin)
{
    std::vector<T> x(rin.size());
    const T limit = (bin == 64) ? ~T(0) : (T(1) << bin);
    for (std::size_t i = 0; i < rin.size(); ++i)
        x[i] = (i % 3 == 0 || rin[i] + 1 >= limit) ? rin[i] : (rin[i] + 1);
    return x;
}

static std::vector<T> build_dcf_rin(int bin, int n)
{
    std::vector<T> rin(n);
    if (bin == 64)
    {
        for (int i = 0; i < n; ++i)
            rin[i] = T(20) + T(2) * i;
        return rin;
    }

    const T limit = T(1) << bin;
    const T span = limit - 1;
    constexpr T k_stride = 104729;
    for (int i = 0; i < n; ++i)
        rin[i] = T(1) + ((T(19) + T(i) * k_stride) % span);
    return rin;
}

static std::vector<T> build_dcf_queries(int bin, const std::vector<T> &rin)
{
    std::vector<T> x(rin.size());
    const T limit = (bin == 64) ? ~T(0) : (T(1) << bin);
    for (std::size_t i = 0; i < rin.size(); ++i)
    {
        if (i % 4 == 0)
            x[i] = rin[i];
        else if (i % 4 == 1)
            x[i] = rin[i] - 1;
        else
            x[i] = (rin[i] + 1 < limit) ? (rin[i] + 1) : rin[i];
    }
    return x;
}

static double elapsed_us(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms) * 1000.0;
}

static void create_events(cudaEvent_t &start, cudaEvent_t &stop)
{
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
}

static void destroy_events(cudaEvent_t start, cudaEvent_t stop)
{
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}

static DpfDeviceBuffers alloc_dpf_buffers(int bin, int n)
{
    DpfDeviceBuffers bufs;
    const u64 mem_scw = u64(n) * u64(bin - LOG_AES_BLOCK_LEN) * sizeof(AESBlock);
    const u64 mem_l = u64(n) * sizeof(AESBlock);
    const u64 mem_t = (((u64(n) - 1) / PACKING_SIZE) + 1) * sizeof(PACK_TYPE) * u64(bin - LOG_AES_BLOCK_LEN);
    const u64 mem_out = (((u64(n) - 1) / PACKING_SIZE) + 1) * sizeof(PACK_TYPE);

    auto rin = build_dpf_rin(bin, n);
    auto x = build_dpf_queries(bin, rin);

    bufs.d_rin = gpu_mpc::standalone::detail::copyVectorToGpu(rin);
    bufs.d_x = gpu_mpc::standalone::detail::copyVectorToGpu(x);
    bufs.d_s0 = randomAESBlockOnGpu(n);
    bufs.d_s1 = randomAESBlockOnGpu(n);
    bufs.d_scw = reinterpret_cast<AESBlock *>(gpuMalloc(mem_scw));
    bufs.d_l0 = reinterpret_cast<AESBlock *>(gpuMalloc(mem_l));
    bufs.d_l1 = reinterpret_cast<AESBlock *>(gpuMalloc(mem_l));
    bufs.d_tR = reinterpret_cast<u32 *>(gpuMalloc(mem_t));
    bufs.d_out = reinterpret_cast<u32 *>(gpuMalloc(mem_out));
    bufs.thread_blocks = (n - 1) / 256 + 1;
    bufs.out_words = mem_out / sizeof(PACK_TYPE);
    return bufs;
}

static void free_dpf_buffers(DpfDeviceBuffers &bufs)
{
    gpuFree(bufs.d_rin);
    gpuFree(bufs.d_x);
    gpuFree(bufs.d_s0);
    gpuFree(bufs.d_s1);
    gpuFree(bufs.d_scw);
    gpuFree(bufs.d_l0);
    gpuFree(bufs.d_l1);
    gpuFree(bufs.d_tR);
    gpuFree(bufs.d_out);
}

static DcfDeviceBuffers alloc_dcf_buffers(int bin, int bout, int n)
{
    DcfDeviceBuffers bufs;
    const int elems_per_block = AES_BLOCK_LEN_IN_BITS / bout;
    const int new_bin = bin - static_cast<int>(std::log2(elems_per_block));
    const u64 mem_scw = u64(n) * u64(new_bin) * sizeof(AESBlock);
    const u64 mem_l = 2 * u64(n) * sizeof(AESBlock);
    const u64 mem_v = (((u64(bout) * u64(n) - 1) / PACKING_SIZE) + 1) * sizeof(PACK_TYPE) * u64(new_bin - 1);
    const u64 mem_out = (((u64(bout) * u64(n) - 1) / PACKING_SIZE) + 1) * sizeof(PACK_TYPE);

    auto rin = build_dcf_rin(bin, n);
    auto x = build_dcf_queries(bin, rin);

    bufs.d_rin = gpu_mpc::standalone::detail::copyVectorToGpu(rin);
    bufs.d_x = gpu_mpc::standalone::detail::copyVectorToGpu(x);
    bufs.d_s0 = randomAESBlockOnGpu(n);
    bufs.d_s1 = randomAESBlockOnGpu(n);
    bufs.d_scw = reinterpret_cast<AESBlock *>(gpuMalloc(mem_scw));
    bufs.d_leaves = reinterpret_cast<AESBlock *>(gpuMalloc(mem_l));
    bufs.d_vcw = reinterpret_cast<u32 *>(gpuMalloc(mem_v));
    bufs.d_out = reinterpret_cast<u32 *>(gpuMalloc(mem_out));
    bufs.thread_blocks = (n - 1) / 256 + 1;
    bufs.out_words = mem_out / sizeof(PACK_TYPE);
    return bufs;
}

static void free_dcf_buffers(DcfDeviceBuffers &bufs)
{
    gpuFree(bufs.d_rin);
    gpuFree(bufs.d_x);
    gpuFree(bufs.d_s0);
    gpuFree(bufs.d_s1);
    gpuFree(bufs.d_scw);
    gpuFree(bufs.d_leaves);
    gpuFree(bufs.d_vcw);
    gpuFree(bufs.d_out);
}

static TimingPair benchmark_dpf_keygen(AESGlobalContext *aes, int bin, int n, int chunks, DpfDeviceBuffers &bufs)
{
    TimingPair t;
    cudaEvent_t start, stop;
    create_events(start, stop);

    keyGenDPFTreeKernel<<<bufs.thread_blocks, 256>>>(SERVER0, bin, n, bufs.d_rin, bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_l0, bufs.d_l1, bufs.d_tR, *aes, false);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < chunks; ++i)
        keyGenDPFTreeKernel<<<bufs.thread_blocks, 256>>>(SERVER0, bin, n, bufs.d_rin, bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_l0, bufs.d_l1, bufs.d_tR, *aes, false);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    t.p0_us = elapsed_us(start, stop);

    keyGenDPFTreeKernel<<<bufs.thread_blocks, 256>>>(SERVER1, bin, n, bufs.d_rin, bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_l0, bufs.d_l1, bufs.d_tR, *aes, false);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < chunks; ++i)
        keyGenDPFTreeKernel<<<bufs.thread_blocks, 256>>>(SERVER1, bin, n, bufs.d_rin, bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_l0, bufs.d_l1, bufs.d_tR, *aes, false);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    t.p1_us = elapsed_us(start, stop);

    destroy_events(start, stop);
    return t;
}

static TimingPair benchmark_dpf_eval(AESGlobalContext *aes, int bin, int n, int chunks, DpfDeviceBuffers &bufs)
{
    TimingPair t;
    cudaEvent_t start, stop;
    create_events(start, stop);

    keyGenDPFTreeKernel<<<bufs.thread_blocks, 256>>>(SERVER0, bin, n, bufs.d_rin, bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_l0, bufs.d_l1, bufs.d_tR, *aes, false);
    checkCudaErrors(cudaDeviceSynchronize());
    dpfTreeEval<T, doDpf><<<bufs.thread_blocks, 256>>>(SERVER0, bin, n, bufs.d_x, bufs.d_scw, bufs.d_l0, bufs.d_l1, bufs.d_tR, bufs.d_out, bufs.out_words, *aes);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < chunks; ++i)
        dpfTreeEval<T, doDpf><<<bufs.thread_blocks, 256>>>(SERVER0, bin, n, bufs.d_x, bufs.d_scw, bufs.d_l0, bufs.d_l1, bufs.d_tR, bufs.d_out, bufs.out_words, *aes);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    t.p0_us = elapsed_us(start, stop);

    keyGenDPFTreeKernel<<<bufs.thread_blocks, 256>>>(SERVER1, bin, n, bufs.d_rin, bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_l0, bufs.d_l1, bufs.d_tR, *aes, false);
    checkCudaErrors(cudaDeviceSynchronize());
    dpfTreeEval<T, doDpf><<<bufs.thread_blocks, 256>>>(SERVER1, bin, n, bufs.d_x, bufs.d_scw, bufs.d_l0, bufs.d_l1, bufs.d_tR, bufs.d_out, bufs.out_words, *aes);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < chunks; ++i)
        dpfTreeEval<T, doDpf><<<bufs.thread_blocks, 256>>>(SERVER1, bin, n, bufs.d_x, bufs.d_scw, bufs.d_l0, bufs.d_l1, bufs.d_tR, bufs.d_out, bufs.out_words, *aes);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    t.p1_us = elapsed_us(start, stop);

    destroy_events(start, stop);
    return t;
}

static TimingPair benchmark_dcf_keygen(AESGlobalContext *aes, int bin, int bout, int n, int chunks, DcfDeviceBuffers &bufs)
{
    TimingPair t;
    cudaEvent_t start, stop;
    create_events(start, stop);

    dcf::keyGenDCFKernel<<<bufs.thread_blocks, 256>>>(SERVER0, bin, bout, n, bufs.d_rin, u64(1), bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_vcw, bufs.d_leaves, *aes, true);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < chunks; ++i)
        dcf::keyGenDCFKernel<<<bufs.thread_blocks, 256>>>(SERVER0, bin, bout, n, bufs.d_rin, u64(1), bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_vcw, bufs.d_leaves, *aes, true);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    t.p0_us = elapsed_us(start, stop);

    dcf::keyGenDCFKernel<<<bufs.thread_blocks, 256>>>(SERVER1, bin, bout, n, bufs.d_rin, u64(1), bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_vcw, bufs.d_leaves, *aes, true);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < chunks; ++i)
        dcf::keyGenDCFKernel<<<bufs.thread_blocks, 256>>>(SERVER1, bin, bout, n, bufs.d_rin, u64(1), bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_vcw, bufs.d_leaves, *aes, true);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    t.p1_us = elapsed_us(start, stop);

    destroy_events(start, stop);
    return t;
}

static TimingPair benchmark_dcf_eval(AESGlobalContext *aes, int bin, int bout, int n, int chunks, DcfDeviceBuffers &bufs)
{
    TimingPair t;
    cudaEvent_t start, stop;
    create_events(start, stop);

    dcf::keyGenDCFKernel<<<bufs.thread_blocks, 256>>>(SERVER0, bin, bout, n, bufs.d_rin, u64(1), bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_vcw, bufs.d_leaves, *aes, true);
    checkCudaErrors(cudaDeviceSynchronize());
    dcf::doDcf<T, 1, dcf::idPrologue, dcf::idEpilogue><<<bufs.thread_blocks, 256>>>(bin, bout, SERVER0, n, bufs.d_x, bufs.d_scw, bufs.d_vcw, bufs.d_leaves, bufs.d_out, bufs.out_words, *aes);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < chunks; ++i)
        dcf::doDcf<T, 1, dcf::idPrologue, dcf::idEpilogue><<<bufs.thread_blocks, 256>>>(bin, bout, SERVER0, n, bufs.d_x, bufs.d_scw, bufs.d_vcw, bufs.d_leaves, bufs.d_out, bufs.out_words, *aes);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    t.p0_us = elapsed_us(start, stop);

    dcf::keyGenDCFKernel<<<bufs.thread_blocks, 256>>>(SERVER1, bin, bout, n, bufs.d_rin, u64(1), bufs.d_s0, bufs.d_s1, bufs.d_scw, bufs.d_vcw, bufs.d_leaves, *aes, true);
    checkCudaErrors(cudaDeviceSynchronize());
    dcf::doDcf<T, 1, dcf::idPrologue, dcf::idEpilogue><<<bufs.thread_blocks, 256>>>(bin, bout, SERVER1, n, bufs.d_x, bufs.d_scw, bufs.d_vcw, bufs.d_leaves, bufs.d_out, bufs.out_words, *aes);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < chunks; ++i)
        dcf::doDcf<T, 1, dcf::idPrologue, dcf::idEpilogue><<<bufs.thread_blocks, 256>>>(bin, bout, SERVER1, n, bufs.d_x, bufs.d_scw, bufs.d_vcw, bufs.d_leaves, bufs.d_out, bufs.out_words, *aes);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    t.p1_us = elapsed_us(start, stop);

    destroy_events(start, stop);
    return t;
}

static void print_result(const char *mode, int bin, int bout, int chunk_n, int chunks, const TimingPair &keygen, const TimingPair &eval)
{
    const double total_n = static_cast<double>(chunk_n) * static_cast<double>(chunks);
    const double keygen_avg = (keygen.p0_us + keygen.p1_us) / 2.0;
    const double eval_avg = (eval.p0_us + eval.p1_us) / 2.0;

    std::printf(
        "compute_only benchmark finished\n"
        "  mode: %s\n"
        "  bin: %d bit\n"
        "  bout: %d bit\n"
        "  chunk_n: %d elem\n"
        "  chunks: %d\n"
        "  total_n: %.0f elem\n"
        "  keygen_p0_kernel: %.3f us\n"
        "  keygen_p1_kernel: %.3f us\n"
        "  keygen_avg_kernel: %.3f us\n"
        "  eval_p0_kernel: %.3f us\n"
        "  eval_p1_kernel: %.3f us\n"
        "  eval_avg_kernel: %.3f us\n"
        "  keygen_avg_us_per_elem: %.9f\n"
        "  eval_avg_us_per_elem: %.9f\n",
        mode,
        bin,
        bout,
        chunk_n,
        chunks,
        total_n,
        keygen.p0_us,
        keygen.p1_us,
        keygen_avg,
        eval.p0_us,
        eval.p1_us,
        eval_avg,
        keygen_avg / total_n,
        eval_avg / total_n);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6)
    {
        print_usage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];
    const int bin = std::atoi(argv[2]);
    const int chunk_n = std::atoi(argv[3]);
    const int chunks = std::atoi(argv[4]);
    const int bout = (argc == 6) ? std::atoi(argv[5]) : 1;

    if ((mode != "dpf" && mode != "dcf") || bin <= 7 || bin > 64 || chunk_n <= 0 || chunks <= 0 || bout <= 0 || bout > 64)
    {
        print_usage(argv[0]);
        return 1;
    }

    gpu_mpc::standalone::Runtime runtime;

    if (mode == "dpf")
    {
        auto bufs = alloc_dpf_buffers(bin, chunk_n);
        auto keygen = benchmark_dpf_keygen(runtime.aes(), bin, chunk_n, chunks, bufs);
        auto eval = benchmark_dpf_eval(runtime.aes(), bin, chunk_n, chunks, bufs);
        print_result("dpf", bin, 1, chunk_n, chunks, keygen, eval);
        free_dpf_buffers(bufs);
    }
    else
    {
        auto bufs = alloc_dcf_buffers(bin, bout, chunk_n);
        auto keygen = benchmark_dcf_keygen(runtime.aes(), bin, bout, chunk_n, chunks, bufs);
        auto eval = benchmark_dcf_eval(runtime.aes(), bin, bout, chunk_n, chunks, bufs);
        print_result("dcf", bin, bout, chunk_n, chunks, keygen, eval);
        free_dcf_buffers(bufs);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
