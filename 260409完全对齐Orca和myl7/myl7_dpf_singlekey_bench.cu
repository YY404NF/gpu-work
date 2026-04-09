#include <cuda_runtime.h>

#include <fss/dpf.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/chacha.cuh>

#include "dpf_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kInBits = dpf_bench::kBin;
constexpr int kThreadsPerBlock = dpf_bench::kThreadsPerBlock;

using InType = dpf_bench::InputType;
using GroupType = fss::group::Uint<std::uint64_t>;
using PrgType = fss::prg::ChaCha<2>;
using SchemeType = fss::Dpf<kInBits, GroupType, PrgType, InType>;

__constant__ int kDeviceNonce[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};
static int kHostNonce[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t err__ = (x);                                             \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr,                                             \
                "CUDA error at %s:%d: %s\n",                                 \
                __FILE__,                                                    \
                __LINE__,                                                    \
                cudaGetErrorString(err__));                                  \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

__global__ void evalBroadcastKernel(
    int4 *ys,
    bool party,
    const int4 *seed,
    const SchemeType::Cw *cws,
    const InType *xs,
    std::uint64_t n)
{
    const std::uint64_t tid = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
        static_cast<std::uint64_t>(threadIdx.x);
    if (tid >= n)
        return;

    PrgType prg(kDeviceNonce);
    SchemeType scheme{prg};
    ys[tid] = scheme.Eval(party, seed[0], cws, xs[tid]);
}

void validateOutputs(
    InType alpha,
    const std::vector<InType> &xs,
    const std::vector<int4> &timedShare,
    const std::vector<int4> &cachedShare)
{
    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        const auto combined =
            (GroupType::From(timedShare[i]) + GroupType::From(cachedShare[i])).val;
        const std::uint64_t expected = static_cast<std::uint64_t>(xs[i] == alpha);
        if (combined != expected)
        {
            std::fprintf(stderr,
                "Validation failed at i=%zu: alpha=%llu x=%llu expected=%llu got=%llu\n",
                i,
                static_cast<unsigned long long>(alpha),
                static_cast<unsigned long long>(xs[i]),
                static_cast<unsigned long long>(expected),
                static_cast<unsigned long long>(combined));
            std::exit(2);
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        dpf_bench::printSingleKeyUsage(argv[0]);
        return 1;
    }

    const std::uint64_t n = std::strtoull(argv[1], nullptr, 10);
    const int evalIters = std::atoi(argv[2]);
    if (n == 0 || evalIters <= 0)
    {
        dpf_bench::printSingleKeyUsage(argv[0]);
        return 1;
    }

    const InType alpha = 10;
    const int4 seed0 = dpf_bench::makeSeed(0, 0x1111111111111111ULL);
    const int4 seed1 = dpf_bench::makeSeed(0, 0x2222222222222222ULL);
    const auto xs = dpf_bench::buildQueries(static_cast<std::size_t>(n), alpha);

    CUDA_CHECK(cudaFree(0));

    PrgType prg(kHostNonce);
    SchemeType scheme{prg};
    SchemeType::Cw h_cws[kInBits + 1];
    int4 seeds[2] = {seed0, seed1};
    scheme.Gen(h_cws, seeds, alpha, dpf_bench::kPayload);

    SchemeType::Cw *d_cws = nullptr;
    InType *d_xs = nullptr;
    int4 *d_seed = nullptr;
    int4 *d_ys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cws, sizeof(h_cws)));
    CUDA_CHECK(cudaMalloc(&d_xs, sizeof(InType) * n));
    CUDA_CHECK(cudaMalloc(&d_seed, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(int4) * n));

    const int numBlocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);

    CUDA_CHECK(cudaMemcpy(d_cws, h_cws, sizeof(h_cws), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seed, &seed1, sizeof(int4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), sizeof(InType) * n, cudaMemcpyHostToDevice));
    evalBroadcastKernel<<<numBlocks, kThreadsPerBlock>>>(d_ys, true, d_seed, d_cws, d_xs, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int4> cachedShare(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy(cachedShare.data(), d_ys, sizeof(int4) * n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(d_seed, &seed0, sizeof(int4), cudaMemcpyHostToDevice));
    evalBroadcastKernel<<<numBlocks, kThreadsPerBlock>>>(d_ys, false, d_seed, d_cws, d_xs, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int4> timedShare(static_cast<std::size_t>(n));
    std::vector<unsigned long long> evalTimes;
    evalTimes.reserve(evalIters);

    for (int i = 0; i < evalIters; ++i)
    {
        evalTimes.push_back(dpf_bench::measureMicros([&] {
            CUDA_CHECK(cudaMemcpy(d_cws, h_cws, sizeof(h_cws), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_seed, &seed0, sizeof(int4), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), sizeof(InType) * n, cudaMemcpyHostToDevice));

            evalBroadcastKernel<<<numBlocks, kThreadsPerBlock>>>(d_ys, false, d_seed, d_cws, d_xs, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(timedShare.data(), d_ys, sizeof(int4) * n, cudaMemcpyDeviceToHost));
            validateOutputs(alpha, xs, timedShare, cachedShare);
        }));
    }

    cudaFree(d_cws);
    cudaFree(d_xs);
    cudaFree(d_seed);
    cudaFree(d_ys);

    std::printf(
        "myl7-gpu DPF single-key benchmark finished\n"
        "  bin: %d bit\n"
        "  n: %llu elem\n"
        "  eval_iters: %d\n"
        "  eval: %.2f us\n",
        kInBits,
        static_cast<unsigned long long>(n),
        evalIters,
        dpf_bench::averageMicros(evalTimes));

    return 0;
}
