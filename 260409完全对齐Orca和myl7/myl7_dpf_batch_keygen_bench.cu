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
constexpr std::size_t kChunkBudgetBytes = 256ULL << 20;

using InType = dpf_bench::InputType;
using GroupType = fss::group::Uint<std::uint64_t>;
using PrgType = fss::prg::ChaCha<2>;
using SchemeType = fss::Dpf<kInBits, GroupType, PrgType, InType>;

__constant__ int kDeviceNonce[2] = {0x12345678, static_cast<int>(0x9abcdef0u)};

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

__global__ void dpfGenKernel(
    SchemeType::Cw *cws,
    const int4 *seeds,
    const InType *alphas,
    const int4 *betas,
    std::uint64_t n)
{
    const std::uint64_t tid = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
        static_cast<std::uint64_t>(threadIdx.x);
    if (tid >= n)
        return;

    PrgType prg(kDeviceNonce);
    SchemeType dpf{prg};
    int4 s[2] = {seeds[tid * 2], seeds[tid * 2 + 1]};
    dpf.Gen(cws + tid * static_cast<std::uint64_t>(kInBits + 1), s, alphas[tid], betas[tid]);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        dpf_bench::printBatchUsage(argv[0]);
        return 1;
    }

    const std::uint64_t n = std::strtoull(argv[1], nullptr, 10);
    if (n == 0)
    {
        dpf_bench::printBatchUsage(argv[0]);
        return 1;
    }

    CUDA_CHECK(cudaFree(0));

    const std::size_t bytesPerElem =
        sizeof(int4) * 2 +
        sizeof(InType) +
        sizeof(int4) +
        sizeof(SchemeType::Cw) * static_cast<std::size_t>(kInBits + 1);
    const std::size_t chunkSize =
        std::max<std::size_t>(1, kChunkBudgetBytes / std::max<std::size_t>(bytesPerElem, 1));

    const auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t offset = 0; offset < static_cast<std::size_t>(n); offset += chunkSize)
    {
        const std::size_t chunk = std::min(chunkSize, static_cast<std::size_t>(n) - offset);
        std::vector<int4> h_seeds(chunk * 2);
        std::vector<InType> h_alphas(chunk);
        std::vector<int4> h_betas(chunk, dpf_bench::kPayload);
        for (std::size_t i = 0; i < chunk; ++i)
        {
            const auto idx = offset + i;
            h_seeds[i * 2] = dpf_bench::makeSeed(idx, 0x1111111111111111ULL);
            h_seeds[i * 2 + 1] = dpf_bench::makeSeed(idx, 0x2222222222222222ULL);
            h_alphas[i] = static_cast<InType>(10ULL + 2ULL * idx);
        }

        int4 *d_seeds = nullptr;
        InType *d_alphas = nullptr;
        int4 *d_betas = nullptr;
        SchemeType::Cw *d_cws = nullptr;
        CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int4) * h_seeds.size()));
        CUDA_CHECK(cudaMalloc(&d_alphas, sizeof(InType) * h_alphas.size()));
        CUDA_CHECK(cudaMalloc(&d_betas, sizeof(int4) * h_betas.size()));
        CUDA_CHECK(cudaMalloc(
            &d_cws,
            sizeof(SchemeType::Cw) * static_cast<std::size_t>(kInBits + 1) * chunk));

        CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds.data(), sizeof(int4) * h_seeds.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_alphas, h_alphas.data(), sizeof(InType) * h_alphas.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_betas, h_betas.data(), sizeof(int4) * h_betas.size(), cudaMemcpyHostToDevice));

        const int numBlocks = static_cast<int>((chunk + kThreadsPerBlock - 1) / kThreadsPerBlock);
        dpfGenKernel<<<numBlocks, kThreadsPerBlock>>>(
            d_cws,
            d_seeds,
            d_alphas,
            d_betas,
            static_cast<std::uint64_t>(chunk));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_seeds);
        cudaFree(d_alphas);
        cudaFree(d_betas);
        cudaFree(d_cws);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto keygenMicros = static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    std::printf(
        "myl7-gpu DPF batch-keygen benchmark finished\n"
        "  bin: %d bit\n"
        "  n: %llu elem\n"
        "  keygen: %llu us\n",
        kInBits,
        static_cast<unsigned long long>(n),
        keygenMicros);

    return 0;
}
