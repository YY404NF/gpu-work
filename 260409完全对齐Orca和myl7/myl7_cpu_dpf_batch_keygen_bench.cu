#include <fss/dpf.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/aes128_mmo.cuh>

#include "dpf_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kInBits = dpf_bench::kBin;

using InType = dpf_bench::InputType;
using GroupType = fss::group::Uint<std::uint64_t>;
using PrgType = fss::prg::Aes128Mmo<2>;
using SchemeType = fss::Dpf<kInBits, GroupType, PrgType, InType>;

constexpr unsigned char kAesKey0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
constexpr unsigned char kAesKey1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        dpf_bench::printBatchUsage(argv[0]);
        return 1;
    }

    const std::size_t n = std::strtoull(argv[1], nullptr, 10);
    if (n == 0)
    {
        dpf_bench::printBatchUsage(argv[0]);
        return 1;
    }

    const auto alphas = dpf_bench::buildPoints(n);

    const auto start = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        const unsigned char *keys[2] = {kAesKey0, kAesKey1};
        auto ctxs = PrgType::CreateCtxs(keys);
        PrgType prg(ctxs);
        SchemeType scheme{prg};

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i)
        {
            int4 localSeeds[2] = {
                dpf_bench::makeSeed(static_cast<std::uint64_t>(i), 0x1111111111111111ULL),
                dpf_bench::makeSeed(static_cast<std::uint64_t>(i), 0x2222222222222222ULL),
            };
            SchemeType::Cw localCws[kInBits + 1];
            scheme.Gen(
                localCws,
                localSeeds,
                alphas[static_cast<std::size_t>(i)],
                dpf_bench::kPayload);
        }

        PrgType::FreeCtxs(ctxs);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto keygenMicros = static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    std::printf(
        "myl7-cpu DPF batch-keygen benchmark finished\n"
        "  bin: %d bit\n"
        "  n: %llu elem\n"
        "  keygen: %llu us\n",
        kInBits,
        static_cast<unsigned long long>(n),
        keygenMicros);

    return 0;
}
