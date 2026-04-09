#include <fss/dpf.cuh>
#include <fss/group/uint.cuh>
#include <fss/prg/aes128_mmo.cuh>

#include "dpf_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kInBits = dpf_bench::kBin;

using InType = dpf_bench::InputType;
using GroupType = fss::group::Uint<std::uint64_t>;
using PrgType = fss::prg::Aes128Mmo<2>;
using SchemeType = fss::Dpf<kInBits, GroupType, PrgType, InType>;

constexpr unsigned char kAesKey0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
constexpr unsigned char kAesKey1[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

void evalParty(
    bool party,
    int4 seed,
    const SchemeType::Cw *cws,
    const std::vector<InType> &xs,
    std::vector<int4> *ys)
{
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
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(xs.size()); ++i)
        {
            (*ys)[static_cast<std::size_t>(i)] =
                scheme.Eval(party, seed, cws, xs[static_cast<std::size_t>(i)]);
        }

        PrgType::FreeCtxs(ctxs);
    }
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
        const auto expected = static_cast<std::uint64_t>(xs[i] == alpha);
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

    const std::size_t n = std::strtoull(argv[1], nullptr, 10);
    const int evalIters = std::atoi(argv[2]);
    if (n == 0 || evalIters <= 0)
    {
        dpf_bench::printSingleKeyUsage(argv[0]);
        return 1;
    }

    const InType alpha = 10;
    const int4 seed0 = dpf_bench::makeSeed(0, 0x1111111111111111ULL);
    const int4 seed1 = dpf_bench::makeSeed(0, 0x2222222222222222ULL);
    const auto xs = dpf_bench::buildQueries(n, alpha);

    const unsigned char *keys[2] = {kAesKey0, kAesKey1};
    auto ctxs = PrgType::CreateCtxs(keys);
    PrgType prg(ctxs);
    SchemeType scheme{prg};

    SchemeType::Cw cws[kInBits + 1];
    int4 seeds[2] = {seed0, seed1};
    scheme.Gen(cws, seeds, alpha, dpf_bench::kPayload);
    PrgType::FreeCtxs(ctxs);

    std::vector<int4> cachedShare(n);
    std::vector<int4> timedShare(n);
    evalParty(true, seed1, cws, xs, &cachedShare);

    std::vector<unsigned long long> evalTimes;
    evalTimes.reserve(evalIters);

    evalParty(false, seed0, cws, xs, &timedShare);

    for (int i = 0; i < evalIters; ++i)
    {
        evalTimes.push_back(dpf_bench::measureMicros([&] {
            evalParty(false, seed0, cws, xs, &timedShare);
            validateOutputs(alpha, xs, timedShare, cachedShare);
        }));
    }

    std::printf(
        "myl7-cpu DPF single-key benchmark finished\n"
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
