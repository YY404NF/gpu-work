#include "dcf_bench_common.h"

#include "fss-client.h"
#include "fss-server.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using InType = dcf_bench::InputType;

void evalParty(Fss *server, ServerKeyLt *key, const std::vector<InType> &xs, std::vector<std::uint64_t> *ys)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(xs.size()); ++i)
    {
        (*ys)[static_cast<std::size_t>(i)] =
            evaluateLt(server, key, xs[static_cast<std::size_t>(i)]);
    }
}

void validateOutputs(
    InType alpha,
    const std::vector<InType> &xs,
    const std::vector<std::uint64_t> &timedShare,
    const std::vector<std::uint64_t> &cachedShare)
{
    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        const auto combined = timedShare[i] - cachedShare[i];
        const auto expected = static_cast<std::uint64_t>(xs[i] <= alpha);
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

void destroyLtKey(ServerKeyLt *key)
{
    free(key->cw[0]);
    free(key->cw[1]);
    key->cw[0] = nullptr;
    key->cw[1] = nullptr;
}

void destroyFssState(Fss *f)
{
    free(f->aes_keys);
    f->aes_keys = nullptr;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        dcf_bench::printSingleKeyUsage(argv[0]);
        return 1;
    }

    const std::size_t n = std::strtoull(argv[1], nullptr, 10);
    const int evalIters = std::atoi(argv[2]);
    if (n == 0 || evalIters <= 0)
    {
        dcf_bench::printSingleKeyUsage(argv[0]);
        return 1;
    }

    const InType alpha = 20;
    const InType keyAlpha = alpha + 1;
    const auto xs = dcf_bench::buildQueries(n, alpha);

    Fss client{};
    Fss server{};
    ServerKeyLt key0{};
    ServerKeyLt key1{};

    initializeClient(&client, dcf_bench::kBin, 2);
    generateTreeLt(&client, &key0, &key1, keyAlpha, 1);
    initializeServer(&server, &client);

    std::vector<std::uint64_t> cachedShare(n);
    std::vector<std::uint64_t> timedShare(n);
    evalParty(&server, &key1, xs, &cachedShare);

    std::vector<unsigned long long> evalTimes;
    evalTimes.reserve(evalIters);

    evalParty(&server, &key0, xs, &timedShare);

    for (int i = 0; i < evalIters; ++i)
    {
        evalTimes.push_back(dcf_bench::measureMicros([&] {
            evalParty(&server, &key0, xs, &timedShare);
            validateOutputs(alpha, xs, timedShare, cachedShare);
        }));
    }

    destroyLtKey(&key0);
    destroyLtKey(&key1);
    destroyFssState(&server);
    destroyFssState(&client);

    std::printf(
        "libfss DCF single-key benchmark finished\n"
        "  bin: %d bit\n"
        "  bout: %d bit\n"
        "  n: %llu elem\n"
        "  eval_iters: %d\n"
        "  eval: %.2f us\n",
        dcf_bench::kBin,
        dcf_bench::kBout,
        static_cast<unsigned long long>(n),
        evalIters,
        dcf_bench::averageMicros(evalTimes));

    return 0;
}
