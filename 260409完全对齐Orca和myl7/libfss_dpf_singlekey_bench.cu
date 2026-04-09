#include "dpf_bench_common.h"

#include "fss-client.h"
#include "fss-server.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using InType = dpf_bench::InputType;

void evalParty(Fss *server, ServerKeyEq *key, const std::vector<InType> &xs, std::vector<mpz_class> *ys)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(xs.size()); ++i)
    {
        (*ys)[static_cast<std::size_t>(i)] =
            evaluateEq(server, key, xs[static_cast<std::size_t>(i)]);
    }
}

void validateOutputs(
    InType alpha,
    const std::vector<InType> &xs,
    const std::vector<mpz_class> &timedShare,
    const std::vector<mpz_class> &cachedShare)
{
    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        const mpz_class combined = timedShare[i] - cachedShare[i];
        const mpz_class expected(static_cast<unsigned long>((xs[i] == alpha) ? 1UL : 0UL));
        if (combined != expected)
        {
            std::fprintf(stderr,
                "Validation failed at i=%zu: alpha=%llu x=%llu expected=%llu got=%s\n",
                i,
                static_cast<unsigned long long>(alpha),
                static_cast<unsigned long long>(xs[i]),
                static_cast<unsigned long long>(xs[i] == alpha),
                combined.get_str().c_str());
            std::exit(2);
        }
    }
}

void destroyEqKey(ServerKeyEq *key)
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
    const auto xs = dpf_bench::buildQueries(n, alpha);

    Fss client{};
    Fss server{};
    ServerKeyEq key0{};
    ServerKeyEq key1{};

    initializeClient(&client, dpf_bench::kBin, 2);
    generateTreeEq(&client, &key0, &key1, alpha, 1);
    initializeServer(&server, &client);

    std::vector<mpz_class> cachedShare(n);
    std::vector<mpz_class> timedShare(n);
    evalParty(&server, &key1, xs, &cachedShare);

    std::vector<unsigned long long> evalTimes;
    evalTimes.reserve(evalIters);

    evalParty(&server, &key0, xs, &timedShare);

    for (int i = 0; i < evalIters; ++i)
    {
        evalTimes.push_back(dpf_bench::measureMicros([&] {
            evalParty(&server, &key0, xs, &timedShare);
            validateOutputs(alpha, xs, timedShare, cachedShare);
        }));
    }

    destroyEqKey(&key0);
    destroyEqKey(&key1);
    destroyFssState(&server);
    destroyFssState(&client);

    std::printf(
        "libfss DPF single-key benchmark finished\n"
        "  bin: %d bit\n"
        "  n: %llu elem\n"
        "  eval_iters: %d\n"
        "  eval: %.2f us\n",
        dpf_bench::kBin,
        static_cast<unsigned long long>(n),
        evalIters,
        dpf_bench::averageMicros(evalTimes));

    return 0;
}
