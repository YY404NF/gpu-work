#include "dpf_bench_common.h"

#include "fss-client.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using InType = dpf_bench::InputType;

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
        Fss client{};
        initializeClient(&client, dpf_bench::kBin, 2);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i)
        {
            ServerKeyEq key0{};
            ServerKeyEq key1{};
            generateTreeEq(
                &client,
                &key0,
                &key1,
                alphas[static_cast<std::size_t>(i)],
                1);
            destroyEqKey(&key0);
            destroyEqKey(&key1);
        }

        destroyFssState(&client);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto keygenMicros = static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    std::printf(
        "libfss DPF batch-keygen benchmark finished\n"
        "  bin: %d bit\n"
        "  n: %llu elem\n"
        "  keygen: %llu us\n",
        dpf_bench::kBin,
        static_cast<unsigned long long>(n),
        keygenMicros);

    return 0;
}
