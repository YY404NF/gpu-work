#include "dcf_bench_common.h"

#include "fss-client.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using InType = dcf_bench::InputType;

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
    if (argc != 2)
    {
        dcf_bench::printBatchUsage(argv[0]);
        return 1;
    }

    const std::size_t n = std::strtoull(argv[1], nullptr, 10);
    if (n == 0)
    {
        dcf_bench::printBatchUsage(argv[0]);
        return 1;
    }

    auto alphas = dcf_bench::buildThresholds(n);
    for (auto &alpha : alphas)
        alpha += 1;

    const auto start = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        Fss client{};
        initializeClient(&client, dcf_bench::kBin, 2);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i)
        {
            ServerKeyLt key0{};
            ServerKeyLt key1{};
            generateTreeLt(
                &client,
                &key0,
                &key1,
                alphas[static_cast<std::size_t>(i)],
                1);
            destroyLtKey(&key0);
            destroyLtKey(&key1);
        }

        destroyFssState(&client);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto keygenMicros = static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    std::printf(
        "libfss DCF batch-keygen benchmark finished\n"
        "  bin: %d bit\n"
        "  bout: %d bit\n"
        "  n: %llu elem\n"
        "  keygen: %llu us\n",
        dcf_bench::kBin,
        dcf_bench::kBout,
        static_cast<unsigned long long>(n),
        keygenMicros);

    return 0;
}
