#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "dpf_bench_common.h"
#include "fss/dpf_api.h"

using T = u64;

namespace {

constexpr int kBin = dpf_bench::kBin;
constexpr std::size_t kChunkSize = 1ULL << 16;

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        dpf_bench::printBatchUsage(argv[0]);
        return 1;
    }

    const int n = std::atoi(argv[1]);
    if (n <= 0)
    {
        dpf_bench::printBatchUsage(argv[0]);
        return 1;
    }

    gpu_mpc::standalone::Runtime runtime;

    const auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t offset = 0; offset < static_cast<std::size_t>(n); offset += kChunkSize)
    {
        const std::size_t chunk = std::min(kChunkSize, static_cast<std::size_t>(n) - offset);
        std::vector<T> rin(chunk);
        for (std::size_t i = 0; i < chunk; ++i)
            rin[i] = static_cast<T>(10ULL + 2ULL * (offset + i));
        auto [dpfKey0, dpfKey1] = gpu_mpc::standalone::generateDpfKeys(runtime, kBin, rin);
        (void)dpfKey0;
        (void)dpfKey1;
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const auto keygenMicros = static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    std::printf(
        "Orca DPF batch-keygen benchmark finished\n"
        "  bin: %d bit\n"
        "  n: %d elem\n"
        "  keygen: %llu us\n",
        kBin,
        n,
        keygenMicros);

    return 0;
}
