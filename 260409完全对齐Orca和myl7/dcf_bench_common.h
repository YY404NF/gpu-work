#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace dcf_bench {

constexpr int kBin = 64;
constexpr int kBout = 1;
constexpr int kThreadsPerBlock = 256;
inline constexpr int4 kPayload{1, 0, 0, 0};

using InputType = std::uint64_t;

template <typename F>
unsigned long long measureMicros(F &&fn)
{
    const auto start = std::chrono::high_resolution_clock::now();
    fn();
    const auto end = std::chrono::high_resolution_clock::now();
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

inline double averageMicros(const std::vector<unsigned long long> &values)
{
    if (values.empty())
        return 0.0;
    const auto total = std::accumulate(values.begin(), values.end(), 0ULL);
    return static_cast<double>(total) / static_cast<double>(values.size());
}

inline std::uint64_t splitMix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline int4 makeSeed(std::uint64_t idx, std::uint64_t salt)
{
    const std::uint64_t a = splitMix64(idx ^ salt);
    const std::uint64_t b = splitMix64(a + salt + 0x9e3779b97f4a7c15ULL);
    return {
        static_cast<int>(static_cast<std::uint32_t>(a)),
        static_cast<int>(static_cast<std::uint32_t>(a >> 32)),
        static_cast<int>(static_cast<std::uint32_t>(b)),
        static_cast<int>(static_cast<std::uint32_t>(b >> 32) & ~1U),
    };
}

inline std::vector<InputType> buildQueries(std::size_t n, InputType alpha)
{
    std::vector<InputType> xs(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (i % 4 == 0)
            xs[i] = alpha;
        else if (i % 4 == 1)
            xs[i] = alpha == 0 ? 0 : alpha - 1;
        else
            xs[i] = alpha + 1 + static_cast<InputType>(i);
    }
    return xs;
}

inline std::vector<InputType> buildThresholds(std::size_t n)
{
    std::vector<InputType> alphas(n);
    for (std::size_t i = 0; i < n; ++i)
        alphas[i] = static_cast<InputType>(20ULL + 2ULL * i);
    return alphas;
}

inline void printBatchUsage(const char *prog)
{
    std::fprintf(stderr, "Usage: %s <n>\n", prog);
}

inline void printSingleKeyUsage(const char *prog)
{
    std::fprintf(stderr, "Usage: %s <n> <eval_iters>\n", prog);
}

} // namespace dcf_bench
