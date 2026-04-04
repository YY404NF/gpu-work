#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "fss-client.h"
#include "fss-server.h"

namespace {

using Clock = std::chrono::high_resolution_clock;

unsigned long long microsBetween(const Clock::time_point& start, const Clock::time_point& end) {
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

std::size_t floorPowerOfTwo(std::size_t value) {
    std::size_t out = 1;
    while ((out << 1) <= value) out <<= 1;
    return out;
}

std::size_t pickDefaultChunk(std::string_view primitive, int bin) {
    constexpr std::size_t kTargetBytes = 256ULL * 1024ULL * 1024ULL;
    const std::size_t per_item_key_bytes =
        primitive == "dpf"
            ? 2ULL * (sizeof(ServerKeyEq) + 2ULL * static_cast<std::size_t>(bin - 1) * sizeof(CWEq))
            : 2ULL * (sizeof(ServerKeyLt) + 2ULL * static_cast<std::size_t>(bin - 1) * sizeof(CWLt));
    const std::size_t rough = std::max<std::size_t>(1, kTargetBytes / std::max<std::size_t>(1, per_item_key_bytes));
    return std::max<std::size_t>(1, floorPowerOfTwo(rough));
}

uint64_t buildDpfAlpha(int bin, std::size_t global_idx) {
    if (bin == 64) return 10ULL + 2ULL * global_idx;
    const uint64_t limit = uint64_t{1} << bin;
    const uint64_t mask = limit - 1;
    constexpr uint64_t kStride = 104729;
    return (10ULL + global_idx * kStride) & mask;
}

uint64_t buildDpfQuery(int bin, std::size_t global_idx, uint64_t alpha) {
    if (bin == 64) return (global_idx % 3 == 0) ? alpha : (alpha + 1);
    const uint64_t limit = uint64_t{1} << bin;
    return (global_idx % 3 == 0 || alpha + 1 >= limit) ? alpha : (alpha + 1);
}

uint64_t buildDcfAlpha(int bin, std::size_t global_idx) {
    if (bin == 64) return 20ULL + 2ULL * global_idx;
    const uint64_t limit = uint64_t{1} << bin;
    const uint64_t span = limit - 1;
    constexpr uint64_t kStride = 104729;
    return 1ULL + ((19ULL + global_idx * kStride) % span);
}

uint64_t buildDcfQuery(int bin, std::size_t global_idx, uint64_t alpha) {
    if (bin == 64) {
        if (global_idx % 4 == 0) return alpha;
        if (global_idx % 4 == 1) return alpha - 1;
        return alpha + 1;
    }
    const uint64_t limit = uint64_t{1} << bin;
    if (global_idx % 4 == 0) return alpha;
    if (global_idx % 4 == 1) return alpha - 1;
    return (alpha + 1 < limit) ? (alpha + 1) : alpha;
}

void freeEqKeys(std::vector<ServerKeyEq>& k0, std::vector<ServerKeyEq>& k1) {
    for (std::size_t i = 0; i < k0.size(); ++i) {
        free(k0[i].cw[0]);
        free(k0[i].cw[1]);
        free(k1[i].cw[0]);
        free(k1[i].cw[1]);
        k0[i].cw[0] = nullptr;
        k0[i].cw[1] = nullptr;
        k1[i].cw[0] = nullptr;
        k1[i].cw[1] = nullptr;
    }
}

void freeLtKeys(std::vector<ServerKeyLt>& k0, std::vector<ServerKeyLt>& k1) {
    for (std::size_t i = 0; i < k0.size(); ++i) {
        free(k0[i].cw[0]);
        free(k0[i].cw[1]);
        free(k1[i].cw[0]);
        free(k1[i].cw[1]);
        k0[i].cw[0] = nullptr;
        k0[i].cw[1] = nullptr;
        k1[i].cw[0] = nullptr;
        k1[i].cw[1] = nullptr;
    }
}

int runDpf(int bin, std::size_t n, std::size_t chunk) {
    Fss client{};
    Fss server{};
    initializeClient(&client, static_cast<uint32_t>(bin), 2);
    initializeServer(&server, &client);

    unsigned long long keygen_us = 0;
    unsigned long long eval_p0_us = 0;
    unsigned long long eval_p1_us = 0;
    const auto total_start = Clock::now();

    for (std::size_t offset = 0; offset < n; offset += chunk) {
        const std::size_t cur = std::min(chunk, n - offset);
        std::vector<uint64_t> alphas(cur);
        std::vector<uint64_t> xs(cur);
        for (std::size_t i = 0; i < cur; ++i) {
            const std::size_t global_idx = offset + i;
            alphas[i] = buildDpfAlpha(bin, global_idx);
            xs[i] = buildDpfQuery(bin, global_idx, alphas[i]);
        }

        std::vector<ServerKeyEq> k0(cur);
        std::vector<ServerKeyEq> k1(cur);
        const std::size_t check_n = std::min<std::size_t>(16, cur);
        std::vector<mpz_class> y0_check(check_n);
        std::vector<mpz_class> y1_check(check_n);

        auto start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i) {
            generateTreeEq(&client, &k0[i], &k1[i], alphas[i], 1);
        }
        keygen_us += microsBetween(start, Clock::now());

        start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i) {
            const mpz_class y = evaluateEq(&server, &k0[i], xs[i]);
            if (i < check_n) y0_check[i] = y;
        }
        eval_p0_us += microsBetween(start, Clock::now());

        start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i) {
            const mpz_class y = evaluateEq(&server, &k1[i], xs[i]);
            if (i < check_n) y1_check[i] = y;
        }
        eval_p1_us += microsBetween(start, Clock::now());

        for (std::size_t i = 0; i < check_n; ++i) {
            const bool hit = xs[i] == alphas[i];
            const mpz_class sum = y0_check[i] - y1_check[i];
            if ((hit && sum != 1) || (!hit && sum != 0)) {
                std::fprintf(stderr, "libfss dpf verification failed at global_idx=%zu\n", offset + i);
                freeEqKeys(k0, k1);
                return 2;
            }
        }

        freeEqKeys(k0, k1);
    }

    const auto total_us = microsBetween(total_start, Clock::now());
    std::printf(
        "primitive=dpf impl=libfss_cpu bin=%d n=%zu chunk=%zu keygen_us=%llu "
        "eval_p0_us=%llu eval_p1_us=%llu total_us=%llu\n",
        bin,
        n,
        chunk,
        keygen_us,
        eval_p0_us,
        eval_p1_us,
        total_us);
    return 0;
}

int runDcf(int bin, int bout, std::size_t n, std::size_t chunk) {
    if (bout != 1) {
        std::fprintf(stderr, "libfss DCF benchmark currently only supports bout=1\n");
        return 1;
    }

    Fss client{};
    Fss server{};
    initializeClient(&client, static_cast<uint32_t>(bin), 2);
    initializeServer(&server, &client);

    unsigned long long keygen_us = 0;
    unsigned long long eval_p0_us = 0;
    unsigned long long eval_p1_us = 0;
    const auto total_start = Clock::now();

    for (std::size_t offset = 0; offset < n; offset += chunk) {
        const std::size_t cur = std::min(chunk, n - offset);
        std::vector<uint64_t> alphas(cur);
        std::vector<uint64_t> xs(cur);
        for (std::size_t i = 0; i < cur; ++i) {
            const std::size_t global_idx = offset + i;
            alphas[i] = buildDcfAlpha(bin, global_idx);
            xs[i] = buildDcfQuery(bin, global_idx, alphas[i]);
        }

        std::vector<ServerKeyLt> k0(cur);
        std::vector<ServerKeyLt> k1(cur);
        const std::size_t check_n = std::min<std::size_t>(16, cur);
        std::vector<uint64_t> y0_check(check_n);
        std::vector<uint64_t> y1_check(check_n);

        auto start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i) {
            generateTreeLt(&client, &k0[i], &k1[i], alphas[i], 1);
        }
        keygen_us += microsBetween(start, Clock::now());

        start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i) {
            const uint64_t y = evaluateLt(&server, &k0[i], xs[i]);
            if (i < check_n) y0_check[i] = y;
        }
        eval_p0_us += microsBetween(start, Clock::now());

        start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i) {
            const uint64_t y = evaluateLt(&server, &k1[i], xs[i]);
            if (i < check_n) y1_check[i] = y;
        }
        eval_p1_us += microsBetween(start, Clock::now());

        for (std::size_t i = 0; i < check_n; ++i) {
            const bool hit = xs[i] < alphas[i];
            const uint64_t sum = y0_check[i] - y1_check[i];
            if ((hit && sum != 1) || (!hit && sum != 0)) {
                std::fprintf(stderr, "libfss dcf verification failed at global_idx=%zu\n", offset + i);
                freeLtKeys(k0, k1);
                return 2;
            }
        }

        freeLtKeys(k0, k1);
    }

    const auto total_us = microsBetween(total_start, Clock::now());
    std::printf(
        "primitive=dcf impl=libfss_cpu bin=%d bout=%d n=%zu chunk=%zu keygen_us=%llu "
        "eval_p0_us=%llu eval_p1_us=%llu total_us=%llu\n",
        bin,
        bout,
        n,
        chunk,
        keygen_us,
        eval_p0_us,
        eval_p1_us,
        total_us);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s dpf <bin> <n> [chunk]\n", argv[0]);
        std::fprintf(stderr, "       %s dcf <bin> <bout> <n> [chunk]\n", argv[0]);
        return 1;
    }

    const std::string_view primitive = argv[1];
    if (primitive == "dpf") {
        if (argc != 4 && argc != 5) {
            std::fprintf(stderr, "Usage: %s dpf <bin> <n> [chunk]\n", argv[0]);
            return 1;
        }
        const int bin = std::atoi(argv[2]);
        const std::size_t n = static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10));
        const std::size_t chunk =
            argc == 5 ? static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 10)) : pickDefaultChunk(primitive, bin);
        if (bin <= 0 || bin > 64 || n == 0 || chunk == 0) {
            std::fprintf(stderr, "Usage: %s dpf <bin> <n> [chunk]\n", argv[0]);
            return 1;
        }
        return runDpf(bin, n, chunk);
    }

    if (primitive == "dcf") {
        if (argc != 5 && argc != 6) {
            std::fprintf(stderr, "Usage: %s dcf <bin> <bout> <n> [chunk]\n", argv[0]);
            return 1;
        }
        const int bin = std::atoi(argv[2]);
        const int bout = std::atoi(argv[3]);
        const std::size_t n = static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 10));
        const std::size_t chunk =
            argc == 6 ? static_cast<std::size_t>(std::strtoull(argv[5], nullptr, 10)) : pickDefaultChunk(primitive, bin);
        if (bin <= 0 || bin > 64 || bout <= 0 || n == 0 || chunk == 0) {
            std::fprintf(stderr, "Usage: %s dcf <bin> <bout> <n> [chunk]\n", argv[0]);
            return 1;
        }
        return runDcf(bin, bout, n, chunk);
    }

    std::fprintf(stderr, "Usage: %s dpf <bin> <n> [chunk]\n", argv[0]);
    std::fprintf(stderr, "       %s dcf <bin> <bout> <n> [chunk]\n", argv[0]);
    return 1;
}
