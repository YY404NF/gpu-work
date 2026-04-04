#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "fss-client.h"
#include "fss-server.h"

namespace {

unsigned long long microsBetween(const std::chrono::high_resolution_clock::time_point &start,
    const std::chrono::high_resolution_clock::time_point &end) {
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

std::vector<uint64_t> buildDpfAlphas(int bin, std::size_t n) {
    std::vector<uint64_t> rin(n);
    const uint64_t limit = uint64_t{1} << bin;
    const uint64_t mask = limit - 1;
    constexpr uint64_t kStride = 104729;
    for (std::size_t i = 0; i < n; ++i) {
        rin[i] = (uint64_t{10} + i * kStride) & mask;
    }
    return rin;
}

std::vector<uint64_t> buildDpfQueries(int bin, const std::vector<uint64_t> &alphas) {
    std::vector<uint64_t> xs(alphas.size());
    const uint64_t limit = uint64_t{1} << bin;
    for (std::size_t i = 0; i < alphas.size(); ++i) {
        xs[i] = (i % 3 == 0 || alphas[i] + 1 >= limit) ? alphas[i] : (alphas[i] + 1);
    }
    return xs;
}

std::vector<uint64_t> buildDcfAlphas(int bin, std::size_t n) {
    std::vector<uint64_t> rin(n);
    const uint64_t limit = uint64_t{1} << bin;
    const uint64_t span = limit - 1;
    constexpr uint64_t kStride = 104729;
    for (std::size_t i = 0; i < n; ++i) {
        rin[i] = 1 + ((19 + i * kStride) % span);
    }
    return rin;
}

std::vector<uint64_t> buildDcfQueries(int bin, const std::vector<uint64_t> &alphas) {
    std::vector<uint64_t> xs(alphas.size());
    const uint64_t limit = uint64_t{1} << bin;
    for (std::size_t i = 0; i < alphas.size(); ++i) {
        if (i % 4 == 0) {
            xs[i] = alphas[i];
        } else if (i % 4 == 1) {
            xs[i] = alphas[i] - 1;
        } else {
            xs[i] = (alphas[i] + 1 < limit) ? (alphas[i] + 1) : alphas[i];
        }
    }
    return xs;
}

void benchDpf(int bin, std::size_t n) {
    Fss client{};
    Fss server{};
    initializeClient(&client, static_cast<uint32_t>(bin), 2);
    initializeServer(&server, &client);

    const auto alphas = buildDpfAlphas(bin, n);
    const auto xs = buildDpfQueries(bin, alphas);

    std::vector<ServerKeyEq> k0(n);
    std::vector<ServerKeyEq> k1(n);
    std::vector<mpz_class> y0(n);
    std::vector<mpz_class> y1(n);

    const auto total_start = std::chrono::high_resolution_clock::now();

    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        generateTreeEq(&client, &k0[i], &k1[i], alphas[i], 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    const auto keygen_us = microsBetween(start, end);

    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        y0[i] = evaluateEq(&server, &k0[i], xs[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    const auto eval_p0_us = microsBetween(start, end);

    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        y1[i] = evaluateEq(&server, &k1[i], xs[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    const auto eval_p1_us = microsBetween(start, end);

    const auto total_us = microsBetween(total_start, std::chrono::high_resolution_clock::now());

    const std::size_t check_n = n < 16 ? n : 16;
    for (std::size_t i = 0; i < check_n; ++i) {
        const bool hit = xs[i] == alphas[i];
        const mpz_class sum = y0[i] - y1[i];
        if ((hit && sum != 1) || (!hit && sum != 0)) {
            std::fprintf(stderr, "libfss dpf verification failed at idx=%zu\n", i);
            std::exit(2);
        }
    }

    std::printf(
        "primitive=dpf impl=libfss_cpu bin=%d n=%zu keygen_us=%llu eval_p0_us=%llu "
        "eval_p1_us=%llu total_us=%llu\n",
        bin,
        n,
        keygen_us,
        eval_p0_us,
        eval_p1_us,
        total_us);
}

void benchDcf(int bin, std::size_t n) {
    Fss client{};
    Fss server{};
    initializeClient(&client, static_cast<uint32_t>(bin), 2);
    initializeServer(&server, &client);

    const auto alphas = buildDcfAlphas(bin, n);
    const auto xs = buildDcfQueries(bin, alphas);

    std::vector<ServerKeyLt> k0(n);
    std::vector<ServerKeyLt> k1(n);
    std::vector<uint64_t> y0(n);
    std::vector<uint64_t> y1(n);

    const auto total_start = std::chrono::high_resolution_clock::now();

    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        generateTreeLt(&client, &k0[i], &k1[i], alphas[i], 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    const auto keygen_us = microsBetween(start, end);

    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        y0[i] = evaluateLt(&server, &k0[i], xs[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    const auto eval_p0_us = microsBetween(start, end);

    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        y1[i] = evaluateLt(&server, &k1[i], xs[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    const auto eval_p1_us = microsBetween(start, end);

    const auto total_us = microsBetween(total_start, std::chrono::high_resolution_clock::now());

    const std::size_t check_n = n < 16 ? n : 16;
    for (std::size_t i = 0; i < check_n; ++i) {
        const bool hit = xs[i] < alphas[i];
        const uint64_t sum = y0[i] - y1[i];
        if ((hit && sum != 1) || (!hit && sum != 0)) {
            std::fprintf(stderr, "libfss dcf verification failed at idx=%zu\n", i);
            std::exit(2);
        }
    }

    std::printf(
        "primitive=dcf impl=libfss_cpu bin=%d n=%zu keygen_us=%llu eval_p0_us=%llu "
        "eval_p1_us=%llu total_us=%llu\n",
        bin,
        n,
        keygen_us,
        eval_p0_us,
        eval_p1_us,
        total_us);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s <dpf|dcf> <bin> <n>\n", argv[0]);
        return 1;
    }

    const std::string_view primitive = argv[1];
    const int bin = std::atoi(argv[2]);
    const std::size_t n = static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10));

    if ((primitive != "dpf" && primitive != "dcf") || bin <= 0 || bin >= 63 || n == 0) {
        std::fprintf(stderr, "Usage: %s <dpf|dcf> <bin> <n>\n", argv[0]);
        return 1;
    }

    if (primitive == "dpf") {
        benchDpf(bin, n);
    } else {
        benchDcf(bin, n);
    }
    return 0;
}
