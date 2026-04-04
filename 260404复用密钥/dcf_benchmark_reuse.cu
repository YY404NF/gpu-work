#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "fss/dcf_api.h"
#include "gpu/gpu_stats.h"

using T = u64;

namespace
{

struct UploadedDcfKey
{
    dcf::GPUDCFKey key{};
};

void printUsage(const char *prog)
{
    std::fprintf(stderr, "Usage: %s <bin> <bout> <n> <eval_iters>\n", prog);
}

std::vector<T> buildRin(int bin, int n)
{
    std::vector<T> rin(n);
    if (bin == 64)
    {
        for (int i = 0; i < n; ++i)
            rin[i] = T(20) + T(2) * i;
        return rin;
    }

    const T limit = T(1) << bin;
    const T span = limit - 1;
    constexpr T kStride = 104729;
    for (int i = 0; i < n; ++i)
        rin[i] = T(1) + ((T(19) + T(i) * kStride) % span);
    return rin;
}

std::vector<T> buildQueries(int bin, const std::vector<T> &rin)
{
    std::vector<T> x(rin.size());
    const T limit = (bin == 64) ? ~T(0) : (T(1) << bin);
    for (std::size_t i = 0; i < rin.size(); ++i)
    {
        if (i % 4 == 0)
            x[i] = rin[i];
        else if (i % 4 == 1)
            x[i] = rin[i] - 1;
        else
            x[i] = (rin[i] + 1 < limit) ? (rin[i] + 1) : rin[i];
    }
    return x;
}

unsigned long long microsBetween(
    const std::chrono::high_resolution_clock::time_point &start,
    const std::chrono::high_resolution_clock::time_point &end)
{
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

double averageMicros(const std::vector<unsigned long long> &values)
{
    if (values.empty())
        return 0.0;
    const auto total = std::accumulate(values.begin(), values.end(), 0ULL);
    return static_cast<double>(total) / static_cast<double>(values.size());
}

u64 outputMask(int bout)
{
    return (bout == 64) ? ~u64(0) : ((u64(1) << bout) - 1);
}

void freeParsedDcfKey(dcf::GPUDCFKey &key)
{
    if (key.bin > 8)
    {
        delete[] key.dcfTreeKey;
        key.dcfTreeKey = nullptr;
    }
}

UploadedDcfKey uploadDcfKeyToGpu(const gpu_mpc::standalone::KeyBlob &hostKey, Stats *stats)
{
    UploadedDcfKey uploaded;

    u8 *cursor = hostKey.data();
    auto parsed = dcf::readGPUDCFKey(&cursor);
    uploaded.key = parsed;

    if (parsed.bin <= 8)
    {
        uploaded.key.ssKey = parsed.ssKey;
        uploaded.key.ssKey.ss = reinterpret_cast<u8 *>(
            moveToGPU(reinterpret_cast<u8 *>(parsed.ssKey.ss), parsed.ssKey.memSzSS, stats));
        return uploaded;
    }

    uploaded.key.dcfTreeKey = new dcf::GPUDCFTreeKey[parsed.B];
    for (int b = 0; b < parsed.B; ++b)
    {
        const auto &src = parsed.dcfTreeKey[b];
        auto &dst = uploaded.key.dcfTreeKey[b];
        dst = src;
        dst.scw = reinterpret_cast<AESBlock *>(moveToGPU(reinterpret_cast<u8 *>(src.scw), src.memSzScw, stats));
        dst.vcw = reinterpret_cast<u32 *>(moveToGPU(reinterpret_cast<u8 *>(src.vcw), src.memSzVcw, stats));
        dst.l = reinterpret_cast<AESBlock *>(moveToGPU(reinterpret_cast<u8 *>(src.l), src.memSzL, stats));
    }

    delete[] parsed.dcfTreeKey;
    return uploaded;
}

void destroyUploadedDcfKey(UploadedDcfKey *uploaded)
{
    if (uploaded->key.bin <= 8)
    {
        gpuFree(uploaded->key.ssKey.ss);
        uploaded->key.ssKey.ss = nullptr;
        return;
    }

    for (int b = 0; b < uploaded->key.B; ++b)
    {
        gpuFree(uploaded->key.dcfTreeKey[b].scw);
        gpuFree(uploaded->key.dcfTreeKey[b].vcw);
        gpuFree(uploaded->key.dcfTreeKey[b].l);
    }
    delete[] uploaded->key.dcfTreeKey;
    uploaded->key.dcfTreeKey = nullptr;
}

namespace dcf_cached
{

template <typename TValue, int E, dcf::dcfPrologue Prologue, dcf::dcfEpilogue Epilogue>
void gpuDcfTreeEval(dcf::GPUDCFTreeKey key, int party, TValue *d_in, u32 *d_out, u64 outStride, AESGlobalContext *aes)
{
    const int tbSize = 256;
    const int numBlocks = (key.N - 1) / tbSize + 1;
    dcf::doDcf<TValue, E, Prologue, Epilogue>
        <<<numBlocks, tbSize>>>(key.bin, key.bout, party, key.N, d_in, key.scw, key.vcw, key.l, d_out, outStride, *aes);
    checkCudaErrors(cudaDeviceSynchronize());
}

template <typename TValue, int E, dcf::dcfPrologue Prologue, dcf::dcfEpilogue Epilogue>
u32 *gpuLookupSSTable(GPUSSTabKey &key, int party, TValue *d_in, Stats *stats)
{
    auto *d_out = moveMasks(key.memSzOut, nullptr, stats);
    dcf::lookupSSTable<TValue, E, Prologue, Epilogue>
        <<<(key.N - 1) / 128 + 1, 128>>>(party, key.bin, key.N, d_in, key.ss, d_out);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_out;
}

template <typename TValue, int E, dcf::dcfPrologue Prologue, dcf::dcfEpilogue Epilogue>
u32 *gpuDcf(const dcf::GPUDCFKey &key, int party, TValue *d_in, AESGlobalContext *aes, Stats *stats)
{
    if (key.bin <= 8)
        return gpuLookupSSTable<TValue, E, Prologue, Epilogue>(const_cast<GPUSSTabKey &>(key.ssKey), party, d_in, stats);

    auto *d_out = moveMasks(key.memSzOut, nullptr, stats);
    const size_t globalWords = key.memSzOut / sizeof(PACK_TYPE);
    const int blockInputSize = key.dcfTreeKey[0].N;
    const size_t blockWords = key.dcfTreeKey[0].memSzOut / sizeof(PACK_TYPE);
    for (int b = 0; b < key.B; ++b)
    {
        gpuDcfTreeEval<TValue, E, Prologue, Epilogue>(
            key.dcfTreeKey[b], party, d_in + b * blockInputSize, d_out + b * blockWords, static_cast<u64>(globalWords), aes);
    }
    return d_out;
}

} // namespace dcf_cached

std::vector<u32> evalPackedWithHostKey(
    gpu_mpc::standalone::Runtime &runtime,
    const gpu_mpc::standalone::KeyBlob &key,
    int party,
    T *d_x)
{
    u8 *cursor = key.data();
    auto parsedKey = dcf::readGPUDCFKey(&cursor);
    u32 *d_out = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(parsedKey, party, d_x, runtime.aes(), nullptr);
    auto packed = gpu_mpc::standalone::detail::copyPackedOutputToHost(d_out, parsedKey.memSzOut);
    freeParsedDcfKey(parsedKey);
    return packed;
}

std::vector<u32> evalPackedWithCachedKey(
    gpu_mpc::standalone::Runtime &runtime,
    const UploadedDcfKey &key,
    int party,
    T *d_x)
{
    u32 *d_out = dcf_cached::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(key.key, party, d_x, runtime.aes(), nullptr);
    return gpu_mpc::standalone::detail::copyPackedOutputToHost(d_out, key.key.memSzOut);
}

void validateOutputs(
    gpu_mpc::standalone::Runtime &runtime,
    const gpu_mpc::standalone::KeyBlob &hostKey0,
    const gpu_mpc::standalone::KeyBlob &hostKey1,
    const UploadedDcfKey &cachedKey0,
    const UploadedDcfKey &cachedKey1,
    T *d_x,
    const std::vector<T> &rin,
    const std::vector<T> &x,
    int bin,
    int bout)
{
    const auto hostPacked0 = evalPackedWithHostKey(runtime, hostKey0, SERVER0, d_x);
    const auto hostPacked1 = evalPackedWithHostKey(runtime, hostKey1, SERVER1, d_x);
    const auto cachedPacked0 = evalPackedWithCachedKey(runtime, cachedKey0, SERVER0, d_x);
    const auto cachedPacked1 = evalPackedWithCachedKey(runtime, cachedKey1, SERVER1, d_x);

    assert(hostPacked0 == cachedPacked0);
    assert(hostPacked1 == cachedPacked1);

    const auto share0 = gpu_mpc::standalone::unpackPackedOutput(cachedPacked0, static_cast<int>(x.size()), bout);
    const auto share1 = gpu_mpc::standalone::unpackPackedOutput(cachedPacked1, static_cast<int>(x.size()), bout);
    const u64 mask = outputMask(bout);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const u64 combined = (share0[i] + share1[i]) & mask;
        const bool expectedBool = (bin <= 8) ? (x[i] < rin[i]) : (x[i] <= rin[i]);
        const u64 expected = static_cast<u64>(expectedBool);
        assert(combined == expected);
    }
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        printUsage(argv[0]);
        return 1;
    }

    const int bin = std::atoi(argv[1]);
    const int bout = std::atoi(argv[2]);
    const int n = std::atoi(argv[3]);
    const int evalIters = std::atoi(argv[4]);
    if (bin <= 0 || bin > 64 || bout <= 0 || bout > 64 || n <= 0 || evalIters <= 0)
    {
        printUsage(argv[0]);
        return 1;
    }

    gpu_mpc::standalone::Runtime runtime;
    const auto rin = buildRin(bin, n);
    const auto x = buildQueries(bin, rin);

    const auto totalStart = std::chrono::high_resolution_clock::now();

    const auto keygenStart = std::chrono::high_resolution_clock::now();
    auto [dcfKey0, dcfKey1] = gpu_mpc::standalone::generateDcfKeys(runtime, bin, bout, rin, T(1), true);
    const auto keygenEnd = std::chrono::high_resolution_clock::now();

    Stats uploadStats0;
    const auto uploadP0Start = std::chrono::high_resolution_clock::now();
    auto uploadedKey0 = uploadDcfKeyToGpu(dcfKey0, &uploadStats0);
    const auto uploadP0End = std::chrono::high_resolution_clock::now();

    Stats uploadStats1;
    const auto uploadP1Start = std::chrono::high_resolution_clock::now();
    auto uploadedKey1 = uploadDcfKeyToGpu(dcfKey1, &uploadStats1);
    const auto uploadP1End = std::chrono::high_resolution_clock::now();

    T *d_x = gpu_mpc::standalone::detail::copyVectorToGpu(x);

    std::vector<unsigned long long> evalTimesP0;
    std::vector<unsigned long long> evalTimesP1;
    std::vector<unsigned long long> transferTimesP0;
    std::vector<unsigned long long> transferTimesP1;
    evalTimesP0.reserve(evalIters);
    evalTimesP1.reserve(evalIters);
    transferTimesP0.reserve(evalIters);
    transferTimesP1.reserve(evalIters);

    for (int i = 0; i < evalIters; ++i)
    {
        Stats stats;
        const auto start = std::chrono::high_resolution_clock::now();
        u32 *d_out = dcf_cached::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(uploadedKey0.key, SERVER0, d_x, runtime.aes(), &stats);
        const auto end = std::chrono::high_resolution_clock::now();
        evalTimesP0.push_back(microsBetween(start, end));
        transferTimesP0.push_back(stats.transfer_time);
        gpuFree(d_out);
    }

    for (int i = 0; i < evalIters; ++i)
    {
        Stats stats;
        const auto start = std::chrono::high_resolution_clock::now();
        u32 *d_out = dcf_cached::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(uploadedKey1.key, SERVER1, d_x, runtime.aes(), &stats);
        const auto end = std::chrono::high_resolution_clock::now();
        evalTimesP1.push_back(microsBetween(start, end));
        transferTimesP1.push_back(stats.transfer_time);
        gpuFree(d_out);
    }

    const auto totalEnd = std::chrono::high_resolution_clock::now();

    validateOutputs(runtime, dcfKey0, dcfKey1, uploadedKey0, uploadedKey1, d_x, rin, x, bin, bout);

    gpuFree(d_x);
    destroyUploadedDcfKey(&uploadedKey0);
    destroyUploadedDcfKey(&uploadedKey1);

    std::printf(
        "DCF cached-key benchmark finished\n"
        "  bin: %d bit\n"
        "  bout: %d bit\n"
        "  n: %d elem\n"
        "  eval_iters: %d\n"
        "  keygen: %llu us\n"
        "  key_upload_p0: %llu us\n"
        "  key_upload_p1: %llu us\n"
        "  avg_eval_p0: %.2f us\n"
        "  avg_eval_p1: %.2f us\n"
        "  avg_transfer_p0: %.2f us\n"
        "  avg_transfer_p1: %.2f us\n"
        "  total: %llu us\n",
        bin,
        bout,
        n,
        evalIters,
        microsBetween(keygenStart, keygenEnd),
        microsBetween(uploadP0Start, uploadP0End),
        microsBetween(uploadP1Start, uploadP1End),
        averageMicros(evalTimesP0),
        averageMicros(evalTimesP1),
        averageMicros(transferTimesP0),
        averageMicros(transferTimesP1),
        microsBetween(totalStart, totalEnd));

    return 0;
}
