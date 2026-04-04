#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "fss/dpf_api.h"
#include "gpu/gpu_stats.h"

using T = u64;

namespace
{

constexpr int kBin = 64;

struct UploadedDpfKey
{
    GPUDPFKey key{};
};

void printUsage(const char *prog)
{
    std::fprintf(stderr, "Usage: %s <n> <eval_iters>\n", prog);
}

std::vector<T> buildRin(int n)
{
    std::vector<T> rin(n);
    for (int i = 0; i < n; ++i)
        rin[i] = T(10) + T(2) * i;
    return rin;
}

std::vector<T> buildQueries(const std::vector<T> &rin)
{
    std::vector<T> x(rin.size());
    const T limit = ~T(0);
    for (std::size_t i = 0; i < rin.size(); ++i)
        x[i] = (i % 3 == 0 || rin[i] + 1 >= limit) ? rin[i] : (rin[i] + 1);
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

void freeParsedDpfKey(GPUDPFKey &key)
{
    if (key.bin > 7)
    {
        delete[] key.dpfTreeKey;
        key.dpfTreeKey = nullptr;
    }
}

UploadedDpfKey uploadDpfKeyToGpu(const gpu_mpc::standalone::KeyBlob &hostKey, Stats *stats)
{
    UploadedDpfKey uploaded;

    u8 *cursor = hostKey.data();
    auto parsed = readGPUDPFKey(&cursor);
    uploaded.key = parsed;

    if (parsed.bin <= 7)
    {
        uploaded.key.ssKey = parsed.ssKey;
        uploaded.key.ssKey.ss = reinterpret_cast<u8 *>(
            moveToGPU(reinterpret_cast<u8 *>(parsed.ssKey.ss), parsed.ssKey.memSzSS, stats));
        return uploaded;
    }

    uploaded.key.dpfTreeKey = new GPUDPFTreeKey[parsed.B];
    for (int b = 0; b < parsed.B; ++b)
    {
        const auto &src = parsed.dpfTreeKey[b];
        auto &dst = uploaded.key.dpfTreeKey[b];
        dst = src;
        dst.scw = reinterpret_cast<AESBlock *>(moveToGPU(reinterpret_cast<u8 *>(src.scw), src.memSzScw, stats));
        dst.l0 = reinterpret_cast<AESBlock *>(moveToGPU(reinterpret_cast<u8 *>(src.l0), src.memSzL, stats));
        dst.l1 = reinterpret_cast<AESBlock *>(moveToGPU(reinterpret_cast<u8 *>(src.l1), src.memSzL, stats));
        dst.tR = reinterpret_cast<u32 *>(moveToGPU(reinterpret_cast<u8 *>(src.tR), src.memSzT, stats));
    }

    delete[] parsed.dpfTreeKey;
    return uploaded;
}

void destroyUploadedDpfKey(UploadedDpfKey *uploaded)
{
    if (uploaded->key.bin <= 7)
    {
        gpuFree(uploaded->key.ssKey.ss);
        uploaded->key.ssKey.ss = nullptr;
        return;
    }

    for (int b = 0; b < uploaded->key.B; ++b)
    {
        gpuFree(uploaded->key.dpfTreeKey[b].scw);
        gpuFree(uploaded->key.dpfTreeKey[b].l0);
        gpuFree(uploaded->key.dpfTreeKey[b].l1);
        gpuFree(uploaded->key.dpfTreeKey[b].tR);
    }
    delete[] uploaded->key.dpfTreeKey;
    uploaded->key.dpfTreeKey = nullptr;
}

template <typename TValue, treeTraversal Traversal>
void gpuDpfTreeEvalCached(GPUDPFTreeKey key, int party, TValue *d_in, AESGlobalContext *aes, u32 *d_out, u64 outStride)
{
    const int tbSize = 256;
    const int numBlocks = (key.N - 1) / tbSize + 1;
    dpfTreeEval<TValue, Traversal><<<numBlocks, tbSize>>>(
        party, key.bin, key.N, d_in, key.scw, key.l0, key.l1, key.tR, d_out, outStride, *aes);
    checkCudaErrors(cudaDeviceSynchronize());
}

template <typename TValue, int E, dpfPrologue Prologue, dpfEpilogue Epilogue>
u32 *gpuLookupSSTableCached(GPUSSTabKey &key, int party, TValue *d_in, Stats *stats)
{
    auto *d_out = moveMasks(key.memSzOut, nullptr, stats);
    lookupSSTable<TValue, E, Prologue, Epilogue>
        <<<(key.N - 1) / 128 + 1, 128>>>(party, key.bin, key.N, d_in, key.ss, d_out);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_out;
}

template <typename TValue>
u32 *gpuDpfCached(const GPUDPFKey &key, int party, TValue *d_in, AESGlobalContext *aes, Stats *stats)
{
    if (key.bin <= 7)
        return gpuLookupSSTableCached<TValue, 1, idPrologue, idEpilogue>(const_cast<GPUSSTabKey &>(key.ssKey), party, d_in, stats);

    auto *d_out = moveMasks(key.memSzOut, nullptr, stats);
    const int blockInputSize = key.dpfTreeKey[0].N;
    const size_t globalWords = key.memSzOut / sizeof(PACK_TYPE);
    const size_t blockWords = key.dpfTreeKey[0].memSzOut / sizeof(PACK_TYPE);
    for (int b = 0; b < key.B; ++b)
    {
        gpuDpfTreeEvalCached<TValue, doDpf>(
            key.dpfTreeKey[b], party, d_in + b * blockInputSize, aes, d_out + b * blockWords, static_cast<u64>(globalWords));
    }
    return d_out;
}

std::vector<u32> evalPackedWithHostKey(
    gpu_mpc::standalone::Runtime &runtime,
    const gpu_mpc::standalone::KeyBlob &key,
    int party,
    T *d_x)
{
    u8 *cursor = key.data();
    auto parsedKey = readGPUDPFKey(&cursor);
    u32 *d_out = gpuDpf(parsedKey, party, d_x, runtime.aes(), nullptr);
    auto packed = gpu_mpc::standalone::detail::copyPackedOutputToHost(d_out, parsedKey.memSzOut);
    freeParsedDpfKey(parsedKey);
    return packed;
}

std::vector<u32> evalPackedWithCachedKey(
    gpu_mpc::standalone::Runtime &runtime,
    const UploadedDpfKey &key,
    int party,
    T *d_x)
{
    u32 *d_out = gpuDpfCached(key.key, party, d_x, runtime.aes(), nullptr);
    return gpu_mpc::standalone::detail::copyPackedOutputToHost(d_out, key.key.memSzOut);
}

void validateOutputs(
    gpu_mpc::standalone::Runtime &runtime,
    const gpu_mpc::standalone::KeyBlob &hostKey0,
    const gpu_mpc::standalone::KeyBlob &hostKey1,
    const UploadedDpfKey &cachedKey0,
    const UploadedDpfKey &cachedKey1,
    T *d_x,
    const std::vector<T> &rin,
    const std::vector<T> &x)
{
    const auto hostPacked0 = evalPackedWithHostKey(runtime, hostKey0, SERVER0, d_x);
    const auto hostPacked1 = evalPackedWithHostKey(runtime, hostKey1, SERVER1, d_x);
    const auto cachedPacked0 = evalPackedWithCachedKey(runtime, cachedKey0, SERVER0, d_x);
    const auto cachedPacked1 = evalPackedWithCachedKey(runtime, cachedKey1, SERVER1, d_x);

    assert(hostPacked0 == cachedPacked0);
    assert(hostPacked1 == cachedPacked1);

    const auto share0 = gpu_mpc::standalone::unpackPackedOutput(cachedPacked0, static_cast<int>(x.size()), 1);
    const auto share1 = gpu_mpc::standalone::unpackPackedOutput(cachedPacked1, static_cast<int>(x.size()), 1);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const u64 combined = (share0[i] + share1[i]) & 1ULL;
        const u64 expected = static_cast<u64>(x[i] == rin[i]);
        assert(combined == expected);
    }
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printUsage(argv[0]);
        return 1;
    }

    const int n = std::atoi(argv[1]);
    const int evalIters = std::atoi(argv[2]);
    if (n <= 0 || evalIters <= 0)
    {
        printUsage(argv[0]);
        return 1;
    }

    gpu_mpc::standalone::Runtime runtime;
    const auto rin = buildRin(n);
    const auto x = buildQueries(rin);

    const auto totalStart = std::chrono::high_resolution_clock::now();

    const auto keygenStart = std::chrono::high_resolution_clock::now();
    auto [dpfKey0, dpfKey1] = gpu_mpc::standalone::generateDpfKeys(runtime, kBin, rin);
    const auto keygenEnd = std::chrono::high_resolution_clock::now();

    Stats uploadStats0;
    const auto uploadP0Start = std::chrono::high_resolution_clock::now();
    auto uploadedKey0 = uploadDpfKeyToGpu(dpfKey0, &uploadStats0);
    const auto uploadP0End = std::chrono::high_resolution_clock::now();

    Stats uploadStats1;
    const auto uploadP1Start = std::chrono::high_resolution_clock::now();
    auto uploadedKey1 = uploadDpfKeyToGpu(dpfKey1, &uploadStats1);
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
        u32 *d_out = gpuDpfCached(uploadedKey0.key, SERVER0, d_x, runtime.aes(), &stats);
        const auto end = std::chrono::high_resolution_clock::now();
        evalTimesP0.push_back(microsBetween(start, end));
        transferTimesP0.push_back(stats.transfer_time);
        gpuFree(d_out);
    }

    for (int i = 0; i < evalIters; ++i)
    {
        Stats stats;
        const auto start = std::chrono::high_resolution_clock::now();
        u32 *d_out = gpuDpfCached(uploadedKey1.key, SERVER1, d_x, runtime.aes(), &stats);
        const auto end = std::chrono::high_resolution_clock::now();
        evalTimesP1.push_back(microsBetween(start, end));
        transferTimesP1.push_back(stats.transfer_time);
        gpuFree(d_out);
    }

    const auto totalEnd = std::chrono::high_resolution_clock::now();

    validateOutputs(runtime, dpfKey0, dpfKey1, uploadedKey0, uploadedKey1, d_x, rin, x);

    gpuFree(d_x);
    destroyUploadedDpfKey(&uploadedKey0);
    destroyUploadedDpfKey(&uploadedKey1);

    std::printf(
        "DPF cached-key benchmark finished\n"
        "  bin: %d bit\n"
        "  n: %d elem\n"
        "  eval_iters: %d\n"
        "  keygen: %llu us\n"
        "  key_upload_p0: %llu us\n"
        "  key_upload_p1: %llu us\n"
        "  key_upload_transfer_p0: %llu us\n"
        "  key_upload_transfer_p1: %llu us\n"
        "  avg_eval_p0: %.2f us\n"
        "  avg_eval_p1: %.2f us\n"
        "  avg_transfer_p0: %.2f us\n"
        "  avg_transfer_p1: %.2f us\n"
        "  total: %llu us\n",
        kBin,
        n,
        evalIters,
        microsBetween(keygenStart, keygenEnd),
        microsBetween(uploadP0Start, uploadP0End),
        microsBetween(uploadP1Start, uploadP1End),
        static_cast<unsigned long long>(uploadStats0.transfer_time),
        static_cast<unsigned long long>(uploadStats1.transfer_time),
        averageMicros(evalTimesP0),
        averageMicros(evalTimesP1),
        averageMicros(transferTimesP0),
        averageMicros(transferTimesP1),
        microsBetween(totalStart, totalEnd));

    return 0;
}
