#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dpf_bench_common.h"
#include "fss/dpf_api.h"

using T = u64;

namespace {

constexpr int kBin = dpf_bench::kBin;
constexpr int kThreadsPerBlock = dpf_bench::kThreadsPerBlock;

struct HostSingleDpfKey
{
    AESBlock *scw = nullptr;
    AESBlock *l0 = nullptr;
    AESBlock *l1 = nullptr;
    std::vector<u32> tRLevels;
    std::size_t memSzScw = 0;
    std::size_t memSzL = 0;
};

struct DeviceSingleDpfKey
{
    AESBlock *scw = nullptr;
    AESBlock *l0 = nullptr;
    AESBlock *l1 = nullptr;
    u32 *tRLevels = nullptr;
    T *x = nullptr;
    u32 *out = nullptr;
    std::size_t outBytes = 0;
};

HostSingleDpfKey parseHostKey(const gpu_mpc::standalone::KeyBlob &hostKey)
{
    u8 *cursor = hostKey.data();
    auto parsed = readGPUDPFKey(&cursor);
    assert(parsed.bin > 7);
    assert(parsed.B == 1);

    HostSingleDpfKey host;
    const auto &src = parsed.dpfTreeKey[0];
    host.scw = src.scw;
    host.l0 = src.l0;
    host.l1 = src.l1;
    host.memSzScw = static_cast<std::size_t>(src.memSzScw);
    host.memSzL = static_cast<std::size_t>(src.memSzL);

    const int levelCount = src.bin - LOG_AES_BLOCK_LEN - 1;
    host.tRLevels.resize(static_cast<std::size_t>(std::max(levelCount, 0)));
    const auto *packed = reinterpret_cast<const u32 *>(src.tR);
    for (int i = 0; i < levelCount; ++i)
        host.tRLevels[static_cast<std::size_t>(i)] = packed[i] & 1U;

    delete[] parsed.dpfTreeKey;
    return host;
}

void uploadKeyToDevice(const HostSingleDpfKey &host, DeviceSingleDpfKey *device)
{
    moveIntoGPUMem(reinterpret_cast<u8 *>(device->scw),
        reinterpret_cast<u8 *>(host.scw),
        host.memSzScw,
        nullptr);
    moveIntoGPUMem(reinterpret_cast<u8 *>(device->l0),
        reinterpret_cast<u8 *>(host.l0),
        host.memSzL,
        nullptr);
    moveIntoGPUMem(reinterpret_cast<u8 *>(device->l1),
        reinterpret_cast<u8 *>(host.l1),
        host.memSzL,
        nullptr);
    if (!host.tRLevels.empty())
    {
        moveIntoGPUMem(reinterpret_cast<u8 *>(device->tRLevels),
            reinterpret_cast<u8 *>(const_cast<u32 *>(host.tRLevels.data())),
            sizeof(u32) * host.tRLevels.size(),
            nullptr);
    }
}

__global__ void dpfSingleKeyEvalKernel(
    int party,
    int queryCount,
    const T *in,
    const AESBlock *scw,
    const AESBlock *l0,
    const AESBlock *l1,
    const u32 *tRLevels,
    u32 *out,
    AESGlobalContext gaes)
{
    AESSharedContext saes;
    loadSbox(&gaes, &saes);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= queryCount)
        return;

    AESBlock s = scw[0];
    auto x = static_cast<u64>(in[tid]);
    auto x1 = __brevll(x) >> (64 - kBin);

    for (int i = 0; i < kBin - LOG_AES_BLOCK_LEN; ++i)
    {
        const u8 keep = lsb(x1);
        if (i < kBin - LOG_AES_BLOCK_LEN - 1)
        {
            const u32 tRLevel = tRLevels[i];
            s = expandDPFTreeNode(kBin, party, s, scw[i + 1], 0, 0, tRLevel, keep, i, &saes);
        }
        else
        {
            s = expandDPFTreeNode(kBin, party, s, 0, l0[0], l1[0], 0, keep, i, &saes);
        }
        x1 >>= 1;
    }

    const auto o = getDPFOutput(&s, x);
    writePackedOp(out, u64(o), 1, queryCount);
}

void runEval(
    const DeviceSingleDpfKey &device,
    int party,
    int queryCount,
    AESGlobalContext *aes)
{
    const int numBlocks = (queryCount - 1) / kThreadsPerBlock + 1;
    dpfSingleKeyEvalKernel<<<numBlocks, kThreadsPerBlock>>>(
        party,
        queryCount,
        device.x,
        device.scw,
        device.l0,
        device.l1,
        device.tRLevels,
        device.out,
        *aes);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void validateOutputs(
    const std::vector<T> &x,
    T alpha,
    const std::vector<u64> &timedShare,
    const std::vector<u64> &cachedShare)
{
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const u64 combined = (timedShare[i] + cachedShare[i]) & 1ULL;
        const u64 expected = static_cast<u64>(x[i] == alpha);
        if (combined != expected)
        {
            std::fprintf(stderr,
                "Validation failed at i=%zu: alpha=%llu x=%llu expected=%llu got=%llu\n",
                i,
                static_cast<unsigned long long>(alpha),
                static_cast<unsigned long long>(x[i]),
                static_cast<unsigned long long>(expected),
                static_cast<unsigned long long>(combined));
            std::exit(2);
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        dpf_bench::printSingleKeyUsage(argv[0]);
        return 1;
    }

    const int n = std::atoi(argv[1]);
    const int evalIters = std::atoi(argv[2]);
    if (n <= 0 || evalIters <= 0)
    {
        dpf_bench::printSingleKeyUsage(argv[0]);
        return 1;
    }

    gpu_mpc::standalone::Runtime runtime;
    const T alpha = 10;
    const std::vector<T> rin = {alpha};
    const auto x = dpf_bench::buildQueries(static_cast<std::size_t>(n), alpha);

    auto [dpfKey0, dpfKey1] = gpu_mpc::standalone::generateDpfKeys(runtime, kBin, rin);

    const auto hostKey0 = parseHostKey(dpfKey0);
    const auto hostKey1 = parseHostKey(dpfKey1);

    DeviceSingleDpfKey device;
    device.scw = reinterpret_cast<AESBlock *>(gpuMalloc(hostKey0.memSzScw));
    device.l0 = reinterpret_cast<AESBlock *>(gpuMalloc(hostKey0.memSzL));
    device.l1 = reinterpret_cast<AESBlock *>(gpuMalloc(hostKey0.memSzL));
    if (!hostKey0.tRLevels.empty())
    {
        device.tRLevels = reinterpret_cast<u32 *>(gpuMalloc(sizeof(u32) * hostKey0.tRLevels.size()));
    }
    device.x = reinterpret_cast<T *>(gpuMalloc(sizeof(T) * static_cast<std::size_t>(n)));
    device.outBytes = ((((std::size_t)n) - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    device.out = reinterpret_cast<u32 *>(gpuMalloc(device.outBytes));

    std::vector<u32> packedTimed(device.outBytes / sizeof(u32));
    std::vector<u32> packedCached(device.outBytes / sizeof(u32));
    std::vector<unsigned long long> evalTimes;
    evalTimes.reserve(evalIters);

    uploadKeyToDevice(hostKey1, &device);
    moveIntoGPUMem(reinterpret_cast<u8 *>(device.x),
        reinterpret_cast<u8 *>(const_cast<T *>(x.data())),
        sizeof(T) * x.size(),
        nullptr);
    runEval(device, SERVER1, n, runtime.aes());
    moveIntoCPUMem(reinterpret_cast<u8 *>(packedCached.data()),
        reinterpret_cast<u8 *>(device.out),
        device.outBytes,
        nullptr);
    const auto cachedShare =
        gpu_mpc::standalone::unpackPackedOutput(packedCached, n, 1);

    uploadKeyToDevice(hostKey0, &device);
    moveIntoGPUMem(reinterpret_cast<u8 *>(device.x),
        reinterpret_cast<u8 *>(const_cast<T *>(x.data())),
        sizeof(T) * x.size(),
        nullptr);
    runEval(device, SERVER0, n, runtime.aes());

    for (int i = 0; i < evalIters; ++i)
    {
        evalTimes.push_back(dpf_bench::measureMicros([&] {
            uploadKeyToDevice(hostKey0, &device);
            moveIntoGPUMem(reinterpret_cast<u8 *>(device.x),
                reinterpret_cast<u8 *>(const_cast<T *>(x.data())),
                sizeof(T) * x.size(),
                nullptr);
            runEval(device, SERVER0, n, runtime.aes());
            moveIntoCPUMem(reinterpret_cast<u8 *>(packedTimed.data()),
                reinterpret_cast<u8 *>(device.out),
                device.outBytes,
                nullptr);
            const auto timedShare =
                gpu_mpc::standalone::unpackPackedOutput(packedTimed, n, 1);
            validateOutputs(x, alpha, timedShare, cachedShare);
        }));
    }

    gpuFree(device.scw);
    gpuFree(device.l0);
    gpuFree(device.l1);
    if (device.tRLevels != nullptr)
        gpuFree(device.tRLevels);
    gpuFree(device.x);
    gpuFree(device.out);

    std::printf(
        "Orca DPF single-key benchmark finished\n"
        "  bin: %d bit\n"
        "  n: %d elem\n"
        "  eval_iters: %d\n"
        "  eval: %.2f us\n",
        kBin,
        n,
        evalIters,
        dpf_bench::averageMicros(evalTimes));

    return 0;
}
