#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dcf_bench_common.h"
#include "fss/dcf_api.h"

using T = u64;

namespace {

constexpr int kBin = dcf_bench::kBin;
constexpr int kBout = dcf_bench::kBout;
constexpr int kThreadsPerBlock = dcf_bench::kThreadsPerBlock;

struct HostSingleDcfKey
{
    AESBlock *scw = nullptr;
    AESBlock *leaves = nullptr;
    std::vector<u32> vcwLevels;
    std::size_t memSzScw = 0;
    std::size_t memSzL = 0;
};

struct DeviceSingleDcfKey
{
    AESBlock *scw = nullptr;
    AESBlock *leaves = nullptr;
    u32 *vcwLevels = nullptr;
    T *x = nullptr;
    u32 *out = nullptr;
    std::size_t outBytes = 0;
    u64 outStride = 0;
};

HostSingleDcfKey parseHostKey(const gpu_mpc::standalone::KeyBlob &hostKey)
{
    u8 *cursor = hostKey.data();
    auto parsed = dcf::readGPUDCFKey(&cursor);
    assert(parsed.bin > 8);
    assert(parsed.B == 1);

    HostSingleDcfKey host;
    const auto &src = parsed.dcfTreeKey[0];
    host.scw = src.scw;
    host.leaves = src.l;
    host.memSzScw = static_cast<std::size_t>(src.memSzScw);
    host.memSzL = static_cast<std::size_t>(src.memSzL);

    const int elemsPerBlock = AES_BLOCK_LEN_IN_BITS / src.bout;
    const int newBin = src.bin - int(log2(elemsPerBlock));
    const int levelCount = std::max(newBin - 1, 0);
    host.vcwLevels.resize(static_cast<std::size_t>(levelCount));
    const auto *packed = reinterpret_cast<const u32 *>(src.vcw);
    for (int i = 0; i < levelCount; ++i)
        host.vcwLevels[static_cast<std::size_t>(i)] = packed[i] & 1U;

    delete[] parsed.dcfTreeKey;
    return host;
}

void uploadKeyToDevice(const HostSingleDcfKey &host, DeviceSingleDcfKey *device)
{
    moveIntoGPUMem(reinterpret_cast<u8 *>(device->scw),
        reinterpret_cast<u8 *>(host.scw),
        host.memSzScw,
        nullptr);
    moveIntoGPUMem(reinterpret_cast<u8 *>(device->leaves),
        reinterpret_cast<u8 *>(host.leaves),
        host.memSzL,
        nullptr);
    if (!host.vcwLevels.empty())
    {
        moveIntoGPUMem(reinterpret_cast<u8 *>(device->vcwLevels),
            reinterpret_cast<u8 *>(const_cast<u32 *>(host.vcwLevels.data())),
            sizeof(u32) * host.vcwLevels.size(),
            nullptr);
    }
}

__global__ void dcfSingleKeyEvalKernel(
    int party,
    int queryCount,
    const T *in,
    const AESBlock *scw,
    const u32 *vcwLevels,
    const AESBlock *leaves,
    u32 *out,
    u64 oStride,
    AESGlobalContext gaes)
{
    AESSharedContext saes;
    loadSbox(&gaes, &saes);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= queryCount)
        return;

    auto x = u64(in[tid]);
    AESBlock s[1];
    u64 x0[1], x1[1], vAlpha[1];
    dcf::idPrologue(party, kBin, queryCount, x, x0);

    s[0] = scw[0];
    gpuMod(x0[0], kBin);
    x1[0] = __brevll(x0[0]) >> (64 - kBin);
    vAlpha[0] = 0;

    const int elemsPerBlock = AES_BLOCK_LEN_IN_BITS / kBout;
    const int levelsPacked = int(ceil(log2((double)elemsPerBlock)));
    for (int i = 0; i < kBin - levelsPacked - 1; ++i)
    {
        const auto curVcw = static_cast<u64>(vcwLevels[i]);
        const auto curScw = scw[i + 1];
        const u8 keep = lsb(x1[0]);
        s[0] = dcf::traverseOneDCF(
            kBin, kBout, party, s[0], curScw, keep, &vAlpha[0], curVcw, i, &saes);
        x1[0] >>= 1;
    }

    AESBlock l[2];
    l[0] = leaves[0];
    l[1] = leaves[1];
    AESBlock ct;
    const int j = x1[0] & 1;
    u64 offset = x0[0];
    gpuMod(offset, levelsPacked);
    const u64 t = lsb(s[0]);
    auto ss = s[0] & ~3;
    applyAESPRG(&saes, (u32 *)&ss, 2 * j + 1, (u32 *)&ct);
    const u64 v = dcf::getGroupElementFromAESBlock(ct, kBout, offset);
    const u64 curLeaf = dcf::getGroupElementFromAESBlock(l[j], kBout, offset);
    const u64 sign = party == SERVER1 ? -1 : 1;
    vAlpha[0] += (sign * (v + (t * curLeaf)));
    gpuMod(vAlpha[0], kBout);

    dcf::idEpilogue(party, kBin, kBout, queryCount, x, vAlpha, out, oStride);
}

void runEval(
    const DeviceSingleDcfKey &device,
    int party,
    int queryCount,
    AESGlobalContext *aes)
{
    const int numBlocks = (queryCount - 1) / kThreadsPerBlock + 1;
    dcfSingleKeyEvalKernel<<<numBlocks, kThreadsPerBlock>>>(
        party,
        queryCount,
        device.x,
        device.scw,
        device.vcwLevels,
        device.leaves,
        device.out,
        device.outStride,
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
        const u64 expected = static_cast<u64>(x[i] <= alpha);
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
        dcf_bench::printSingleKeyUsage(argv[0]);
        return 1;
    }

    const int n = std::atoi(argv[1]);
    const int evalIters = std::atoi(argv[2]);
    if (n <= 0 || evalIters <= 0)
    {
        dcf_bench::printSingleKeyUsage(argv[0]);
        return 1;
    }

    gpu_mpc::standalone::Runtime runtime;
    const T alpha = 20;
    const std::vector<T> rin = {alpha};
    const auto x = dcf_bench::buildQueries(static_cast<std::size_t>(n), alpha);

    auto [dcfKey0, dcfKey1] =
        gpu_mpc::standalone::generateDcfKeys(runtime, kBin, kBout, rin, T(1), true);

    const auto hostKey0 = parseHostKey(dcfKey0);
    const auto hostKey1 = parseHostKey(dcfKey1);

    DeviceSingleDcfKey device;
    device.scw = reinterpret_cast<AESBlock *>(gpuMalloc(hostKey0.memSzScw));
    device.leaves = reinterpret_cast<AESBlock *>(gpuMalloc(hostKey0.memSzL));
    if (!hostKey0.vcwLevels.empty())
    {
        device.vcwLevels = reinterpret_cast<u32 *>(
            gpuMalloc(sizeof(u32) * hostKey0.vcwLevels.size()));
    }
    device.x = reinterpret_cast<T *>(gpuMalloc(sizeof(T) * static_cast<std::size_t>(n)));
    device.outBytes = ((((std::size_t)kBout * static_cast<std::size_t>(n)) - 1) / PACKING_SIZE + 1) *
        sizeof(PACK_TYPE);
    device.out = reinterpret_cast<u32 *>(gpuMalloc(device.outBytes));
    device.outStride = device.outBytes / sizeof(PACK_TYPE);

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
        gpu_mpc::standalone::unpackPackedOutput(packedCached, n, kBout);

    uploadKeyToDevice(hostKey0, &device);
    moveIntoGPUMem(reinterpret_cast<u8 *>(device.x),
        reinterpret_cast<u8 *>(const_cast<T *>(x.data())),
        sizeof(T) * x.size(),
        nullptr);
    runEval(device, SERVER0, n, runtime.aes());

    for (int i = 0; i < evalIters; ++i)
    {
        evalTimes.push_back(dcf_bench::measureMicros([&] {
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
                gpu_mpc::standalone::unpackPackedOutput(packedTimed, n, kBout);
            validateOutputs(x, alpha, timedShare, cachedShare);
        }));
    }

    gpuFree(device.scw);
    gpuFree(device.leaves);
    if (device.vcwLevels != nullptr)
        gpuFree(device.vcwLevels);
    gpuFree(device.x);
    gpuFree(device.out);

    std::printf(
        "Orca DCF single-key benchmark finished\n"
        "  bin: %d bit\n"
        "  bout: %d bit\n"
        "  n: %d elem\n"
        "  eval_iters: %d\n"
        "  eval: %.2f us\n",
        kBin,
        kBout,
        n,
        evalIters,
        dcf_bench::averageMicros(evalTimes));

    return 0;
}
