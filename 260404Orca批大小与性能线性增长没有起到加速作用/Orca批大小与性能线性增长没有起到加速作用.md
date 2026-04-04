- 我的研究方向是 `DPF` / `DCF` 的 `GPU` 加速实现

- `Orca` 项目提取出来的项目为 `/GPU-MPC` ，剥离出来的FSS函数为 `/FSS`

- `myl7` 项目为 `/fss`

- `libfss` 项目为 `/libfss`

- 研究该问题产生的测试文件只能写入当前文档所在文件夹，非必要不要更改其他文件夹内容（其他文件夹有其他AI在工作，你只能看）

- 临时文件请写入

- 研究该问题的流程写入本文档

- 不要修改本条及以上条目内容

# `Orca` 增加批大小好像不能等价于提高 GPU 并行能力，批大小从一百万到一千万这一块，基本上是线性增长的，似乎没有起到加速作用

研究并解释，给出为什么线性增长原因

## 2026-04-04 第一轮结论

先说当前结论：

- 这里看到的“一百万到一千万基本线性增长”，大概率不是 `GPU` 没并行，而是已经进入了“吞吐基本稳定”的区间
- 在 `Orca/FSS` 这条链路里，增大 `n` 不只是“多给 GPU 一些线程”
- 它同时意味着：
  - 更多独立实例要做完整的 `DPF/DCF` 树遍历
  - 更大的 key blob 要从 `CPU` 搬到 `GPU`
  - 更大的 key blob 要在 keygen 阶段从 `GPU` 拷回 `CPU`
- 当前实现的 `eval` 不是纯 kernel benchmark，而是“搬 key 到 GPU + 启动 kernel + 同步”的整段调用
- 实测里 `eval` 时间有大约 `85% ~ 91%` 都落在 `transfer_time` 上，说明百万级以后主要被 host-device 数据搬运拖着线性增长

所以：

- “批大小变大为什么没有额外加速”
- 更准确的说法不是“GPU 没并行”
- 而是“在一百万这个量级上，并行度已经足够，继续增大 `n` 主要是在增加总工作量和总搬运量，因此总时间接近线性增长”

## 1. 先澄清这里的“批大小”到底是什么

这里先统一术语：

- 本文讨论的“批大小”，对应 `benchmark` 入口里的参数 `n`
- 也就是一次 `DPF/DCF keygen/eval` 中要处理的独立实例数
- 不是神经网络框架意义上的 `batchSz`

从 benchmark 入口就能直接看到：

- `dpf_benchmark` 接口是 `<bin> <n>`
- `dcf_benchmark` 接口是 `<bin> <bout> <n>`

对应代码：

- `FSS/dpf_benchmark.cu:64-65`
- `FSS/dcf_benchmark.cu:71-73`

## 2. 代码里已经能直接确认的几件事

### 2.1 `eval_p0 / eval_p1` 不是纯 kernel 时间

`benchmark` 的计时方式是：

- 先构造 host 侧 `KeyBlob`
- 再解析 key
- 再调用 `gpuDpf(...)` 或 `gpuDcf(...)`
- 真正的输出回传和 CPU 解包发生在 `evalP0End / evalP1End` 之后

对应代码：

- `FSS/dpf_benchmark.cu:80-97`
- `FSS/dpf_benchmark.cu:105-109`
- `FSS/dcf_benchmark.cu:88-105`
- `FSS/dcf_benchmark.cu:113-117`

因此：

- `eval_p0 / eval_p1` 统计的是 `gpuDpf/gpuDcf` 这层 facade 调用
- 它里面已经包含了 key 从 `CPU -> GPU` 的搬运
- 但不包含最后 `d_out -> host` 的输出回传

### 2.2 `transfer_p0 / transfer_p1` 主要统计的是 key 搬运，不是完整总传输

当前 `transfer_time` 来自 `moveToGPU / moveIntoGPUMem / moveToCPU / moveIntoCPUMem` 内部对 `cudaMemcpy` 的计时：

- `FSS/gpu/gpu_mem.cu:87-129`

但 benchmark 里的两段传输没有记到 `Stats` 里：

- 输入 `x` 拷到 GPU 时传的是 `nullptr`
- 输出 `d_out` 拷回 host 时传的也是 `nullptr`

对应代码：

- `FSS/runtime/standalone_runtime.h:109-115`
- `FSS/runtime/standalone_runtime.h:118-125`

所以当前字段含义更准确地说是：

- `transfer_p0 / transfer_p1`
- 主要是在统计 `gpuDpf/gpuDcf` 内部把 key 材料从 host 搬到 device 的时间
- 它其实低估了整条链路的总搬运量

### 2.3 `eval` 时每次都会把整份 tree key 从 host 再搬回 GPU

`DPF` 的 tree eval 路径：

- `gpuDpfTreeEval(...)` 一上来就把 `scw / l0 / l1 / tR` 全部 `moveToGPU`
- 然后启动 kernel
- 然后 `cudaDeviceSynchronize()`
- 然后释放这些临时 GPU 缓冲区

对应代码：

- `FSS/fss/gpu_dpf.cu:212-230`

`DCF` 也是同样结构：

- `gpuDcfTreeEval(...)` 每次都把 `scw / vcw / l` 搬到 GPU
- 然后启动 kernel
- 然后同步

对应代码：

- `FSS/fss/gpu_dcf.cu:123-140`

这意味着：

- key 并不是“长期驻留在 GPU 上”
- 每次 eval 都要重新拷一遍 key
- 如果 key 的大小和 `n` 线性增长，那么 eval 的一大块时间天然也会线性增长

### 2.4 没有把传输和计算 overlap 起来

当前实现里用的是：

- 同步 `cudaMemcpy`
- 显式 `cudaDeviceSynchronize()`

对应代码：

- `FSS/gpu/gpu_mem.cu:121-129`
- `FSS/fss/gpu_dpf.cu:224-225`
- `FSS/fss/gpu_dcf.cu:135-137`

所以这条链路是：

- 先搬
- 再算
- 再同步

而不是：

- 一边搬下一批
- 一边算当前批

## 3. 为什么从算法和 kernel 结构上看，本来就会接近线性

### 3.1 每个线程就是处理一个实例

`DPF`：

- `dpfTreeEval` 里 `tid` 对应一个输入
- `doDpf(...)` 会沿着整棵树一路往下走

对应代码：

- `FSS/fss/gpu_dpf.cu:202-207`
- `FSS/fss/gpu_dpf.cu:77-102`

`DCF`：

- `doDcf` kernel 里也是 `tid` 对应一个输入
- 每个线程自己把这一条实例的树遍历做完

对应代码：

- `FSS/fss/gpu_dcf.cu:58-118`

### 3.2 固定 `bin=64` 时，每个实例的工作量基本是常数

对 `DPF`：

- 主循环次数是 `bin - LOG_AES_BLOCK_LEN`
- 当前代码里 `LOG_AES_BLOCK_LEN = 7`
- 所以 `bin=64` 时，每条实例大约要走 `57` 层

对应代码：

- `FSS/fss/gpu_dpf.cu:85`

对 `DCF`：

- 主循环次数是 `bin - levelsPacked - 1`
- `bout=1` 时 `elemsPerBlock = 128`，`levelsPacked = 7`
- 所以 `bin=64, bout=1` 时也是 `56` 层主循环，再加末层处理

对应代码：

- `FSS/fss/gpu_dcf.cu:85-98`

因此在固定 `bin/bout` 下：

- 单条实例成本近似常数
- 总计算量近似 `O(n)`

如果一百万时已经把 GPU 填满了，那么再把 `n` 增到一千万：

- 不会把“每条实例更快做完”
- 只会在相近吞吐率下做十倍左右的实例

### 3.3 一百万个元素早就足够把 launch 网格铺得很满

当前线程块大小固定是 `256`：

- `FSS/fss/gpu_dpf.cu:222-223`
- `FSS/fss/gpu_dcf.cu:125-127`

所以：

- `n = 1,000,000` 时，线程块数约为 `3907`
- `n = 10,000,000` 时，线程块数约为 `39063`

这说明从 launch 数量上看：

- 一百万这个量级已经远远不是“并行度不够”
- 后面增长的主要意义不是“解锁更多并行”
- 而是“继续塞更多总工作量”

## 4. key 大小本身就是线性增长，这一点非常关键

`standalone_runtime.h` 里已经把 key 大小公式写出来了：

- `DPF`: `dpfTreeKeyBytes(...)`
- `DCF`: `dcfTreeKeyBytes(...)`

对应代码：

- `FSS/runtime/standalone_runtime.h:54-61`
- `FSS/runtime/standalone_runtime.h:79-86`

对当前最关心的口径：

- `bin = 64`
- `bout = 1`

可直接化简出：

- `DPF` 每方 key 大约是 `951` byte / 元素
- `DCF` 每方 key 大约也是 `951` byte / 元素

所以量级上就是：

- `100,000` 个元素时，每方 key 大约 `90.7 MiB`
- `1,000,000` 个元素时，每方 key 大约 `906.9 MiB`
- `10,000,000` 个元素时，每方 key 大约 `9.07 GiB`

这件事非常重要，因为当前 `eval` 会在每次调用时把这整份 key 再搬到 GPU。

也就是说：

- 你把 `n` 从一百万加到一千万
- 不只是让 kernel 多算了十倍元素
- 还让每方要搬的 key 从不到 `1 GiB` 变成了大约 `9 GiB`

## 5. 当前实现里的“batch”并不会在一千万这里帮你拆开

代码里确实有 batch 预算：

- `kDpfBatchBudgetBytes = 24 GiB`

对应代码：

- `FSS/runtime/standalone_runtime.h:22-25`

而 `gpuKeyGenBatchedDPF` / `gpuKeyGenDCF` 里真正决定 `B` 的公式是：

- `m = (24 GiB) / bytes_per_elem`
- `B = ceil(N / m)`

对应代码：

- `FSS/fss/gpu_dpf.cu:408-420`
- `FSS/fss/gpu_dcf.cu:333-356`

对 `bin=64, bout=1`，每元素预算约 `944` byte，算出来：

- `m = 27,298,496`

也就是：

- `N = 10,000,000` 时，仍然有 `B = 1`
- 所以代码并不会把一千万自动切成多个更小的 GPU batch
- 它仍然会把这一整个超大 key 当成一批来做

这进一步解释了为什么：

- 这里的“增大批大小”
- 并不等价于“更高效地利用 GPU batch pipeline”
- 它更像是在把“单次任务的数据体量”直接做大

## 6. 本地实测

### 6.1 实测环境

- 日期：`2026-04-04`
- GPU：`NVIDIA GeForce RTX 3070 Laptop GPU`
- 显存：`8192 MiB`
- Driver：`591.74`
- CUDA：`13.1`

### 6.2 新增分析文件

本轮只在当前目录新增了分析文件：

- `run_orca_scaling_analysis.py`
- `orca_scaling_results.json`
- `orca_scaling_results_2m.json`

脚本作用：

- 调现成 `FSS/build/dpf_benchmark` 和 `FSS/build/dcf_benchmark`
- 采集不同 `n` 的时间
- 按代码公式计算理论 key 大小

### 6.3 DCF 实测摘要

`DCF` 更接近 `Orca` 的 `relu/maxpool` 路径，所以先看它：

| n | 每方 key 大小 (MiB) | eval_avg (us) | transfer_avg (us) | transfer 占比 | eval-transfer (us) | eval us/elem |
|---:|---:|---:|---:|---:|---:|---:|
| 100000 | 90.694 | 16929.5 | 14398.5 | 85.0% | 2531.0 | 0.169295 |
| 200000 | 181.389 | 33037.5 | 28643.0 | 86.7% | 4394.5 | 0.165187 |
| 500000 | 453.472 | 81172.0 | 71062.0 | 87.5% | 10110.0 | 0.162344 |
| 1000000 | 906.944 | 151713.0 | 132096.5 | 87.1% | 19616.5 | 0.151713 |
| 2000000 | 1813.889 | 329442.0 | 287874.0 | 87.4% | 41568.0 | 0.164721 |

观察：

- `eval_avg` 随 `n` 基本线性增长
- `eval us/elem` 基本稳定在 `0.15 ~ 0.17`
- `transfer_avg` 长期占 `eval` 的大头，约 `85% ~ 87%`
- 把 `transfer` 扣掉以后，剩下的“更像纯计算”的部分只占很小一块

### 6.4 DPF 实测摘要

`DPF` 也有同样趋势：

| n | 每方 key 大小 (MiB) | eval_avg (us) | transfer_avg (us) | transfer 占比 | eval-transfer (us) | eval us/elem |
|---:|---:|---:|---:|---:|---:|---:|
| 100000 | 90.706 | 16516.0 | 14967.5 | 90.6% | 1548.5 | 0.165160 |
| 200000 | 181.413 | 32976.5 | 29573.0 | 89.7% | 3403.5 | 0.164882 |
| 500000 | 453.532 | 95059.5 | 85544.0 | 90.0% | 9515.5 | 0.190119 |
| 1000000 | 907.063 | 156759.0 | 142052.0 | 90.6% | 14707.0 | 0.156759 |
| 2000000 | 1814.127 | 315837.0 | 288071.5 | 91.2% | 27765.5 | 0.157918 |

结论和 `DCF` 一样：

- 百万级以后，`eval us/elem` 没有因为 `n` 更大而明显下降
- 说明吞吐率已经比较稳定
- 继续增大 `n` 只是在稳定吞吐率下做更多元素

### 6.5 第二轮：只看 compute-only kernel

你提的这个点是对的：

- 传输线性，容易理解
- 但“纯计算为什么也近似线性”，需要单独拿出来看

所以我又补了一轮 `compute-only` benchmark。

新增文件：

- `orca_compute_only_benchmark.cu`
- `run_orca_compute_only_scaling.py`
- `orca_compute_only_results.json`

这轮 benchmark 的做法是：

- 先把输入和中间 key 缓冲区都准备在 GPU 上
- 然后用 `CUDA event` 只包住 kernel
- 不再把 `CPU <-> GPU` 搬运时间算进来

具体量的 kernel 是：

- `DPF keygen`: `keyGenDPFTreeKernel`
- `DPF eval`: `dpfTreeEval<T, doDpf>`
- `DCF keygen`: `dcf::keyGenDCFKernel`
- `DCF eval`: `dcf::doDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>`

这几个 kernel 的调用都在：

- `orca_compute_only_benchmark.cu`

因为本机只有 `8 GiB` 显存，`10M` 的整份 key 放不下，所以这轮 `10M` 纯计算测试用的是：

- `chunk_n = 1,000,000`
- 重放 `10` 次同规格 kernel
- 把纯 kernel 时间直接相加

这个做法对“计算是否线性”是成立的，因为这里没有跨元素依赖，也没有跨 chunk 共享状态。

#### DPF compute-only 结果

| 总元素数 total_n | keygen_avg_kernel (us) | eval_avg_kernel (us) | keygen us/elem | eval us/elem |
|---:|---:|---:|---:|---:|
| 1000000 | 49619.455 | 14758.912 | 0.049619455 | 0.014758912 |
| 2000000 | 100172.798 | 30010.368 | 0.050086399 | 0.015005184 |
| 5000000 | 250023.422 | 77167.103 | 0.050004684 | 0.015433421 |
| 10000000 | 510792.175 | 146991.615 | 0.051079218 | 0.014699162 |

观察：

- `1M -> 10M` 时，`eval_avg_kernel` 从 `14.76 ms` 变成 `146.99 ms`
- 增长约 `9.96x`
- 和元素数增长的 `10x` 基本一致
- `eval us/elem` 基本稳定在 `0.0147 ~ 0.0154`

#### DCF compute-only 结果

| 总元素数 total_n | keygen_avg_kernel (us) | eval_avg_kernel (us) | keygen us/elem | eval us/elem |
|---:|---:|---:|---:|---:|
| 1000000 | 100618.240 | 23298.048 | 0.100618240 | 0.023298048 |
| 2000000 | 201956.352 | 47304.705 | 0.100978176 | 0.023652352 |
| 5000000 | 513498.108 | 113275.906 | 0.102699622 | 0.022655181 |
| 10000000 | 1011999.756 | 226885.124 | 0.101199976 | 0.022688512 |

观察：

- `1M -> 10M` 时，`eval_avg_kernel` 从 `23.30 ms` 变成 `226.89 ms`
- 增长约 `9.74x`
- 也和元素数增长的 `10x` 很接近
- `eval us/elem` 基本稳定在 `0.0227 ~ 0.0237`

#### direct 2M 单次 launch 交叉验证

为了避免“1M chunk 重放”本身带来的疑问，我又额外做了：

- `2M` 单次 direct launch
- 每个模式跑 `3` 次，取 `eval_avg_kernel` 中位数对应的结果

结果：

- `DPF direct 2M`
  - `eval us/elem = 0.014900224`
  - `keygen us/elem = 0.050360577`
- `DCF direct 2M`
  - `eval us/elem = 0.022883584`
  - `keygen us/elem = 0.101121281`

这和上面 `1M chunk` 重放出来的 per-element 时间非常接近，说明：

- “纯计算近似线性”不是 chunk 重放方法伪造出来的
- 而是 kernel 本身就确实表现成近似线性的 per-element 固定成本

#### WSL 安全参数复测

考虑到大参数会把当前 `WSL` 顶得很紧，我又补了一轮只用小参数的 direct-launch 复测，并把脚本默认值也改成了安全档：

- `run_orca_compute_only_scaling.py` 默认现在跑：
  - `100000`
  - `200000`
  - `500000`
  - `1000000`

对应结果文件：

- `orca_compute_only_results_small.json`

`DPF` compute-only 小参数结果：

| total_n | keygen_avg_kernel (us) | eval_avg_kernel (us) | keygen us/elem | eval us/elem |
|---:|---:|---:|---:|---:|
| 100000 | 4174.336 | 1333.760 | 0.041743360 | 0.013337600 |
| 200000 | 8465.920 | 2828.800 | 0.042329600 | 0.014144000 |
| 500000 | 25614.336 | 7843.840 | 0.051228672 | 0.015687680 |
| 1000000 | 51266.048 | 15519.744 | 0.051266048 | 0.015519744 |

`DCF` compute-only 小参数结果：

| total_n | keygen_avg_kernel (us) | eval_avg_kernel (us) | keygen us/elem | eval us/elem |
|---:|---:|---:|---:|---:|
| 100000 | 8711.168 | 2237.440 | 0.087111678 | 0.022374400 |
| 200000 | 18496.511 | 4830.208 | 0.092482557 | 0.024151040 |
| 500000 | 50120.192 | 11914.752 | 0.100240383 | 0.023829504 |
| 1000000 | 98922.497 | 23627.264 | 0.098922497 | 0.023627264 |

这轮的意义是：

- 不需要碰 `2M / 5M / 10M`
- 单看 `100k -> 1M`，也已经能看到纯计算在快速收敛到稳定的 per-element 成本
- 所以就算不再跑更大的参数，当前结论也已经比较稳

#### 为什么纯计算本身也会近似线性

结合 kernel 代码，原因其实很直接：

1. `DPF/DCF eval` 都是“一线程处理一个实例”
2. 固定 `bin=64` 时，每个线程都要走固定深度的树
3. 不存在随着 `n` 变大而减少的共享工作
4. 也没有全局 reduction / scan / sort 这类会改变复杂度形态的步骤
5. 输出打包 `writePackedOp(...)` 虽然用了 warp shuffle，但它是每个 warp 局部的常数开销，不会随着总 `n` 增大而摊薄出新的加速

也就是说，纯计算更像下面这个模型：

- 总计算时间 ≈ `n * 每元素固定工作量 / GPU 稳定吞吐`

当 `1M` 已经足够把 GPU 填满时：

- 再把规模提到 `10M`
- 不会让“每个元素更便宜”
- 只会让 GPU 在大致相同吞吐下持续做更久

所以纯计算本身也会近似线性增长。

## 7. 这组数据怎么解释“一百万到一千万为什么近似线性”

现在可以把原因说得更具体一些：

### 7.1 不是“并行没有发生”，而是“并行早就已经够了”

一百万规模时：

- launch 的 block 数已经很多
- 每线程处理一条实例
- 每条实例的树遍历深度又是固定的

所以此时再增大 `n`：

- 不会显著降低单元素成本
- 更像是在相同吞吐率下继续灌入更多任务

而且第二轮 `compute-only` 结果已经直接说明：

- 即使把 host-device 传输全部剥掉
- `DPF/DCF` 的 kernel 时间也依然接近按 `n` 线性增长
- 这说明线性不只是“传输问题”
- 也是当前算法映射到 GPU 后的自然复杂度形态

### 7.2 `Orca/FSS` 的 `eval` 被 key 搬运强烈主导

实测里：

- `DCF transfer / eval` 约 `85% ~ 87%`
- `DPF transfer / eval` 约 `90%`

这说明“线性增长”的核心原因之一非常朴素：

- key 大小本身随 `n` 线性增长
- 每次 eval 又要把整份 key 从 `CPU` 搬到 `GPU`
- 而且没有做传输-计算 overlap

因此总时间自然会很接近线性增长。

### 7.3 `10,000,000` 这个量级下，key 已经大到不能忽略

按代码公式：

- `10,000,000` 个元素时
- 每方 key 大约 `9.07 GiB`

这意味着即使不考虑 kernel：

- 光把 key 搬一遍到 GPU
- 就已经是一个很重的成本

结合本地实测中大约 `6 GiB/s` 左右的有效搬运速度看：

- `10M` 时单方 eval 的 host->device 搬运时间就会在 `1.4s` 左右

这部分是基于 `1M ~ 2M` 实测的推断，不是本机直接跑出来的 `10M` 数据。

### 7.4 当前机器上其实也不适合直接验证 `10M`

当前本机只有 `8 GiB` 显存。

而在当前实现里：

- `10M` 时每方 key 大约 `9.07 GiB`
- 且 `B=1`

所以这台机器不适合直接跑 `10M` 来做本地复现实验。

但这不影响对“为什么会线性增长”的解释，因为：

- 代码的数据流已经说明了 key 大小和搬运量是线性的
- `100k -> 2M` 的本地实测也已经显示出同样趋势

## 8. 为什么这个结论会传到 `Orca` 上层算子

`Orca` 上层的 `relu` / `maxpool` 走的就是 `DCF` 相关路径：

- 推理/执行时读 key，再调用 `gpuRelu* / gpuMaxPool`
- keygen 时生成对应的 `DCF` key

对应代码：

- `GPU-MPC/backend/orca.h:44-77`
- `GPU-MPC/backend/orca.h:153-180`

所以如果上层算子底下还是：

- key 先落在 host
- eval 时再搬回 GPU
- 批量规模变大时 key 大小线性增长

那么上层看见的时间趋势也会继承这种线性特征。

## 9. 当前阶段的结论

可以先把结论定成下面这句：

- `Orca` 里把 `n` 从一百万加到一千万，没有换来额外“加速”，主要不是因为 GPU 不并行
- 而是因为这一档规模下 GPU 吞吐已经接近稳定，继续增大 `n` 主要是在增加总实例数和总 key 搬运量
- 第二轮 `compute-only` benchmark 进一步表明：就算把传输剥掉，`DPF/DCF` 的 keygen/eval kernel 也依然近似线性，因为它们本身就是“每元素固定成本、无跨元素共享摊薄”的实现
- 当前实现又恰好把 eval 做成了“每次重新把整份 key 从 CPU 搬回 GPU”的形态，所以总体时间就表现为近似线性增长

## 10. 下一步可继续验证的方向

如果继续深挖，可以优先做下面几件事：

1. 做一个“key 常驻 GPU”的对照 benchmark，看线性增长里到底有多少是 key 搬运造成的
2. 把 `eval` 拆成：
   - key host->device
   - kernel
   - output device->host
   三段单独计时
3. 做一个纯 kernel benchmark，再和当前 facade benchmark 对照
4. 看能不能用 stream / async memcpy 把搬运和计算 overlap 起来

当前第一轮研究先到这里。
