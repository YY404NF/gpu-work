- 我的研究方向是 `DPF` / `DCF` 的 `GPU` 加速实现

- `Orca` 相关代码现整理在 `/FSS/Orca`

- `myl7` 项目为 `/FSS/myl7`

- `libfss` 项目为 `/FSS/libfss`

- 研究该问题产生的测试文件只能写入当前文档所在文件夹，非必要不要更改其他文件夹内容（其他文件夹有其他AI在工作，你只能看）

- 临时文件请写入当前文档所在文件夹的 `.gitignore`

- 研究该问题的流程写入本文档

- 不要修改本条及以上条目内容

# `Orca` 增加批大小好像不能等价于提高 GPU 并行能力，批大小从一百万到一千万这一块，基本上是线性增长的，似乎没有起到加速作用

研究并解释，给出为什么线性增长原因

## 2026-04-04 第一轮结论

先说当前结论：

- 这里看到的“一百万到一千万基本线性增长”，大概率不是 `GPU` 没并行，而是已经进入了“吞吐基本稳定”的区间
- 在 `FSS/Orca` 这条链路里，增大 `n` 不只是“多给 GPU 一些线程”
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

## 1. 问题定义

这里的“批大小”指 benchmark 里的参数 `n`：

- `DPF`: `dpf_benchmark <bin> <n>`
- `DCF`: `dcf_benchmark <bin> <bout> <n>`

它表示一次 `keygen/eval` 中要处理的独立实例数，不是深度学习框架里的 `batchSz`。

## 2. 为什么会近似线性增长

把代码和实验放在一起看，核心原因其实可以压缩成 4 点：

1. `DPF/DCF` 的 kernel 是“一线程处理一个实例”，固定 `bin=64` 时每个实例都要走固定深度的树，所以总计算量天然近似 `O(n)`。
2. `eval_p0 / eval_p1` 不是纯 kernel 时间，而是包含了把 key 从 `CPU -> GPU` 搬过去、启动 kernel、同步的整段调用。
3. 当前实现里 key 不是常驻 `GPU`，每次 `eval` 都会把整份 tree key 重新搬到 `GPU`，而 key 大小本身也随 `n` 线性增长。
4. 当前实现没有做 `async memcpy` 和传输-计算 overlap，所以链路更接近“先搬、再算、再同步”。

因此：

- 当 `n = 1,000,000` 时，GPU 并行度通常已经够了
- 再把 `n` 提到 `10,000,000`
- 不会显著降低单元素成本
- 更像是在稳定吞吐率下继续处理更多元素、搬运更大的 key

## 3. 代码层面的关键信息

只保留最重要的几条：

- `eval_p0 / eval_p1` 不是纯 kernel 时间，对应 `gpuDpf(...) / gpuDcf(...)` 这层 facade 调用。
- `transfer_p0 / transfer_p1` 主要统计的是 key 搬运时间，不是完整总传输。
- `gpuDpfTreeEval(...)` / `gpuDcfTreeEval(...)` 每次都会把整份 tree key 从 host 再搬回 device。
- `bin=64, bout=1` 时，每方 key 大小约 `951 byte / elem`。
- 所以：
  - `100,000` 元素时，每方 key 约 `90.7 MiB`
  - `1,000,000` 元素时，每方 key 约 `906.9 MiB`
  - `10,000,000` 元素时，每方 key 约 `9.07 GiB`
- 当前 batch 预算下，`10,000,000` 这个规模仍然是 `B = 1`，不会自动拆成多个更小的 GPU batch。

## 4. 本轮怎么做的

### 4.1 第一轮：`RTX 3070 Laptop GPU`

- 日期：`2026-04-04`
- GPU：`NVIDIA GeForce RTX 3070 Laptop GPU`
- 显存：`8192 MiB`
- Driver：`591.74`
- CUDA：`13.1`

这轮做了两类测试：

- 端到端 benchmark：看 `eval` 与 `transfer`
- compute-only benchmark：只看 kernel

### 4.2 第二轮：`L20`

- 日期：`2026-04-04`
- GPU：`NVIDIA L20`
- 显存：`49140 MiB`
- Driver：`570.195.03`
- CUDA：`12.8`

这轮补了两个点：

- `10M` 的端到端 `DPF/DCF` direct 复测
- `20M` 的 compute-only direct 复测

另外，当时本机上的 `FSS` 子模块还是空壳，而且 `cmake` 版本偏旧，所以这轮改成了：

- 自动下载 `FSS` tarball
- 用 `nvcc` 直接编 benchmark

当前目录里的两个脚本已经改好，可以直接复现：

- `run_orca_scaling_analysis.py`
- `run_orca_compute_only_scaling.py`

## 5. 实验结论

### 5.1 端到端 benchmark

第一轮 `RTX 3070 Laptop GPU`：

- `DPF eval us/elem` 约 `0.156 ~ 0.190`
- `DCF eval us/elem` 约 `0.152 ~ 0.169`
- `transfer / eval`
  - `DPF` 约 `89.7% ~ 91.2%`
  - `DCF` 约 `85.0% ~ 87.4%`

第二轮 `L20`：

- `DPF eval us/elem`
  - `100k`: `0.078165`
  - `1M`: `0.041592`
  - `10M`: `0.041460`
- `DCF eval us/elem`
  - `100k`: `0.045245`
  - `1M`: `0.044820`
  - `10M`: `0.043607`
- `transfer / eval`
  - `DPF` 约 `90.7% ~ 91.6%`
  - `DCF` 约 `84.7% ~ 87.1%`

可以看到：

- 换到更强的 `GPU` 后，绝对时间明显下降
- 但 `eval us/elem` 很快收敛到稳定区间
- `transfer` 仍然占 `eval` 的大头

### 5.2 compute-only benchmark

第一轮 `RTX 3070 Laptop GPU`：

- `DPF eval us/elem` 约 `0.0147 ~ 0.0154`
- `DCF eval us/elem` 约 `0.0227 ~ 0.0237`

第二轮 `L20`：

- `DPF eval us/elem`
  - `1M`: `0.003598336`
  - `10M`: `0.003661926`
  - `20M`: `0.003710439`
- `DCF eval us/elem`
  - `1M`: `0.005800448`
  - `10M`: `0.005831117`
  - `20M`: `0.005853261`

这说明：

- 纯 kernel 也不是“批越大单元素越便宜很多”
- 它同样会很快进入稳定的 per-element 成本区间
- 所以近似线性不只是传输问题，也是当前实现本身的复杂度形态

## 6. 一句话解释“为什么 1M 到 10M 近似线性”

可以把结论压缩成一句话：

- `1M` 左右时 GPU 基本已经吃满，继续增大 `n` 不会明显降低单元素成本；而当前实现又会让 key 大小、key 搬运量和总实例数都随 `n` 线性增长，所以端到端时间和纯 kernel 时间都会表现为近似线性增长。

## 7. 为什么这个结论会传到 `Orca`

`Orca` 上层的 `relu` / `maxpool` 走的是 `DCF` 相关路径。

所以只要底层还是：

- key 先在 host 上
- `eval` 时再搬回 `GPU`
- 实例数变大时 key 大小也线性变大

那么上层算子看到的时间趋势也会继承这种近似线性增长。

## 8. 下一步最值得做的事

如果继续深挖，我认为最有价值的是下面 3 件事：

1. 做一个“key 常驻 GPU”的对照 benchmark，直接量化搬运到底占了多少。
2. 把 `eval` 拆成 `key host->device / kernel / output device->host` 三段分别计时。
3. 用 `stream + async memcpy` 做一次 overlap 对照，看线性增长里有没有一块是实现方式带来的。
