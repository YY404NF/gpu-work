- 我的研究方向是 `DPF` / `DCF` 的 `GPU` 加速实现

- `Orca` 相关代码现整理在 `/FSS/Orca`

- `myl7` 项目为 `/FSS/myl7`

- `libfss` 项目为 `/FSS/libfss`

- 研究该问题产生的测试文件只能写入当前文档所在文件夹，非必要不要更改其他文件夹内容（其他文件夹有其他AI在工作，你只能看）

- 临时文件请写入当前文档所在文件夹的 `.gitignore`

- 研究该问题的流程只能写入本文档，不要动别的文件夹的文档，你可以了解一下其他研究文件夹的研究内容以供借鉴参考

- 不要修改本条及以上条目内容

# 完全对齐 `Orca` 和 `myl7`

1. 批量 `keygen`
2. 单 key、多 `x` 的单边 `eval`

## 当前范围

- 比较四个实现：
- `Orca`
- `myl7-gpu`
- `myl7-cpu`
- `libfss`
- 参数统一：
- `bin=64`
- `bout=1`
- `n` 按 `10` 的梯度取值
- 当前设备已补到 `10^7`
- 超过 `60 s` 直接记为 `超时`，不继续等待

## 输出字段

### 维度一：批量 keygen

- `keygen`

说明：

- 统计批量生成 `n` 组两方密钥的总耗时

### 维度二：单 key、多 x

- `eval`

说明：

- 统一按单边 `p0/party0` 计时

- `GPU` 实现的 `eval` 包含：
1. 单边密钥上传
2. `x` 上传
3. `GPU Eval`
4. 结果回传
5. 重构结果并验证正确性

- `CPU` 实现的 `eval` 仅包含：
1. 单边 `Eval`
2. 重构结果并验证正确性

- 另一边 share 只在计时外计算一次，用于正确性校验
- 表内统一使用 `us`
- 超过阈值直接写 `超时`

## 当前结果

## 测试环境1

- 主机内存：`8059220 kB`（约 `7.7 GiB`）
- 显卡：`NVIDIA GeForce RTX 3070 Laptop GPU`
- 显存：`8192 MiB`
- `GPU-ARCH`：`86`（`sm_86`）

### DCF - 批量 keygen

| 实现 | `10^4` | `10^5` | `10^6` |
| --- | ---: | ---: | ---: |
| `Orca` | `52158 us` | `160620 us` | `1030351 us` |
| `myl7-gpu` | `17012 us` | `24984 us` | `160758 us` |
| `myl7-cpu` | `89262 us` | `835704 us` | `6952504 us` |
| `libfss` | `5135442 us` | `51850448 us` | `超时` |

### DCF - 单 key，多 `x`

| 实现 | `10^4` | `10^5` | `10^6` |
| --- | ---: | ---: | ---: |
| `Orca` | `562 us` | `2997 us` | `25854 us` |
| `myl7-gpu` | `400 us` | `2363 us` | `17705 us` |
| `myl7-cpu` | `41354 us` | `400130 us` | `4283677 us` |
| `libfss` | `986328 us` | `10222797 us` | `超时` |

### DPF - 批量 keygen

| 实现 | `10^4` | `10^5` | `10^6` |
| --- | ---: | ---: | ---: |
| `Orca` | `25856 us` | `132319 us` | `1023764 us` |
| `myl7-gpu` | `16842 us` | `23990 us` | `167510 us` |
| `myl7-cpu` | `75650 us` | `439129 us` | `4971823 us` |
| `libfss` | `4199744 us` | `39609605 us` | `超时` |

说明：

- `DPF` 忽略 `bout`

### DPF - 单 key，多 `x`

| 实现 | `10^4` | `10^5` | `10^6` |
| --- | ---: | ---: | ---: |
| `Orca` | `534 us` | `2242 us` | `20648 us` |
| `myl7-gpu` | `545 us` | `1932 us` | `17762 us` |
| `myl7-cpu` | `22398 us` | `248820 us` | `2313794 us` |
| `libfss` | `1001609 us` | `9151654 us` | `超时` |

说明：

- `DPF` 忽略 `bout`

## 测试环境2

- 主机内存：`64416496 kB`（约 `61 GiB`）
- 显卡：`NVIDIA L20`
- 显存：`49140 MiB`
- `GPU-ARCH`：`89`（`sm_89`）
- 超时阈值：`60 s`

### DCF - 批量 keygen

| 实现 | `10^4` | `10^5` | `10^6` | `10^7` |
| --- | ---: | ---: | ---: | ---: |
| `Orca` | `47993 us` | `64553 us` | `559461 us` | `5480995 us` |
| `myl7-gpu` | `6344 us` | `8761 us` | `56704 us` | `529511 us` |
| `myl7-cpu` | `19732 us` | `154134 us` | `1461861 us` | `14596526 us` |
| `libfss` | `7212928 us` | `超时` | `超时` | `超时` |

### DCF - 单 key，多 `x`

| 实现 | `10^4` | `10^5` | `10^6` | `10^7` |
| --- | ---: | ---: | ---: | ---: |
| `Orca` | `196 us` | `965 us` | `8865 us` | `105145 us` |
| `myl7-gpu` | `165 us` | `893 us` | `6844 us` | `69097 us` |
| `myl7-cpu` | `7247 us` | `74039 us` | `743463 us` | `7419505 us` |
| `libfss` | `2132097 us` | `超时` | `超时` | `超时` |

### DPF - 批量 keygen

| 实现 | `10^4` | `10^5` | `10^6` | `10^7` |
| --- | ---: | ---: | ---: | ---: |
| `Orca` | `8599 us` | `56318 us` | `526866 us` | `5267669 us` |
| `myl7-gpu` | `6378 us` | `10023 us` | `62371 us` | `587853 us` |
| `myl7-cpu` | `12297 us` | `82248 us` | `777281 us` | `7802276 us` |
| `libfss` | `6175551 us` | `超时` | `超时` | `超时` |

说明：

- `DPF` 忽略 `bout`

### DPF - 单 key，多 `x`

| 实现 | `10^4` | `10^5` | `10^6` | `10^7` |
| --- | ---: | ---: | ---: | ---: |
| `Orca` | `153 us` | `736 us` | `6710 us` | `85492 us` |
| `myl7-gpu` | `155 us` | `842 us` | `6634 us` | `65925 us` |
| `myl7-cpu` | `3927 us` | `38727 us` | `390925 us` | `3947306 us` |
| `libfss` | `2286638 us` | `超时` | `超时` | `超时` |

说明：

- `DPF` 忽略 `bout`

## 续跑提示

- 前置：
- `/root/gpu-work/260409完全对齐Orca和myl7/prepare_deps.sh`
- 构建：
- `cmake -S /root/gpu-work/260409完全对齐Orca和myl7 -B /root/gpu-work/260409完全对齐Orca和myl7/.build`
- `cmake --build /root/gpu-work/260409完全对齐Orca和myl7/.build -j`
- 跑表：
- `TIMEOUT_SECS=60 N_LIST="10000 100000 1000000 10000000" /root/gpu-work/260409完全对齐Orca和myl7/run_dcf_benchmarks.sh`
- `TIMEOUT_SECS=60 N_LIST="10000 100000 1000000 10000000" /root/gpu-work/260409完全对齐Orca和myl7/run_dpf_benchmarks.sh`
- 输出：
- `DCF` 表写到 `results/dcf_bench_latest.md`
- `DPF` 表写到 `results/dpf_bench_latest.md`
- 原始日志写到 `results/raw/`
- 说明：
- 当前目录的 `CMakeLists.txt` 已兼容到 `cmake 3.18`
- `myl7` 相关 target 在 `CMake` 里直接调用 `nvcc`，用来绕开老版 `cmake` 对 `CUDA20` 标准旗标支持不完整的问题
- 超时统一写 `超时`
- `libfss` 编译依赖当前文件夹下的 `libfss_aesni_shim.cpp`，这是为了兼容它老的 `aesni_*` 符号，不要删
- 为避免大 `n` 时内存炸掉，当前 batch keygen benchmark 已做了保护：
- `Orca` 外层按块生成后立即丢弃
- `myl7-gpu` 按 `256 MiB` 预算分块
- `myl7-cpu` 和 `libfss` 都改成“生成后立即丢弃”，不再把整批 key 留在内存里
- 后续新设备优先补：
- 更大 `n` 的两张表
- `libfss` 超时点的精确值，如果想把 `超时` 换成真实值

## 为什么 `10^7` 时 `Orca` 比 `myl7` 慢一些

先说结论：

- 在当前这份 benchmark 口径下，`Orca` 比 `myl7` 慢，主要不是慢在“纯 eval kernel 算得更多”，而是慢在 key 的生命周期更重
- 到 `10^7` 时，真正拉开差距的主要是 `keygen`
- `eval` 也比 `myl7` 慢，但差距明显小于 `keygen`

### 1. 先看 `10^7` 的实测差距

`DCF`：

- `keygen`: `Orca 5480995 us`，`myl7-gpu 529511 us`
- `eval`: `Orca 105145 us`，`myl7-gpu 69097 us`
- `keygen` 倍率约 `10.35x`
- `eval` 倍率约 `1.52x`

`DPF`：

- `keygen`: `Orca 5267669 us`，`myl7-gpu 587853 us`
- `eval`: `Orca 85492 us`，`myl7-gpu 65925 us`
- `keygen` 倍率约 `8.96x`
- `eval` 倍率约 `1.30x`

直接看就很清楚：

- `Orca` 和 `myl7` 的 `eval` 确实有差距，但只是 `1.3x ~ 1.5x`
- 真正把总时间拉开的，是 `keygen` 的近 `9x ~ 10x` 差距
- 在 `10^7` 这个点上，`Orca` 的已统计时间里，`keygen` 占比已经到 `98%` 左右；`myl7` 这边也以 `keygen` 为主，但占比只在 `88% ~ 90%`

### 2. 为什么 `Orca keygen` 更重

看当前目录 benchmark 对接的 `Orca` facade：

- `generateDpfKeys(...)` 在 [dpf_api.h](/root/gpu-work/260409完全对齐Orca和myl7/.deps/FSS/Orca/fss/dpf_api.h)
- `generateDcfKeys(...)` 在 [dcf_api.h](/root/gpu-work/260409完全对齐Orca和myl7/.deps/FSS/Orca/fss/dcf_api.h)

这两条路径的接口语义都很明确：

- 先在 GPU 上生成两方 key
- 再把结果封装成可独立保存/传递的 `KeyBlob`

也就是说，当前 `Orca` 的 `keygen` 不是“只把后续 eval 会直接用到的 device key 留在 GPU 上”，而是：

- 生成 GPU key
- 序列化成 host 侧 blob
- 为可传递/可保存的 facade 形式付出额外整理成本

而当前 `myl7` 的 batch keygen benchmark 则更直接：

- [myl7_dpf_batch_keygen_bench.cu](/root/gpu-work/260409完全对齐Orca和myl7/myl7_dpf_batch_keygen_bench.cu)
- [myl7_dcf_batch_keygen_bench.cu](/root/gpu-work/260409完全对齐Orca和myl7/myl7_dcf_batch_keygen_bench.cu)

它做的是：

- host 侧准备 `seed / alpha / beta`
- 上传到 GPU
- kernel 直接生成 `SchemeType::Cw`
- 生成后立即丢弃

所以这两边比较的不是同一种“key 成品”：

- `Orca` 更接近“可导出、可保存、可重新装载”的 key
- `myl7` 更接近“后续马上 eval 的 GPU 工作格式”

这也是为什么在 `10^7` 时，`Orca keygen` 会比 `myl7` 重很多。

### 3. 为什么 `Orca eval` 也更慢

当前单 key benchmark 里，`Orca` 的 eval 路径是：

- `generate*Keys(...)`
- 把 `KeyBlob` 解析成 host 侧结构
- 再把 key 的各段拷回 GPU
- 上传 `x`
- 跑 GPU eval
- 回传结果并校验

对应代码在：

- [orca_dpf_singlekey_bench.cu](/root/gpu-work/260409完全对齐Orca和myl7/orca_dpf_singlekey_bench.cu)
- [orca_dcf_singlekey_bench.cu](/root/gpu-work/260409完全对齐Orca和myl7/orca_dcf_singlekey_bench.cu)

里面能直接看到：

- `parseHostKey(...)`
- `uploadKeyToDevice(...)`
- 每轮 `eval` 都会再次 `uploadKeyToDevice(...)`

而 `myl7` 的单 key benchmark：

- [myl7_dpf_singlekey_bench.cu](/root/gpu-work/260409完全对齐Orca和myl7/myl7_dpf_singlekey_bench.cu)
- [myl7_dcf_singlekey_bench.cu](/root/gpu-work/260409完全对齐Orca和myl7/myl7_dcf_singlekey_bench.cu)

这边的 key 形式就是：

- 一个 `SchemeType::Cw[kInBits + 1]`
- 加上一份 `seed`

每轮只是：

- `cudaMemcpy(d_cws, h_cws, ...)`
- `cudaMemcpy(d_seed, ...)`
- `cudaMemcpy(d_xs, ...)`
- kernel eval

所以 `Orca eval` 比 `myl7` 慢，本质上不是只慢在 kernel，而是它的 eval benchmark 还包含了：

- `KeyBlob -> host 结构` 的解析成本
- `scw / l0 / l1 / tR` 或 `scw / leaves / vcwLevels` 这类多段 key 的重新上传成本

### 4. 复用密钥实验给出的关键证据

这个结论不是只靠静态读代码猜的，之前已经有直接证据：

- [Orca复用密钥.md](/root/gpu-work/260404Orca复用密钥/Orca复用密钥.md)

那份实验做的事情是：

- key 只生成一次
- 两方 key 只上传一次
- 后续重复 eval 时，GPU 端直接复用常驻 key

在那份 L20 数据里，`N=10^7` 时：

`DPF`：

- `Orca` 复用 key 后的 `avg steady eval = 36415.90 us`
- 当前主表里的 `myl7 eval = 65925 us`

`DCF`：

- `Orca` 复用 key 后的 `avg steady eval = 57471.60 us`
- 当前主表里的 `myl7 eval = 69097 us`

这说明：

- 一旦把“重复上传 key”从 `Orca eval` 路径里拿掉，`Orca` 的 steady-state eval 不但会大幅下降
- 而且在 `10^7` 时，`Orca` 的 steady-state eval 其实已经不比 `myl7` 慢，甚至还略快

所以最稳妥的归因是：

- 当前主表里 `Orca` 比 `myl7` 慢，主要是 benchmark 口径里 `Orca` 背了更重的 key 生命周期成本
- 这个成本在 `10^7` 时主要体现为：
- `keygen` 要产出 `KeyBlob`
- `eval` 每轮要重新上传并装配 key
- 不能把当前主表里的全部差距简单解释成“`Orca` 的纯 GPU 计算一定比 `myl7` 差”

### 5. 当前最简结论

- 如果看当前主表这个“端到端口径”，`myl7` 的确更快
- 如果只看 key 已常驻 GPU 之后的 steady-state eval，`Orca` 并不慢，甚至在 `10^7` 时更快
- 因此，`10^7` 时 `Orca` 比 `myl7` 慢一些，核心原因是：
- `Orca` 当前 benchmark 统计进来了更多 key 序列化、解析、重传相关成本
- `myl7` 当前 benchmark 更接近“key 直接以 GPU 工作格式存在并被立即使用”的路径
