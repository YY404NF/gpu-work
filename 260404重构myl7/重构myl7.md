- 我的研究方向是 `DPF` / `DCF` 的 `GPU` 加速实现

- `Orca` 项目提取出来的项目为 `/GPU-MPC` ，剥离出来的FSS函数为 `/FSS`

- `myl7` 项目为 `/fss`

- `libfss` 项目为 `/libfss`

- 研究该问题产生的测试文件只能写入当前文档所在文件夹，非必要不要更改其他文件夹内容（其他文件夹有其他AI在工作，你只能看）

- 临时文件请写入当前文档所在文件夹的 `.gitignore`

- 研究该问题的流程写入本文档

- 不要修改本条及以上条目内容

#  重写 `myl7` 项目的 benchmark，实现对 bin、bout 及测试规模等参数的可控配置。

实现myl7和orca的dpf/dcf性能对比

orca测试参数：
./dpf 64 10000000
./dpf 64 1 10000000

myl7似乎无法这样设定参数，难以横向比较

## 2026-04-04 当前进展

### 结论先记

- `myl7/fss` 现有 `src/bench_gpu.cu` 是 Google Benchmark 风格，`bin` 基本写死在模板实例里，`n` 也固定成 `1 << 20`，不适合直接拿来和 Orca 的命令行 benchmark 横向对比。
- 直接把 `/fss` 原 benchmark 改成“单二进制 + 任意运行时 `bin/bout`”会引入大量模板实例化；在当前机器上更稳妥的方案是：
  在当前文档目录里按 `(scheme, bin, bout)` 生成专用 CUDA benchmark 源文件并缓存编译结果，而把 `n` 和 `chunk_n` 保持成运行时参数。
- 这样可以同时满足：
  1. 不去打扰 `/fss` 目录本身。
  2. `bin/bout` 真正作用在模板实例上，不是“表面可配、内部没变”。
  3. 可以通过 `chunk_n` 控制单次并行规模，避免 `bin=64,n=10^7` 时在 8GB 显存机器上直接 OOM。

### 当前目录新增文件

- `run_myl7_bench.py`
  负责按参数生成、编译并运行专用 benchmark。
- `myl7_gpu_bench_template.cu.in`
  benchmark 模板，实际实例化为 `(scheme, bin, bout)` 固定的 `.cu` 文件。
- `.gitignore`
  忽略当前目录下的 `.build/` 缓存产物。

### 实现要点

- benchmark 使用 `fss::Dpf` / `fss::Dcf` + `fss::prg::ChaCha`，保持和 `myl7` GPU 路径一致。
- `bout` 通过 `fss::group::Uint<uint64_t, 2^bout>` 进入模板；`bout=64` 时退化为 `Uint<uint64_t>`。
- `n` 和 `chunk_n` 保持运行时可配：
  - `n` 表示总测试规模。
  - `chunk_n` 表示单次 kernel 处理的条数；不传时根据 `cudaMemGetInfo` 自动估算安全值。
- 每个 batch 都会把两方结果拷回主机，并对前 16 个样本做重构校验，避免“只快但算错”。
- 当前机器上 `nvcc` 默认接的是 `g++-9.5`，会卡在 `<concepts>`；脚本已自动切到 `/usr/bin/g++-13`。

### 使用方式

DPF 对齐 Orca 的命令：

```bash
python3 260404重构myl7/run_myl7_bench.py dpf 64 10000000 --chunk 1000000
```

DCF 对齐 Orca 的命令：

```bash
python3 260404重构myl7/run_myl7_bench.py dcf 64 1 10000000 --chunk 1000000
```

如果想扫不同 `bout`：

```bash
python3 260404重构myl7/run_myl7_bench.py dcf 64 8 1000000 --chunk 500000
python3 260404重构myl7/run_myl7_bench.py dcf 64 16 1000000 --chunk 500000
```

说明：

- 不传 `--chunk` 时，脚本会自动给出当前显存下的安全 `chunk_n`。
- 生成出来的专用源码和二进制会缓存在 `260404重构myl7/.build/`。
- DPF 这边额外保留了 `--bout`（默认 `1`），便于后面需要时统一输出群位宽；和 Orca 对齐时直接用默认值即可。

### 当前机器环境

- GPU: `NVIDIA GeForce RTX 3070 Laptop GPU`
- 显存: `8 GiB`
- CUDA: `13.1`
- host compiler: `g++-13`

### 已完成验证

以下命令已经实际跑通并通过重构校验：

```bash
python3 260404重构myl7/run_myl7_bench.py dpf 64 1024 --chunk 256
python3 260404重构myl7/run_myl7_bench.py dcf 64 1 1024 --chunk 256
python3 260404重构myl7/run_myl7_bench.py dcf 64 8 1024 --chunk 256
```

说明 `bin=64`、`bout!=1`、以及批量分块逻辑都已经打通。

### Orca vs myl7 初步对比

对齐参数：

- `bin=64`
- `bout=1`（DCF）
- `n=1,000,000`
- myl7 这边显式设置 `chunk_n=1,000,000`，即单批直接跑满 100 万条，不做额外分块

#### DPF

实际命令：

```bash
/home/yy404nf/FSS-Work/FSS/build/dpf_benchmark 64 1000000
python3 260404重构myl7/run_myl7_bench.py dpf 64 1000000 --chunk 1000000
```

结果：

| 指标 | Orca/FSS | myl7 | 倍率（Orca / myl7） |
| --- | ---: | ---: | ---: |
| keygen | 1676558 us | 43241 us | 38.77x |
| eval_p0 | 170382 us | 14176 us | 12.02x |
| eval_p1 | 169509 us | 14539 us | 11.66x |
| total | 2044271 us | 142568 us | 14.34x |

#### DCF

实际命令：

```bash
/home/yy404nf/FSS-Work/FSS/build/dcf_benchmark 64 1 1000000
python3 260404重构myl7/run_myl7_bench.py dcf 64 1 1000000 --chunk 1000000
```

结果：

| 指标 | Orca/FSS | myl7 | 倍率（Orca / myl7） |
| --- | ---: | ---: | ---: |
| keygen | 1683435 us | 46868 us | 35.92x |
| eval_p0 | 170874 us | 15228 us | 11.22x |
| eval_p1 | 170528 us | 15104 us | 11.29x |
| total | 2039111 us | 149546 us | 13.64x |

### 当前判断

- 在这台 `RTX 3070 Laptop GPU (8GB)` 上，`myl7` 的 DPF/DCF 在 `bin=64,n=10^6` 下都显著快于当前 Orca/FSS 提取版。
- 目前差距最大的部分是 `keygen`，大约在 `36x-39x`。
- `eval_p0/eval_p1` 也有大约 `11x-12x` 的差距。
- 这组对比已经足够说明“myl7 不是因为 benchmark 不可配置才看不出优势”；配置问题现在已经被拆掉了。

## 2026-04-04 按新要求收窄测试范围

### 本轮只看这一组参数

- DPF: `bin=64`
- DCF: `bin=64, bout=1`
- `N` 不再强行只跑 `10,000,000`，改为多档梯度
- myl7 统一使用 `--chunk 1000000`

### 梯度选择与边界

- 先尝试了 `100,000 / 500,000 / 1,000,000 / 2,000,000 / 5,000,000`
- 小规模档位 `100,000 / 500,000` 抖动明显，尤其 Orca 的 DPF `keygen`，很容易被 CUDA 初始化、Pinned Memory 注册和主机端杂项开销污染
- 因此主表采用单独复测后的 `1,000,000 / 2,000,000 / 3,000,000`
- 额外探了 `5,000,000`：
  - `Orca DPF @ 5,000,000`：OOM
  - `Orca DCF @ 5,000,000`：OOM
  - `myl7 DPF/DCF @ 5,000,000`：都能正常跑完

### 说明

- 下面优先比较 `keygen / eval_p0 / eval_p1`
- `total` 只作为参考，因为两边 benchmark harness 的主机端工作并不完全一致：
  - Orca 这边包含自己的 key 解析、打包输出搬运等流程
  - myl7 这边包含 Python 包装层、host 侧批量构造、结果重构校验等流程
- 所以更适合把 `keygen` 和两侧 `eval` 看成主比较指标，`total` 作为补充观察

### DPF 主表（bin=64）

| N | Orca keygen | myl7 keygen | Orca eval_p0 | myl7 eval_p0 | Orca eval_p1 | myl7 eval_p1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 1,755,354 us | 46,321 us | 166,623 us | 13,429 us | 164,620 us | 13,933 us |
| 2,000,000 | 2,909,821 us | 68,344 us | 325,494 us | 26,504 us | 325,875 us | 29,841 us |
| 3,000,000 | 4,519,245 us | 93,217 us | 504,015 us | 39,526 us | 506,388 us | 44,315 us |

倍率（Orca / myl7）：

| N | keygen | eval_p0 | eval_p1 | total 参考 |
| --- | ---: | ---: | ---: | ---: |
| 1,000,000 | 37.90x | 12.41x | 11.82x | 14.43x |
| 2,000,000 | 42.58x | 12.28x | 10.92x | 16.67x |
| 3,000,000 | 48.48x | 12.75x | 11.43x | 17.87x |

补充：

- `myl7 DPF @ 5,000,000`：
  - keygen: `142,880 us`
  - eval_p0: `67,976 us`
  - eval_p1: `72,370 us`
  - total: `422,867 us`
- `Orca DPF @ 5,000,000`：
  - 直接报 `cudaErrorMemoryAllocation`

### DCF 主表（bin=64, bout=1）

| N | Orca keygen | myl7 keygen | Orca eval_p0 | myl7 eval_p0 | Orca eval_p1 | myl7 eval_p1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 1,594,029 us | 42,330 us | 173,476 us | 11,949 us | 171,039 us | 14,742 us |
| 2,000,000 | 2,245,127 us | 67,042 us | 339,736 us | 26,600 us | 342,425 us | 30,527 us |
| 3,000,000 | 3,481,303 us | 97,803 us | 513,167 us | 40,029 us | 511,205 us | 43,260 us |

倍率（Orca / myl7）：

| N | keygen | eval_p0 | eval_p1 | total 参考 |
| --- | ---: | ---: | ---: | ---: |
| 1,000,000 | 37.66x | 14.52x | 11.60x | 14.98x |
| 2,000,000 | 33.49x | 12.77x | 11.22x | 13.91x |
| 3,000,000 | 35.60x | 12.82x | 11.82x | 16.01x |

补充：

- `myl7 DCF @ 5,000,000`：
  - keygen: `152,091 us`
  - eval_p0: `67,569 us`
  - eval_p1: `76,442 us`
  - total: `442,652 us`
- `Orca DCF @ 5,000,000`：
  - 直接报 `cudaErrorMemoryAllocation`

### 当前判断更新

- 只看用户指定的 `bin=64, bout=1`，在这台 `RTX 3070 Laptop GPU 8GB` 上，myl7 相比 Orca 依然稳定占优
- `keygen` 的优势最大，基本落在 `33x-48x`
- `eval_p0 / eval_p1` 的优势更稳定，基本落在 `11x-15x`
- `N=1e6 -> 3e6` 这段里，两边都大致呈线性增长，但 myl7 的斜率明显更小
- `N=5e6` 时，Orca 的 DPF 和 DCF 都已经因为显存/HostRegister 相关分配失败而无法继续；myl7 借助分块还可以继续跑

### 如果后面还继续做

- 如果目标是“这台机器上的最大可比规模”，下一步最值得做的是补一个 `N=4,000,000`
  这样可以把 Orca 的可运行上界夹在 `3e6` 和 `5e6` 之间
- 如果目标是“尽量贴近 Orca 原始命令形式”，可以在当前目录再加两个小壳脚本：
  - `dpf 64 N`
  - `dcf 64 1 N`

## 2026-04-04 WSL 友好小梯度补测

用户补充要求：

- 继续测
- 但参数不要太大，避免把 WSL 卡死

因此本轮改成更保守的设置：

- 只测到 `N=1,000,000`
- `N` 梯度：`100,000 / 250,000 / 500,000 / 750,000 / 1,000,000`
- myl7 统一使用 `--chunk 250000`
- 正式测量前先跑一轮 `N=10000` warm-up，尽量把首次 CUDA 初始化抖动压掉

### DPF 小梯度（bin=64）

| N | Orca keygen | myl7 keygen | Orca eval_p0 | myl7 eval_p0 | Orca eval_p1 | myl7 eval_p1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 176,611 us | 18,541 us | 17,490 us | 2,280 us | 17,455 us | 1,155 us |
| 250,000 | 415,836 us | 24,358 us | 43,221 us | 3,585 us | 43,262 us | 3,746 us |
| 500,000 | 867,363 us | 27,944 us | 85,828 us | 6,915 us | 85,472 us | 6,748 us |
| 750,000 | 1,068,291 us | 32,526 us | 118,033 us | 9,786 us | 117,939 us | 9,120 us |
| 1,000,000 | 1,468,038 us | 37,753 us | 160,675 us | 14,500 us | 161,075 us | 11,362 us |

倍率（Orca / myl7）：

| N | keygen | eval_p0 | eval_p1 | total 参考 |
| --- | ---: | ---: | ---: | ---: |
| 100,000 | 9.53x | 7.67x | 15.11x | 6.79x |
| 250,000 | 17.07x | 12.06x | 11.55x | 10.15x |
| 500,000 | 31.04x | 12.41x | 12.67x | 16.37x |
| 750,000 | 32.84x | 12.06x | 12.93x | 15.63x |
| 1,000,000 | 38.89x | 11.08x | 14.18x | 18.66x |

### DCF 小梯度（bin=64, bout=1）

| N | Orca keygen | myl7 keygen | Orca eval_p0 | myl7 eval_p0 | Orca eval_p1 | myl7 eval_p1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 206,565 us | 23,542 us | 18,968 us | 1,528 us | 18,371 us | 2,472 us |
| 250,000 | 358,691 us | 32,086 us | 45,254 us | 4,743 us | 45,020 us | 3,465 us |
| 500,000 | 480,441 us | 31,974 us | 88,516 us | 7,990 us | 88,768 us | 7,578 us |
| 750,000 | 1,056,215 us | 38,125 us | 125,326 us | 11,347 us | 125,506 us | 10,492 us |
| 1,000,000 | 1,152,562 us | 40,069 us | 158,037 us | 11,950 us | 158,469 us | 12,220 us |

倍率（Orca / myl7）：

| N | keygen | eval_p0 | eval_p1 | total 参考 |
| --- | ---: | ---: | ---: | ---: |
| 100,000 | 8.77x | 12.41x | 7.43x | 6.77x |
| 250,000 | 11.18x | 9.54x | 12.99x | 7.64x |
| 500,000 | 15.03x | 11.08x | 11.71x | 9.23x |
| 750,000 | 27.70x | 11.04x | 11.96x | 14.85x |
| 1,000,000 | 28.76x | 13.22x | 12.97x | 14.37x |

### 这一轮的判断

- 如果只想在 WSL 里稳定地复现实验，`N <= 1,000,000` 会比前面的上限探测友好很多
- 在这组更保守的参数下，myl7 相比 Orca 仍然稳定占优
- 当 `N` 增长到 `500,000` 以上后，倍率基本稳定下来：
  - DPF `keygen` 大约 `31x-39x`
  - DPF `eval` 大约 `11x-14x`
  - DCF `keygen` 大约 `15x-29x`
  - DCF `eval` 大约 `11x-13x`
- 小于 `250,000` 时，倍率波动更明显，说明固定开销和调度噪声占比更高，不适合拿来做主结论

### 如果后面继续沿着“别把 WSL 卡死”的方向做

- 优先推荐的参数区间：
  - `100,000 <= N <= 1,000,000`
  - myl7 使用 `--chunk 250000`
- 如果还想再往上试，不要直接跳 `5,000,000`
  - 更稳妥的是先补 `1,250,000 / 1,500,000 / 2,000,000`
