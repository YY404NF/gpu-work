- 我的研究方向是 `DPF` / `DCF` 的 `GPU` 加速实现

- `Orca` 相关代码现整理在 `/FSS/Orca`

- `myl7` 项目为 `/FSS/myl7`

- `libfss` 项目为 `/FSS/libfss`

- 研究该问题产生的测试文件只能写入当前文档所在文件夹，非必要不要更改其他文件夹内容（其他文件夹有其他AI在工作，你只能看）

- 临时文件请写入当前文档所在文件夹的 `.gitignore`

- 研究该问题的流程只能写入本文档，不要动别的文件夹的文档，你可以了解一下其他研究文件夹的研究内容以供借鉴参考

- 不要修改本条及以上条目内容

#  重写 `myl7` 项目的 benchmark，实现对 bin、bout 及测试规模等参数的可控配置。

实现myl7和orca的dpf/dcf性能对比

orca测试参数：
./dpf 64 10000000
./dpf 64 1 10000000

myl7似乎无法这样设定参数，难以横向比较

## 2026-04-04 收口后的主文档

只保留最终要用的口径：

- `DPF: bin=64`
- `DCF: bin=64, bout=1`
- `N = 10^5 / 10^6 / 10^7`

`FSS/Orca/README.md` 里的论文主点是 `bin=64, bout=1, n=10,000,000`，所以这里最重要的是 `N=10^7` 这一行。当前环境不是论文原环境，因此这里是“同参数对比”，不是“论文环境完全复现”。

### 1. 当前工具

- GPU benchmark: `run_myl7_bench.py`
- CPU benchmark: `run_myl7_cpu_bench.py`

命令形式：

```bash
python3 260404重构myl7/run_myl7_bench.py dpf 64 N --chunk N
python3 260404重构myl7/run_myl7_bench.py dcf 64 1 N --chunk N
python3 260404重构myl7/run_myl7_cpu_bench.py dpf 64 N --chunk CHUNK_N
python3 260404重构myl7/run_myl7_cpu_bench.py dcf 64 1 N --chunk CHUNK_N
```

CPU harness 也已经对齐到同一组参数，当前是单线程 host loop，可作为后续 `myl7 GPU/CPU` 优化倍数的 CPU baseline。已跑通的 CPU 点：

| 模式 | `N` | `chunk_n` | `keygen` | `eval_p0` | `eval_p1` | `total` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DPF` | `10^5` | `100000` | `355,865 us` | `186,642 us` | `184,091 us` | `784,142 us` |
| `DCF` | `10^5` | `100000` | `706,055 us` | `339,542 us` | `334,622 us` | `1,437,388 us` |
| `DPF` | `10^6` | `100000` | `3,545,239 us` | `1,861,758 us` | `1,825,125 us` | `7,294,168 us` |
| `DCF` | `10^6` | `100000` | `6,997,790 us` | `3,402,986 us` | `3,320,988 us` | `13,783,718 us` |
| `DPF` | `10^7` | `100000` | `35,671,696 us` | `18,820,584 us` | `18,597,108 us` | `73,198,447 us` |
| `DCF` | `10^7` | `100000` | `70,426,022 us` | `34,221,537 us` | `33,421,118 us` | `138,176,898 us` |

### 2. 本轮 GPU 对比环境

- GPU: `NVIDIA L20`
- 显存: `49140 MiB`
- Driver: `570.195.03`
- CUDA runtime: `12.8`

`Orca` 数据来自当前目录已有的 L20 端到端结果，`myl7` 数据来自本目录 benchmark 实测。

### 3. DPF 对比表

| N | Orca keygen | myl7 keygen | Orca eval_p0 | myl7 eval_p0 | Orca eval_p1 | myl7 eval_p1 | Orca transfer_p0 | myl7 transfer_p0 | Orca transfer_p1 | myl7 transfer_p1 | Orca total | myl7 total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10^5` | `51,405 us` | `6,832 us` | `4,354 us` | `939 us` | `11,279 us` | `886 us` | `3,854 us` | `149 us` | `10,833 us` | `145 us` | `102,473 us` | `12,261 us` |
| `10^6` | `433,901 us` | `19,232 us` | `41,597 us` | `8,019 us` | `41,587 us` | `7,976 us` | `38,023 us` | `781 us` | `38,038 us` | `769 us` | `525,161 us` | `67,730 us` |
| `10^7` | `4,376,312 us` | `151,487 us` | `414,646 us` | `87,309 us` | `414,555 us` | `85,415 us` | `379,884 us` | `7,152 us` | `379,827 us` | `7,138 us` | `5,280,636 us` | `641,560 us` |

### 4. DCF 对比表

| N | Orca keygen | myl7 keygen | Orca eval_p0 | myl7 eval_p0 | Orca eval_p1 | myl7 eval_p1 | Orca transfer_p0 | myl7 transfer_p0 | Orca transfer_p1 | myl7 transfer_p1 | Orca total | myl7 total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10^5` | `56,355 us` | `5,489 us` | `4,556 us` | `932 us` | `4,493 us` | `911 us` | `3,842 us` | `152 us` | `3,820 us` | `182 us` | `66,435 us` | `10,999 us` |
| `10^6` | `466,778 us` | `16,345 us` | `44,321 us` | `9,217 us` | `45,319 us` | `8,220 us` | `38,127 us` | `780 us` | `39,177 us` | `747 us` | `565,677 us` | `88,818 us` |
| `10^7` | `4,565,051 us` | `73,206 us` | `435,929 us` | `80,440 us` | `436,210 us` | `84,665 us` | `379,841 us` | `7,148 us` | `379,795 us` | `7,121 us` | `5,511,430 us` | `555,337 us` |

### 5. `myl7 GPU / CPU` 优化倍数

这里统一按 `total` 来算：

- `myl7 GPU / CPU = myl7_cpu_total / myl7_gpu_total`

| 模式 | `N=10^5` | `N=10^6` | `N=10^7` |
| --- | ---: | ---: | ---: |
| `DPF` | `784,142 / 12,261 = 63.95x` | `7,294,168 / 67,730 = 107.69x` | `73,198,447 / 641,560 = 114.09x` |
| `DCF` | `1,437,388 / 10,999 = 130.68x` | `13,783,718 / 88,818 = 155.19x` | `138,176,898 / 555,337 = 248.82x` |

直接观察：

- `myl7` 的 `GPU / CPU` 优化倍数会随着 `N` 增大继续抬升
- `DCF` 的 `GPU / CPU` 优化倍数始终高于 `DPF`
- 到和 README 参数直接对齐的 `N=10^7` 时：
  - `DPF` 已经到 `114.09x`
  - `DCF` 已经到 `248.82x`

### 6. 直接结论

- 现在主文档只保留 `10` 的幂次 `N`
- 各维度时间都已经把 `Orca` 和 `myl7` 并排放在同一张表里
- `myl7` 的 CPU baseline 也已经按同参数补齐到 `10^7`
- 所以这份文档里现在同时具备：
  - `Orca / myl7` 的端到端横向对比
  - `myl7 GPU / CPU` 的项目内优化倍数对比
- 只看当前这组参数，`myl7` 在 `keygen / eval_p0 / eval_p1 / transfer / total` 上都明显快于 `Orca`
- `N=10^7` 这一行最重要，因为它和 `README` 里的论文参数直接对齐
- 在 `N=10^7` 时：
  - `DPF total`: `Orca 5,280,636 us`，`myl7 641,560 us`
  - `DCF total`: `Orca 5,511,430 us`，`myl7 555,337 us`
  - `myl7 GPU / CPU`: `DPF 114.09x`，`DCF 248.82x`
- 从 `10^5 -> 10^7`，两边都大致线性增长，但 `myl7` 的斜率明显更小

### 7. 为什么 `myl7` 更快

压缩成一句话：

- 当前端到端 benchmark 里，`myl7` 更快，主要赢在 key 生命周期更轻，而不是单看某个 kernel 就一定更强。

关键点只有 3 个：

1. `Orca keygen` 更重  
   `Orca` 会先在 GPU 上生成 key，再把 tree key 回写成 host 侧 `KeyBlob`；`myl7` 则是直接把后续 eval 要用的 `cws` 留在 GPU 内存里。

2. `Orca eval` 大头是重新上传 key  
   `Orca` 的 `eval` 会先解析 `KeyBlob`，再把 `scw/l0/l1/tR` 或 `scw/l/vcw` 重新传回 GPU；`myl7` 的 `eval` 直接读 GPU 常驻 `cws`。  
   在 `N=10^7`：
   - `DPF`: `Orca transfer_p0 / eval_p0 ≈ 91.6%`，`myl7 ≈ 8.2%`
   - `DCF`: `Orca transfer_p0 / eval_p0 ≈ 87.1%`，`myl7 ≈ 8.9%`

3. `Orca复用密钥` 证明差距主要来自 key 传输  
   在 `260404Orca复用密钥/Orca复用密钥.md` 里，一旦 key 只上传一次并常驻 GPU，`avg_transfer_p0/p1` 会降到 `0.00 us`，而 `Orca` 的 steady-state eval 会大幅下降。  
   例如 `N=10^7`：
   - `DPF`: `Orca 原始 eval_avg = 414600.5 us`，`Orca 复用 key 后 = 36415.90 us`，`myl7 eval_avg = 86362.0 us`
   - `DCF`: `Orca 原始 eval_avg = 436069.5 us`，`Orca 复用 key 后 = 57471.60 us`，`myl7 eval_avg = 82552.5 us`

因此最稳妥的结论是：

- 当前主表里 `myl7` 明显快于 `Orca`
- 这主要是因为 `myl7` 的 benchmark 更接近“key 已在 GPU 上”的使用方式
- 不能简单把全部差距归因成“`myl7` 的纯计算 kernel 一定更强”
