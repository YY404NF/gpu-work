- 我的研究方向是 `DPF` / `DCF` 的 `GPU` 加速实现

- `Orca` 相关代码现整理在 `/FSS/Orca`

- `myl7` 项目为 `/FSS/myl7`

- `libfss` 项目为 `/FSS/libfss`

- 研究该问题产生的测试文件只能写入当前文档所在文件夹，非必要不要更改其他文件夹内容（其他文件夹有其他AI在工作，你只能看）

- 临时文件请写入当前文档所在文件夹的 `.gitignore`

- 研究该问题的流程只能写入本文档，不要动别的文件夹的文档

- 不要修改本条及以上条目内容

#  重写 `myl7` 项目的 benchmark，实现对 bin、bout 及测试规模等参数的可控配置。

实现myl7和orca的dpf/dcf性能对比

orca测试参数：
./dpf 64 10000000
./dpf 64 1 10000000

myl7似乎无法这样设定参数，难以横向比较

## 2026-04-04 收口后的主文档

用户新要求：

- `N` 只保留 `10` 的幂次
- 各维度时间都要把 `Orca` 和 `myl7` 并排对比
- 文档尽量压缩

因此下面只保留：

- `DPF: bin=64`
- `DCF: bin=64, bout=1`
- `N = 10^5 / 10^6 / 10^7`
- `Orca` 与 `myl7` 的端到端 benchmark 对比

### 1. 与论文参数的关系

`FSS/Orca/README.md` 里的论文主参数是：

- 环境：`Ubuntu 20.04`、`RTX A6000`、`CUDA 11.7`、`CMake 3.27.2`、`g++-9`
- 参数：`bin=64, bout=1, n=10,000,000`
- 命令：
  - `./build/dpf_benchmark 64 10000000`
  - `./build/dcf_benchmark 64 1 10000000`

本目录重构 `myl7` 的目的，就是把 `myl7` 的 benchmark 口径收敛到能直接对齐这组 `Orca` 参数。

当前保留口径：

- `DPF` 只看 `bin=64`
- `DCF` 只看 `bin=64, bout=1`
- `N` 只看 `10^5 / 10^6 / 10^7`

其中：

- `N=10^7` 是和 `README` 论文参数直接对齐的主点
- 当前环境不是论文原环境，所以这是“同参数对比”，不是“论文环境完全复现”

### 2. 当前目录里的 benchmark

- `run_myl7_bench.py`
  - 用于生成、编译、运行 `myl7` benchmark
- `myl7_gpu_bench_template.cu.in`
  - benchmark 模板

当前命令形式：

```bash
python3 260404重构myl7/run_myl7_bench.py dpf 64 N --chunk N
python3 260404重构myl7/run_myl7_bench.py dcf 64 1 N --chunk N
```

### 3. 本轮对比环境

- GPU: `NVIDIA L20`
- 显存: `49140 MiB`
- Driver: `570.195.03`
- CUDA runtime: `12.8`

说明：

- `Orca` 数据来自当前目录现有的 L20 端到端结果文件 `orca_scaling_results_l20.json`
- `myl7` 数据来自当前目录 benchmark 的实际运行结果
- 两边都只保留 `DPF bin=64` 与 `DCF bin=64,bout=1`

### 4. DPF 对比表

| N | Orca keygen | myl7 keygen | Orca eval_p0 | myl7 eval_p0 | Orca eval_p1 | myl7 eval_p1 | Orca transfer_p0 | myl7 transfer_p0 | Orca transfer_p1 | myl7 transfer_p1 | Orca total | myl7 total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10^5` | `51,405 us` | `6,832 us` | `4,354 us` | `939 us` | `11,279 us` | `886 us` | `3,854 us` | `149 us` | `10,833 us` | `145 us` | `102,473 us` | `12,261 us` |
| `10^6` | `433,901 us` | `19,232 us` | `41,597 us` | `8,019 us` | `41,587 us` | `7,976 us` | `38,023 us` | `781 us` | `38,038 us` | `769 us` | `525,161 us` | `67,730 us` |
| `10^7` | `4,376,312 us` | `151,487 us` | `414,646 us` | `87,309 us` | `414,555 us` | `85,415 us` | `379,884 us` | `7,152 us` | `379,827 us` | `7,138 us` | `5,280,636 us` | `641,560 us` |

### 5. DCF 对比表

| N | Orca keygen | myl7 keygen | Orca eval_p0 | myl7 eval_p0 | Orca eval_p1 | myl7 eval_p1 | Orca transfer_p0 | myl7 transfer_p0 | Orca transfer_p1 | myl7 transfer_p1 | Orca total | myl7 total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10^5` | `56,355 us` | `5,489 us` | `4,556 us` | `932 us` | `4,493 us` | `911 us` | `3,842 us` | `152 us` | `3,820 us` | `182 us` | `66,435 us` | `10,999 us` |
| `10^6` | `466,778 us` | `16,345 us` | `44,321 us` | `9,217 us` | `45,319 us` | `8,220 us` | `38,127 us` | `780 us` | `39,177 us` | `747 us` | `565,677 us` | `88,818 us` |
| `10^7` | `4,565,051 us` | `73,206 us` | `435,929 us` | `80,440 us` | `436,210 us` | `84,665 us` | `379,841 us` | `7,148 us` | `379,795 us` | `7,121 us` | `5,511,430 us` | `555,337 us` |

### 6. 结论

- 现在主文档只保留 `10` 的幂次 `N`
- 各维度时间都已经把 `Orca` 和 `myl7` 并排放在同一张表里
- 只看当前这组参数，`myl7` 在 `keygen / eval_p0 / eval_p1 / transfer / total` 上都明显快于 `Orca`
- `N=10^7` 这一行最重要，因为它和 `README` 里的论文参数直接对齐
- 在 `N=10^7` 时：
  - `DPF total`: `Orca 5,280,636 us`，`myl7 641,560 us`
  - `DCF total`: `Orca 5,511,430 us`，`myl7 555,337 us`
- 从 `10^5 -> 10^7`，两边都大致线性增长，但 `myl7` 的斜率明显更小
