- 我的研究方向是 `DPF` / `DCF` 的 `GPU` 加速实现

- `Orca` 相关代码现整理在 `/FSS/Orca`

- `myl7` 项目为 `/FSS/myl7`

- `libfss` 项目为 `/FSS/libfss`

- 研究该问题产生的测试文件只能写入当前文档所在文件夹，非必要不要更改其他文件夹内容（其他文件夹有其他AI在工作，你只能看）

- 临时文件请写入当前文档所在文件夹的 `.gitignore`

- 研究该问题的流程只能写入本文档，不要动别的文件夹的文档，你可以了解一下其他研究文件夹的研究内容以供借鉴参考

- 不要修改本条及以上条目内容

# 重构 `libfss`，使其尽可能收敛到 `Orca / myl7` 的通用参数口径，并用于后续三方横向耗时对比

## 工作记录

### 1. 目标

- 当前最终目标不是单独优化 `libfss`
- 而是让它能够参与后续三方统一参数的横向比较
- 当前优先对齐的公共口径是：
  - `DPF: bin=64, n=10^5 / 10^6 / 10^7`
  - `DCF: bin=64, bout=1, n=10^5 / 10^6 / 10^7`

### 2. 已确认的当前问题

- 现有本地 harness 直接拒绝 `bin=64`
- 现有本地 harness 没有独立的 `bout` 参数入口
- 上游 `libfss` 的 64 位高位路径不稳定：
  - 低位样例有时能过
  - 但高位 `alpha = 1ULL << 63` 的 `DPF/DCF` 探针会直接段错误
- 按旧 harness 的“一次性把整批 key 和输出都放进内存”写法，`n=10^7` 的内存量级不可接受

### 3. 当前重构方向

- 在本目录 `.deps/` 下放一份可修改的 `libfss` 本地副本
- 只在本地副本上修补 `bin=64` 相关问题
- 新写一个分块 CPU benchmark：
  - 支持 `DPF`
  - 支持 `DCF`
  - CLI 尽量向 `Orca` 靠拢
  - `DCF` 侧显式接受 `bout` 参数，但当前只支持 `bout=1`
- benchmark 需要按 chunk 流式处理，避免 `n=10^7` 时内存炸掉

### 4. 当前状态

- 目录骨架已建立
- 已把 `FSS/libfss` 拷到当前目录的 `.deps/libfss-src/`
- 已在本地副本上修补 `getBit(...)` 的 64 位边界问题
- 已新增：
  - [libfss_batch_bench.cpp](/root/gpu-work/260404重构libfss/libfss_batch_bench.cpp)
  - [run_libfss_bench.sh](/root/gpu-work/260404重构libfss/run_libfss_bench.sh)

### 5. 当前这版重构具体做了什么

#### 5.1 修了 `bin=64` 的高位边界

上游 `libfss` 的 `getBit(...)` 原来是：

- `return (n & (1 << (64-pos))) >> (64-pos);`

这里的 `1` 是 `int`，所以：

- `pos=1` 时会走到 `1 << 63`
- 这在 32 位 `int` 上是未定义行为
- 低位样例有时看起来能过
- 但高位样例会直接炸

本地副本里已经改成：

- 右移取位：`(n >> (64 - pos)) & 1ULL`

这轮本地探针结果：

- `DPF`
  - `alpha = 1ULL << 63`
  - `x = alpha` 时输出 `1`
  - 其他测试点输出 `0`
- `DCF`
  - `alpha = 1ULL << 63`
  - `x < alpha` 时输出 `1`
  - `x >= alpha` 时输出 `0`

也就是说：

- 当前主路径上的 `bin=64` 高位段错误已经被修掉了

#### 5.2 改成分块 benchmark

新 benchmark 不再一次性把 `n` 条实例的全部 key 和输出都堆进内存。

当前做法是：

- 对总规模 `n` 做 chunk 流式处理
- 每个 chunk 内部分别计：
  - `keygen`
  - `eval_p0`
  - `eval_p1`
- 最后把所有 chunk 的分项时间累加成整批总时间

这一步是必要的，因为旧写法在 `n=10^7, bin=64` 时，仅两方 key 的理论下界内存就已经非常夸张：

- `DPF`：约 `81.14 GiB`
- `DCF`：约 `132.77 GiB`

所以如果不分块：

- 就算 `bin=64` 的逻辑 bug 修好
- 也仍然不适合直接拿去跑通用参数

#### 5.3 统一了 CLI 形态

当前命令已经尽量向 `Orca` 靠拢：

```bash
./run_libfss_bench.sh dpf <bin> <n> [chunk]
./run_libfss_bench.sh dcf <bin> <bout> <n> [chunk]
```

其中：

- `DPF` 已支持 `bin=64`
- `DCF` 当前显式接受 `bout`
- 但目前只支持 `bout=1`

这里要特别说明：

- 这不是因为 CLI 偷懒
- 而是当前 `libfss` 这套 2-party `DCF` API 本身就没有一个独立的通用 `bout` 维度设计
- 在我们当前关心的公共参数里，`bout=1` 已经够用

### 6. 已完成的功能验证

已验证通过的点：

1. `DPF bin=64` 高位样例不再段错误
2. `DCF bin=64,bout=1` 高位样例不再段错误
3. 新 benchmark 跑通：
   - `dpf 64 1024`
   - `dcf 64 1 1024`
   - `dpf 64 100000`
   - `dcf 64 1 100000`
   - `dpf 64 1000000`
   - `dcf 64 1 1000000`

原始结果文件：

- [results_dpf_bin64_n1024.txt](/root/gpu-work/260404重构libfss/results_dpf_bin64_n1024.txt)
- [results_dcf_bin64_bout1_n1024.txt](/root/gpu-work/260404重构libfss/results_dcf_bin64_bout1_n1024.txt)
- [results_dpf_bin64_n100000.txt](/root/gpu-work/260404重构libfss/results_dpf_bin64_n100000.txt)
- [results_dcf_bin64_bout1_n100000.txt](/root/gpu-work/260404重构libfss/results_dcf_bin64_bout1_n100000.txt)
- [results_dpf_bin64_n1000000.txt](/root/gpu-work/260404重构libfss/results_dpf_bin64_n1000000.txt)
- [results_dcf_bin64_bout1_n1000000.txt](/root/gpu-work/260404重构libfss/results_dcf_bin64_bout1_n1000000.txt)

### 7. 当前实测结果

当前默认 chunk：

- `chunk = 16384`

#### 7.1 `DPF`

| `N` | `keygen` | `eval_p0` | `eval_p1` | `total` |
| ---: | ---: | ---: | ---: | ---: |
| `1024` | `189419 us` | `17555 us` | `17782 us` | `224969 us` |
| `100000` | `18516817 us` | `1732626 us` | `1732131 us` | `22012036 us` |
| `1000000` | `188605786 us` | `17323836 us` | `17329267 us` | `223560982 us` |

#### 7.2 `DCF`

| `N` | `keygen` | `eval_p0` | `eval_p1` | `total` |
| ---: | ---: | ---: | ---: | ---: |
| `1024` | `278549 us` | `22763 us` | `22781 us` | `324729 us` |
| `100000` | `27201944 us` | `2240775 us` | `2240443 us` | `31723915 us` |
| `1000000` | `270090035 us` | `22360876 us` | `22358452 us` | `315133624 us` |

直接观察：

- `10^5 -> 10^6` 基本严格线性
- 所以当前这版 benchmark 的 chunk 化没有引入明显额外非线性
- 也说明 `libfss` 现在已经可以在通用参数接口下稳定处理到 `10^6`

### 8. 当前结论

- `libfss` 现在已经不再是“不能吃 `bin=64,bout=1`”的状态
- 至少在当前主路径上，它已经可以参与通用参数横向比较
- 当前真正剩下的问题已经从“功能不可用”变成了“CPU 实现非常慢”

按 `10^5 / 10^6` 的实际结果看：

- `DPF n=10^7` 预计会是 `~2200 s` 量级，也就是 `~37 min`
- `DCF n=10^7` 预计会是 `~3150 s` 量级，也就是 `~52 min`

所以后续三方横向比较里：

- `libfss` 这边的主要代价已经不是重构代价
- 而是大参数 CPU 跑数本身就会非常耗时
