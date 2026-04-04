- 我的研究方向是 `DPF` / `DCF` 的 `GPU` 加速实现

- `Orca` 相关代码现整理在 `/FSS/Orca`

- `myl7` 项目为 `/FSS/myl7`

- `libfss` 项目为 `/FSS/libfss`

- 研究该问题产生的测试文件只能写入当前文档所在文件夹，非必要不要更改其他文件夹内容（其他文件夹有其他AI在工作，你只能看）

- 临时文件请写入当前文档所在文件夹的 `.gitignore`

- 研究该问题的流程只能写入本文档，不要动别的文件夹的文档

- 不要修改本条及以上条目内容

# `Orca` 中每次评估的密钥都是重新生成的，如果允许所有评估都复用同一个密钥，能不能提高评估阶段的性能，减少CPU、GPU间数据传输的开销

## 2026-04-04 收口后的研究记录

当前只保留大参数路径，只做 `N` 梯度测试。

### 1. 当前范围

- `DPF` 只保留：`bin=64`
- `DCF` 只保留：`bin=64,bout=1`
- 只把 `N` 当自变量，其他参数不再测试、不再记录
- 这里只说“1份密钥”时，实际指 `1` 组固定密钥，也就是两方 share 各 `1` 份：
  - 每个 `N` 只 `keygen` 一次
  - 两方 key 各上传一次到 GPU
  - 然后复用这一组 GPU 常驻 key 连续做 `eval_iters` 次评估

### 2. 当前目录里的 benchmark

- `dpf_benchmark_reuse.cu`
  - 固定 `bin=64`
  - 用法：`./dpf_benchmark_reuse <n> <eval_iters>`
- `dcf_benchmark_reuse.cu`
  - 固定 `bin=64,bout=1`
  - 用法：`./dcf_benchmark_reuse <n> <eval_iters>`

当前输出字段：

- `keygen`
- `key_upload_p0/p1`
- `key_upload_transfer_p0/p1`
- `avg_eval_p0/p1`
- `avg_transfer_p0/p1`
- `total`

说明：

- `key_upload_p0/p1` 是整段上传 wall time
- `key_upload_transfer_p0/p1` 是纯 `H2D memcpy` 时间
- 当前统计口径下，复用后重复 `eval` 的 `avg_transfer_p0/p1` 应该接近 `0`

### 3. L20 环境

- GPU：`NVIDIA L20`
- `compute capability`: `8.9`
- 驱动显示 `CUDA 12.8`
- 当时这台机器上的 `/root/gpu-work/FSS` 还是空壳子模块，所以这轮没有直接复用主仓库里的 `FSS/Orca`
- 为了不改动其他目录，本轮依赖只落在当前目录的 `.deps/`
- 这台机器的 `cmake` 是 `3.22.1`，低于该版 `FSS` 所需的 `3.24`
- 所以本轮继续使用手工 `nvcc` 编译

### 4. N 梯度测试

测试口径：

- `eval_iters=10`
- `N` 按 `10` 的幂次取：
  - `10^4 = 10000`
  - `10^5 = 100000`
  - `10^6 = 1000000`
  - `10^7 = 10000000`
- 每个点连续跑两次，以下表格记录第二次结果，用来避开首轮 warm-up / mempool 抖动

#### `DPF`：固定 `bin=64`

| `N` | `keygen` (us) | `avg key upload transfer` (us) | `avg steady eval` (us) | `avg_transfer_p0/p1` |
| ---: | ---: | ---: | ---: | ---: |
| `10000` | `7500` | `402.50` | `94.10` | `0.00` |
| `100000` | `48008` | `3824.00` | `426.10` | `0.00` |
| `1000000` | `430512` | `37946.50` | `3548.35` | `0.00` |
| `10000000` | `4300131` | `379569.50` | `36415.90` | `0.00` |

#### `DCF`：固定 `bin=64,bout=1`

| `N` | `keygen` (us) | `avg key upload transfer` (us) | `avg steady eval` (us) | `avg_transfer_p0/p1` |
| ---: | ---: | ---: | ---: | ---: |
| `10000` | `13685` | `398.50` | `156.15` | `0.00` |
| `100000` | `54108` | `3816.00` | `669.60` | `0.00` |
| `1000000` | `462107` | `37980.00` | `5718.50` | `0.00` |
| `10000000` | `4553417` | `379512.00` | `57471.60` | `0.00` |

直接观察：

- `DPF` 和 `DCF` 两条路径里，`avg_transfer_p0/p1` 在所有 `N` 点上都稳定为 `0.00 us`
- `avg key upload transfer` 随 `N` 基本线性增长
- `avg steady eval` 也随 `N` 基本线性增长
- 同一 `N` 下，`DCF` 的 steady-state `eval` 明显慢于 `DPF`
- `N=10^7` 也已经跑通，说明这台 `L20` 上当前实现能稳定撑到千万级批量
- 到 `N=10^6` 时：
  - `DPF avg steady eval = 3548.35 us`
  - `DCF avg steady eval = 5718.50 us`
- 到 `N=10^7` 时：
  - `DPF avg steady eval = 36415.90 us`
  - `DCF avg steady eval = 57471.60 us`

### 5. 当前结论

- 对当前真正关心的参数区间，也就是 `bin=64` / `bout=1` 这类大参数路径，`device-side key cache` 已经稳定可用
- 只要 key 不变，重复评估阶段的 CPU/GPU key 传输已经被压到 `0`
- 现在最适合继续观察的自变量就是 `N`
- 从这组 L20 数据看，后续如果还要继续扩展实验，优先做：
  - 更大的 `N`
  - 更细的 `N` 梯度
  - Orca 上层真实调用场景里的多轮复用

### 6. 当前状态

- 新 benchmark 仍然只放在 `260404复用密钥/` 下
- 当前目录 `.gitignore` 已加入 `.deps/`
- 当前目录里的二进制和临时依赖都只留在当前目录内
- 还没有放回 `FSS/Orca/`
