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
- 本机先只跑到 `10^6`
- 超过 `60 s` 直接记为 `>60000000 us`，不继续等待

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
- 单边密钥上传
- `x` 上传
- `GPU Eval`
- 结果回传
- 重构结果并验证正确性

- `CPU` 实现的 `eval` 仅包含：
- 单边 `Eval`
- 重构结果并验证正确性

- 另一边 share 只在计时外计算一次，用于正确性校验
- 表内统一使用 `us`
- 超过阈值直接写 `>60000000 us`

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
| `libfss` | `5135442 us` | `51850448 us` | `>60000000 us` |

### DCF - 单 key，多 `x`

| 实现 | `10^4` | `10^5` | `10^6` |
| --- | ---: | ---: | ---: |
| `Orca` | `562 us` | `2997 us` | `25854 us` |
| `myl7-gpu` | `400 us` | `2363 us` | `17705 us` |
| `myl7-cpu` | `41354 us` | `400130 us` | `4283677 us` |
| `libfss` | `986328 us` | `10222797 us` | `>60000000 us` |

### DPF - 批量 keygen

| 实现 | `10^4` | `10^5` | `10^6` |
| --- | ---: | ---: | ---: |
| `Orca` | `25856 us` | `132319 us` | `1023764 us` |
| `myl7-gpu` | `16842 us` | `23990 us` | `167510 us` |
| `myl7-cpu` | `75650 us` | `439129 us` | `4971823 us` |
| `libfss` | `4199744 us` | `39609605 us` | `>60000000 us` |

说明：

- `DPF` 忽略 `bout`

### DPF - 单 key，多 `x`

| 实现 | `10^4` | `10^5` | `10^6` |
| --- | ---: | ---: | ---: |
| `Orca` | `534 us` | `2242 us` | `20648 us` |
| `myl7-gpu` | `545 us` | `1932 us` | `17762 us` |
| `myl7-cpu` | `22398 us` | `248820 us` | `2313794 us` |
| `libfss` | `1001609 us` | `9151654 us` | `>60000000 us` |

说明：

- `DPF` 忽略 `bout`

## 续跑提示

- 当前文件夹里的 `CMakeLists.txt` 已自动读取本机 `CUDA toolkit version` 和 `GPU arch`
- 正常续跑命令：
- `cmake -S /home/yy404nf/gpu-work/260409完全对齐Orca和myl7 -B /home/yy404nf/gpu-work/260409完全对齐Orca和myl7/.build`
- `cmake --build /home/yy404nf/gpu-work/260409完全对齐Orca和myl7/.build -j`
- 自动跑表脚本：
- `TIMEOUT_SECS=60 N_LIST="10000 100000 1000000" /home/yy404nf/gpu-work/260409完全对齐Orca和myl7/run_dcf_benchmarks.sh`
- 脚本输出：
- markdown 表格写到 `results/dcf_bench_latest.md`
- 原始日志写到 `results/raw/`
- 如果要和本文档保持一致，建议脚本后处理也统一保留 `us`
- 脚本后续建议保留自动写环境信息，这样换设备时不会把结果混在一起
- 当前可直接使用的 target：
- `orca_dpf_batch_keygen_bench`
- `orca_dpf_singlekey_bench`
- `orca_dcf_batch_keygen_bench`
- `orca_dcf_singlekey_bench`
- `myl7_dpf_batch_keygen_bench`
- `myl7_dpf_singlekey_bench`
- `myl7_dcf_batch_keygen_bench`
- `myl7_dcf_singlekey_bench`
- `myl7_cpu_dpf_batch_keygen_bench`
- `myl7_cpu_dpf_singlekey_bench`
- `myl7_cpu_dcf_batch_keygen_bench`
- `myl7_cpu_dcf_singlekey_bench`
- `libfss_dpf_batch_keygen_bench`
- `libfss_dpf_singlekey_bench`
- `libfss_dcf_batch_keygen_bench`
- `libfss_dcf_singlekey_bench`
- 当前本机超时规则是单项 `60 s`，建议后续设备先沿用；如果改阈值，也要在表里显式写出来
- `libfss` 编译依赖当前文件夹下的 `libfss_aesni_shim.cpp`，这是为了兼容它老的 `aesni_*` 符号，不要删
- 为避免大 `n` 时内存炸掉，当前 batch keygen benchmark 已做了保护：
- `Orca` 外层按块生成后立即丢弃
- `myl7-gpu` 按 `256 MiB` 预算分块
- `myl7-cpu` 和 `libfss` 都改成“生成后立即丢弃”，不再把整批 key 留在内存里
- 后续新设备优先补：
- `10^7` 的两张表
- `libfss` 在 `10^6` 的精确值，如果想把 `>60 s` 换成真实值
