# 性能对比工作进度

## 工作目标

在仓库根目录提供一个统一的性能对比程序，对齐并比较以下两套实现的 GPU DPF/DCF 性能：

- `./fss`
  - 对应 myl7 的 FSS 代码
  - 当前以 `bench_gpu` 为主要 GPU benchmark 入口
- `./FSS`
  - 对应从 Orca / GPU-MPC 中剥离出的最小运行子集
  - 当前以 `dpf_benchmark` / `dcf_benchmark` 为 benchmark 入口

目标不是只“手工跑几条命令”，而是先在根目录建立一个可重复执行、能自动汇总结果的统一对比程序，为后续分析优化点打基础。

## 当前已确认信息

- `fss` 当前 GPU benchmark 位于 `fss/src/bench_gpu.cu`
- `fss` 需要通过 `-DBUILD_BENCH=ON` 构建 `bench_gpu`
- `fss` 当前已注册、且和 Orca 口径最接近的条目包括：
  - `BM_DpfGen_Uint/20`
  - `BM_DpfEval_Uint/20`
  - `BM_DcfGen_Uint/20`
  - `BM_DcfEval_Uint/20`
- `fss` 当前 benchmark 内部批大小固定为 `kN = 1 << 20`
- `FSS` 当前 benchmark 可执行文件为：
  - `FSS/build/dpf_benchmark`
  - `FSS/build/dcf_benchmark`
- 当前机器 GPU 已确认：
  - `NVIDIA GeForce RTX 3070 Laptop GPU`
  - compute capability `8.6`

## 当前口径约束

为了先完成“可比的第一版根目录对比程序”，当前默认对齐到以下口径：

- `bin = 64`
- `n = 100000`
- `bout = 1`（DCF 默认先对齐这一档）

原因：

- `fss` 当前工作区里的 GPU benchmark 已调整为 `64` bit 对齐口径
- `fss` 的 benchmark 内部数据规模当前固定为 `100000`
- `FSS` 的 benchmark 支持参数化输入，因此可以向 `fss` 的固定口径靠拢

## 当前进展

### 已完成

- 已梳理两边 benchmark 入口和构建方式
- 已确认第一版统一比较应优先覆盖：
  - DPF keygen
  - DPF eval
  - DCF keygen
  - DCF eval
- 已确认需要在根目录新增统一对比脚本，而不是修改两个子仓库的 benchmark 逻辑
- 已在根目录新增统一对比脚本：
  - [perf_compare.py](/home/yy404nf/FSS-Work/perf_compare.py)
- 已生成首版对比结果：
  - [perf_compare_report.md](/home/yy404nf/FSS-Work/perf_compare_report.md)
  - [perf_compare_report.json](/home/yy404nf/FSS-Work/perf_compare_report.json)
- 已将 `perf_compare.py` 生成的 Markdown 报告模板改为中文
- 已将当前统一对比口径切换为：
  - `bin=64`
  - `bout=1`
  - `n=100000`
- 已将 `fss/src/bench_gpu.cu` 中用于对比的 DPF/DCF benchmark 改为 64 位输入对齐配置
- 已修复 `FSS` benchmark 在 `bin=20, n=2^20` 下的数据生成越界问题：
  - [dpf_benchmark.cu](/home/yy404nf/FSS-Work/FSS/dpf_benchmark.cu)
  - [dcf_benchmark.cu](/home/yy404nf/FSS-Work/FSS/dcf_benchmark.cu)

### 进行中

- 正在根据首轮结果分析 `FSS/Orca` 是否有可以借鉴的优化技术
- 已定位 `fss` 当前初始化时的实际环境问题：
  - 机器默认 `c++` 指向 `g++-9`
  - `fss` 的 benchmark 需要可用的 C++20 主机编译器
  - 已在根目录脚本里自动优先选择 `g++-13/gcc-13` 进行构建

## 首轮实测结果

测试环境：

- GPU: `NVIDIA GeForce RTX 3070 Laptop GPU`
- CUDA: `13.1`
- 对齐参数：
  - `bin=64`
  - `bout=1`
  - `n=100000`

首轮结果摘要：

- DPF
  - `fss` keygen: `1418058.47 us`
  - `FSS/Orca` keygen: `199335 us`
  - `FSS / fss`: `0.14x`
  - `fss` eval: `1412.23 us`
  - `FSS/Orca` eval_avg: `22106.00 us`
  - `FSS / fss`: `15.65x`
- DCF
  - `fss` keygen: `2673.13 us`
  - `FSS/Orca` keygen: `112322 us`
  - `FSS / fss`: `42.02x`
  - `fss` eval: `1491.15 us`
  - `FSS/Orca` eval_avg: `22360.50 us`
  - `FSS / fss`: `15.00x`

当前初步结论：

- 在当前 `bin=64, bout=1, n=100000` 口径下，`fss` 的 `eval` 与 `DCF keygen` 仍明显快于 `FSS/Orca`
- 但 `DPF keygen` 出现了明显反转：
  - `fss` 约 `1418 ms`
  - `FSS/Orca` 约 `199 ms`
- `FSS/Orca` 的 `transfer_p0/p1` 当前约 `14.6 ms`，仍然是后续重点分析对象
- `fss` 的 `DPF keygen` 异常偏慢，值得优先排查 benchmark 是否触发了 64 位路径上的实现或测量问题
- 但两边当前仍不是完全同口径：
  - `fss` 更接近 kernel 级 benchmark
  - `FSS/Orca` 更接近 runtime/facade 级 benchmark
  - 因此这些数据适合作为第一版工程对比，不应直接当作论文级公平对比

## 结果解释注意事项

第一版对比需要明确区分下面两类时间：

- `fss/bench_gpu`
  - Google Benchmark 风格
  - 主要是 kernel 级别的 `gen` / 单方 `eval` 时间
  - 当前不包含 Orca benchmark 那样拆开的 host-device 传输字段
- `FSS/*_benchmark`
  - 程序级 benchmark
  - 输出 `keygen`、`eval_p0`、`eval_p1`、`transfer_p0`、`transfer_p1`、`total`

因此第一版统一脚本会：

- 直接保留两边原始字段
- 对 `eval` 增加一个便于比较的 Orca 汇总值，例如 `eval_avg = (eval_p0 + eval_p1) / 2`
- 在输出中明确标注“`fss` 的 eval 为单方 kernel 时间，`FSS` 的 eval 为单方整段调用时间”

## 下一步

1. 在根目录新增统一性能对比脚本
2. 先完成 DPF/DCF 的一键跑通与结果解析
3. 输出首版汇总表
4. 再根据结果继续分析 Orca 是否有可借鉴的优化技术
