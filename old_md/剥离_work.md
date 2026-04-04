# DCF/DPF 剥离工作进度

## 维护提醒

每一次对话结束前，都必须先检查并更新本文件，再给出回复。
每完成一项分析、移动、裁剪、编译验证或依赖变更，都必须立即更新本文件，不允许等到阶段结束后再补写。
后续每一轮工作都要优先补充中文注释，并把已补充的范围记录到本文档。

## 工作目标

本工作的最终目标不是在 `剥离工作/` 中继续“调用仓库外部现成实现”，而是把 `DPF` 和 `DCF` 运行所需的最小集合逐步剥离到 `剥离工作/` 目录内部，形成一个自洽、可单独构建、可继续裁剪的最小运行子集。

目标约束如下：

1. 工作成果范围限定在 `剥离工作/`
2. `剥离工作/` 根目录只允许保留：
   - `Makefile`
   - `dpf.cu`
   - `dcf.cu`
   - `README.md`
   - `工作进度.md`
3. 其余代码必须按功能放入子目录
4. 当前已保留 `fss/` 目录，后续可继续创建：
   - `aes/`
   - `gpu/`
   - `runtime/`
   - `common/`
5. 最终目标是得到 `DCF/DPF` 可运行的最小集合，而不是继续依赖 `剥离工作/` 外部的大工程结构

## 当前目录规范

### 根目录允许内容

- `Makefile`
- `dpf.cu`
- `dcf.cu`
- `README.md`
- `工作进度.md`

### 子目录规划

- `fss/`
  - 存放 DPF/DCF/SSTable 等 FSS 核心逻辑
- `aes/`
  - 存放 AES table、AES context、PRG/AES backend
- `gpu/`
  - 存放 GPU 内存、随机数、CUDA helper、公共 GPU 类型
- `runtime/`
  - 存放 facade、初始化、key blob、结果打包/解包等运行时胶水层
- `common/`
  - 存放纯通用定义、基础类型、与 CUDA 无关的小工具

说明：

- 现在只是建立规划，不会一次性搬完
- 每完成一步，都需要更新本文档

## 当前状态

### 已完成

- 已将 `dpf` 和 `dcf` 拆成两个独立测试入口：
  - [dpf.cu](/home/yy404nf/FSS/剥离工作/dpf.cu)
  - [dcf.cu](/home/yy404nf/FSS/剥离工作/dcf.cu)
- 已将原先合并的 facade 拆分为：
  - [fss/dpf_api.h](/home/yy404nf/FSS/剥离工作/fss/dpf_api.h)
  - [fss/dcf_api.h](/home/yy404nf/FSS/剥离工作/fss/dcf_api.h)
- 已新增：
  - [runtime/standalone_runtime.h](/home/yy404nf/FSS/剥离工作/runtime/standalone_runtime.h)
- 已新增本地 GPU 基础设施目录：
  - `gpu/`
- 已修正测试入口的 include 路径，保证根目录只保留测试入口和文档
- 已将第一批 FSS 核心文件复制到本地：
  - `fss/gpu_dpf.h`
  - `fss/gpu_dpf.cu`
  - `fss/gpu_sstab.h`
  - `fss/gpu_fss_helper.h`
  - `fss/gpu_dpf_templates.h`
  - `fss/gpu_dcf.h`
  - `fss/gpu_dcf.cu`
  - `fss/gpu_dcf_templates.h`
  - `fss/gpu_dcf_sstab.h`
- 已将第一批 GPU 基础设施文件复制到本地：
  - `gpu/gpu_mem.h`
  - `gpu/gpu_mem.cu`
  - `gpu/gpu_random.h`
  - `gpu/gpu_random.cu`
  - `gpu/gpu_data_types.h`
  - `gpu/gpu_stats.h`
  - `gpu/helper_cuda.h`
  - `gpu/helper_string.h`
  - `gpu/curand_utils.h`
  - `gpu/misc_utils.h`
- 已将 AES backend 文件迁移到本地：
  - `aes/gpu_aes_shm.h`
  - `aes/gpu_aes_shm.cu`
  - `aes/gpu_aes_table.h`
- 已将 `Makefile` 的 include 搜索顺序改为优先搜索 `剥离工作/` 本地头文件
- 已将本地 `fss/` 与 `runtime/` 中对 `utils/...` 的 include 批量切换为 `gpu/...`
- 已将 `Makefile` 的额外编译单元从根目录 `utils/gpu_mem.cu + gpu_file_utils.cpp + sigma_comms.cpp` 缩减为本地 `gpu/gpu_mem.cu`
- 已将 `gpu/gpu_data_types.h` 中对 `sytorch/tensor.h` 的依赖替换为本地基础类型定义
- 已将 `Makefile` 的链接库从 `sytorch/cryptoTools/LLAMA/bitpack + CUDA` 压缩为纯 CUDA 基础库：
  - `-lcuda`
  - `-lcudart`
  - `-lcurand`
- 已将 `Makefile` 的 include 路径进一步压缩为仅保留：
  - `剥离工作/`
- 已确认当前 `DCF/DPF` 仍然大量依赖 `剥离工作/` 外部的：
  - `fss/`
  - `utils/`
  - 若干链接库

### 当前结论

当前 `剥离工作/` 目录已经不再只是“剥离工作台”，而是已经形成了一条可以独立构建的本地最小运行链。

当前仍未完全结束的部分主要是：

1. 继续识别并删除已经复制进来但未必还需要的遗留文件
2. 继续压缩头文件包含关系
3. 把“当前最小运行文件集”和“历史残留文件集”彻底区分清楚

新增决策：

- 对明确可公开拉取、版本清晰的第三方库，优先按“外部依赖”处理，不在 `剥离工作/` 内手拆
- 对项目自研逻辑、项目内 glue code、或无法明确稳定拉取的部分，继续由我手动剥离到 `剥离工作/`
- 当前已确认 `cutlass` 属于前者；`sytorch` 暂时还不能直接归类为“完全外置即可”

## 最小集合定义

这里的“最小集合”定义为：

- 能完成 `DPF keygen + eval`
- 能完成 `DCF keygen + eval`
- 能跑通当前的 [dpf.cu](/home/yy404nf/FSS/剥离工作/dpf.cu) 和 [dcf.cu](/home/yy404nf/FSS/剥离工作/dcf.cu)
- 只保留这两条功能链必需的源码、头文件、CUDA 运行时依赖
- 尽量消除与主工程其余模块、神经网络模块、通信模块、训练模块的耦合

不属于最小集合的内容包括：

- 通用 NN 层实现
- 通信与多方协议框架
- 与 DPF/DCF 无关的 cutlass/llama/orca 功能
- 与当前测试入口无关的其它 FSS primitive

## 依赖拆解结论

### DPF 当前主链

`dpf.cu`
-> `剥离工作/fss/dpf_api.h`
-> `fss/gpu_dpf.h`
-> `fss/gpu_dpf.cu`
-> `fss/gpu_sstab.h`
-> `aes/gpu_aes_shm.h`
-> `gpu/gpu_mem.h`
-> `gpu/gpu_random.h`
-> `gpu/gpu_data_types.h`

### DCF 当前主链

`dcf.cu`
-> `剥离工作/fss/dcf_api.h`
-> `fss/gpu_dcf.h`
-> `fss/gpu_dcf.cu`
-> `fss/gpu_sstab.h`
-> `aes/gpu_aes_shm.h`
-> `gpu/gpu_mem.h`
-> `gpu/gpu_random.h`
-> `gpu/gpu_data_types.h`

### 当前共享硬依赖

- `fss/gpu_sstab.h`
- `aes/gpu_aes_shm.h` 和 `aes/gpu_aes_table.h`
- `gpu/gpu_mem.h/.cu`
- `gpu/gpu_random.h/.cu`
- `gpu/gpu_data_types.h`
- CUDA runtime / driver / curand

### 当前需要重点去耦的外部依赖

- 当前主链已不再依赖 `sytorch`
- 当前主链已不再依赖 `cutlass`
- 当前主链已不再依赖 `LLAMA`
- 当前主链已不再依赖 `cryptoTools`
- 当前主链已不再依赖 `bitpack`

说明：

- 上述依赖都已经从当前最小构建链中移除
- 保留这些分析记录，是为了说明剥离过程中已经完成过哪些去耦步骤

## 公开依赖处理原则

### 原则

- 公开库：
  - 能稳定通过 `git clone`、submodule 或 release 包获取
  - 且这里只是编译期或链接期依赖
  - 则不在 `剥离工作/` 内细拆，只记录获取方式和版本
- 非公开或高耦合部分：
  - 只存在于当前项目
  - 或与当前项目代码强耦合
  - 或无法确认是否能稳定独立获取
  - 则继续手拆进 `剥离工作/`

### 当前分类

- 已从当前最小链路中移除：
  - `cutlass`
  - `sytorch`
  - `LLAMA`
  - `cryptoTools`
  - `bitpack`
- 明确需要手拆：
  - DPF/DCF 主链
  - SSTable
  - AES glue
  - GPU runtime 基础设施
  - `剥离工作/` 自己的 runtime/facade

## 当前最小运行文件清单

### 入口

- `dpf.cu`
- `dcf.cu`

### 运行时胶水

- `runtime/standalone_runtime.h`
- `fss/dpf_api.h`
- `fss/dcf_api.h`

### FSS 核心

- `fss/gpu_dpf.h`
- `fss/gpu_dpf.cu`
- `fss/gpu_dpf_templates.h`
- `fss/gpu_dcf.h`
- `fss/gpu_dcf.cu`
- `fss/gpu_dcf_templates.h`
- `fss/gpu_dcf_sstab.h`
- `fss/gpu_sstab.h`
- `fss/gpu_fss_helper.h`

### GPU 基础设施

- `gpu/gpu_mem.h`
- `gpu/gpu_mem.cu`
- `gpu/gpu_random.h`
- `gpu/gpu_random.cu`
- `gpu/gpu_data_types.h`
- `gpu/gpu_stats.h`
- `gpu/helper_cuda.h`
- `gpu/helper_string.h`
- `gpu/curand_utils.h`
- `gpu/misc_utils.h`

### AES backend

- `aes/gpu_aes_shm.h`
- `aes/gpu_aes_shm.cu`
- `aes/gpu_aes_table.h`

说明：

- 这是“当前可编译运行链”对应的文件集合
- 后续仍可能继续缩减

## 中文注释覆盖清单

### 已覆盖的主链文件

- facade：
  - `fss/dpf_api.h`
  - `fss/dcf_api.h`
- runtime：
  - `runtime/standalone_runtime.h`
- FSS 主入口：
  - `fss/gpu_dpf.h`
  - `fss/gpu_dcf.h`
- FSS 模板与主流程：
  - `fss/gpu_dpf_templates.h`
  - `fss/gpu_dcf_templates.h`
  - `fss/gpu_dpf.cu`
  - `fss/gpu_dcf.cu`
- 小 bin / SSTable 路径：
  - `fss/gpu_sstab.h`
  - `fss/gpu_dcf_sstab.h`
- FSS helper：
  - `fss/gpu_fss_helper.h`
- GPU 基础设施：
  - `gpu/gpu_mem.h`
  - `gpu/gpu_random.h`
  - `gpu/gpu_random.cu`
- AES：
  - `aes/gpu_aes_shm.h`
  - `aes/gpu_aes_shm.cu`

### 当前暂缓补充的文件

- `gpu/gpu_data_types.h`
- `gpu/gpu_stats.h`
- `gpu/helper_cuda.h`
- `gpu/helper_string.h`
- `gpu/curand_utils.h`
- `gpu/misc_utils.h`
- `gpu/gpu_mem.cu`
- `aes/gpu_aes_table.h`

说明：

- 上述“暂缓补充”文件大多属于基础类型、常量表、通用 helper 或薄封装
- 它们不在“理解 DPF/DCF 主逻辑”的最短阅读路径上
- 除非后续发现这些文件重新成为裁剪阻塞点，否则中文注释优先级低于继续做减法

## 当前裁剪观察

- `gpu_fss_helper.h` 当前主要承载：
  - `lsb`
  - `gpuMsb`
  - `getVCW`
  - `writeVCW`
  - `moveMasks`
  其中前三类更偏 FSS 打包逻辑，`moveMasks` 更偏 GPU 输出缓冲管理，后续存在继续拆分的空间
- `gpu_random.cu` 对 `misc_utils.h` 的主要实际依赖是 `modKernel`，而 `modKernel` 本身又只是 `randomGEOnGpu` 的位宽裁剪辅助
- `misc_utils.h` 当前仍是一个“混合型工具头”，同时放了：
  - 打包输出写入
  - 线性组合 kernel
  - 模运算/位宽裁剪
  - share 打包辅助
  后续如果继续做减法，可以优先考虑把“随机数位宽裁剪”和“FSS 输出打包”从中拆出来

## 公开库获取方式

### cutlass

当前仓库通过 submodule 使用 `cutlass`，`.gitmodules` 记录的地址为：

- `https://github.com/NVIDIA/cutlass.git`

独立获取方式：

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v2.11.0
```

如果按当前仓库方式初始化：

```bash
git submodule update --init --recursive ext/cutlass
```

### sytorch

当前 `ext/sytorch` 在这个仓库里是 vendored 工作树，但不是顶层 `.gitmodules` 中已声明的公开 submodule。

目前只能确认：

- 它在当前项目中被直接 vendor 进来了
- 它不像 `cutlass` 一样已经有清晰的 submodule 获取路径

当前策略：

- 不先把 `sytorch` 视为“只写拉取命令即可”的公开依赖
- 优先尝试在剥离过程中削弱或消除它的必要性
- 如果后面证明仍然必须依赖它，再补充它的独立获取方案

## 剥离计划表

| 阶段 | 目标 | 主要动作 | 产出 | 状态 |
| --- | --- | --- | --- | --- |
| P0 | 整理工作区结构 | 清理根目录，只保留测试入口和文档；把实现迁入子目录 | 规范化目录骨架 | 已开始 |
| P1 | 画清真实依赖图 | 明确 DPF/DCF 主链、共享依赖、可删依赖 | 本文档依赖分析部分 | 已完成首版 |
| P2 | 剥离 facade/runtime | 将 facade 拆成 DPF/DCF 两个独立 API，并保留 `runtime/` 作为共享运行时组件 | `runtime/` 初版 + 独立 API facade | 已完成 |
| P3 | 剥离 FSS 核心 | 将 DPF、DCF、SSTable 从仓库根目录 `fss/` 复制/裁剪到 `剥离工作/fss/` | `剥离工作/fss/` 最小核心 | 已开始 |
| P4 | 剥离 GPU 基础设施 | 将最小必需的 `gpu_mem`、`gpu_random`、helper、类型定义移入 `gpu/` | `剥离工作/gpu/` | 已开始 |
| P5 | 剥离 AES backend | 将 `gpu_aes_shm` 和最小 AES table 移入 `aes/` | `剥离工作/aes/` | 已完成 |
| P6 | 去掉 sytorch 耦合 | 重写最小 `gpu_data_types` 或替换掉 `sytorch/tensor.h` 依赖 | 无 `sytorch` 头依赖 | 已完成 |
| P7 | 外部依赖分类 | 区分哪些公开库只记录获取方式，哪些必须继续手拆 | 公开依赖清单 | 已开始 |
| P8 | 最小链接实验 | 逐步去掉 `LLAMA`、`cryptoTools`、`bitpack`、`cutlass` 等非必需项 | 最小 Makefile | 已大幅完成 |
| P9 | 自包含验证 | 确认构建只依赖 `剥离工作/` 内源码、CUDA 基础环境，以及必要公开库 | 可独立构建运行 | 已开始 |

## 近期执行顺序

### 第一优先级

1. 补充 `dpf_api.h`、`dcf_api.h`、`runtime/` 的中文注释，先把 facade 层读顺
2. 继续检查 DPF/DCF 真实主链需要的源码是否还能进一步缩减
3. 确定 `gpu_sstab.h` 和相关共享头文件是否还能继续拆薄

### 第二优先级

1. 把 `utils/gpu_mem.*`、`utils/gpu_random.*`、`utils/gpu_data_types.h` 中的最小子集迁入 `剥离工作/gpu/`
2. 把 `gpu_aes_shm.*` 和最小必要 table 迁入 `剥离工作/aes/`
3. 去掉对仓库根目录 `fss/` 和 `utils/` 的直接 include

### 第三优先级

1. 去掉 `sytorch/tensor.h` 依赖
2. 验证 `cutlass` 对当前 DPF/DCF 主链是否真必需
3. 验证 `LLAMA`、`cryptoTools`、`bitpack` 是否只是历史保守链接项

## 后续工作方向

### 方向 A：补中文注释

目标：

- 优先给 facade、runtime、FSS 主入口和关键 GPU/AES 胶水补中文注释
- 把“最小运行链怎么走”直接写在代码附近

优点：

- 后续继续裁剪时不需要反复重新读旧实现
- 方便快速判断某个文件/函数是不是还在主链上

### 方向 B：继续做减法

目标：

- 在已有自包含最小链基础上继续删减冗余 include、接口和文件

优点：

- 能继续逼近真正的最小集合

### 当前建议

建议按“先补关键中文注释，再继续做减法”的顺序推进：

1. 先把最小运行链的入口和 facade 说明白
2. 再继续压缩共享头文件和历史兼容残留

原因：

- 当前链路已经自包含，短板不再是“搬不进来”，而是“可读性和可维护性不足”
- 先补注释能显著降低后续裁剪和交接成本

## 具体任务清单

### T1 目录整理

- 将所有非测试入口代码迁出根目录
- 按 `fss/`、`gpu/`、`aes/`、`runtime/` 分类

状态：

- 已部分完成

### T2 runtime/facade 拆分

- 将 facade 和 runtime 中的：
  - `Runtime`
  - `KeyBlob`
  - `copyVectorToGpu`
  - `copyPackedOutputToHost`
  - `unpackPackedOutput`
  保留在 `runtime/`
- 将 DPF/DCF facade 分拆为：
  - `fss/dpf_api.h`
  - `fss/dcf_api.h`

状态：

- 已完成

### T3 FSS 主链复制

- 复制并裁剪 DPF 相关最小文件到 `剥离工作/fss/`
- 复制并裁剪 DCF 相关最小文件到 `剥离工作/fss/`
- 复制共享 `SSTable` 逻辑到 `剥离工作/fss/`

状态：

- 已开始

### T4 GPU 基础设施复制

- 复制并裁剪：
  - `gpu_mem`
  - `gpu_random`
  - `helper_cuda`
  - 最小 `gpu_data_types`

状态：

- 已开始

### T5 AES 复制

- 复制并裁剪：
  - `gpu_aes_shm`
  - `gpu_aes_table`

状态：

- 未开始

### T6 依赖去耦

- 去掉 `sytorch/tensor.h`
- 检查并去掉不必要的 `cutlass`
- 去掉非必需链接库

状态：

- 已开始

### T7 最小构建验证

- 修改 `Makefile`
- 保证构建只依赖 `剥离工作/` 内源码、CUDA 基础环境，以及必要公开库

状态：

- 已开始

### T8 公开依赖说明

- 为明确可公开拉取的依赖记录获取方式
- 对不能明确外置的依赖继续做裁剪或手拆

状态：

- 已开始

## 需要持续记录的事项

从现在开始，每一轮工作都要把下面内容写回本文档：

1. 本轮做了什么
2. 移动或新增了哪些文件
3. 删掉了哪些外部依赖
4. 还剩哪些阻塞项
5. 下一步准备做什么
6. 本轮补充了哪些中文注释

## 进度日志

### 2026-03-19 第 1 次更新

本轮完成：

- 建立了当前工作总目标
- 明确了根目录文件约束
- 明确了后续子目录规划
- 补全了阶段计划表、方向、任务清单
- 将根目录 `fss_api.h` 移入 [fss/fss_api.h](/home/yy404nf/FSS/剥离工作/fss/fss_api.h)
- 更新了测试入口对 `fss/fss_api.h` 的引用

当前阻塞：

- 真实 DPF/DCF 实现还在 `剥离工作/` 外部
- GPU/AES/runtime/common 还没有真正拆进本目录
- 当前构建仍依赖外部 include 和 link 路径

下一步：

- 开始执行 P2 和 P3
- 先拆 runtime，再复制最小 FSS 主链

### 2026-03-19 第 2 次更新

本轮完成：

- 明确了公开依赖和手拆部分的处理边界
- 确认 `cutlass` 可以按公开依赖处理，并已记录获取方式
- 确认 `sytorch` 当前还不能直接视为“只需要拉取即可”的稳定外部依赖
- 将“公开依赖说明”加入计划表和任务清单

当前结论：

- 可公开拉取的第三方库不必在 `剥离工作/` 内细拆
- 项目自研逻辑、项目内 glue code、以及高耦合部分仍然需要继续手拆

下一步：

- 优先继续推进 runtime/FSS/GPU/AES 的手拆
- 并在过程中尽量消除 `sytorch` 这类不稳定外部依赖

### 2026-03-19 第 3 次更新

本轮完成：

- 将 `fss_api.h` 中的 runtime/gpu glue 抽出到 [runtime/standalone_runtime.h](/home/yy404nf/FSS/剥离工作/runtime/standalone_runtime.h)
- 保留 [fss/fss_api.h](/home/yy404nf/FSS/剥离工作/fss/fss_api.h) 作为更薄的 FSS facade
- 将第一批 DPF/DCF/AES/SSTable 相关源码复制到 `剥离工作/fss/`
- 将 `Makefile` 改为优先搜索 `剥离工作/` 本地头文件
- 进行了 `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86` 验证

编译验证结果：

- 编译已经开始命中 `剥离工作/fss/` 中的本地副本
- 首轮缺失的 `gpu_fss_helper.h` 已补齐
- 当前已经越过“本地 FSS 文件缺失”阶段
- 当前卡在链接阶段，仍缺：
  - `-lsytorch`
  - `-lcryptoTools`
  - `-lLLAMA`
  - `-lbitpack`

当前结论：

- `runtime` 和 `fss` 的本地接管已经开始生效
- 下一阶段的重点应转向：
  - 继续本地化 `gpu/` 与 `aes/`
  - 然后再压缩链接库集合

下一步：

- 继续剥离 `utils/` 中 DPF/DCF 真正需要的 GPU 基础设施到 `剥离工作/gpu/`
- 继续摆脱对根目录 `utils/` 的直接依赖

### 2026-03-19 第 4 次更新

本轮完成：

- 将 `gpu_mem`、`gpu_random`、`gpu_data_types`、`gpu_stats`、`helper_cuda`、`helper_string`、`curand_utils`、`misc_utils`、`helper_cutlass` 复制到 `剥离工作/gpu/`
- 将本地 `fss/` 与 `runtime/` 中已接管文件的 `utils/...` include 改为 `gpu/...`
- 将 `Makefile` 的额外编译单元缩减为本地 [gpu/gpu_mem.cu](/home/yy404nf/FSS/剥离工作/gpu/gpu_mem.cu)
- 移除了当前 DPF/DCF 主链对 `gpu_file_utils.cpp` 和 `sigma_comms.cpp` 的构建期依赖
- 将 `OneGB` 的使用替换为本地常量表达式，避免重新引入通信/文件工具链

编译验证结果：

- `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86` 已命中本地 `gpu/` 副本
- 当前编译可通过到链接阶段
- 当前主要阻塞仍是外部链接库缺失：
  - `-lsytorch`
  - `-lcryptoTools`
  - `-lLLAMA`
  - `-lbitpack`

当前结论：

- 现在已经初步摆脱了对根目录 `utils/gpu_mem.cu`、`gpu_file_utils.cpp`、`sigma_comms.cpp` 的直接依赖
- 下一步重点应从“复制源码”转向“压缩链接库集合”和“继续清理 `gpu_data_types.h` 带来的外部耦合”

下一步：

- 检查 `sytorch`、`cryptoTools`、`LLAMA`、`bitpack` 是否仍是当前 DPF/DCF 链接必需
- 优先处理 `gpu_data_types.h -> sytorch/tensor.h` 这条依赖链

### 2026-03-19 第 5 次更新

本轮完成：

- 将 [gpu/gpu_data_types.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_data_types.h) 中对 `sytorch/tensor.h` 的依赖替换为本地 `u64/u32/u16/u8/i64/i32` typedef
- 完成最小链接实验，验证 `dpf` 和 `dcf` 均不再需要：
  - `-lsytorch`
  - `-lcryptoTools`
  - `-lLLAMA`
  - `-lbitpack`
- 将 [Makefile](/home/yy404nf/FSS/剥离工作/Makefile) 固化为最小版本：
  - 仅保留本地源码
  - 仅保留 `cutlass` 头文件路径
  - 仅保留 CUDA 基础库 `-lcuda -lcudart -lcurand`
- 使用精简后的 `Makefile` 成功完成：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`

当前结论：

- `sytorch` 已不再是当前 DPF/DCF 主链的编译期或链接期依赖
- `cryptoTools`、`LLAMA`、`bitpack` 已确认不是当前 DPF/DCF 主链的必需链接库
- 当前外部公开依赖已大幅压缩，核心只剩：
  - CUDA 基础环境
  - `cutlass` 头文件

下一步：

- 继续检查 `gpu_linear_helper` 引入的 `cutlass` 是否真是 DPF/DCF 当前功能链的必需依赖
- 继续清理 `fss/` 和 `gpu/` 中仍然保留但可能未使用的历史代码

### 2026-03-19 第 6 次更新

本轮完成：

- 按当前剥离工作自己的结构需要，将原先 `剥离工作/fss/dcf/` 下的 DCF 文件拍平到 `剥离工作/fss/`
- 当前 DCF 相关文件改为与 DPF 同层放置：
  - `fss/gpu_dcf.h`
  - `fss/gpu_dcf.cu`
  - `fss/gpu_dcf_templates.h`
  - `fss/gpu_dcf_sstab.h`
- 更新了 [fss/fss_api.h](/home/yy404nf/FSS/剥离工作/fss/fss_api.h) 中的 include 路径
- 删除了空目录 `剥离工作/fss/dcf`
- 重新验证：
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 结果通过

当前结论：

- `剥离工作/` 的目录结构不再继续模仿原项目的 `fss/dcf/` 分层
- 当前 `fss/` 已按剥离目标改为更扁平的结构

下一步：

- 继续检查 `cutlass` 是否还能从当前链路中剔除
- 继续清理 `fss/` 内部沿袭自原项目但未必必要的文件拆分方式

### 2026-03-19 第 7 次更新

本轮完成：

- 分析确认 [fss/gpu_dpf.cu](/home/yy404nf/FSS/剥离工作/fss/gpu_dpf.cu) 中对 `gpu_linear_helper.h` 的包含没有对应的实际调用
- 移除了 [fss/gpu_dpf.cu](/home/yy404nf/FSS/剥离工作/fss/gpu_dpf.cu) 中对 `gpu_linear_helper.h` 的依赖
- 通过临时最小 include 实验确认：
  - 当前 `dpf`
  - 当前 `dcf`
  都不需要 `cutlass` 头文件路径即可编译
- 将 [Makefile](/home/yy404nf/FSS/剥离工作/Makefile) 的 include 路径进一步压缩为只保留 `剥离工作/`
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- `cutlass` 已不再是当前 DPF/DCF 主链的必需外部依赖
- 当前构建所需外部环境已进一步压缩为 CUDA 基础环境

下一步：

- 继续清理 `fss/gpu_linear_helper.*`、`gpu/helper_cutlass.h` 这类已经脱离主链的遗留文件
- 继续识别并删除当前最小链路中未被使用的复制文件

### 2026-03-19 第 8 次更新

本轮完成：

- 删除了已经脱离当前主链的遗留文件：
  - `fss/gpu_linear_helper.h`
  - `fss/gpu_linear_helper.cu`
  - `gpu/helper_cutlass.h`
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过
- 更新了 [README.md](/home/yy404nf/FSS/剥离工作/README.md)，反映当前最小构建已只依赖 `剥离工作/` 内源码和 CUDA 基础环境

当前结论：

- `gpu_linear_helper` 和 `helper_cutlass` 已确认不是当前 DPF/DCF 最小链路的一部分
- 当前目录已经进一步收紧到更接近实际使用集合

下一步：

- 继续检查是否还有复制进来的头文件只是“历史兼容残留”而非当前必需
- 开始考虑把 `aes` 和 `gpu` 中仍然混杂的内容再进一步归并或删减

### 2026-03-19 第 9 次更新

本轮完成：

- 将 AES backend 文件从 `fss/` 迁移到独立的 `aes/` 目录：
  - `aes/gpu_aes_shm.h`
  - `aes/gpu_aes_shm.cu`
  - `aes/gpu_aes_table.h`
- 修正了 `runtime/`、`fss/` 中对 AES 头文件的 include 路径
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过
- 更新了 [README.md](/home/yy404nf/FSS/剥离工作/README.md)，补充 `aes/` 目录职责

当前结论：

- 当前目录职责已经更清晰：
  - `fss/` 负责 FSS 主逻辑
  - `gpu/` 负责 GPU 基础设施
  - `aes/` 负责 AES backend
- `P5` 已进入完成状态

下一步：

- 继续检查 `gpu/` 与 `aes/` 中是否还有可进一步压缩的头文件
- 继续清理历史兼容残留

### 2026-03-19 第 10 次更新

本轮完成：

- 删除了 [Makefile](/home/yy404nf/FSS/剥离工作/Makefile) 中已经无用的 `ROOT` 变量
- 清理了 [fss/gpu_dpf.cu](/home/yy404nf/FSS/剥离工作/fss/gpu_dpf.cu) 中未使用的标准库 include
- 清理了 [fss/gpu_dcf.cu](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf.cu) 中未使用的标准库 include
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前最小链路不仅在目录结构上更清晰，构建脚本和源码头依赖也进一步收紧
- 现阶段剩余工作更偏向“继续识别细碎残留”而不是大块依赖剥离

下一步：

- 继续检查 `gpu/`、`aes/`、`fss/` 中剩余文件的 include 是否还能继续精简
- 评估是否要开始整理一份“当前最小运行文件清单”

### 2026-03-19 第 11 次更新

本轮完成：

- 补充了“当前最小运行文件清单”，将入口、runtime、FSS、GPU、AES 五类文件明确列出
- 更新了 [README.md](/home/yy404nf/FSS/剥离工作/README.md)，同步记录当前最小运行文件集合
- 修正文档中仍停留在早期阶段的旧表述，使其反映当前真实状态：
  - 当前主链已可独立构建
  - 当前主链已不再依赖 `sytorch/cutlass/LLAMA/cryptoTools/bitpack`

当前结论：

- 现在已经有一份可直接作为后续裁剪基线的“最小运行文件清单”
- 后续清理工作可以围绕这份清单继续做减法，而不需要再重新梳理主链

下一步：

- 开始对照这份最小清单，检查是否仍有文件只是“被列入但未实际必需”
- 优先排查 `gpu_random.*`、`misc_utils.h`、`gpu_fss_helper.h` 等公共文件中的可删部分

### 2026-03-19 第 12 次更新

本轮完成：

- 对公共文件做了函数级瘦身，删除了当前主链未使用的接口与工具函数
- 在 [gpu/gpu_random.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_random.h) / [gpu/gpu_random.cu](/home/yy404nf/FSS/剥离工作/gpu/gpu_random.cu) 中删除了未被当前 DPF/DCF 主链使用的部分：
  - `randomGEOnCpu`
  - `initCPURandomness`
  - `destroyCPURandomness`
  - `getMaskedInputOnGpu`
  - `getMaskedInputOnCpu`
  - `writeShares`
  - `randomUIntsOnCpu`
- 在 [gpu/misc_utils.h](/home/yy404nf/FSS/剥离工作/gpu/misc_utils.h) 中删除了未被当前主链使用的部分：
  - `xorKernel`
  - `gpuXor`
  - `unmaskKernel`
  - `unmaskValues`
  - `cpuMod`
  - `cpuArs`
  - `asFloat`
  - `dropOSPageCache`
- 在 [fss/gpu_fss_helper.h](/home/yy404nf/FSS/剥离工作/fss/gpu_fss_helper.h) 中删除了未被当前主链使用的 `printAESBlock`
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前最小链路中的公共基础文件已经不再携带大量明显无关的历史接口
- 现在剩余的清理工作将更偏向“精细确认是否还可继续收缩”，而不是大面积删减

下一步：

- 继续检查最小文件清单中的每个文件是否仍包含多余 include
- 评估 `gpu_random.cu` 与 `misc_utils.h` 中剩余接口是否还能进一步按用途拆小

### 2026-03-19 第 13 次更新

本轮完成：

- 继续做头文件传播面收缩
- 在 [gpu/gpu_data_types.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_data_types.h) 中移除了对 `gpu_stats.h` 的直接包含
- 在 [gpu/gpu_mem.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_mem.h) 中将 `Stats` 改为前置声明
- 在 [fss/gpu_dcf_templates.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf_templates.h) 中去掉了不必要的 `gpu_stats.h` 直接包含
- 在 [fss/fss_api.h](/home/yy404nf/FSS/剥离工作/fss/fss_api.h) 中将 `Stats` 改为前置声明
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前最小链路的头文件传播面继续缩小
- `Stats` 这类只出现在指针签名中的类型已经不再被无意义地大范围 include 传播

下一步：

- 继续检查剩余头文件是否还能前置声明化
- 继续对照最小文件清单，判断是否还有文件虽然参与 include 链但可继续拆薄

### 2026-03-19 第 14 次更新

本轮完成：

- 继续做 include 级收缩
- 在 [gpu/gpu_random.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_random.h) 中去掉了不必要的 `<curand.h>` 传播
- 在 [fss/gpu_dcf_templates.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf_templates.h) 中去掉了未使用的：
  - `gpu/gpu_mem.h`
  - `<vector>`
- 在 [fss/gpu_dcf.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf.h) 中去掉了未使用的 `<cassert>`
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前最小链路的头文件传播面继续缩小
- 现在的清理已经进入“剩余依赖逐项抠细节”的阶段

下一步：

- 继续排查其余头文件中的无用标准库包含
- 开始考虑是否要把“当前最小运行文件清单”进一步细化成“必需文件”和“待确认文件”

### 2026-03-19 第 15 次更新

本轮完成：

- 移除了 `剥离工作/` 本地副本中保留的作者与版权声明信息
- 清理范围覆盖了当前 `aes/`、`fss/`、`gpu/` 下的本地源码/头文件副本
- 额外补清了 [gpu/curand_utils.h](/home/yy404nf/FSS/剥离工作/gpu/curand_utils.h) 中残留的 NVIDIA 版权声明块
- 通过全局搜索确认 `剥离工作/` 中已不再包含：
  - `Author:`
  - `Copyright:`
  - `Permission is hereby granted`
  - `All rights reserved`
  - `SPDX-License-Identifier`
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前 `剥离工作/` 目录内的本地副本已经移除了作者及版权声明信息
- 这次清理没有影响当前最小运行链的构建

下一步：

- 继续做最小集合收缩，重点仍是“必需文件”和“待确认文件”的进一步区分
- 如果需要，可以开始把文档中的历史分析段再压缩成更贴近当前状态的版本

### 2026-03-19 第 16 次更新

本轮完成：

- 将“后续每一轮优先补充中文注释”写入本文档开头的维护提醒
- 将后续工作方向调整为：
  - 先补关键中文注释
  - 再继续做减法
- 将 facade 层现状补回文档，明确当前入口已改为：
  - [fss/dpf_api.h](/home/yy404nf/FSS/剥离工作/fss/dpf_api.h)
  - [fss/dcf_api.h](/home/yy404nf/FSS/剥离工作/fss/dcf_api.h)
- 在 [fss/dpf_api.h](/home/yy404nf/FSS/剥离工作/fss/dpf_api.h) 中补充了中文注释，说明 DPF keygen/eval 的职责
- 在 [fss/dcf_api.h](/home/yy404nf/FSS/剥离工作/fss/dcf_api.h) 中补充了中文注释，说明 DCF keygen/eval 的职责
- 更新了 [README.md](/home/yy404nf/FSS/剥离工作/README.md) 中对 facade 文件的说明
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- “补充中文注释”已经从口头要求变成文档中的正式后续方向
- facade 层已经从旧的合并接口切换为 DPF/DCF 两个独立入口
- 当前改动没有影响最小运行链，`dcf` 仍只保留已有的符号号变换警告

下一步：

- 继续给 `runtime/standalone_runtime.h` 补中文注释
- 继续给 `fss/gpu_dpf.h`、`fss/gpu_dcf.h` 这类主入口头文件补中文注释
- 在补注释过程中同步标记哪些共享类型/函数仍值得继续裁剪

### 2026-03-19 第 17 次更新

本轮完成：

- 在 [runtime/standalone_runtime.h](/home/yy404nf/FSS/剥离工作/runtime/standalone_runtime.h) 中补充了首批中文注释，覆盖：
  - 批大小估算常量
  - `Runtime` 的职责
  - `KeyBlob` 的用途
  - `unpackPackedOutput` 的作用
- 在 [fss/gpu_dpf.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dpf.h) 中补充了中文注释，说明：
  - `GPUDPFTreeKey`/`GPUDPFKey` 的含义
  - `readGPUDPFTreeKey`/`readGPUDPFKey` 的解析职责
  - 兼容别名 `readGPUDcfKey` 的保留原因
- 在 [fss/gpu_dcf.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf.h) 中补充了中文注释，说明：
  - `GPUDCFTreeKey`/`GPUDCFKey` 的含义
  - `readGPUDCFTreeKey`/`readGPUDCFKey` 的解析职责
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前最小运行链的 runtime 层和 FSS 主入口层已经开始具备基本中文可读性
- 本轮只补充注释，没有引入新的源码依赖或结构变化
- `dcf` 构建仍只保留已有的符号号变换 warning，没有新增问题

下一步：

- 继续给 [fss/gpu_sstab.h](/home/yy404nf/FSS/剥离工作/fss/gpu_sstab.h) 补中文注释
- 继续给 [aes/gpu_aes_shm.h](/home/yy404nf/FSS/剥离工作/aes/gpu_aes_shm.h) 和 [gpu/gpu_mem.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_mem.h) 补中文注释
- 在补注释过程中记录哪些公共结构体仍然偏大，后续可继续拆薄

### 2026-03-19 第 18 次更新

本轮完成：

- 在 [fss/gpu_sstab.h](/home/yy404nf/FSS/剥离工作/fss/gpu_sstab.h) 中补充了中文注释，覆盖：
  - `GPUSSTabKey` 的用途
  - `readGPUSSTabKey` 的解析职责
  - `lookupSSTable` / `gpuLookupSSTable` 的执行流程
  - `dpfShares` / `dcfShares` 两种小 bin 编码方式
  - `genSSTableKernel` / `genSSTable` 的生成职责
- 在 [aes/gpu_aes_shm.h](/home/yy404nf/FSS/剥离工作/aes/gpu_aes_shm.h) 中补充了中文注释，说明：
  - `AESGlobalContext` 保存的全局 GPU 查表数据
  - `AESSharedContext` 在 shared memory 中的用途
- 在 [gpu/gpu_mem.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_mem.h) 中补充了中文注释，说明：
  - 这一层是最小运行链统一使用的内存申请/拷贝接口
  - `initGPUMemPool` 的职责
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前最小运行链的入口、runtime、FSS 主入口、SSTable、AES 头部和 GPU 内存接口都已有中文注释
- 现在最缺注释的部分开始下沉到模板实现和 `.cu` 实现内部
- `dcf` 构建仍只保留已有的符号号变换 warning，没有新增问题

下一步：

- 继续给 [fss/gpu_dpf_templates.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dpf_templates.h) 和 [fss/gpu_dcf_templates.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf_templates.h) 补中文注释
- 继续给 [gpu/gpu_random.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_random.h) 补中文注释
- 在补注释过程中同步记录哪些模板/辅助函数仍然耦合偏重，后续可以继续裁剪

### 2026-03-19 第 19 次更新

本轮完成：

- 在 [fss/gpu_dpf_templates.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dpf_templates.h) 中补充了中文注释，说明：
  - `idPrologue` 的默认输入透传作用
  - `dReluPrologue` / `geluPrologue` 的比较点构造职责
  - `dReluEpilogue` / `geluEpilogue` / `maskEpilogue` / `idEpilogue` 的结果合成职责
- 在 [fss/gpu_dcf_templates.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf_templates.h) 中补充了中文注释，说明：
  - prologue/epilogue 模板在 DCF 里的复用方式
  - `idPrologue` / `idEpilogue` / `maskEpilogue` / `dReluPrologue` / `dReluEpilogue` 各自的语义
- 在 [gpu/gpu_random.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_random.h) 中补充了中文注释，说明：
  - `randomGEOnGpu` 的用途
  - `randomAESBlockOnGpu` 的用途
  - GPU 随机源初始化/销毁接口的职责
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前最小运行链的主要头文件已经基本具备中文导航信息
- 现在更缺解释的是 `gpu_dpf.cu`、`gpu_dcf.cu`、`gpu_random.cu` 这类实现文件中的主流程
- `dcf` 构建仍只保留已有的符号号变换 warning，没有新增问题

下一步：

- 继续给 [fss/gpu_dpf.cu](/home/yy404nf/FSS/剥离工作/fss/gpu_dpf.cu) 和 [fss/gpu_dcf.cu](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf.cu) 补主流程中文注释
- 继续给 [gpu/gpu_random.cu](/home/yy404nf/FSS/剥离工作/gpu/gpu_random.cu) 补随机数实现相关注释
- 在补实现文件注释的同时，标记是否存在可以进一步拆成更小 helper 的代码段

### 2026-03-19 第 20 次更新

本轮完成：

- 在 [fss/gpu_dpf.cu](/home/yy404nf/FSS/剥离工作/fss/gpu_dpf.cu) 中补充了中文注释，说明：
  - `expandDPFTreeNode` 的单层展开职责
  - `doDpf` 的默认 DPF 树遍历流程
  - `doDcf` 的 DCF-on-DPF 复用路径
  - `dpfTreeEval` / `gpuDpfTreeEval` 的 kernel 与数据搬运关系
  - `gpuDpf` / `gpuDcf` 总入口中“小 bin 查表 / 大 bin 走树”的分支
- 在 [fss/gpu_dcf.cu](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf.cu) 中补充了中文注释，说明：
  - `getGroupElementFromAESBlock` 的位切片职责
  - `traverseOneDCF` 的单层遍历职责
  - `doDcf` / `gpuDcfTreeEval` / `gpuDcf` 的主流程
  - `keyGenDCFKernel` / `doDcfTreeKeyGen` / `gpuKeyGenDCF` 的 keygen 主路径
- 在 [gpu/gpu_random.cu](/home/yy404nf/FSS/剥离工作/gpu/gpu_random.cu) 中补充了中文注释，说明：
  - 原始随机数生成流程
  - 指定位宽随机元素生成方式
  - AES block 随机生成方式
  - CURAND 初始化/销毁策略
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前最小运行链的头文件和主要 `.cu` 实现文件都已经有了主流程级中文注释
- 现在剩余更适合补注释的部分主要是更细的 helper 和个别小型实现文件
- `dcf` 构建仍只保留已有的符号号变换 warning，行号因注释插入变为 `gpu_dcf.cu:278`

下一步：

- 继续检查 [fss/gpu_fss_helper.h](/home/yy404nf/FSS/剥离工作/fss/gpu_fss_helper.h) 和 [fss/gpu_dcf_sstab.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf_sstab.h) 是否需要补充中文注释
- 继续检查 [aes/gpu_aes_shm.cu](/home/yy404nf/FSS/剥离工作/aes/gpu_aes_shm.cu) 是否需要补充主流程注释
- 在补注释过程中同步观察是否还有可以进一步拆分的小 helper

### 2026-03-19 第 21 次更新

本轮完成：

- 在 [fss/gpu_fss_helper.h](/home/yy404nf/FSS/剥离工作/fss/gpu_fss_helper.h) 中补充了中文注释，说明：
  - `getVCW` 如何从打包数组中取出当前线程对应的值
  - `writeVCW` 如何按 `bout` 和 warp 聚合方式回写结果
  - `moveMasks` 如何统一分配并搬运输出 mask
- 在 [fss/gpu_dcf_sstab.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf_sstab.h) 中补充了中文注释，说明：
  - 这是 DCF 小 `bin` 路径对 SSTable 的专用 wrapper
  - `lookupSSTable` / `gpuLookupSSTable` 的执行职责
- 在 [aes/gpu_aes_shm.cu](/home/yy404nf/FSS/剥离工作/aes/gpu_aes_shm.cu) 中补充了中文注释，说明：
  - key schedule 的职责
  - 表驱动 AES 加密的用途
  - `loadSbox` 的 shared memory 复用逻辑
  - `applyAESPRG` / `applyAESPRGTwoTimes` / `applyAESPRGFourTimes` 的输出模式
  - `initAESContext` / `freeAESGlobalContext` 的资源生命周期
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前最小运行链中最关键的 helper、SSTable wrapper 和 AES PRG 实现都已经补上了主职责级中文注释
- 现在还没补到的内容大多是细粒度表达式、局部算术细节，优先级已经明显下降
- `dcf` 构建仍只保留已有的符号号变换 warning，没有新增问题

下一步：

- 继续做一轮全局扫描，确认还缺中文注释的文件是否值得继续补
- 如果剩余都是低价值的细节函数，可以开始把文档里的“中文注释覆盖范围”整理成汇总清单
- 同步观察是否有哪几个 helper 已经可以进一步合并或裁剪

### 2026-03-19 第 22 次更新

本轮完成：

- 对 `剥离工作/` 当前最小运行链做了一次全局扫描，覆盖目录：
  - `fss/`
  - `gpu/`
  - `aes/`
  - `runtime/`
- 确认当前主链中的关键文件都已经具备中文注释，覆盖范围包括：
  - facade：`dpf_api.h`、`dcf_api.h`
  - runtime：`standalone_runtime.h`
  - FSS 主入口：`gpu_dpf.h`、`gpu_dcf.h`
  - FSS 模板与主流程：`gpu_dpf_templates.h`、`gpu_dcf_templates.h`、`gpu_dpf.cu`、`gpu_dcf.cu`
  - 小 bin/SSTable 路径：`gpu_sstab.h`、`gpu_dcf_sstab.h`
  - helper：`gpu_fss_helper.h`
  - GPU 基础设施：`gpu_mem.h`、`gpu_random.h`、`gpu_random.cu`
  - AES：`gpu_aes_shm.h`、`gpu_aes_shm.cu`
- 确认当前更适合暂缓补注释的文件主要是：
  - `gpu_data_types.h`
  - `gpu_stats.h`
  - `helper_cuda.h`
  - `helper_string.h`
  - `curand_utils.h`
  - `misc_utils.h`
  - `gpu_mem.cu`
  - `gpu_aes_table.h`
  这些文件大多属于基础定义、常量表、通用 helper 或薄封装，继续补中文注释的收益相对较低

当前结论：

- 中文注释工作已经覆盖了最小运行链的主要阅读路径
- 剩余未重点注释的文件大多不在“理解 DPF/DCF 主逻辑”所必需的阅读路径上
- 后续继续补注释的优先级已经低于整理汇总和再次审视 helper 裁剪空间

下一步：

- 开始把“中文注释覆盖范围”整理成一个固定清单，写在本文档中
- 继续观察 `gpu_fss_helper.h`、`misc_utils.h`、`gpu_random.cu` 之间是否还有可继续裁剪或合并的 helper
- 如无新的高价值注释点，后续重点将从“补注释”转向“整理覆盖清单 + 再做减法”

### 2026-03-19 第 23 次更新

本轮完成：

- 在本文档中新增“中文注释覆盖清单”，把当前已经覆盖注释的主链文件固定记录下来
- 在本文档中新增“当前暂缓补充的文件”清单，明确以下文件暂不继续补中文注释：
  - `gpu/gpu_data_types.h`
  - `gpu/gpu_stats.h`
  - `gpu/helper_cuda.h`
  - `gpu/helper_string.h`
  - `gpu/curand_utils.h`
  - `gpu/misc_utils.h`
  - `gpu/gpu_mem.cu`
  - `aes/gpu_aes_table.h`
- 在本文档中新增“当前裁剪观察”，记录了：
  - `gpu_fss_helper.h` 内部仍可考虑再按职责拆分
  - `gpu_random.cu` 对 `misc_utils.h` 的主要依赖集中在 `modKernel`
  - `misc_utils.h` 当前仍是混合型工具头，后续适合优先继续拆薄

当前结论：

- 中文注释工作已经有了固定覆盖清单，后续不需要再反复人工确认“哪些文件已经补过”
- 后续继续补注释的收益已经低于重新审视 helper 拆分和头文件裁剪
- 下一阶段可以把重点重新切回 `gpu_fss_helper.h` / `misc_utils.h` / `gpu_random.cu` 这一带的减法工作

下一步：

- 优先检查能否把 `misc_utils.h` 中只服务于随机数路径的内容继续拆小
- 优先检查 `moveMasks` 是否值得从 `gpu_fss_helper.h` 中迁到更贴近 runtime/GPU 输出管理的位置
- 如开始做减法，仍需在每次变更后重新验证 `dpf/dcf` 构建

### 2026-03-19 第 24 次更新

本轮完成：

- 对 `misc_utils.h` 的实际使用点做了进一步核查，确认当前主链里：
  - `gpu_random.cu` 只依赖其中的 `modKernel`
  - FSS 主链主要依赖 `gpuMod`、`writeInt`、`writePackedOp`
- 在 [gpu/gpu_random.cu](/home/yy404nf/FSS/剥离工作/gpu/gpu_random.cu) 中新增了本地 `randomModKernel`
- 将 [gpu/gpu_random.cu](/home/yy404nf/FSS/剥离工作/gpu/gpu_random.cu) 中对 `misc_utils.h` 的依赖移除
- 同时清掉了 [gpu/gpu_random.cu](/home/yy404nf/FSS/剥离工作/gpu/gpu_random.cu) 中未使用的 `<cassert>` include
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 已经切断了一条明确的依赖链：`gpu_random.cu -> misc_utils.h`
- `misc_utils.h` 现在更清晰地收敛为 FSS/GPU 打包相关工具，而不再同时承载随机数位宽裁剪实现
- 当前剩余更值得继续观察的点是：
  - `gpuMod`
  - `writeInt`
  - `writePackedOp`
  - `moveMasks`

下一步：

- 继续评估 `misc_utils.h` 中 `writeInt` / `writePackedOp` 是否值得再拆为更小的打包工具头
- 继续评估 `moveMasks` 是否应从 `gpu_fss_helper.h` 迁到更贴近输出缓冲管理的位置
- 若下一轮继续拆分，同样需要保持 `dpf/dcf` 构建回归验证

### 2026-03-19 第 25 次更新

本轮完成：

- 新增 [gpu/packing_utils.h](/home/yy404nf/FSS/剥离工作/gpu/packing_utils.h)，把以下打包相关工具从 `misc_utils.h` 中独立出来：
  - `writeInt`
  - `writePackedOp`
- 将以下文件改为直接依赖新的 [packing_utils.h](/home/yy404nf/FSS/剥离工作/gpu/packing_utils.h)：
  - [fss/gpu_dpf.cu](/home/yy404nf/FSS/剥离工作/fss/gpu_dpf.cu)
  - [fss/gpu_dcf.cu](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf.cu)
  - [fss/gpu_sstab.h](/home/yy404nf/FSS/剥离工作/fss/gpu_sstab.h)
  - [fss/gpu_dpf_templates.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dpf_templates.h)
  - [fss/gpu_dcf_templates.h](/home/yy404nf/FSS/剥离工作/fss/gpu_dcf_templates.h)
- 更新了 [gpu/misc_utils.h](/home/yy404nf/FSS/剥离工作/gpu/misc_utils.h)：
  - 移除本地 `writeInt`
  - 移除本地 `writePackedOp`
  - 去掉不再需要的 `<cstring>` 和 `gpu_mem.h`
  - 改为包含 [packing_utils.h](/home/yy404nf/FSS/剥离工作/gpu/packing_utils.h)
- 重新验证：
  - `make -C 剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- `misc_utils.h` 已进一步瘦身，不再同时承担“位宽裁剪 + 打包写回 + 序列化写整数”三类职责
- 当前“打包输出/写 key 字节流”这条逻辑已经被收拢到 [packing_utils.h](/home/yy404nf/FSS/剥离工作/gpu/packing_utils.h)
- `dcf` 构建仍只保留已有的符号号变换 warning，行号随插入内容变为 `gpu_dcf.cu:279`

下一步：

- 继续评估 `moveMasks` 是否应从 [gpu_fss_helper.h](/home/yy404nf/FSS/剥离工作/fss/gpu_fss_helper.h) 迁到更贴近输出缓冲管理的位置
- 继续评估 `misc_utils.h` 中剩余的 `gpuMod` / `modKernel` / `getPackedSharesKernel` 是否还值得再拆
- 如果继续做减法，保持 `dpf/dcf` 构建回归验证不变

### 2026-03-19 第 26 次更新

本轮完成：

- 根据当前最小构建链，整理并确认了 `dpf` 和 `dcf` 的直接构建/运行命令
- 将这次对话作为“运行方式说明”记录到本文档，便于后续直接复用

当前结论：

- 当前运行 `dpf/dcf` 只需要 CUDA 编译环境和 `剥离工作/` 目录内源码
- 推荐直接在仓库根目录通过 `make -C 剥离工作 ...` 构建，再执行对应二进制

下一步：

- 如继续做剥离，回到 `moveMasks` 和 `misc_utils.h` 剩余内容的减法工作

### 2026-03-19 第 27 次更新

本轮完成：

- 在根目录新增两个批量参数测试入口：
  - [dpf_batch.cu](/home/yy404nf/FSS/剥离工作/dpf_batch.cu)
  - [dcf_batch.cu](/home/yy404nf/FSS/剥离工作/dcf_batch.cu)
- 更新了 [Makefile](/home/yy404nf/FSS/剥离工作/Makefile)，新增目标：
  - `dpf_batch`
  - `dcf_batch`
  - `all` 现在会一并构建这两个批量测试程序
- 更新了 [README.md](/home/yy404nf/FSS/剥离工作/README.md)，加入批量测试入口及运行方式说明
- 将 [gpu/gpu_mem.cu](/home/yy404nf/FSS/剥离工作/gpu/gpu_mem.cu) 的内存池预留策略调整为：
  - 默认使用 `min(4 GiB, 显存一半)`
  - 可通过环境变量 `GPU_MEM_POOL_BYTES` 覆盖
  这样避免当前机器上固定预留 `40 GiB` 导致的启动即 OOM
- 重新验证：
  - `make -C 剥离工作 dpf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C 剥离工作 dcf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - `./剥离工作/dpf_batch`
  - `./剥离工作/dcf_batch`
  - 均可完成运行

当前结论：

- 当前新增的 `dpf_batch` / `dcf_batch` 更适合作为“批量参数烟雾测试”，用于扫描多组 `bin` / `bout` / `n` 组合是否能跑通
- 由于当前 facade 主链和原项目底层测试语义并不完全一致，这两个批量入口目前输出结果摘要，不做强语义断言
- 当前批量烟雾测试已经可以稳定执行并打印每组参数的 sample 输出

下一步：

- 如果后续需要更强的批量回归，可以再对照原项目测试语义补“逐 case 断言版”批量测试
- 如继续做剥离，回到 `moveMasks` 和 `misc_utils.h` 剩余内容的减法工作


### 2026-03-19 第 28 次更新

本轮完成：

- 确认 `剥离工作/` 已从原位置迁移到：
  - `/home/yy404nf/FSS/剥离工作`
- 在不移动文件夹的前提下，完成了新位置下的环境检查：
  - [Makefile](/home/yy404nf/FSS/剥离工作/Makefile) 仍可直接在新位置构建
  - `$(CURDIR)` 相关相对路径策略仍然有效
  - 新位置下 `dpf` / `dcf` 重新编译通过
- 修正了 [工作进度.md](/home/yy404nf/FSS/剥离工作/工作进度.md) 中残留的旧绝对路径引用：
  - 从 `/home/yy404nf/GPU-MPC/剥离工作/...`
  - 改为 `/home/yy404nf/FSS/剥离工作/...`
- 验证了新位置下可以直接拉起以下程序：
  - `/home/yy404nf/FSS/剥离工作/dpf`
  - `/home/yy404nf/FSS/剥离工作/dcf`
  - `/home/yy404nf/FSS/剥离工作/dpf_batch`
  - `/home/yy404nf/FSS/剥离工作/dcf_batch`

当前结论：

- 这次“环境坏了”的核心原因确实是目录位置变更后，旧会话仍在引用原来的 `GPU-MPC/剥离工作` 路径
- 当前构建环境已经修复到适配新位置，不需要把 `剥离工作/` 再移回 `GPU-MPC/`
- 当前剩余现象不是路径引用错误，而是程序运行时在 GPU 计算阶段耗时较长；从日志看已经成功越过初始化并打印了 `reserved memory`

下一步：

- 如果后续需要继续验证运行结果，统一使用新路径：
  - `make -C /home/yy404nf/FSS/剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dpf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dcf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
- 如继续做剥离，后续重点重新回到 helper 裁剪和批量测试语义增强


### 2026-03-19 第 30 次更新

本轮完成：

- 将 [dpf_batch.cu](/home/yy404nf/FSS/剥离工作/dpf_batch.cu) 改为运行时参数模式：
  - 用法：`dpf_batch <bin> <n>`
- 将 [dcf_batch.cu](/home/yy404nf/FSS/剥离工作/dcf_batch.cu) 改为运行时参数模式：
  - 用法：`dcf_batch <bin> <bout> <n>`
- 为两个批量入口补充了参数校验和 usage 提示
- 更新了 [README.md](/home/yy404nf/FSS/剥离工作/README.md)，同步说明新的运行方式示例：
  - `./剥离工作/dpf_batch 64 10000000`
  - `./剥离工作/dcf_batch 64 1 10000000`
- 重新验证：
  - `make -C /home/yy404nf/FSS/剥离工作 dpf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dcf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 批量测试入口已经不再写死内置 case，而是改成了你要求的运行时输入形式
- 现在可以直接通过命令行指定大规模参数，例如 `64 1 10000000`
- `dcf_batch` 编译仍保留已有的符号号变换 warning，但不影响当前构建通过

下一步：

- 如果你要，我可以继续把 `dpf` / `dcf` 单例入口也改成同样的命令行参数风格
- 或者继续回到 helper 裁剪和批量测试输出格式优化


### 2026-03-19 第 31 次更新

本轮完成：

- 明确了当前 `dpf_batch` / `dcf_batch` 输出中的 `sample` 字段含义
- 确认当前批量入口还没有输出耗时字段，这属于当前测试入口信息不足，不是底层库缺失

当前结论：

- `sample` 只是为了快速确认程序确实跑出了结果，打印的是前 4 个位置的合并输出摘要
- 当前批量入口还没有统计 keygen / eval / 传输时间
- 如果后续需要做性能测试，需要在批量入口里显式补充计时逻辑和输出字段

下一步：

- 可继续为 `dpf_batch` / `dcf_batch` 增加耗时字段，例如 keygen 时间、P0 eval 时间、P1 eval 时间、总时间、transfer 时间


### 2026-03-19 第 32 次更新

本轮完成：

- 按最新要求移除了 `dpf_batch` / `dcf_batch` 输出中的 `sample` 字段
- 将 [dpf_batch.cu](/home/yy404nf/FSS/剥离工作/dpf_batch.cu) 补成完整性能字段输出，当前输出包括：
  - `bin`
  - `n`
  - `keygen_us`
  - `eval_p0_us`
  - `eval_p1_us`
  - `transfer_p0_us`
  - `transfer_p1_us`
  - `total_us`
- 将 [dcf_batch.cu](/home/yy404nf/FSS/剥离工作/dcf_batch.cu) 补成完整性能字段输出，当前输出包括：
  - `bin`
  - `bout`
  - `n`
  - `keygen_us`
  - `eval_p0_us`
  - `eval_p1_us`
  - `transfer_p0_us`
  - `transfer_p1_us`
  - `total_us`
- 更新了 [README.md](/home/yy404nf/FSS/剥离工作/README.md)，说明批量入口现在输出耗时字段
- 重新验证：
  - `make -C /home/yy404nf/FSS/剥离工作 dpf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dcf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 批量入口现在不再输出临时 `sample` 摘要，而是直接输出你要的性能字段
- 其中 `transfer_*_us` 来自 `Stats.transfer_time`，`keygen/eval/total` 来自入口层显式计时
- `dcf_batch` 编译仍保留已有的符号号变换 warning，但不影响当前构建通过

下一步：

- 如果你需要，我可以继续把 `dpf` / `dcf` 单例入口也统一成相同的性能字段输出风格


### 2026-03-19 第 33 次更新

本轮完成：

- 在完善 `dpf_batch` / `dcf_batch` 耗时字段后，继续为之前没有中文注释的文件补充了职责级中文注释，包括：
  - [dpf_batch.cu](/home/yy404nf/FSS/剥离工作/dpf_batch.cu)
  - [dcf_batch.cu](/home/yy404nf/FSS/剥离工作/dcf_batch.cu)
  - [gpu/gpu_data_types.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_data_types.h)
  - [gpu/gpu_stats.h](/home/yy404nf/FSS/剥离工作/gpu/gpu_stats.h)
  - [gpu/gpu_mem.cu](/home/yy404nf/FSS/剥离工作/gpu/gpu_mem.cu)
  - [gpu/misc_utils.h](/home/yy404nf/FSS/剥离工作/gpu/misc_utils.h)
  - [gpu/packing_utils.h](/home/yy404nf/FSS/剥离工作/gpu/packing_utils.h)
  - [gpu/helper_cuda.h](/home/yy404nf/FSS/剥离工作/gpu/helper_cuda.h)
  - [gpu/helper_string.h](/home/yy404nf/FSS/剥离工作/gpu/helper_string.h)
  - [gpu/curand_utils.h](/home/yy404nf/FSS/剥离工作/gpu/curand_utils.h)
  - [aes/gpu_aes_table.h](/home/yy404nf/FSS/剥离工作/aes/gpu_aes_table.h)
- 做了一次全局扫描，确认当前 `fss/`、`gpu/`、`aes/`、`runtime/` 以及两个 batch 入口中，不再存在“完全没有中文注释”的 `.h/.cu` 文件
- 重新验证：
  - `make -C /home/yy404nf/FSS/剥离工作 dpf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dcf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前最小运行链中你实际会接触到的源码/头文件，已经都具备至少一层中文职责说明
- 后续如果再补注释，重点将不再是“有没有中文注释”，而是“是否还需要把某些实现细节讲得更细”
- `dcf_batch` 编译仍保留已有的符号号变换 warning，但不影响当前构建通过

下一步：

- 如继续做剥离，可重新把重点切回 helper 裁剪和头文件瘦身
- 如继续做测试体验优化，可考虑把 `dpf` / `dcf` 单例入口也统一成和 batch 一样的性能字段输出风格


### 2026-03-20 第 1 次更新

本轮完成：

- 按“为剥离工作的 dpf/dcf 增加完整中文注释”的要求，进一步补充了入口级完整注释，重点覆盖：
  - [dpf.cu](/home/yy404nf/FSS/剥离工作/dpf.cu)
  - [dcf.cu](/home/yy404nf/FSS/剥离工作/dcf.cu)
  - [dpf_batch.cu](/home/yy404nf/FSS/剥离工作/dpf_batch.cu)
  - [dcf_batch.cu](/home/yy404nf/FSS/剥离工作/dcf_batch.cu)
- 这轮新增的注释不再只是“文件职责级”说明，而是补到了：
  - 输入数据的语义
  - 两方 keygen / eval 的流程解释
  - share 合并后的结果语义
  - batch 入口里各耗时字段的含义
  - usage/参数校验逻辑的作用
- 重新验证：
  - `make -C /home/yy404nf/FSS/剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dpf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dcf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 当前 `剥离工作` 中的 DPF/DCF 入口层已经具备完整中文注释，直接阅读入口文件就能理解“输入是什么、流程怎么走、输出字段是什么意思”
- 之前已经完成的底层注释和本轮入口级注释现在形成了完整链路
- `dcf` / `dcf_batch` 编译仍保留已有的符号号变换 warning，但不影响当前构建通过

下一步：

- 如果后续还要继续补注释，重点可以转向“把单例入口也补成和 batch 一样的性能字段输出风格”
- 如果继续做剥离，则重新回到 helper 裁剪和头文件瘦身


### 2026-03-20 第 2 次更新

本轮完成：

- 将 [dpf_batch.cu](/home/yy404nf/FSS/剥离工作/dpf_batch.cu) 的性能字段输出改为逐行显示
- 将 [dcf_batch.cu](/home/yy404nf/FSS/剥离工作/dcf_batch.cu) 的性能字段输出改为逐行显示
- 为输出字段直接补上单位，当前格式包括：
  - `bin: ... bit`
  - `bout: ... bit`
  - `n: ... elem`
  - `keygen: ... us`
  - `eval_p0: ... us`
  - `eval_p1: ... us`
  - `transfer_p0: ... us`
  - `transfer_p1: ... us`
  - `total: ... us`
- 重新验证：
  - `make -C /home/yy404nf/FSS/剥离工作 dpf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dcf_batch CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过

当前结论：

- 批量测试输出现在更适合人工阅读，字段不再挤在一行里
- 单位已经直接体现在输出文本里，不需要再靠字段名猜测
- `dcf_batch` 编译仍保留已有的符号号变换 warning，但不影响当前构建通过

下一步：

- 如果你需要，我可以继续把 `dpf` / `dcf` 单例入口的输出也统一成同样的多行字段风格


### 2026-03-20 第 3 次更新

本轮完成：

- 专门核实了当前 [dpf](/home/yy404nf/FSS/剥离工作/dpf.cu) / [dcf](/home/yy404nf/FSS/剥离工作/dcf.cu) 是否还能脱离 `GPU-MPC` 目录独立工作
- 重新验证构建：
  - `make -C /home/yy404nf/FSS/剥离工作 dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS/剥离工作 dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过；`dcf` 仍只有已有的符号变换 warning
- 检查 [Makefile](/home/yy404nf/FSS/剥离工作/Makefile) 后确认：当前编译只依赖 `剥离工作/` 本地源码和 CUDA 基础库 `-lcuda -lcudart -lcurand`
- 扫描本地 include 关系后确认：当前源码内部只在 `剥离工作/` 内部的 `fss/`、`gpu/`、`aes/`、`runtime/` 之间互相引用，不再从 `GPU-MPC` 根目录取源码或头文件
- 发现两类残留但不影响当前独立运行：
  - [工作进度.md](/home/yy404nf/FSS/剥离工作/工作进度.md) 中还保留了对旧 `GPU-MPC` 路径的历史记录文字
  - [helper_string.h](/home/yy404nf/FSS/剥离工作/gpu/helper_string.h) 里仍有 NVIDIA sample 风格的相对路径查找字符串，这部分是通用辅助代码残留，不是当前 DPF/DCF 主链依赖

当前结论：

- `dpf` / `dcf` 现在已经可以脱离 `GPU-MPC` 目录独立工作
- 更准确地说，当前它们已经不依赖 `GPU-MPC` 仓库内的其他源码、头文件或本地库产物
- 仍然需要的外部环境只有 CUDA 工具链和运行时环境：
  - `nvcc`
  - CUDA driver/runtime
  - `libcuda` / `libcudart` / `libcurand`
- 所以它们已经实现了“脱离 `GPU-MPC`”，但还不是“零外部依赖的纯 C++ 程序”

下一步：

- 如果你要把结论做得更彻底，可以再清理 [helper_string.h](/home/yy404nf/FSS/剥离工作/gpu/helper_string.h) 里的历史 sample 路径字符串
- 也可以把 [工作进度.md](/home/yy404nf/FSS/剥离工作/工作进度.md) 里的旧 `GPU-MPC` 路径描述改成“历史记录”措辞，避免后续阅读时混淆


### 2026-03-20 第 4 次更新

本轮完成：

- 按要求将原 `/home/yy404nf/FSS/剥离工作/` 下的全部文件和子目录整体上移到当前工作区根目录 `/home/yy404nf/FSS/`
- 当前根目录直接包含：`Makefile`、`README.md`、`工作进度.md`、`dpf.cu`、`dcf.cu`、`dpf_batch.cu`、`dcf_batch.cu`、`fss/`、`gpu/`、`aes/`、`runtime/` 以及已生成的可执行文件
- 旧目录 `/home/yy404nf/FSS/剥离工作/` 已删除，不再保留一层额外包装目录
- 重新验证构建路径切换后的根目录编译：
  - `make -C /home/yy404nf/FSS dpf CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS dcf CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过；`dcf` 仍只有已有的符号变换 warning

当前结论：

- 当前最小可运行集合已经直接落在 `/home/yy404nf/FSS/` 根目录
- 后续使用、编译、运行都不再需要经过 `剥离工作/` 子目录
- `Makefile` 仍然可以正常工作，因为它基于 `CURDIR` 组织 include 和编译输入，目录上移后无需额外修改

下一步：

- 后续如果继续整理，可以逐步把文档中的旧 `/home/yy404nf/FSS/剥离工作/...` 绝对路径引用更新为新的根目录路径，避免阅读混淆


### 2026-03-20 第 5 次更新

本轮完成：

- 开始将当前 `/home/yy404nf/FSS/` 根目录整理为独立 Git 仓库并准备推送到 GitHub
- 确认当前目录原本不是 Git 仓库，已在根目录执行 `git init`
- 为当前仓库设置本地提交身份：`YY404NF <YY404NF@users.noreply.github.com>`
- 当前计划将根目录下的全部文件一并提交，包括源码、文档、目录结构以及当前生成的可执行文件

当前结论：

- `/home/yy404nf/FSS/` 现在已经具备独立仓库初始化条件
- 下一步将执行首个提交、绑定远端 `git@github.com:YY404NF/FSS.git` 并推送到 `main`


### 2026-03-20 第 6 次更新

本轮完成：

- 按要求改用 HTTP 远端地址，而不是 SSH
- 已完成首个 Git 提交：`Initial commit`
- 当前远端已设置为：`https://github.com/YY404NF/FSS.git`
- 当前分支已整理为 `main`
- 下一步将从本地 `main` 推送到 GitHub 远端 `origin/main`

当前结论：

- 仓库初始化和本地提交已经完成
- 还差最后一步远端推送；若本机已具备 GitHub HTTP 凭据，则可直接完成


### 2026-03-20 第 7 次更新

本轮完成：

- 已尝试通过 HTTP 执行 `git push -u origin main`
- 推送失败，错误为：`fatal: could not read Username for 'https://github.com': No such device or address`
- 已额外检查当前环境：
  - 未安装 `gh`
  - 未配置 Git `credential.helper`
  - 未发现可直接用于 GitHub 推送的 HTTP 凭据环境变量

当前结论：

- 本地仓库初始化、分支整理、远端配置和本地提交都已完成
- 当前唯一未完成步骤是 GitHub HTTP 鉴权，因此还没有真正推送到远端
- 只要本机补上 GitHub 用户名 + PAT，或配置好 credential helper，就可以直接继续执行：
  - `git push -u origin main`


### 2026-03-20 第 8 次更新

本轮完成：

- 更新了 [README.md](/home/yy404nf/FSS/README.md)，使其匹配当前仓库已经上移到根目录后的实际结构
- 将构建和运行命令从旧的 `剥离工作/` 路径写法改为当前根目录直接执行的写法
- 将 CUDA 示例环境更新为当前实际使用的 `CUDA_VERSION=13.1`
- 在 README 中补充了 `dpf_batch` / `dcf_batch` 的输出字段说明和单位说明
- 准备将 README 变更与当前工作区里的其余改动一并提交

当前结论：

- README 现在已经能正确反映当前仓库布局、构建命令和批量测试输出格式
- 下一步将提交当前工作区内全部已修改文件


### 2026-03-20 第 9 次更新

本轮完成：

- 按最新命名要求，将批量测试入口统一重命名为 benchmark：
  - `dpf_batch.cu` -> `dpf_benchmark.cu`
  - `dcf_batch.cu` -> `dcf_benchmark.cu`
- 同步更新了 [Makefile](/home/yy404nf/FSS/Makefile) 中的构建目标：
  - `dpf_benchmark`
  - `dcf_benchmark`
- 同步更新了 [README.md](/home/yy404nf/FSS/README.md) 中的文件说明、构建命令和输出字段说明
- 将程序运行结束提示从 `batch finished` 改为 `benchmark finished`
- 重新验证构建：
  - `make -C /home/yy404nf/FSS dpf_benchmark CUDA_VERSION=13.1 GPU_ARCH=86`
  - `make -C /home/yy404nf/FSS dcf_benchmark CUDA_VERSION=13.1 GPU_ARCH=86`
  - 均通过；`dcf_benchmark` 仍只有已有的符号变换 warning

当前结论：

- 当前主用名称已经统一为 `dpf_benchmark` / `dcf_benchmark`
- `Makefile` 的 `clean` 仍保留旧 `dpf_batch` / `dcf_batch` 二进制名的清理项，仅用于兼容清除历史产物


### 2026-03-20 第 10 次更新

本轮完成：

- 检查了用户手动修改后的 [README.md](/home/yy404nf/FSS/README.md)
- 本轮以 review 方式给出问题和改进建议，不主动覆盖用户文案
- 额外确认到原 [工作进度.md](/home/yy404nf/FSS/工作进度.md) 已被替换为 [work.md](/home/yy404nf/FSS/work.md)，因此本轮记录继续写入 `work.md`

当前结论：

- README 的主方向是对的，内容比之前更短更聚焦
- 但仍有几处会影响可用性或一致性的细节，需要在后续修订中处理


### 2026-03-20 第 11 次更新

本轮完成：

- 准备将当前工作区内全部未提交修改统一提交
- 待提交内容包括：
  - `batch` -> `benchmark` 命名切换后的相关改动
  - [README.md](/home/yy404nf/FSS/README.md) 的用户自定义修订
  - [Makefile](/home/yy404nf/FSS/Makefile) 的目标名同步更新
  - 新增 [.gitignore](/home/yy404nf/FSS/.gitignore)
  - 旧 [工作进度.md](/home/yy404nf/FSS/work.md) 删除，并继续使用 [work.md](/home/yy404nf/FSS/work.md)

当前结论：

- 当前工作区中的改动已经整理完成，可以进行一次统一提交
