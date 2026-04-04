# `Orca/README.md` 论文参数对照

## 1. README 里写明的论文口径

来源：

- [README.md](/root/gpu-work/FSS/Orca/README.md)

README 当前明确给出的论文环境和参数是：

- OS: `Ubuntu 20.04`
- GPU: `NVIDIA RTX A6000`
- CUDA: `11.7`
- CMake: `3.27.2`
- host compiler: `g++-9`
- primitive: README 当前说明这张表对应 `DCF`
- 参数：
  - `bin = 64`
  - `bout = 1`
  - `n = 10,000,000`

README 里的论文结果表：

| 模式 | Naive | AES | AES+LAYOUT | AES+LAYOUT+MEM |
| --- | ---: | ---: | ---: | ---: |
| Time (ms) | 3305 | 840 | 716 | 523 |

## 2. 和我们当前两类实验的关系

### 2.1 和横向比较实验的关系

来源：

- [三个实现横向比较.md](/root/gpu-work/260404三个实现横向比较/三个实现横向比较.md)

当前横向比较实验用的是：

- `bin = 20`
- `bout = 1`
- `n = 131072`

所以这份横向比较和 README 论文表的关系是：

- primitive 体系相关，但参数没有对齐
- 它主要回答的是：
  - 在相同机器上，`myl7 GPU/CPU` 和 `Orca/libfss` 的项目级优化倍数怎么比
- 它不适合直接拿来对 README 那张论文表做复现对照

### 2.2 和复用密钥实验的关系

来源：

- [复用密钥.md](/root/gpu-work/260404复用密钥/复用密钥.md)

当前复用密钥实验用的是：

- `DPF`: `bin = 64`
- `DCF`: `bin = 64, bout = 1`
- `N` 梯度覆盖到 `10,000,000`

所以这份实验和 README 论文表的关系是：

- 参数点已经覆盖到论文参数点
  - `bin = 64`
  - `bout = 1`
  - `n = 10,000,000`
- 但实验目标已经变了
  - README 是原始 `Orca` 多档实现的论文结果
  - 复用密钥实验测的是 `device-side key cache` 之后的 steady-state `eval`

因此：

- 可以说我们已经覆盖了 README 关心的那组大参数
- 但不能把复用密钥实验结果直接当成 README 表格的复现值

## 3. 目前最稳妥的解读

现在可以明确说的只有三点：

1. README 论文表对应的大参数点是 `DCF, bin=64, bout=1, n=10^7`。
2. 我们的横向比较实验没有对齐这组参数，所以不要和 README 表格直接混读。
3. 我们的复用密钥实验已经覆盖到这组参数，但它测的是另一件事：
   `GPU 常驻 key` 下，多轮评估能把 steady-state `eval` 压到什么程度。

如果后面要做真正严格的 README 论文复现，应该补一组单独实验，要求同时满足：

- primitive 对齐到 `DCF`
- 参数对齐到 `bin=64, bout=1, n=10^7`
- 环境尽量贴近 README：
  - 原始 `Orca` 路径
  - 非复用 key
  - 尽量单独对应 README 里的 `Naive / AES / AES+LAYOUT / AES+LAYOUT+MEM`
