# aicb/core 模块设计与使用总结

## 目录功能概述

`core` 是 AICB 的核心工具模块，主要提供 **Grouped GEMM**（分组通用矩阵乘法）的可用性检测与封装功能。该模块作为 MoE（Mixture of Experts）计算的底层依赖支撑。

## 目录结构

```
core/
├── __init__.py                 # 空文件，标记为 Python 包
└── grouped_gemm_util.py        # 分组 GEMM 工具封装（20 行）
```

## 文件详细说明

### `__init__.py`
- 空文件，仅用于将 `core` 目录标记为 Python 包。

### `grouped_gemm_util.py`
- **来源**：NVIDIA（版权声明 2023）
- **功能**：封装第三方 `grouped_gemm` 库，提供可用性检查接口。

#### 核心函数

| 函数名 | 功能 |
|--------|------|
| `grouped_gemm_is_available()` | 检查 `grouped_gemm` 库是否已安装，返回布尔值 |
| `assert_grouped_gemm_is_available()` | 断言 `grouped_gemm` 可用，不可用时抛出 `AssertionError` 并提示安装命令 |

#### 模块级变量

- `ops`：若 `grouped_gemm` 可用，则指向 `grouped_gemm.ops`（提供分组矩阵乘法算子）；否则为 `None`。

## 依赖关系

- **外部依赖**：`grouped_gemm@v1.0`（通过 `pip install git+https://github.com/fanshiqing/grouped_gemm@v1.0` 安装），需要 NVIDIA Hopper 架构 GPU（SM90）或更新架构
- **被调用方**：`workload_generator/mocked_model/training/AiobMegatron.py` 中的 `GroupedMLP` 类

## 使用方式

```python
from core import grouped_gemm_util as gg

# 检查可用性
if gg.grouped_gemm_is_available():
    # 使用 ops 执行分组矩阵乘法（MoE 的 FC1/FC2 线性变换）
    result = gg.ops.gmm(permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False)

# 或强制断言
gg.assert_grouped_gemm_is_available()
```

## 设计说明

采用 try-except 模式进行可选依赖导入，使得在未安装 `grouped_gemm` 的环境下程序不会直接崩溃，而是在实际需要使用时通过 assert 给出明确错误提示。这是一种常见的可选依赖管理模式，为高性能分组 GEMM 提供抽象层。
