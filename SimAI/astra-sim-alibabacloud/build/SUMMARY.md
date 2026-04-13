# astra-sim-alibabacloud/build 模块设计与使用总结

## 目录功能概述

`build` 是 astra-sim 的**构建配置目录**，包含三种网络后端模式的 CMake 配置和构建脚本。

## 目录结构

```
build/
├── astra_ns3/                    # NS-3 网络后端构建
│   ├── CMakeLists.txt            # CMake 配置
│   └── build.sh                  # 构建脚本
├── simai_analytical/             # 解析网络模型构建
│   ├── CMakeLists.txt
│   └── build.sh
└── simai_phy/                    # 物理网络后端构建
    ├── CMakeLists.txt
    └── build.sh
```

## 三种构建模式

| 模式 | 说明 | 网络后端 |
|------|------|---------|
| `ns3` | NS-3 网络仿真 | 包级网络仿真，精确模拟 RDMA/拥塞控制 |
| `analytical` | 解析模型 | 基于数学模型的快速估算，无需 NS-3 |
| `phy` | 物理网络 | 真实 RDMA 网络，需要 MPI |

## 构建流程

### NS-3 模式（`astra_ns3/build.sh`）
1. `compile_astrasim` — 编译 AstraSim 库
2. 将 `network_frontend/ns3` 代码复制到 NS-3 的 scratch 目录
3. 配置 NS-3（启用 MTP 多线程）并编译

### 构建命令

```bash
# 编译
./build.sh -c ns3          # NS-3 模式
./build.sh -c phy          # 物理网络模式
./build.sh -c analytical   # 解析模型模式

# 清理
./build.sh -l ns3          # 清理构建产物
./build.sh -lr ns3         # 清理构建产物和结果
```

## CMakeLists.txt 说明

主 CMakeLists.txt 编译 AstraSim 库，根据模式定义不同宏：
- `PHY_RDMA` + `PHY_MTP`：物理 RDMA 模式
- `ANALYTI`：解析模型模式（排除 RDMA 和多线程相关文件）

输出目录约定：`/etc/astra-sim/` 下存放输入配置、仿真文件和结果。
