# ns-3-alibabacloud/simulation 模块设计与使用总结

## 目录功能概述

`simulation` 是基于 NS-3 (Network Simulator 3) 深度定制的**RDMA 数据中心网络包级仿真器**，在标准 NS-3 框架之上扩展了完整的 RDMA 传输、多种拥塞控制算法（DCQCN、HPCC、TIMELY、DCTCP、HPCC-PINT）、PFC 流控、ECN 标记、NVSwitch 节点，以及 MTP（Multi-Threaded Parallel）多线程仿真加速能力。

## 目录结构

```
simulation/
├── CMakeLists.txt              # NS-3 CMake 主构建配置
├── VERSION                     # NS-3 版本号
├── ns3/                        # NS-3 入口脚本（ns3 CLI）
├── src/                        # 核心源代码模块
│   ├── point-to-point/model/   # ★ RDMA 网络核心实现（35 个源文件）
│   ├── network/utils/          # 自定义数据包头和 INT 头
│   ├── applications/model/     # RDMA 客户端应用
│   ├── internet/model/         # RDMA 宏定义
│   ├── mtp/model/              # 多线程并行仿真（MTP）
│   ├── core/                   # NS-3 核心框架
│   └── ...                     # 其他标准 NS-3 模块
├── scratch/                    # 用户仿真脚本入口
├── contrib/                    # 扩展模块（空）
├── examples/                   # 示例程序
├── build-support/              # 构建辅助脚本
└── doc/                        # 文档
```

## 核心自定义模块详细说明

### `src/point-to-point/model/` — RDMA 网络核心

这是 SimAI 网络仿真的**最核心模块**，包含 35 个 C++ 源文件，实现了完整的 RDMA over Converged Ethernet (RoCE) 协议栈。

#### `RdmaHw` — RDMA 硬件抽象层
- 管理 RDMA 网卡（NIC）接口和 Queue Pair 映射
- 处理数据包收发（`ReceiveUdp`、`ReceiveAck`、`ReceiveCnp`）
- 实现 **5 种拥塞控制算法**：
  - **DCQCN (Mellanox)**：基于 CNP 的速率控制，alpha 更新、速率递减/递增、快速恢复
  - **HPCC (High Precision)**：基于 INT（In-Network Telemetry）的高精度拥塞控制
  - **TIMELY**：基于 RTT 梯度的速率调整
  - **DCTCP**：基于 ECN 标记比例的窗口调整
  - **HPCC-PINT**：HPCC 的概率 INT 采样版本，降低头部开销
- 支持 **PCIe 暂停/恢复** 机制
- 支持 **NVLS（NVLink Switch）** 路由：同服务器内 GPU 通过 NVSwitch 通信
- 提供带宽、QP 速率、CNP 计数的监控输出

#### `RdmaQueuePair` — RDMA 队列对
- 模拟 RDMA Queue Pair（QP）的发送/接收状态
- 维护序列号（`snd_nxt`、`snd_una`）、窗口大小、速率
- 为每种 CC 算法维护独立运行时状态：`mlx`（DCQCN）、`hp`（HPCC）、`tmly`（TIMELY）、`dctcp`
- 支持 NVLS 模式标记

#### `SwitchNode` — 数据中心交换机
- 继承自 `Node`，实现 L2/L3 交换机逻辑
- 支持 **ECMP（Equal-Cost Multi-Path）** 路由
- **PFC（Priority Flow Control）** 发送和处理
- 端口队列长度和带宽监控（`PrintSwitchQlen`、`PrintSwitchBw`）
- 每端口最多 1025 端口、8 优先级队列

#### `NVSwitchNode` — NVSwitch 节点
- 模拟 NVIDIA NVSwitch，用于服务器内 GPU 间通信
- 与 `SwitchNode` 类似的 ECMP 路由和包转发
- 不实现 PFC/ECN（NVLink 域无需拥塞控制）

#### `SwitchMmu` — 交换机内存管理单元
- 入口/出口准入控制（`CheckIngressAdmission` / `CheckEgressAdmission`）
- PFC 暂停/恢复阈值计算
- ECN 标记判断（`ShouldSendCN`）
- 共享缓冲区和 headroom 管理
- 可配置 ECN 参数：kmin、kmax、pmax（按端口）

#### `QbbNetDevice` — 802.1Qbb 网络设备
- 继承自 `PointToPointNetDevice`，实现 **Priority-based Flow Control (PFC)**
- `RdmaEgressQueue` 管理 8 优先级出口队列
- 支持 ACK 高优先级队列和 Round-Robin 调度

#### 其他重要文件
| 文件 | 说明 |
|------|------|
| `cn-header.h/cc` | 拥塞通知（CNP）报文头定义 |
| `pause-header.h/cc` | PFC 暂停帧头定义 |
| `pint.h/cc` | PINT（Probabilistic INT）编解码 |
| `qbb-channel.h/cc` | QBB 通道（支持 PFC 的链路） |
| `rdma-driver.h/cc` | RDMA 驱动，管理多 NIC |
| `trace-format.h` | 仿真 Trace 输出格式定义 |

### `src/network/utils/` — 自定义网络工具

#### `IntHeader` — In-Network Telemetry 头
- 最多记录 **5 跳** 交换机遥测数据
- 每跳记录：链路速率（25G/50G/100G/200G/400G）、时间戳、字节计数、队列长度
- 支持三种模式：`NORMAL`（完整 INT）、`TS`（仅时间戳）、`PINT`（概率采样）

#### `CustomHeader` — 统一数据包头
- 合并 L2（PPP）、L3（IPv4）、L4（UDP/TCP/ACK/NACK/PFC/QCN）头解析
- 支持 ECN 字段：NotECT、ECT0、ECT1、CE

#### `BroadcomEgressQueue` — Memory Management
- Memory Management 出口队列模型

### `src/applications/model/` — RDMA 应用

#### `RdmaClient` — RDMA 客户端
- 继承自 `Application`，发起 RDMA 传输
- 配置参数：源/目的 IP 和端口、PG（优先级组）、传输大小、窗口大小、base RTT
- 支持完成回调和发送回调

### `src/mtp/` — 多线程并行仿真

#### `MtpInterface` — MTP 接口
- 提供 NS-3 仿真的**多线程并行执行**能力
- 自动/手动拓扑分区
- 基于原子操作的临界区（`CriticalSection`）
- 可配置线程数和 Logical Process 数量

#### `MultithreadedSimulatorImpl` — 多线程仿真器实现
- NS-3 `SimulatorImpl` 的多线程版本

#### `LogicalProcess` — 逻辑进程
- PDES（Parallel Discrete Event Simulation）的逻辑进程抽象

## 标准 NS-3 模块

simulation 包含完整的 NS-3 标准模块集：

| 模块 | 说明 |
|------|------|
| `core` | 事件调度器、随机变量、日志系统 |
| `network` | 数据包、节点、通道基础设施 |
| `internet` | IPv4/IPv6、TCP/UDP 协议栈 |
| `traffic-control` | 流量控制（队列规则） |
| `flow-monitor` | 流监控统计 |
| `mobility` | 节点移动模型 |
| `stats` | 统计工具 |
| `bridge` / `csma` | 以太网桥接和 CSMA |
| 其他 | wifi, lte, mesh, wave 等（本项目未使用） |

## 构建方式

```bash
# 通过 CMake 构建
cmake -B build -DNS3_ENABLED_MODULES="point-to-point;network;internet;applications;mtp"
cmake --build build

# 通常通过 astra-sim 的 build.sh 自动构建
./build.sh -c ns3
```

## 依赖关系

- **构建依赖**：CMake 3.10+、C++17、ccache（可选）
- **上游调用方**：astra-sim 的 `network_frontend/ns3` 通过 scratch 脚本与此仿真器交互
- **输出数据**：FCT 文件、带宽/队列长度/QP 速率/CNP 监控文件 → 供 `analysis/` 工具分析

## 在项目中的角色

作为 SimAI 的**包级网络仿真后端**，精确模拟 RDMA 数据中心网络的传输行为、拥塞控制和流控机制，为分布式 AI 训练的网络性能评估提供高精度仿真能力。支持 NVSwitch 服务器内通信和 MTP 多线程加速，适配阿里巴巴 HPN 网络架构。
