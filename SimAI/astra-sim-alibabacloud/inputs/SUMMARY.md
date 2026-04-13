# astra-sim-alibabacloud/inputs 模块设计与使用总结

## 目录功能概述

`inputs` 是 astra-sim 的**仿真输入配置目录**，包含网络仿真配置文件、通信性能比率表和拓扑生成工具。

## 目录结构

```
inputs/
├── config/
│   └── SimAI.conf           # NS-3 网络仿真主配置文件
├── ratio/                   # 通信性能比率表
│   ├── nic_ratio.csv        # NIC 带宽利用率表
│   ├── nvlink_ratio.csv     # NVLink 带宽利用率表
│   └── ata_ratio.csv        # All-to-All 比率表
└── topo/
    └── gen_Topo_Template.py # 拓扑生成脚本
```

## 文件详细说明

### `config/SimAI.conf` — NS-3 仿真配置

RDMA 网络仿真的核心配置文件，包含：

| 参数类别 | 关键参数 |
|---------|---------|
| QCN 拥塞控制 | ENABLE_QCN, CC_MODE, RATE_AI, RATE_HAI, MIN_RATE |
| PFC 流控 | USE_DYNAMIC_PFC_THRESHOLD, BUFFER_SIZE |
| 数据包 | PACKET_PAYLOAD_SIZE (9000 MTU) |
| ECN 标记阈值 | KMAX_MAP, KMIN_MAP, PMAX_MAP（按带宽分级） |
| 仿真控制 | SIMULATOR_STOP_TIME, ENABLE_TRACE |
| 输出文件 | FLOW_FILE, TRACE_FILE, FCT/PFC/QLEN/BW/RATE 监控文件 |
| 监控间隔 | QP_MON_INTERVAL, QLEN_MON_INTERVAL, BW_MON_INTERVAL |

### `ratio/` — 带宽利用率表

CSV 格式，记录不同消息大小和集群规模下的带宽利用率：
- **nic_ratio.csv** — NIC 在不同节点数（1~128节点）下各消息大小的带宽利用率
- **nvlink_ratio.csv** — NVLink 节点内通信带宽利用率
- **ata_ratio.csv** — All-to-All 通信的带宽利用率

消息大小范围：16MB ~ 4GB，利用率范围：0.05 ~ 0.95

### `topo/gen_Topo_Template.py` — 拓扑生成器

根据参数自动生成 NS-3 可用的网络拓扑文件。支持三种拓扑模板：

| 拓扑类型 | 说明 |
|---------|------|
| `Rail_Opti_SingleToR` | 阿里巴巴 HPN Rail-Optimized 单 ToR 拓扑 |
| `Spectrum-X` | NVIDIA Spectrum-X 拓扑 |
| `DCN+` | 传统 DCN+ Fat-Tree 拓扑 |

**拓扑参数**：
- `gpu`：总 GPU 数量
- `gpu_per_server`：每服务器 GPU 数
- `nv_switch_per_server`：每服务器 NVSwitch 数
- `asw_switch_num`：接入交换机数
- `psw_switch_num`：骨干交换机数
- `bandwidth`：链路带宽（如 200G/400G）
- `gpu_type`：GPU 类型（A100/H800）

**生成内容**：节点数、交换机编号、链路连接关系、带宽配置

## 使用方式

```bash
# 生成拓扑文件
cd inputs/topo/
python gen_Topo_Template.py --gpu 1024 --gpu_per_server 8 --bandwidth 400G --gpu_type H800

# 仿真配置通常放到 /etc/astra-sim/ 目录
```

## 在项目中的角色

为 astra-sim + NS-3 网络仿真提供必要的输入文件：网络配置、带宽校准数据和拓扑描述。
