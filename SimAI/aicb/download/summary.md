# download 模块总结

## 概述

`download` 目录包含 AICB（AI Comm Benchmark）项目的 Debian 安装包，用于在集群环境中快速部署。

## 目录结构

```
download/
└── AICB_v1.0.deb               # Debian 安装包（约 11 MB）
```

## 包内容

**AICB_v1.0.deb** 是一个 Debian 二进制包（format 2.0），包含 AICB v1.0.0 通信基准测试套件。

安装后解压到 `/opt/AICB/`，包含以下核心组件：

- **aicb.py**：主入口，初始化分布式 PyTorch 进程组，运行基准测试
- **run_suites.py**：测试套件运行器，管理多种基准场景
- **workload_generator/**：工作负载生成器（Megatron、DeepSpeed、集合通信测试）
- **workload_applyer.py**：工作负载执行器，基于 NCCL 在物理集群上运行
- **log_analyzer/**：日志分析与性能统计
- **utils/**：核心工具（通信类型枚举、参数解析、计时器）
- **workload/**：预生成的工作负载数据（GPT、LLaMA 等模型）

## 在项目中的角色

提供 AICB 的**打包分发版本**，方便在 GPU 集群上快速部署和运行通信性能基准测试，无需从源码安装。
