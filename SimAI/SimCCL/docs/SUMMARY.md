# SimCCL/docs 模块设计与使用总结

## 目录功能概述

`docs` 是 SimCCL 项目的**文档与社区资源目录**，当前仅包含社区联系方式的图片资源。

## 目录结构

```
docs/
└── images/
    ├── simai_dingtalk.jpg   # SimAI 钉钉社区群二维码
    └── simai_wechat.jpg     # SimAI 微信社区群二维码
```

## 说明

SimCCL（Simulated Collective Communication Library）是 SimAI 的**集合通信模拟组件**，负责将集合通信操作（AllReduce、AllGather、ReduceScatter、AllToAll 等）转换为点对点通信。

当前状态：
- 基础版本（`mocknccl` 前缀文件）已集成在 `astra-sim-alibabacloud` 仓库中（`MockNcclChannel`、`MockNcclGroup`）
- 完整版本计划独立发布到此仓库，尚未上传代码

## 在项目中的角色

作为 SimAI 的**集合通信模拟层**（占位仓库），未来将提供完整的集合通信算法到点对点通信的转换能力，替代目前内嵌在 astra-sim 中的基础版本。
