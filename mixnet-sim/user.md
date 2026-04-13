# MixNet-Sim 使用说明

本文档覆盖 `mixnet-sim` 仓库的**完整使用流程**：依赖、编译、两个子模块（`mixnet-flexflow` / `mixnet-htsim`）各自的命令行参数、脚本用法、输出文件说明以及常见问题。

如果你只是想"读代码 + 搞清原理"，请同时参考：

- `README.md`（官方简介）
- `ocs.md`（OCS/EPS 动态选择与重配置逻辑）
- `paper.pdf`（SIGCOMM'25 原文）

---

## 0. 仓库结构

```
mixnet-sim/
├── README.md                 # 官方简介（中文/英文对照用例）
├── paper.pdf                 # SIGCOMM'25 论文原文
├── ocs.md                    # OCS/EPS 动态选择与重配置逻辑分析
├── user.md                   # ← 本文
├── DAG.markdown              # FlexFlow 生成 DAG 的记录
├── mixnet-flexflow/          # 任务图生成器（修改版 FlexFlow）
│   ├── INSTALL.md            # FlexFlow 原版安装指南
│   ├── config/config.linux   # CMake 配置脚本
│   ├── examples/cpp/mixture_of_experts/   # MoE 入口 (moe.cc)
│   └── src/runtime/          # taskgraph flatbuf 导出实现
└── mixnet-htsim/             # 包级网络模拟器
    ├── README.md             # 官方简介
    ├── mixnet_scripts/       # 编译+运行脚本
    │   ├── compile.sh
    │   ├── mixtral_8x22B_mixnet.sh           # 全阶段
    │   └── onestage_mixtral_8x22B_mixnet.sh  # 单阶段快速测试
    ├── test/                 # expert 权重矩阵样本
    └── src/clos/             # 模拟器核心
        ├── ffapp.{h,cpp}                # 任务图 + 集合通信建模
        ├── mixnet_topomanager.{h,cpp}   # OCS 重配置控制器
        └── datacenter/
            ├── mixnet.{h,cpp}           # MixNet 拓扑
            ├── fat_tree_topology.{h,cpp}
            ├── main_tcp_mixnet.cpp      # htsim_tcp_mixnet 入口
            ├── main_tcp_fattree.cpp     # htsim_tcp_fattree 入口
            ├── main_tcp_flat.cpp        # htsim_tcp_flat 入口（-ar 开关）
            └── Makefile
```

---

## 1. 依赖与环境

### 1.1 系统依赖

| 依赖            | 最低版本                  | 用途                                |
| --------------- | ------------------------- | ----------------------------------- |
| GCC / G++       | 7+（支持 `-std=c++17`） | 编译 mixnet-htsim                   |
| CMake           | 3.10+                     | 编译 FlexFlow                       |
| Make            | 任意                      | htsim Makefile                      |
| CUDA            | 11.0+（可选）             | 如果要跑真正的 FlexFlow Python path |
| CUDNN           | 与 CUDA 匹配              | 同上                                |
| Python          | 3.8+                      | FlexFlow Python 接口（可选）        |
| FlatBuffers     | 2.0+（header-only）       | 任务图序列化（随 FlexFlow 一同装）  |
| Legion / GASNet | 随 FlexFlow 自带          | FlexFlow 运行时                     |

> **注意**：如果你只想跑**已有** `.fbuf` 文件上的 htsim 模拟（即只关心 MixNet 的网络层评估），**不需要** CUDA / FlexFlow 构建；直接跳到 §3 构建 `mixnet-htsim` 即可。

### 1.2 获取源码

```bash
git clone --recursive <your-repo-url> mixnet-sim
cd mixnet-sim
git submodule update --init --recursive
```

如果之前没加 `--recursive`，手动补 submodule：

```bash
cd mixnet-sim
git submodule update --init --recursive
```

---

## 2. 构建 `mixnet-flexflow`（任务图生成器）

这是**可选**步骤。如果要自己生成新模型的 `.fbuf`，才需要构建 FlexFlow；否则用官方提供的 [Google Drive 预生成文件](https://drive.google.com/drive/folders/1hChT-tVYJwBSCAC_hTm3x99JLcnl3vRk)即可。

### 2.1 用 CMake 构建

```bash
cd mixnet-flexflow
mkdir build && cd build
../config/config.linux
make -j 8
```

常用环境变量（执行 `config.linux` 前 export）：

| 变量                  | 默认                | 说明                                      |
| --------------------- | ------------------- | ----------------------------------------- |
| `CUDA_DIR`          | `/usr/local/cuda` | CUDA 安装路径                             |
| `CUDNN_DIR`         | 同上                | CUDNN 路径                                |
| `FF_CUDA_ARCH`      | `autodetect`      | 目标 GPU 架构，如 `70,80,90` 或 `all` |
| `FF_USE_PYTHON`     | `ON`              | 是否构建 Python 接口                      |
| `FF_USE_NCCL`       | `ON`              | 是否链接 NCCL                             |
| `FF_BUILD_EXAMPLES` | `OFF`             | **必须开启**才会编译 `moe`        |
| `FF_HOME`           | 仓库根              | FlexFlow 根路径，模拟器 Makefile 会用     |

参考 `mixnet-flexflow/INSTALL.md` 获取完整选项说明。

构建完成后，确认：

```bash
ls mixnet-flexflow/build/examples/cpp/mixture_of_experts/moe
export FF_HOME=$(pwd)/../mixnet-flexflow
```

### 2.2 FlexFlow 端命令行参数

FlexFlow `moe` 可执行文件接收 **两层**参数：Legion 运行时参数（`-ll:*`）+ MoE 训练配置（`--*`）。

#### Legion 运行时(生成轨迹要用的配置)

| 参数          | 示例      | 说明                                              |
| ------------- | --------- | ------------------------------------------------- |
| `-ll:gpu`   | `1`     | 每节点使用的 GPU 数（生成 taskgraph 阶段 1 就够） |
| `-ll:cpu`   | `4`     | CPU 数据加载 worker                               |
| `-ll:fsize` | `31000` | 每 GPU 帧缓冲（MB）                               |
| `-ll:zsize` | `24000` | 零拷贝（pinned DRAM）大小（MB）                   |
| `-ll:util`  | `1`     | 每进程 utility 线程数                             |

#### FlexFlow 训练参数

| 参数                     | 示例    | 说明                                                          |
| ------------------------ | ------- | ------------------------------------------------------------- |
| `--budget`             | `20`  | MCMC 并行策略搜索预算                                         |
| `--only-data-parallel` | –      | 只探索 DP 策略（生成 task graph 用）                          |
| `--batchsize`          | `128` | 全局 batch size                                               |
| `--microbatchsize`     | `8`   | microbatch size（PP 切分用）                                  |
| `--taskgraph`          | –      | **触发任务图导出**（将写 `./results/taskgraph.fbuf`） |

#### 并行度（**关键**，需和 htsim 侧严格对齐）

| 参数           | 示例  | 说明                                           |
| -------------- | ----- | ---------------------------------------------- |
| `--train_dp` | `2` | 数据并行度                                     |
| `--train_tp` | `8` | 张量并行度                                     |
| `--train_pp` | `8` | 流水并行度（`pp=1` 即禁用 PP，跑 one-stage） |

集群 GPU 总数 = `dp × tp × pp × ep`（`ep` 在代码中等于 `tp`，因 expert 切分沿 tp 轴）。

#### MoE 模型超参

| 参数                     | 示例      | 说明                    |
| ------------------------ | --------- | ----------------------- |
| `--num-layers`         | `56`    | Transformer 层数        |
| `--embedding-size`     | `1024`  | 嵌入维度                |
| `--hidden-size`        | `6144`  | Attention hidden dim    |
| `--expert-hidden-size` | `16384` | FFN expert hidden dim   |
| `--num-heads`          | `32`    | Attention head 数       |
| `--sequence-length`    | `4096`  | 每 sample 序列长度      |
| `--topk`               | `2`     | 每 token 路由 expert 数 |
| `--expnum`             | `8`     | 每层 expert 数          |

### 2.3 生成任务图示例（Mixtral-8x22B）

```bash
cd mixnet-flexflow
./build/examples/cpp/mixture_of_experts/moe \
    -ll:gpu 1 -ll:fsize 31000 -ll:zsize 24000 \
    --budget 20 --only-data-parallel \
    --batchsize 128 --microbatchsize 8 \
    --train_dp 2 --train_tp 8 --train_pp 8 \
    --num-layers 56 --embedding-size 1024 \
    --expert-hidden-size 16384 --hidden-size 6144 \
    --num-heads 32 --sequence-length 4096 \
    --topk 2 --expnum 8

mkdir -p results
mv output.txt       results/mixtral8x22B_dp2_tp8_pp8_ep8_8.txt
mv taskgraph.fbuf   results/mixtral8x22B_dp2_tp8_pp8_ep8_8.fbuf
```

产物：

- `mixtral8x22B_dp2_tp8_pp8_ep8_8.fbuf` —— FlatBuffer 任务图，`mixnet-htsim` 的输入
- `mixtral8x22B_dp2_tp8_pp8_ep8_8.txt`  —— 人类可读的 task list（调试用）

> **提示**：命名惯例 `<model>_dp<X>_tp<Y>_pp<Z>_ep<W>_<mb>.fbuf` 会被脚本隐式依赖，改名请同时改脚本里的 pattern。

---

## 3. 构建 `mixnet-htsim`（网络模拟器）

### 3.1 一键脚本（推荐）

```bash
cd mixnet-htsim
bash mixnet_scripts/compile.sh
```

脚本会依次：

1. `make clean` 清理 `src/clos/datacenter`
2. 在 `src/clos` 下 `make -j` 构建通用库 + `libhtsim.a`
3. 在 `src/clos/datacenter` 下 `make -j` 生成所有 `htsim_tcp_*` 可执行文件

### 3.2 手动分步

```bash
cd mixnet-htsim/src/clos
make clean && make -j
cd datacenter
make clean && make -j
```

如果 FlexFlow 不在默认路径，需要覆盖 `FF_HOME`（因为 `Makefile` 要找 `fbuf/include`）：

```bash
cd mixnet-htsim/src/clos/datacenter
make FF_HOME=/absolute/path/to/mixnet-flexflow -j
```

### 3.3 产物

位于 `mixnet-htsim/src/clos/datacenter/`：

| 可执行                   | 拓扑                       | 用途                                  |
| ------------------------ | -------------------------- | ------------------------------------- |
| `htsim_tcp_mixnet`     | **MixNet** (OCS+EPS) | MixNet 评估的主力                     |
| `htsim_tcp_fattree`    | 纯 Fat-Tree                | baseline 对照                         |
| `htsim_tcp_os_fattree` | 过订 Fat-Tree              | baseline 对照                         |
| `htsim_tcp_flat`       | Flat (single switch)       | 调试用（支持 `-ar` 选集合通信算法） |
| `htsim_tcp_dyn_flat`   | 动态 Flat                  | SiP-ML 风格基线                       |
| `htsim_tcp_fc`         | Full-Connect               | 调试                                  |
| `htsim_tcp_aggosft`    | 聚合过订 Fat-Tree          | 调试                                  |

---

## 4. `htsim_tcp_mixnet` 命令行参数详解

位置：`mixnet-htsim/src/clos/datacenter/htsim_tcp_mixnet`
源码：`main_tcp_mixnet.cpp` (line 152-299)

### 4.1 必填参数

| 参数                 | 类型   | 默认                             | 说明                                                                                 |
| -------------------- | ------ | -------------------------------- | ------------------------------------------------------------------------------------ |
| `-flowfile <path>` | string | `../../../test/taskgraph.fbuf` | FlexFlow 导出的 `.fbuf` 任务图                                                     |
| `-speed <Mbps>`    | uint32 | **必填**                   | 每 NIC 带宽，单位**Mbps**（脚本里 `$((bw*1000))`，即 `100Gbps → 100000`） |
| `-nodes <N>`       | int    | 16                               | 集群**GPU 总数**（不是 server 数！内部自动除 `NUM_GPU_PER_NODE=8`）          |
| `-simtime <sec>`   | double | 未设                             | 模拟上限，单位秒；一般给个大数如 `3600.1`                                          |
| `-dp_degree <N>`   | int    | –                               | 数据并行度，**必须**和 fbuf 对齐                                               |
| `-tp_degree <N>`   | int    | –                               | 张量并行度                                                                           |
| `-pp_degree <N>`   | int    | –                               | 流水并行度                                                                           |
| `-ep_degree <N>`   | int    | –                               | Expert 并行度                                                                        |

### 4.2 MixNet 物理/拓扑参数

| 参数               | 类型   | 默认        | 说明                                                                                 |
| ------------------ | ------ | ----------- | ------------------------------------------------------------------------------------ |
| `-alpha <N>`     | int    | `6`       | 每节点**OCS 端口数**；EPS 会占用 `8-alpha` 个 NIC。**值越大 OCS 越强** |
| `-rdelay <ns>`   | uint32 | `100`     | OCS 重配置时延（代码里乘以 `1e9`，传入时单位实际是 **ms**，见下方注释）      |
| `-deg <N>`       | int    | `4`       | 拓扑度数（保留参数）                                                                 |
| `-omethod <str>` | string | `sipring` | 初始 OCS 连接方式                                                                    |

> **⚠️ 关于 `-rdelay`**：`main_tcp_mixnet.cpp:398/400` 里做的是 `1000000000ULL * reconf_delay`，单位是 **ns×1e9 = ms**。脚本里传 `-rdelay 25` 表示 25 ms（paper §7 一致）。

### 4.3 TCP / 队列参数

| 参数              | 类型   | 默认     | 说明                                      |
| ----------------- | ------ | -------- | ----------------------------------------- |
| `-ssthresh <N>` | int    | `15`   | TCP slow-start 阈值（pkt）                |
| `-q <pkt>`      | int    | `100`  | 每队列大小（pkt 数，内部 `memFromPkt`） |
| `-rtt <ns>`     | uint32 | `1000` | 基础 RTT（ns）                            |

### 4.4 输入/输出文件

| 参数                     | 类型   | 默认                                               | 说明                                              |
| ------------------------ | ------ | -------------------------------------------------- | ------------------------------------------------- |
| `-weightmatrix <path>` | string | `../../../test/num_global_tokens_per_expert.txt` | 8×8 expert 权重矩阵（决定 A2A 的 per-pair 尺寸） |
| `-logdir <dir>`        | string | 自动 `./logs/<exe>_<date>`                       | 日志目录；**不存在会自动创建**              |
| `-ocs_file <name>`     | string | `fct_util_out.txt`                               | OCS 流完成时间日志的文件名（会写到 logdir 下）    |
| `-ecs_file <name>`     | string | `fct_util_out.txt`                               | EPS 流完成时间日志                                |
| `-utiltime <sec>`      | double | `0.01`                                           | 链路利用率采样间隔（保留）                        |

### 4.5 MPTCP 算法开关（大多用不上）

| 参数                                                                             | 说明               |
| -------------------------------------------------------------------------------- | ------------------ |
| `UNCOUPLED`                                                                    | 默认               |
| `COUPLED_INC` / `FULLY_COUPLED` / `COUPLED_TCP` / `COUPLED_SCALABLE_TCP` | MPTCP 变体         |
| `COUPLED_EPSILON [eps]`                                                        | 带 ε 参数的 MPTCP |

MixNet 里 TCP 走的是 DCTCP（`src/clos/dctcp.cpp`），这几个开关实际不影响 per-flow 行为。

### 4.6 最小完整示例

```bash
cd mixnet-htsim/src/clos/datacenter
mkdir -p logs/demo
./htsim_tcp_mixnet \
    -simtime 3600.1 \
    -flowfile /abs/path/to/mixtral8x22B_dp2_tp8_pp8_ep8_8.fbuf \
    -speed 100000 \
    -nodes 1024 \
    -ssthresh 10000 \
    -rtt 1000 \
    -q 10000 \
    -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 \
    -alpha 6 \
    -rdelay 25 \
    -weightmatrix ../../../test/num_global_tokens_per_expert.txt \
    -logdir logs/demo \
    -ocs_file nwsim_ocs_100.txt \
    -ecs_file nwsim_ecs_100.txt \
    > logs/demo/output.log 2>&1
```

---

## 5. `htsim_tcp_flat` 参数（调试/AR 算法对比）

`main_tcp_flat.cpp` 的主要差异是有 `-ar <strategy>` 开关用来选 AllReduce 算法。

| 参数            | 值                | 说明                       |
| --------------- | ----------------- | -------------------------- |
| `-ar ring`    | ring AllReduce    | NCCL-style 环形，2(N-1) 轮 |
| `-ar ps`      | parameter-server  | 2 轮，node_group[0] 做 PS  |
| `-ar dps`     | direct-PS         | 全 mesh 两两交换           |
| `-ar`（缺省） | `FF_DEFAULT_AR` | 保留行为                   |

其他参数（`-flowfile`、`-nodes`、`-speed`、`-ssthresh`、`-q`、`-logdir`、`-ofile`、`-weightmatrix`、`-simtime`、`-utiltime`）与 mixnet 基本一致，但**没有** DP/TP/PP/EP、`-rdelay`、`-alpha`、`-ocs_file` / `-ecs_file`。

> MixNet（`htsim_tcp_mixnet`）内部把 AllReduce 强制走 EPS，所以 `-ar` 开关**只对 flat topology 生效**。

---

## 6. 脚本使用

### 6.1 编译脚本

```bash
mixnet-htsim/mixnet_scripts/compile.sh
```

无参数，直接执行即可；脚本里用 `$BASH_SOURCE` 自动解析仓库路径，所以在任意 cwd 下运行都没问题。

### 6.2 跑 Mixtral-8x22B (全 PP) / MixNet

```bash
mixnet-htsim/mixnet_scripts/mixtral_8x22B_mixnet.sh
```

**在运行前请修改脚本里的两处路径：**

```bash
# 行 5: htsim_tcp_mixnet 所在目录
dir="/usr/wkspace/mixnet-sim/mixnet-htsim/src/clos/datacenter"

# 行 6: .fbuf 所在目录
new_fbuf_dir="/usr/wkspace/mixnet-sim/mixnet-flexflow/results"
```

脚本会扫 `bw × microbatchsize × rdelay × workloads` 的**笛卡尔积**并**并行** (`&`) 启动所有实验，每组结果单独落到 `logs/mixtral8x22B_<mb>_mixnet_<bw>_<workload>/`。

默认 sweep：

- 带宽：`100` Gbps
- microbatch：`8`
- 重配置延迟：`25` ms
- workload：`num_global_tokens_per_expert`
- 固定 `dp=2 tp=8 pp=8 ep=8`，GPU 数 `1024`

### 6.3 快速单阶段测试（`pp=1`）

```bash
mixnet-htsim/mixnet_scripts/onestage_mixtral_8x22B_mixnet.sh
```

同样需要改 `dir` 和 `new_fbuf_dir`。相比全 PP 版：

- `pp_degree=1`（跳过 PP 划分）
- `-nodes 512`
- fbuf 文件名带 `_onestage_`

单阶段跑得快，用于 smoke test / 调试；全阶段用于论文评估。

### 6.4 Fat-Tree baseline 脚本

```bash
mixnet-htsim/mixnet_scripts/mixtral_8x22B_fattree.sh
mixnet-htsim/mixnet_scripts/onestage_mixtral_8x22B_fattree.sh
```

参数结构类似，但跑的是 `htsim_tcp_fattree`（纯 EPS），没有 `-rdelay`、`-alpha`、`-ocs_file`、`-ecs_file`；用于 MixNet 的对照基线。

---

## 7. 输入文件格式

### 7.1 任务图 `.fbuf`

二进制 FlatBuffer，schema 来自 `mixnet-flexflow/fbuf/taskgraph.fbs`（或 `include/flexflow/taskgraph_generated.h`）。包含：

- `tasks[]`：每个 task 有 `guid`、`type`（FORWARD/BACKWARD/UPDATE/ALLREDUCE/ALLTOALL/DP_ALLREDUCE/...）、`runtime`、`from_node_ids`、`to_node_ids`、`info`、`operator_sizes[]` 等字段
- `devices[]`：device 描述（GPU index、node index）
- `next_tasks[]`：DAG 依赖边

`FFApplication::load_taskgraph_flatbuf()`（`ffapp.cpp:184`）负责解析并实例化 `FFTask` / `FFAlltoAll` / `FFRingAllreduce` / ... 等对象。

### 7.2 Expert 权重矩阵 `-weightmatrix`

纯文本，8×8 整数矩阵（例如 `mixnet-htsim/test/num_global_tokens_per_expert.txt`），表示**每对 expert 之间的 token 流量权重**。A2A 的 per-pair 传输量按公式：

```cpp
xfer_size = total_xfer_size * weight_matrix[i%8][j%8] / 32768
```

这也是模拟器里 MoE A2A traffic matrix 的来源。替换这个文件可以模拟不同的 routing skew。

---

## 8. 输出日志

每次运行都会在 `-logdir` 下产生：

| 文件                                    | 内容                                                  |
| --------------------------------------- | ----------------------------------------------------- |
| `output.log`                          | stdout/stderr 合并（调试参数、进度、finishtime）      |
| `nwsim_ocs_<bw>.txt`（`-ocs_file`） | 走 OCS 的 flow 的 FCT、start、size、路径              |
| `nwsim_ecs_<bw>.txt`（`-ecs_file`） | 走 EPS 的 flow 的同上字段                             |
| （可选）`fct_util_out.txt`            | 当 `-ocs_file` / `-ecs_file` 未指定时的默认文件名 |

在 `output.log` 末尾会有：

```
FinalFinish <simulated_seconds>
Total simulation duration: HHhMMmSSs
```

- **FinalFinish**：一次完整 iteration（1 step）的模拟耗时，单位秒
- **Total simulation duration**：真实墙钟时间

### 8.1 解析日志

官方没有自带 parser，但字段都是纯文本 `key=value` / 列格式，常用 awk / pandas 就能做。常见指标：

- **step time**：FinalFinish
- **A2A 完成比例**：`grep -c "ALLTOALL" nwsim_ocs_*.txt`
- **OCS 利用率**：对 `nwsim_ocs_*.txt` 里 size/FCT 做加权平均
- **重配置次数**：`grep -c "reconf" output.log`

---

## 9. 端到端走一遍（TL;DR）

```bash
# 0. clone + submodule
git clone --recursive <repo> mixnet-sim && cd mixnet-sim

# 1. 构建 FlexFlow（可选，如果用 Google Drive 预生成 fbuf 可跳过）
cd mixnet-flexflow && mkdir build && cd build
../config/config.linux && make -j 8
cd ../..

# 2. 生成任务图（可选）
cd mixnet-flexflow
./build/examples/cpp/mixture_of_experts/moe \
    -ll:gpu 1 -ll:fsize 31000 -ll:zsize 24000 \
    --budget 20 --only-data-parallel \
    --batchsize 128 --microbatchsize 8 \
    --train_dp 2 --train_tp 8 --train_pp 8 \
    --num-layers 56 --embedding-size 1024 \
    --expert-hidden-size 16384 --hidden-size 6144 \
    --num-heads 32 --sequence-length 4096 \
    --topk 2 --expnum 8
mkdir -p results
mv taskgraph.fbuf results/mixtral8x22B_dp2_tp8_pp8_ep8_8.fbuf
cd ..

# 3. 编译 htsim
bash mixnet-htsim/mixnet_scripts/compile.sh

# 4. 改脚本路径并运行
vim mixnet-htsim/mixnet_scripts/mixtral_8x22B_mixnet.sh   # 改 dir / new_fbuf_dir
bash mixnet-htsim/mixnet_scripts/mixtral_8x22B_mixnet.sh

# 5. 看结果
cd mixnet-htsim/src/clos/datacenter
tail -20 logs/mixtral8x22B_8_mixnet_100_num_global_tokens_per_expert/output.log
```

---

## 10. 参数选择指引

| 场景                                 | 推荐参数                                                                                        |
| ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| **单机快速 smoke test**        | `onestage_mixtral_8x22B_mixnet.sh`，`-nodes 128`，`-simtime 100`                          |
| **复现 paper Fig. 12**         | `-nodes 1024`，`-speed 100000`，`-rdelay 25`，`-alpha 6`，`dp=2 tp=8 pp=8 ep=8`       |
| **测不同带宽收益**             | sweep `-speed` ∈ {100000, 200000, 400000, 800000}                                            |
| **测重配置敏感度**             | sweep `-rdelay` ∈ {10, 25, 50, 100, 200}                                                     |
| **测 alpha 对 OCS 收益的影响** | sweep `-alpha` ∈ {2, 4, 6, 8}；`alpha=8` 表示**全 OCS**，`alpha=0` 退化为 fat-tree |
| **测不同 workload skew**       | 替换 `-weightmatrix`（从均匀到极端偏斜）                                                      |

---

## 11. 常见问题 (FAQ)

### Q1. `make` 报错找不到 `flexflow/taskgraph_generated.h`

FlexFlow 没构建完整，或 `FF_HOME` 没设。检查：

```bash
ls $FF_HOME/fbuf/include/flexflow/taskgraph_generated.h
# 或手动指定
make FF_HOME=/abs/path/to/mixnet-flexflow -j
```

### Q2. `htsim_tcp_mixnet: error: Bad parameter: -xxx`

`main_tcp_mixnet.cpp` 的参数解析是严格顺序匹配，任何未识别参数都会让它 `exit(1)`。检查拼写，特别是 `-dp_degree` / `-ep_degree`（带下划线，不是连字符）。

### Q3. 跑出的 FinalFinish 和 paper 差距很大

- **`-speed` 单位是 Mbps 不是 Gbps**，100Gbps 要传 `100000`
- `-rdelay` 实际语义是 **毫秒**（内部乘 `1e9` 把 ms 变 ns）
- 确认 fbuf 文件里的 `dp_degree × tp_degree × pp_degree × 8` 等于 `-nodes`
- 检查 `-weightmatrix` 是否和 fbuf 生成时用的 workload 一致

### Q4. A2A flow 全部走 EPS，OCS 没流量

- 检查 `-alpha` 是否 > 0
- 检查 fbuf 里 `ep_degree` 是否 > 1（单 expert 没 A2A）
- 开头几轮 iteration `conn` 矩阵可能还是初始值，等 `RegionalTopoManager` 第一次重配后才会有 OCS 流量
- 详见 `ocs.md` §3 的 per-flow `topo->conn[src][dst]>0` 判断逻辑

### Q5. `logdir` 里没有 `nwsim_ocs_*.txt`

- 默认情况下 `-ocs_file` 没指定会创建 `fct_util_out.txt`
- 如果指定了带路径的文件名（比如 `./foo/bar.txt`），只会取 basename 写到 logdir 下

### Q6. FlexFlow 构建太慢/太难

用官方提供的 [Google Drive 预生成 fbuf](https://drive.google.com/drive/folders/1hChT-tVYJwBSCAC_hTm3x99JLcnl3vRk)，完全跳过 §2。只构建 `mixnet-htsim` 就够跑 MixNet 评估。

### Q7. 如何只跑几个 layer 做快速验证

FlexFlow 端改 `--num-layers 4` 重新生成 fbuf；htsim 端不支持"只跑前 N 层"的开关（task DAG 已经在 fbuf 里定好）。

### Q8. 多个 sweep 并行时 CPU 打爆

`mixtral_8x22B_mixnet.sh` 默认用 `&` 后台并行启动所有配置；如果组合多，改成串行或用 `wait` 控制并发数：

```bash
N=4  # 最多并发
sem() { while [ $(jobs -rp | wc -l) -ge $N ]; do wait -n; done; }
for ... do
    sem
    ./htsim_tcp_mixnet ... &
done
wait
```

---

## 12. 代码入口速查

| 功能                                   | 文件 : 行                                                                             |
| -------------------------------------- | ------------------------------------------------------------------------------------- |
| `htsim_tcp_mixnet` main              | `mixnet-htsim/src/clos/datacenter/main_tcp_mixnet.cpp:119`                          |
| `htsim_tcp_flat` main (`-ar` 开关) | `mixnet-htsim/src/clos/datacenter/main_tcp_flat.cpp:105`                            |
| 任务图加载                             | `mixnet-htsim/src/clos/ffapp.cpp:184` `FFApplication::load_taskgraph_flatbuf`     |
| A2A per-flow OCS/EPS 选择              | `mixnet-htsim/src/clos/ffapp.cpp:2185` `FFAlltoAll::start_flow`                   |
| OCS 重配触发（前向 GROUP_BY）          | `mixnet-htsim/src/clos/ffapp.cpp:1066` `FFTask::cleanup`                          |
| OCS 重配触发（反向 Aggregate）         | `mixnet-htsim/src/clos/ffapp.cpp:997` `FFTask::execute_compute`                   |
| 贪心 OCS 重配算法                      | `mixnet-htsim/src/clos/mixnet_topomanager.cpp:370` `regional_topo_reconfig`       |
| MixNet 拓扑构造                        | `mixnet-htsim/src/clos/datacenter/mixnet.cpp:115` `init_network`                  |
| FlexFlow taskgraph 导出                | `mixnet-flexflow/src/runtime/substitution.cc:1193` `Graph::get_taskgraph_flatbuf` |
| MoE 示例入口                           | `mixnet-flexflow/examples/cpp/mixture_of_experts/moe.cc:25` `parse_input_args`    |

更深的代码走读见 `ocs.md`。

---

## 13. 参考

- **论文**：Liao et al., *MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training*, ACM SIGCOMM 2025. [DOI](https://doi.org/10.1145/3718958.3750465)
- **TopoOpt**（上游基线）：NSDI'23 [github](https://github.com/hipersys-team/TopoOpt)
- **Opera**（htsim 上游）：NSDI'20
- **FlexFlow**（上游）：OSDI'22 Unity / MLSys'19

有问题可先查：

1. `mixnet-htsim/README.md` / `OPERA_README.md`
2. `mixnet-flexflow/INSTALL.md` / `MULTI-NODE.md`
3. 源码里 `main_tcp_*.cpp` 的参数解析块
4. 本仓库的 `ocs.md`（原理分析）

Contact：

- Xudong Liao — xudong.liao.cs@gmail.com
- Yijun Sun  — yijun.sun@connect.ust.hk
