# aicb/workload_generator 模块设计与使用总结

## 目录功能概述

`workload_generator` 是 AICB 的**核心工作负载生成模块**，通过模拟（Mocked）模型架构和训练/推理流程，自动生成通信和计算工作负载文件，用于物理集群基准测试或 SimAI 网络仿真。

## 目录结构

```
workload_generator/
- 仿真路径(4 个):产 workload 给模拟器吃                                                                                                
    - SimAI_training_workload_generator.py → SimAI(训练)
    - SimAI_inference_workload_generator.py → SimAI(推理)                                                                                
    - Vidur_workload_generator.py → Vidur(推理)                                                                                          
    - (上述共用 mocked_model/ 推算出通信序列)                                                                                            
  - 真机路径(4 个):产 LogItem 给 aicb.py 的 WorkloadApplyer 执行真 NCCL                                                                  
    - generate_megatron_workload.py           
    - generate_deepspeed_stage1_2_workload.py                                                                                            
    - generate_deepspeed_stage3_workload.py                                                                                              
    - generate_collective_test.py                                                                                                        
  - 辅助/基础设施(3 个):                                                                                                                 
    - workload_generator.py(基类,不跑)                                                                                                   
    - generate_ds_trace_replay_workload.py(从 DS log 解析)
    - analysis_pytorch_trace.py(从 PyTorch trace 解析)   
└── mocked_model/                            # 模拟模型架构
    ├── MockedModel.py                       # 基类定义
    ├── training/                            # 训练场景模拟模型
    │   ├── MockedMegatron.py                # Megatron 框架模型
    │   ├── MockedDeepspeed.py               # DeepSpeed 框架模型
    │   ├── MockedDeepSeek.py                # DeepSeek MLA+MoE 模型
    │   ├── AiobMegatron.py                  # Megatron AIOB 计算测量模型
    │   └── AiobDeepSeek.py                  # DeepSeek AIOB 计算测量模型
    └── inference/                           # 推理场景模拟模型
        ├── MockedDeepSeek.py                # DeepSeek 推理模型
        ├── MockedQwen3Moe.py                # Qwen3-MoE 推理模型
        ├── MockedQwen3Next.py               # Qwen3-Next 推理模型
        ├── AiobDeepSeek.py                  # DeepSeek 推理 AIOB 测量
        ├── AiobQwen3Moe.py                  # Qwen3-MoE 推理 AIOB 测量
        └── AiobQwen3Next.py                 # Qwen3-Next 推理 AIOB 测量
```

## 核心架构设计

### 基类层

#### `MockedParam` — 模拟参数
- 模拟 `torch.nn.Parameter`，记录 shape 和 element size
- `msg_size()` 返回参数传输的字节数

#### `MockedModel` — 模拟模型
- 模拟 `torch.nn.Module`，支持 `parameters()` 和 hook 注册
- `Linear` 子类模拟线性层

#### `MockedParamsBase` — 模拟参数配置基类
- 支持从 JSON 配置文件或命令行参数加载模型配置
- 用于推理场景的参数管理

#### `WorkloadGenerator` — 工作负载生成器基类
- 模拟训练流程：`init → [forward → backward] × microbatches → step` × epochs
- 支持 pipeline 并行的 `with_pipeline_forward_backward()`
- 子类重写 `forward()`, `backward()`, `step()` 填充具体通信操作

### SimAI 工作负载格式

#### `Work_Item` 数据类
每个工作项包含：
- `name`：操作名称
- `forward_compute_time`：前向计算时间（微秒）
- `forward_comm` / `forward_comm_size`：前向通信类型和大小
- `backward_compute_time` / `backward_comm` / `backward_comm_size`：反向传播
- `dp_compute_time` / `dp_comm` / `dp_comm_size`：数据并行通信
- `process_time`：处理时间

### 训练工作负载生成

#### `SimAI_training_workload_generator.py`
主入口，根据框架类型（Megatron/DeepSpeed/DeepSeek）选择对应的模拟模型，生成 SimAI 格式的工作负载文件。支持 AIOB 计算时间填充。

#### `generate_megatron_workload.py`
继承 `WorkloadGenerator`，实现 Megatron 框架的通信模式：
- Forward: TP all-gather/reduce-scatter（SP 启用时）、MoE all-to-all
- Backward: 与 forward 对应的反向通信
- Step: DP all-reduce / reduce-scatter + all-gather（分布式优化器）

#### `generate_deepspeed_stage1_2_workload.py` / `stage3`
实现 DeepSpeed ZeRO-1/2/3 的通信模式（bucket-based reduce/all-gather）

### 推理工作负载生成

#### `SimAI_inference_workload_generator.py`
生成推理场景（prefill/decode）的工作负载，支持 DeepSeek-671B、Qwen3-MoE、Qwen3-Next 等模型。

### 模拟模型实现

| 类 | 模拟内容 |
|-----|---------|
| `MockedMegatron` | Megatron GPT/LLaMA 的 Transformer 层、注意力、MLP、MoE |
| `MockedDeepspeed` | DeepSpeed ZeRO 的 LLaMA 模型架构 |
| `MockedDeepSeek` | DeepSeek 的 MLA（Multi-head Latent Attention）+ MoE 架构 |
| `AiobMegatron` | 真实 PyTorch 模型，用于在 GPU 上测量计算时间 |
| `AiobDeepSeek` | DeepSeek 的 GPU 计算时间测量模型 |

## 依赖关系

- **内部依赖**：`utils.utils`（参数解析、枚举）、`log_analyzer.log`（Workload, LogItem）、`core`（grouped_gemm）
- **外部依赖**：torch（AIOB 模式时需要）
- **被调用方**：`aicb.py`、`scripts/` 中的启动脚本

## 使用方式

```bash
# 生成 Megatron 训练工作负载（SimAI 格式）
python -m workload_generator.SimAI_training_workload_generator \
  --frame=Megatron --model_name=GPT-13B --world_size=16 \
  --tensor_model_parallel_size=2 --pipeline_model_parallel=1 \
  --global_batch=16 --micro_batch=1 --num_layers=40 --seq_length=2048 \
  --hidden_size=5120 --num_attention_heads=40

# 生成推理工作负载
python -m workload_generator.SimAI_inference_workload_generator \
  DeepSeek-671B configs/deepseek_default.json --phase decode

# 生成通用 Megatron 工作负载
python -m workload_generator.generate_megatron_workload
```

## 使用范围

采用**模拟模型 + 模板方法**模式：
1. `MockedModel` 构建模型架构（无真实权重），只记录参数 shape
2. `WorkloadGenerator` 定义训练循环模板
3. 具体生成器在 forward/backward/step 中填充通信操作
4. AIOB 模式下使用真实 PyTorch 模型测量 GPU 计算时间
5. 输出标准化的工作负载文件供仿真或回放使用

───────────────┬─────────────────────────────────────────────────────────────────────────┐   
  │     分支      │                               覆盖的模型                                │   
  ├───────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ --frame       │ GPT / LLaMA / LLaMA2/3 / Mistral / Mixtral-8x7B(MoE)/                   │   
  │ Megatron      │ Qwen3-MoE(训练)等所有 Megatron 风格架构                                 │   
  ├───────────────┼─────────────────────────────────────────────────────────────────────────┤   
  │ --frame       │ DeepSeek V3/R1(MLA + DeepSeekMoE,有独立的 mocked 模型)                  │   
  │ DeepSeek      │                                                                         │
  └───────────────┴─────────────────────────────────────────────────────────────────────────┘   

  推理支持(走 SimAI_inference_workload_generator.py)                                            
                                                                                                
  靠 args.model_name 字符串子串匹配(L301-312):                                                  
                                                                                                
  model_name 子串: "Qwen3-Moe"                              
  Mocked Model: MockedQwen3Moe.Qwen3MoeModel(MockedQwen3Moe.py:49)
  Aiob Model: AiobQwen3Moe.Qwen3MoeModel                                                        
  架构特点: Qwen3 MoE(标准 attention + sparse MoE block)                                        
  ────────────────────────────────────────                                                      
  model_name 子串: "Qwen3-Next"                                                                 
  Mocked Model: MockedQwen3Next.Qwen3NextModel(MockedQwen3Next.py:58)
  Aiob Model: AiobQwen3Next.Qwen3NextModel
  架构特点: Qwen3-Next(含 GatedDeltaNet 线性注意力 + MoE)                                       
  ────────────────────────────────────────
  model_name 子串: "DeepSeek"                                                                   
  Mocked Model: MockedDeepSeek.DeepSeekModel(MockedDeepSeek.py:471)(推理版,与训练版同名但不同类)
  Aiob Model: AiobDeepSeek.DeepSeekModel
  架构特点: MLA + DeepSeekMoE(推理简化版)                                                       
  ────────────────────────────────────────
  model_name 子串: 其它                                                                         
  Mocked Model: 无                        
  Aiob Model: —
  架构特点: 报错退出 (L311 sys.exit(1))       

## deepspeed
  elif args.frame == "DeepSpeed":                                                                                                                               
      model = DeepspeedForCausalLM(args)              # training/MockedDeepspeed.py                                                                           
      if args.stage == 1: workload_generator = DeepSpeedStage1(args, model)                                                                                     
      elif args.stage == 2: workload_generator = DeepSpeedStage2(args, model)                                                                                   
      elif args.stage == 3: workload_generator = DeepSpeedStage3(args, model)                                                                                   
  - DeepspeedForCausalLM / DeepspeedModel(MockedDeepspeed.py:46+)mock 出模型架构(只记 param shape,不带真权重)                                                   
  - DeepSpeedStage3(generate_deepspeed_stage3_workload.py:29+)继承 WorkloadGenerator,在 forward/backward/step 里按 DeepSpeed engine 的行为(bucket               
  reduce、_param_queue、prefetch/release、__most_recent_step_id_param_fetched_for)吐出一串 collective LogItem                                                   
  - aicb.py:78 applyer = WorkloadApplyer(...); applyer.apply_workload() 把这些 LogItem 在真 GPU 上用 NCCL 回放                                                  
                                                                                                                                                              
  换言之 aicb 的 DeepSpeed 路径靠自己 mock DeepSpeed engine 的参数管理逻辑,把它展开成一串普通的 NCCL collective 调用;底层只要有 torch.distributed +             
  NCCL,就能在真 GPU 集群跑。astra-sim / htsim 那种"纯仿真"完全不涉及。          
  ZeRO-3 的通信由这些运行时状态决定:bucket 累积、max_live_parameters 触发的 release、prefetch 队列、param access 顺序。这些都是跨层的、动态的、取决于 bucket    
  size / threshold 调优参数,没办法压成 "第 i 层 fp_comm = ALLGATHER 357468 字节" 这种静态表。