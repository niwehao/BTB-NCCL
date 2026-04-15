                                                                               
  - D.1 奇数 a2a 层会产生孤儿 block：最后一个单独成 block 只有 dispatch 无 combine，close_block        
  只累一层 → pred 量级减半。deepseek 不触发（偶数），但无 warning。
  - A.5 其他 iterate path 没挂                                                                         
  hook：iterate_data_parallel、iterate_model_parallel、iterate_hybrid_parallel_DLRM 等 6 条路径没加    
  on_rank_pass_end_hook。如果未来用这些路径 + mixnet，structure 永不 freeze、reconfig 永不
  fire、layer_tm 无限增长。deepseek/mistral 走 Transformer 路径，当前无影响。                          
  - 关键不变量确认：SimAI 的 CollectiveBarrier::Blocking 保证 rank A 的 pass_counter++ 时，它所有 pass
  P 的 a2a send 都已同步进入 on_a2a_flow_start 并 accumulate。pass_end_ranks[P] 达到 g_total_gpus 时   
  pass P 全部 flow 已在 layer_tm 里 —— 这正是 #4/#6 修复所依赖的核心假设。