# TAGE Predictor

## 定位

TAGE, Tagged Geometric History Length predictor, 是现代高精度分支预测器的核心家族之一。它通过多张带 tag 的预测表覆盖不同长度的历史，并以几何级数选择历史长度。

## 核心思想

TAGE 包含：

- 一个 base predictor，通常类似 bimodal。
- 多个 tagged predictor tables。
- 每张 tagged table 使用不同的历史长度，长度按几何级数增长。
- 每个 tagged entry 通常包含 tag、预测计数器和 usefulness 计数器。

预测时，硬件从多张表中查找 tag 匹配项，优先使用历史最长的匹配表；同时保留较短历史的 alternate prediction 用于低置信度或新分配表项。

可以把 TAGE 理解成“多层历史记忆”：

- T0/base 只学当前 key 的长期偏向。
- T1 学很短历史，例如最近 4 次跳转。
- T2 学稍长历史，例如最近 8 次跳转。
- T3 学更长历史，例如最近 16 次跳转。
- T4 学更长历史，例如最近 32 次跳转。

如果一个分支只需要短历史，短表会命中并稳定预测；如果它需要长距离相关性，长表一旦学会，就会覆盖短表结果。

## 数据结构

一个简化 TAGE 可以这样设计：

```text
base_table[key_index] = 2-bit counter

tagged_table[i][index] = {
  tag: partial tag from key + compressed history,
  ctr: signed prediction counter, e.g. 3-bit signed counter,
  u: usefulness counter, e.g. 1-bit or 2-bit
}

history_lengths = [4, 8, 16, 32]
global_history = last N branch outcomes
```

对第 i 张 tagged table：

```text
index_i = hash(key, compressed(global_history, history_lengths[i]))
tag_i   = tag_hash(key, compressed(global_history, history_lengths[i]))
```

这里的 key 可以是 PC，也可以是你自己的组合 key。关键点是：index 用来定位表项，tag 用来确认“这个表项真的属于当前 key + 当前历史上下文”，避免 gshare 那种只靠索引导致的严重 aliasing。

## 预测流程

预测时并行查询 base table 和所有 tagged tables：

1. 计算每张 tagged table 的 index 和 tag。
2. 查出每张表的 entry。
3. 找出 tag 匹配的表项。
4. 最长历史长度的匹配项叫 provider。
5. 次长历史长度的匹配项叫 alternate；如果没有次长匹配，就用 base predictor 当 alternate。
6. 通常使用 provider 的 ctr 符号作为最终预测。
7. 如果 provider 是刚分配的新表项或置信度很低，可以临时使用 alternate，避免新表项污染预测。

伪代码：

```text
base_pred = base_table[base_index(key)].predict()

provider = none
alternate = base_pred

for table in tagged_tables from shortest to longest:
  entry = table[index(table, key, global_history)]
  if entry.tag == tag(table, key, global_history):
    alternate = provider.pred if provider exists else base_pred
    provider = entry

if provider does not exist:
  final_pred = base_pred
else if provider_is_weak_and_new(provider):
  final_pred = alternate
else:
  final_pred = provider.ctr >= 0 ? taken : not_taken
```

## 更新流程

分支真实结果出来后，需要更新三类状态：

- 更新 provider 的预测计数器。
- 更新 usefulness。
- 如果预测错了，尝试在更长历史表中分配新 entry。

计数器更新：

```text
if actual == taken:
  ctr = min(ctr + 1, max)
else:
  ctr = max(ctr - 1, min)
```

usefulness 的直觉是：如果 provider 比 alternate 更有用，就提高 u；如果 provider 没带来额外价值，就降低或不提高 u。

常见规则：

```text
if provider_pred != alternate_pred:
  if provider_pred == actual:
    provider.u += 1
  else:
    provider.u -= 1
```

分配规则的直觉是：如果当前最长命中的 provider 都预测错了，说明现有历史上下文不够好，就尝试在比 provider 更长历史的表里开一个新表项。

简化规则：

```text
if final_pred != actual:
  for table longer than provider_table:
    entry = table[index(table, key, global_history)]
    if entry.u == 0:
      entry.tag = tag(table, key, global_history)
      entry.ctr = weak_taken if actual == taken else weak_not_taken
      entry.u = 0
      break
```

如果找不到 u 为 0 的表项，通常会周期性衰减 usefulness，给新模式腾位置。

## 例子

假设你的 key 是 `K = 0x1234`，全局历史最近 16 位是：

```text
global_history = T N T T N T N N T T T N T N T T
```

简化配置：

```text
base: 2-bit counter table
T1: history length 4
T2: history length 8
T3: history length 16
```

预测某次分支时，查询结果如下：

```text
base predicts: taken

T1 index/tag match:
  history length = 4
  ctr = +1
  prediction = taken
  u = 1

T2 index/tag match:
  history length = 8
  ctr = -2
  prediction = not taken
  u = 1

T3 index hit but tag mismatch:
  ignored
```

此时：

- provider 是 T2，因为它是 tag 匹配里历史最长的表。
- alternate 是 T1，因为它是次长匹配。
- 最终预测是 T2 的 not taken。

如果真实结果也是 not taken：

- T2 的 ctr 从 -2 继续向 not-taken 方向强化。
- 因为 T2 和 T1 意见不同，且 T2 正确，T2.u 可以增加。

如果真实结果是 taken：

- T2 的 ctr 向 taken 方向移动。
- 因为 T2 和 T1 意见不同，且 T2 错误，T2.u 可以减少。
- 最终预测错误时，TAGE 会尝试在比 T2 更长历史的表中分配 entry；这里 T3 比 T2 更长，因此可能在 T3 的对应 index 写入：

```text
T3.entry.tag = tag(K, history16)
T3.entry.ctr = weak_taken
T3.entry.u = 0
```

下一次遇到相同 key 和相似 16-bit 历史时，T3 可能 tag 命中，并作为更长历史的 provider 覆盖 T2。

## 和 gshare 的差异

如果只看组合 key 和历史，gshare 通常是：

```text
index = key XOR history
prediction = counter_table[index]
```

TAGE 则是多组：

```text
index_i = hash(key, history_length_i)
tag_i = tag_hash(key, history_length_i)
prediction = longest_matching_tagged_entry
```

所以 TAGE 相比 gshare 的关键增强是：

- 同时尝试多个历史长度，不需要提前赌一个固定历史长度。
- 表项带 tag，减少不同 key/history 映射到同一 index 后互相污染。
- usefulness 让替换策略更聪明，尽量保留真正有用的长历史模式。

## 为什么流行/成功

- 带 tag 的表显著降低 aliasing。
- 几何历史长度让短历史和长历史相关性都能被捕捉，容量效率高。
- usefulness 机制帮助替换真正无用的表项。
- TAGE 及其变体长期是分支预测研究和竞赛中的标杆，并影响了现代工业级 predictor 设计。

## 局限

- 结构复杂，索引、tag 生成、分配和更新策略都需要精细设计。
- 多表访问带来能耗和时序挑战。
- 对某些统计偏置或特殊模式，常需要 statistical corrector、loop predictor 等组件补强。

## 实现要点

- 历史长度通常按几何级数从短到长配置。
- 使用压缩历史生成各表 index 和 tag。
- provider prediction 来自最长 tag 命中表；alternate prediction 来自次长命中或 base predictor。
- 分配新表项时应优先选择 usefulness 低的表项。
- 初版实现可以先做 2 到 4 张 tagged tables，历史长度选 `[4, 8, 16, 32]`，每个 entry 用 `tag + 3-bit ctr + 1-bit u`。
- 初版可以先不做复杂的 alternate-on-new-entry 逻辑，只要保留 provider/alternate 概念，后续再优化低置信度选择。
- 历史寄存器要跟投机执行配合；如果先做模拟器或离线 predictor，可以在分支提交后再更新历史，逻辑会简单很多。

## 主要来源

- André Seznec and Pierre Michaud, "A case for partially tagged Geometric History Length Branch Prediction", JILP 2006: https://jilp.org/vol8/v8paper1.pdf
- André Seznec, "The L-TAGE Branch Predictor", JILP 2007: https://jilp.org/vol9/v9paper6.pdf
- André Seznec, "A New Case for the TAGE Branch Predictor", MICRO 2011: https://www.cs.cmu.edu/~18742/papers/Seznec2011.pdf
