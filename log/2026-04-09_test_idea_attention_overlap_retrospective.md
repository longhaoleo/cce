# 实验记录：test_idea 注意力重叠原型复盘

日期：2026-04-09  
负责人：Codex（根据仓库现存原型与输出结果整理）

## 1. 背景

`test_idea/` 是一次早期原型尝试，目标不是直接做概念擦除，而是先回答一个更前置的问题：

- 在 SDXL 的生成过程中，两个 prompt 概念的注意力区域是否会在时间维上出现可解释的“交互”？
- 这种交互能否帮助我们判断某个概念是在什么阶段、什么层里发生耦合？
- 如果能观察到稳定模式，是否可以把它作为后续概念擦除或概念分离的依据？

这个原型没有接到现在的 `SDXL + WSAE` 主线里，而是单独依赖 `attention_map_diffusers`，从 cross-attention / self-attention 的注意力图出发，做“概念 A vs 概念 B”的时序可视化分析。

当前整理结论是：它更像一个“注意力诊断工具原型”，不是一个稳定的概念定位或擦除方法。

## 2. 对应代码与产物

- 原型脚本：`test_idea/.ipynb_checkpoints/vslz_wosae-checkpoint.py`
- 主要输出目录：
  - `test_idea/concept_vis_output/kids + knife`
  - `test_idea/concept_vis_output/red cat + blue dog `

仓库状态信息：

- 代码版本：`3ea188c`
- `test_idea/` 目录体积约 `529M`
- `kids + knife` 输出约 `163M`
- `red cat + blue dog` 输出约 `194M`

这已经说明一个很现实的问题：这种按层、按时间步导出热图的方案，很容易在一次实验里生成大量中间产物。

## 3. 目标与假设

### 3.1 目标

- 在单次生成过程中抓取 attention maps
- 将 prompt 中两个概念对应的 token 激活单独抽出来
- 按时间步比较两个概念的空间重叠程度
- 按层或按块观察“交互 strongest 的位置和阶段”

### 3.2 核心假设

这个原型隐含了两个假设：

1. 文本概念之间的空间耦合，会在 cross-attention 图中表现为可观测的重叠
2. 这种重叠随扩散 step 的变化，可能揭示“概念绑定”是在早期形成、还是在后期细化时增强

## 4. 实现方案

### 4.1 总体流程

原型的主类是 `ConceptInteractionVisualizer`。整体流程如下：

1. 加载 `diffusers` pipeline
2. 用 `attention_map_diffusers.init_pipeline(...)` 给 pipeline 打 hook
3. 生成图像时把每个 timestep 的 attention maps 收集到全局 `attn_maps`
4. 在 prompt 中找到概念 A / B 对应的 token 索引
5. 对每个 timestep、每个层分组聚合 attention
6. 提取概念 A / B 的 token attention map
7. 计算二者的 IoU、连续 overlap、平均强度
8. 输出时间曲线、逐步热图和 top interaction csv

这本质上是一个“生成时同步观测”的分析器，不修改模型权重，也不做任何干预。

### 4.2 模型加载与 hook

实现上做了几件实用但偏原型化的事情：

- 自动兼容 `StableDiffusionXLPipeline` 与 `StableDiffusionPipeline`
- `cuda` 不可用时自动退回 CPU
- 优先尝试 safetensors / fp16 变体，失败时继续回退
- 初始化 `attention_map_diffusers` 的 hook，让生成过程自动记录 attention

这一层的优点是部署快，缺点是强依赖第三方 hook 库的张量格式和行为。

### 4.3 token 对齐

概念对齐方式不是语义匹配，而是“token 子序列匹配”：

- 先对 prompt 和 concept phrase 做文本规范化
- 用 tokenizer 把二者转为 token id
- 在 prompt token 序列中查找 phrase token 序列的连续子串
- SDXL 额外尝试 `tokenizer_2`

这一步实现简单，但有明显局限：

- 概念短语必须能作为连续 token 序列被找回
- 对 prompt 改写、同义词、插入修饰词非常敏感
- 只要 token 切分稍微变化，就可能完全匹配不到

所以它更适合明确短语，不适合开放式概念定义。

### 4.4 attention 张量归一化

原型里专门写了 `_attn_to_bthw(...)`，目的就是把不同实现、不同版本下的 attention tensor 尽量归一到统一形状：

- 目标格式是 `(B, T, H, W)`
- 允许输入来自不同头数、不同维度顺序
- 通过启发式规则判断哪一维像 token、哪一维像空间

这是原型里最“工程补丁味”的部分之一，说明作者当时已经遇到：

- hook 库返回格式不稳定
- 不同层可能空间分辨率不同
- token 维和 spatial 维不总是显式可分

这也直接导致结果解释时要比较保守，因为底层张量整理包含经验规则。

### 4.5 层筛选与分组

原型支持几种分组方式：

- `none`：所有层合并
- `block`：按 `down/mid/up`
- `block_attn`：按块和注意力类型组合
- `layer`：每一层单独输出

同时支持：

- `attn_kind="cross" | "self" | "any"`
- `layer_regex` 对层名做进一步筛选

默认思路其实很合理：如果要分析“某个文本 concept 的 token 对应空间区域”，优先看 `cross-attn`，因为只有它与文本 token 的对应关系最明确。

### 4.6 每个时间步的指标

对每个 timestep、每个 group，原型做了下面的计算：

1. 聚合组内所有层的 attention map
2. 取概念 A 对应 token 的 attention，做 token 维均值，得到 `map_a`
3. 取概念 B 对应 token 的 attention，做 token 维均值，得到 `map_b`
4. 对 `map_a` 和 `map_b` 各自做 `normalize_01`
5. 计算：
   - `IoU`：把 attention map 二值化后计算交并比
   - `overlap`：连续值重叠度，代码里用 `min(map_a, map_b)` 的聚合
   - `act_a_mean` / `act_b_mean`：归一化后平均激活强度

这里最值得注意的一点是：`map_a` 和 `map_b` 是分别归一化的。

这意味着：

- `overlap` 更像“形状与区域共现程度”
- 不再代表真实的绝对能量大小
- 当两个 map 都很 diffuse 时，也可能出现不低的 overlap

所以这些指标更适合做相对比较，不适合直接当成“因果耦合强度”。

### 4.7 可视化与输出

每个 group 会输出：

- `metrics.csv`
- `top_interaction.csv`
- `temporal_interaction.png`
- `timestep_XXXX_tYYYY.png`

其中：

- `metrics.csv` 保存全时间序列数值
- `top_interaction.csv` 按 `overlap` 取 top-k 时间步
- `temporal_interaction.png` 同时画交互强度和概念强度曲线
- 单步图里会把概念 A、概念 B、以及二者 overlap 区域并排展示

这套输出对“诊断某一层某一时刻到底发生了什么”很直观，但也正因为每层都保存图片，所以产物膨胀非常快。

## 5. 实际实验设置

从脚本结尾和现有输出可以确认，至少做过两类实验：

### 5.1 示例一：`kids + knife`

可推断的 prompt 设计接近：

- prompt：`a child hold kitchen knife on the table, scary lighting.`
- concept A：`child`
- concept B：`knife`

现存输出目录：

- `test_idea/concept_vis_output/kids + knife`

该输出目录下顶层约有 `10` 个层分组目录，说明这一组实验更像是对筛选后 cross-attn 层的有限观察。

### 5.2 示例二：`red cat + blue dog`

脚本里写死的示例是：

- prompt：`A red cat and a blue dog are playing together in a sunny park.`
- concept A：`red cat`
- concept B：`blue dog`
- `steps=20`
- `group_mode="layer"`
- `attn_kind="any"`

现存输出目录：

- `test_idea/concept_vis_output/red cat + blue dog `

这一组顶层约有 `70` 个层分组目录，说明当时采用的是“每层单独看”的最细粒度模式，输出也最重。

## 6. 结果观察

### 6.1 `kids + knife` 的数值特征

抽样文件：

- `test_idea/concept_vis_output/kids + knife/down_blocks.0.attentions.0.transformer_blocks.0.attn2/metrics.csv`
- `test_idea/concept_vis_output/kids + knife/down_blocks.0.attentions.0.transformer_blocks.0.attn2/top_interaction.csv`

从数值上看：

- `overlap` 在中后段明显升高，大致从 `0.39` 上升到 `0.72` 左右
- `IoU` 在大多数步骤并不高，很多时候只有 `0.03 ~ 0.20`
- 末端个别步骤 `IoU` 突然升到 `0.5`、`0.65`

这说明一件很关键的事：

- 连续 overlap 很容易高
- 但二值 IoU 往往不高，表示两张注意力图在“高值核心区域”并没有稳定重合

换句话说，它更像“同图共存导致的广域共现”，不一定真的是“child 和 knife 在模型内部形成了稳定绑定区域”。

### 6.2 `red cat + blue dog` 的数值特征

抽样文件：

- `test_idea/concept_vis_output/red cat + blue dog /down_blocks.1.attentions.0.transformer_blocks.0.attn2/metrics.csv`
- `test_idea/concept_vis_output/red cat + blue dog /down_blocks.1.attentions.0.transformer_blocks.0.attn2/top_interaction.csv`

这组的特点更明显：

- `overlap` 在前几步就已经很高，大约 `0.67`
- 但 `IoU` 初期非常低，只有 `0.01 ~ 0.07`
- 随时间推进，`IoU` 慢慢升到 `0.18 ~ 0.21`

这反映出：

- 两个主体的注意力分布在整体空间上共处一张图，所以连续 overlap 偏高
- 但它们的强响应中心其实是分离的，至少在很多层里没有明显“粘连”

对于“红猫 + 蓝狗”这种天然多主体场景，这个结果其实是符合直觉的。

### 6.3 指标层面的共同问题

两组实验都暴露出同一个核心问题：

- `overlap` 很容易偏高
- `IoU` 对阈值敏感
- 二者并不稳定对应“概念耦合强度”

原因大概有三类：

1. 两张 map 各自独立归一化，导致 overlap 偏向反映共现区域，而不是绝对注意力强度
2. 在同一张图里，不同对象共享背景、布局和构图，天然会产生空间重叠
3. cross-attention 热区并不直接等同于“可擦除的概念特征”

## 7. 方法层面的优点

虽然它没进主线，但这个原型不是没有价值。

它的优点主要在“诊断”而不是“控制”：

- 可以快速看某个 prompt 里两个概念在什么时间步最容易共现
- 可以帮助挑选值得重点观察的 layer / block
- 对理解 SDXL cross-attn 的时序变化很直观
- 对排查 prompt token 对齐是否合理也有帮助

如果把它定位成“可视化诊断工具”，是成立的。

## 8. 为什么没有进入当前主线

这是这份复盘最重要的一部分。

结合现在仓库的方向看，这个原型没有进入主线是合理的，主要原因如下：

### 8.1 它观测的是注意力，不是可干预的概念特征

当前主线 `exp53 / exp54 / exp55` 是：

- 用 SAE 在 block delta 上做特征分解
- 找到可排序、可黑名单、可复用的 feature id
- 再做 feature-level ablation / injection

而 `test_idea` 观测的是 token attention map。

两者的语义层级不同：

- attention map 更偏解释
- SAE feature 更偏可操作单元

### 8.2 指标不够稳定

不论是 IoU 还是 overlap，都很容易受到这些因素影响：

- 二值阈值
- token 切分
- 层的空间分辨率
- 是否包含 self-attn
- 独立归一化带来的量纲偏移

这导致它不太适合成为一个稳定的“概念定位指标”。

### 8.3 工程成本高，产物膨胀快

一次 `20 step × 多层 × 每步热图` 的实验就能产出上百张图。

在当前仓库里已经直接体现为：

- 单个输出目录 160M 到 190M 级别
- `group_mode="layer"` 时目录数量暴涨
- 文件名里还带空格，后续自动化处理不方便

这对于长期维护不是好信号。

### 8.4 依赖链偏脆弱

这个原型依赖：

- `attention_map_diffusers`
- 外部 hook 返回张量格式的隐式约定

相比之下，当前主线已经切到更可控的：

- `SDLens` hook pipeline
- SAE 编码与 delta 提取
- 统一 CLI 与配置对象

主线明显更稳定。

## 9. 最终结论

### 9.1 结论摘要

这次 `test_idea` 尝试验证了一个方向：

- “用 attention overlap 观察两个 prompt 概念的时序关系”是可实现的
- 结果在可视化层面是直观的
- 但它更适合做分析辅助，不适合直接作为概念擦除主线方法

### 9.2 具体判断

我的判断是：

- 它是一个成功的原型验证
- 但不是一个适合作为主线继续投入的实现路线

成功之处在于：

- 证明 attention hook + token 对齐 + 时序指标这套链路是能跑通的
- 产出了有解释性的图和时间序列

不足之处在于：

- 指标的因果意义不够强
- 可复用性不高
- 工程维护成本偏大

## 10. 如果以后还想继续这个方向，应该怎么改

如果未来想把这条线重新拾起来，不建议直接复活 `test_idea`，而应该重做成一个更轻的分析实验。建议如下：

1. 只保留 `cross-attn`
当前 token 对齐只对 cross-attn 语义最明确，`attn_kind="any"` 会把解释空间弄混。

2. 只保留 block 级分组
不要默认 `group_mode="layer"`，否则输出膨胀太严重。

3. 不默认保存逐 timestep 图片
先只保存 `metrics.csv` 和总曲线，按需回放个别关键时间步。

4. 不再把 overlap 当主指标
可以改成：
   - 峰值区域中心距离
   - top-k token heat concentration
   - 结合 baseline/control prompt 的差分 attention

5. 如果要接入主线，最好只当辅助诊断
例如：
   - 用它帮助判断 exp53 候选 block
   - 用它分析 exp54 干预前后的 cross-attn 变化
   - 而不是让它直接决定“擦除哪个概念”

## 11. 建议的仓库处理方式

基于这次复盘，推荐这样处理 `test_idea`：

- 保留这份日志
- 如无继续投入计划，可以删除 `test_idea/`
- 如果未来还想继续 attention 诊断方向，再从头在 `scripts/sdxl_wsae/experiments/` 里做一个正式、轻量、低产物版本

这样最符合现在仓库已经形成的主线结构。
