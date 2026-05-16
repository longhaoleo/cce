# NSFW Selective Erasure Research Plan

日期：2026-05-15

## 1. 当前判断

下一阶段主线不应继续默认追随 `decorr03`，而应回到目前擦除效果更好的 checkpoint：

```text
train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772
```

当前证据支持：

- `x8/top20_half` 更接近真正有用的 concept erasure：
  - 能擦除目标概念；
  - 图像更容易继续保持可读；
  - 多个 case 会自动补出合理的安全替代内容。
- `decorr03` 是有价值的负例：
  - latent decorrelation 指标更好；
  - 但 `nudity` 这种复合属性的擦除变差；
  - 说明“更低相关性”不等于“更好的 selective erasure”。

因此，下一阶段的核心问题不再是：

> 如何让 SAE 更独立？

而是：

> 如何用 SharedSAE 找到真正对 NSFW 有因果作用、同时对无关概念副作用最小的 feature，并把生成轨迹引向合理的安全替代？

## 2. 论文定位

`SharedSAE` 可以作为方法创新成立，但论文不能只写成：

> 用 SharedSAE 擦除 NSFW。

更强的论文主张应是：

> 跨 block、跨 denoising time 共享的稀疏语义表示，能够支持可解释、低副作用、可评估的 NSFW selective editing。

建议暂定三条贡献：

1. **Shared sparse concept paths**
   - 用统一 feature id 描述 NSFW 语义在多个 block 和多个 timestep 上的路径，而不是只看单层 neuron。
2. **Selectivity-aware feature discovery**
   - 不只找“对 NSFW 最强”的 feature，而是找“目标强、generic 弱、benign neighbor 弱、因果副作用小”的 feature。
3. **Trajectory-aware safe replacement**
   - 不只做 `ablation`，而是研究如何把 unsafe trajectory 推向 matched safe counterpart，减少空洞化和语义重写。

## 3. 同行当前在解决什么

| 方向 | 代表工作 | 他们重点解决的问题 | 对本项目的含义 |
| --- | --- | --- | --- |
| 经典 concept erasure | ESD | 从权重中删除目标概念 | 只是能擦掉已经不够 |
| mass erasure / generality-specificity | MACE | 同时擦多个概念，并兼顾同义概念与无关概念 | 需要报告 specificity，而不只是 efficacy |
| utility-preserving erasure | EraseDiff | 在删除目标信息时保留模型原有能力 | 必须把“非目标保留”作为主指标 |
| fine-grained retention | FADE | 保护邻近概念，避免误伤相似语义 | NSFW 尤其需要 benign neighbor benchmark |
| localized / training-free erasure | GLoCE | 强调局部性、图像 fidelity、specificity、robustness | 论文需要证明不是大范围破坏 |
| adversarial robustness | STEREO、TRCE | 对抗改写、隐式 prompt、embedding attack 下仍能擦除 | NSFW 论文不能只测显式 prompt |
| benchmark / spillover | EraseBench、Holistic Unlearning Benchmark | 系统测非目标退化、鲁棒性和多面向能力 | 需要完整 benchmark，不可只报少量视觉例子 |
| SAE-based erasure（当前主要是预印本） | SAEmnesia、Sparse Autoencoder as a Zero-Shot Classifier、A Single Neuron Works | 用 SAE 做定位、绑定或单 neuron 擦除 | SharedSAE 必须清楚证明“共享、时序、低副作用”这三个差异点 |

### 对同行工作的归纳

当前领域的共同趋势不是“擦得更狠”，而是同时追求：

- efficacy；
- specificity；
- prior preservation；
- robustness；
- fine-grained control；
- 更系统的 benchmark。

如果本项目只展示 `nudity` 被擦掉，标准不够。要进入论文级别，必须证明：

- 为什么 SharedSAE 不是普通 SAE 的替代实现；
- 为什么它能给出更好的机制理解；
- 为什么它能在同等 NSFW suppression 下减少 collateral damage。

## 4. 下一阶段的核心假设

### H1. 旧主线 checkpoint 的“自动替代”能力是真实优势

与 `decorr03` 相比，`x8/top20_half` 可能没有最漂亮的 disentanglement proxy，但更保留了安全编辑所需的语义通路。

需要验证：

- 它是否在同等 suppress 条件下有更低的图像漂移；
- 它是否更容易生成 safe counterpart；
- 它的 feature 是否在 benign neighbor 上更少误伤。

### H2. NSFW 不应被当作单一概念

`nudity` 至少包含：

- `full_body_unclothed`
- `bare_torso`
- `breast_exposure`
- `classical_nude`
- `nude_silhouette`

不同子概念的 feature 干净程度不同。后续要先分析它们，而不是急着把它们硬合成一个父概念。

### H3. 最优 feature 不是 raw TARIS 最高，而是 causal selectivity 最好

一个 feature 是否值得干预，应同时考虑：

- 目标概念提升；
- generic prompt 上是否常亮；
- benign neighbor 上是否误亮；
- 真正 ablate 后对目标与非目标的因果影响。

### H4. 只做 ablation 可能不是最优 NSFW 干预

如果模型已经自然倾向把不安全内容替换成衣物、遮挡物或安全构图，那么应该研究：

- 这个 safe replacement shift 是否能显式估计；
- 是否能比 pure ablation 更好地保住原 prompt 语义。

## 5. 必须建立的 benchmark

下一阶段先不要继续大规模训练，先把 benchmark 建立起来。

### 5.1 Prompt 分组

| split | 用途 | 需要准备 |
| --- | --- | --- |
| `target_explicit` | 测显式 NSFW 擦除 | 直接裸露、明确 sexual exposure prompt |
| `target_implicit` | 测隐式表达 | 不出现 `nude` 但语义等价的 prompt |
| `target_subconcepts` | 测细粒度覆盖 | 5 个子概念各自一组 |
| `benign_neighbors` | 测误伤 | swimsuit、shirtless、classical art、beachwear、dancewear、pregnancy/body silhouette 等 |
| `generic_preservation` | 测普通生成能力 | COCO / 普通人物 / 普通场景 |
| `adversarial` | 测鲁棒性 | paraphrase、隐喻、jailbreak、prompt mutation |

### 5.2 你需要做的人工工作

1. 先手工整理每个 split 的 prompt。
2. 每个 split 至少保留：
   - quick set：20 条，用于日常迭代；
   - paper set：100 条以上，用于正式表格。
3. 对 `benign_neighbors` 不要偷懒：
   - 它决定论文能否证明“不是把人体相关概念一起毁掉”。
4. 对 `target_implicit / adversarial` 做版本控制：
   - prompt 不要在实验中途改；
   - 否则旧结果无法比较。

## 6. 需要报告的指标

### 6.1 主表必须有的指标

| 目标 | 指标 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| NSFW efficacy | `NSFW detection rate` / `target suppression rate` | 已补第一版 | `evaluation.eval_nsfw` 当前用 NudeNet；后续仍需正式 benchmark |
| 目标语义压制 | `CLIP target_prob_drop`、`target_margin_drop` | 已有 | 现有 `evaluation.eval_clip` 可用 |
| Prompt 保留 | `delta_clip_prompt_logit` | 已有 | 越接近 0 越好 |
| 感知保真 | `LPIPS` | 已有 | 适合同 seed 原图对照 |
| 语义保真 | `DreamSim` | 已有 | 比像素指标更重要 |
| 非目标保留 | `benign-neighbor retention rate` | 需要补 | 对邻近但安全概念单独测 |
| 泛化能力 | `generic prompt CLIP / quality retention` | 部分已有 | 在 COCO / generic split 上测 |
| 鲁棒性 | `attack success rate` 或 `unsafe generation rate under attack` | 需要补 | 对 implicit / jailbreak split 测 |
| safe replacement | `safe replacement success rate` | 需要补 | 本项目最有区分度的新指标 |

### 6.2 机制分析指标

| 目的 | 指标 |
| --- | --- |
| feature 是否过泛 | `generic_active_ratio`、`generic_mean_activation` |
| feature 是否对目标专一 | `target_lift = pos - matched_neg` |
| feature 是否误伤邻近概念 | `neighbor_lift` / `neighbor_activation` |
| feature 是否真正因果有效 | `causal_target_drop`、`causal_preservation_cost` |
| 子概念是否分开 | 子概念 top-k Jaccard overlap |
| 时间上是否集中 | feature temporal profile、peak timestep、temporal entropy |
| 层间是否共享 | cross-block feature overlap、共享 feature 的 block coverage |

### 6.3 不应当单独作为论文主指标的东西

- `pixel_l1 / mse / psnr`
  - 只适合作为 debugging；
  - 不能单独说明语义保持。
- `diag_delta_over_x`
  - 只能说明干预打进去了；
  - 不能说明擦除有效或安全。
- 训练中的 `latent_decorr`
  - 只是 surrogate；
  - 当前实验已经说明它不等价于 selective erasure quality。

## 7. 现有代码能做什么，缺什么

### 7.1 已经可直接复用

| 能力 | 位置 |
| --- | --- |
| feature frequency / blacklist | `tools/feature_frequency/` |
| concept locator | `runtime/shared/locator.py` |
| batch erasure | `runtime/shared/batch.py` |
| pixel / diag | `evaluation/eval_pixel_diag.py` |
| CLIP target / prompt | `evaluation/eval_clip.py` |
| LPIPS | `evaluation/eval_lpips.py` |
| DreamSim | `evaluation/eval_dreamsim.py` |

### 7.2 需要补的新模块

1. `evaluation/eval_nsfw.py`
   - 已补第一版 NudeNet 后端；
   - 输入 batch 输出；
   - 输出 NSFW detector 分数、阈值命中率、per-split summary。
2. `evaluation/eval_neighbor_retention.py`
   - 对 benign neighbor split 做 target-specific classifier / CLIP retention。
3. `evaluation/eval_replacement.py`
   - 判断是否从 unsafe concept 转向 matched safe counterpart。
4. `tools/analyze_feature_selectivity.py`
   - 汇总：
     - `target_lift`
     - `generic_active_ratio`
     - `neighbor_activation`
     - 子概念 overlap
5. `data/nsfw_benchmark/`
   - 固定所有 prompt split。

## 8. 推荐实验顺序

### Phase A. 先把当前事实讲清楚

**目标**：确认真正值得作为主线的 checkpoint。

实验：

1. 比较：
   - `x8/top20_half`
   - `decorr03`
2. 保持完全一致：
   - prompt
   - seed
   - blacklist：默认 `ar95_all`
   - top-k
   - scale
   - blocks
3. 输出：
   - NSFW efficacy
   - prompt preservation
   - LPIPS / DreamSim
   - safe replacement rate

**成功标准**

- 如果 `x8/top20_half` 在相近 efficacy 下 preservation 更好，则冻结它为主线；
- `decorr03` 保留为 ablation，不继续追加训练。

### Phase B. 做 feature selectivity study

**目标**：把“哪些 feature 值得擦”从经验判断变成可量化结论。

实验：

1. 对 5 个 nudity 子概念分别定位；
2. 对每个 candidate feature 统计：
   - `target_lift`
   - `generic_active_ratio`
   - `neighbor_activation`
   - 单 feature ablation 的 `target_drop / preservation_cost`
3. 画图：
   - `target_drop` vs `preservation_cost`
   - `generic_active_ratio` vs `causal_target_drop`
   - 各子概念 feature overlap heatmap
   - timestep profile heatmap

**成功标准**

- 能明确指出一批：
  - target effect 高；
  - generic / neighbor effect 低；
  - preservation cost 低；
  的 feature。

### Phase C. 做 NSFW selective erasure 主实验

**目标**：证明 SharedSAE 不只是能擦，而是能更有选择地擦。

对比：

- pure ablation
- stricter blacklist
- selectivity-aware feature selection
- baseline methods

主结果表：

- `target_explicit`
- `target_implicit`
- `target_subconcepts`
- `benign_neighbors`
- `generic_preservation`

**成功标准**

- 在相近 NSFW suppression 下：
  - benign neighbor retention 更高；
  - DreamSim / prompt CLIP 更好；
  - 图像副作用更小。

### Phase D. 做 safe replacement pilot

**目标**：把当前偶然观察到的“自动替代”变成明确方法。

最小版本：

1. 准备 matched pair：
   - unsafe：`unclothed adult`
   - safe：`fully clothed adult`
2. 计算 paired trajectory shift；
3. 对比：
   - pure ablation
   - safe shift
   - ablation + safe shift

指标：

- NSFW detection drop；
- safe counterpart alignment；
- prompt preservation；
- DreamSim；
- 人工小样本偏好评测。

**成功标准**

- safe replacement 至少在一部分 split 上能比 pure ablation 更少破坏原图。

### Phase E. 做论文级鲁棒性与 baseline

**目标**：补齐顶会审稿人会追问的部分。

需要：

- explicit / implicit / adversarial prompts；
- 对强 baseline 的统一比较；
- 至少一组 attack / paraphrase robustness；
- 失败案例与边界分析。

## 9. 你接下来具体要做什么

### 立即做

1. 冻结主线 checkpoint：
   - 暂定 `x8/top20_half` 为主线；
   - `decorr03` 只做对照。
2. 建 `data/nsfw_benchmark/`：
   - 先整理 quick set；
   - 不再只依赖当前 `nudity.csv`。
3. 补 NSFW detector 评测：
   - 这是下一阶段最先缺的硬指标。
4. 用同一套 benchmark 重跑：
   - `x8/top20_half`
   - `decorr03`

### 接着做

5. 做 `feature selectivity` 分析脚本。
6. 先不继续改训练 loss，先回答：
   - 哪些 feature 真正低副作用；
   - 旧 checkpoint 为什么更会 replacement。
7. 选一组最稳的 feature selection 规则，重跑 NSFW 主实验。

### 暂时不要做

- 不要继续无目标扫 `latent_decorr_weight`；
- 不要只凭 `diag_delta_over_x` 判断方法优劣；
- 不要急着把 5 个 nudity 子概念强行合并成一个父概念；
- 不要在 benchmark 固定之前开始大量 baseline 复现。

## 10. 最终论文应有的图表

### 主表

1. NSFW suppression + preservation + robustness 总表
2. Benign neighbor retention 表
3. Explicit / implicit / adversarial 分 split 表

### 分析图

4. Shared feature path over blocks and time
5. Feature selectivity scatter：
   - `causal_target_drop` vs `preservation_cost`
6. 子概念 overlap heatmap
7. old checkpoint vs `decorr03` 机制对比
8. safe replacement qualitative grid

### 消融表

9. no blacklist
10. loose vs default vs strict blacklist
11. no time branch
12. no shared dictionary
13. pure ablation vs safe replacement
14. `x8/top20_half` vs `decorr03`

## 11. 这条路线做到什么程度，才算论文成熟

### 还不能投稿的状态

- 只有 `nudity` 几张图；
- 只有 CLIP；
- 没有 benign neighbor；
- 没有 adversarial / implicit；
- 没有 baseline；
- 不知道为什么旧 checkpoint 更好。

### 可以写成完整论文的状态

- 有固定 benchmark；
- 有 NSFW efficacy、preservation、neighbor retention、robustness；
- 有 SharedSAE 机制分析；
- 有强 baseline；
- 有一条明确优于 pure ablation 的 intervention；
- 能解释：
  - 什么 feature 值得擦；
  - 为什么 SharedSAE 帮助低副作用；
  - 为什么某些更“解耦”的训练反而伤害 selective erasure。

## 12. 当前最重要的决策

下一步不要把主精力继续放在“再训一个 SAE”。

优先级应改成：

```text
benchmark > evaluation > mechanism > intervention > retraining
```

只有当 benchmark 明确告诉我们：

- 当前 feature 选择已经受限于表示质量；
- 而不是受限于定位与干预策略；

再回头训练下一版 SAE，才有意义。

## 13. 参考工作

### 已发表主线

- Gandikota et al., **Erasing Concepts from Diffusion Models**, ICCV 2023.
- Schramowski et al., **Safe Latent Diffusion**, CVPR 2023.
- Lu et al., **MACE: Mass Concept Erasure in Diffusion Models**, CVPR 2024.
- Wu et al., **EraseDiff**, CVPR 2025.
- Thakral et al., **FADE**, CVPR 2025.
- Lee et al., **GLoCE**, CVPR 2025.
- Srivatsan et al., **STEREO**, CVPR 2025.
- Wang et al., **Precise, Fast, and Low-cost Concept Erasure in Value Space**, CVPR 2025.
- Chen et al., **TRCE**, ICCV 2025.
- Amara et al., **Erasing More Than Intended?**, ICCV 2025.
- Moon et al., **Holistic Unlearning Benchmark**, ICCV 2025.

### 与本项目最接近的 SAE 预印本

- **Sparse Autoencoder as a Zero-Shot Classifier for Concept Erasing in Text-to-Image Diffusion Models**.
- **SAEmnesia: Erasing Concepts in Diffusion Models with Sparse Autoencoders**.
- **A Single Neuron Works: Precise Concept Erasure in Text-to-Image Diffusion Models**.
