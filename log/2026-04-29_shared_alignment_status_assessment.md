# 2026-04-29 SharedSAE Alignment Status Assessment

## 结论先说

当前这版 SharedSAE：

- **重建/可干预性：已经基本达到了“能用”**
- **跨层共享 feature id 对齐：还没有真正站稳**

也就是说，它不是“没学到东西”，但也还不能很有把握地说：

- 同一个 global feature id
- 在不同 block 上
- 已经稳定承载了可比较的相同语义

## 这次判断基于什么

### 1. 训练损失层面：模型是能学的

从 [stage_metrics.jsonl](/root/cce/train/output_exp_c_adapter_align/metrics/stage_metrics.jsonl) 看：

- `stage2 mean_recon ≈ 0.486`
- `stage3 mean_recon ≈ 0.448`

说明训练并没有崩，重建能力是在正常提升的。

从 [step_metrics.jsonl](/root/cce/train/output_exp_c_adapter_align/metrics/step_metrics.jsonl) 尾部看：

- `loss_recon` 常见在 `0.43 ~ 0.53`
- `dead_feature_frac` 常见在 `0.013` 左右
- `latent_active_frac = 0.001953125`

这说明：

- 稀疏激活是正常存在的
- 死特征比例不高
- 不是“整个字典没工作”

### 2. 实验现象层面：已经能做出有效擦除

前面 Shared 擦除实验已经确认：

- 选对 block 和参数以后
- `car` 等概念是可以开始被擦掉的

这说明 SharedSAE 至少已经学到了一批**可重建、可操控**的特征方向。

如果模型完全没学到，后面不可能进入“能擦除但副作用偏大”这个阶段。

## 但为什么说“对齐还不够”

最关键的证据是：

- 同一个 concept 在不同 block 上的 top TARIS 特征
- 几乎没有 feature id 重合

我检查了 `car` 和 `dog`：

- [concept_dict/down.2.1/car/top_positive_features_taris.csv](/root/cce/concept_dict/down.2.1/car/top_positive_features_taris.csv)
- [concept_dict/mid.0/car/top_positive_features_taris.csv](/root/cce/concept_dict/mid.0/car/top_positive_features_taris.csv)
- [concept_dict/up.0.0/car/top_positive_features_taris.csv](/root/cce/concept_dict/up.0.0/car/top_positive_features_taris.csv)
- [concept_dict/up.0.1/car/top_positive_features_taris.csv](/root/cce/concept_dict/up.0.1/car/top_positive_features_taris.csv)

以及 `dog` 对应的同类文件。

结果是：

- 四层两两之间的 `top20` overlap 基本都是 `0`

这意味着什么：

- 共享字典虽然存在
- 但不同层仍然更像在使用“各自的一组 feature id”表达概念
- 而不是把同一概念自然收拢到同一批 global id 上

## 对齐损失有没有学到

有，但还不够强。

从 [stage_metrics.jsonl](/root/cce/train/output_exp_c_adapter_align/metrics/stage_metrics.jsonl) 看：

- `stage2 mean_align ≈ 0.655`
- `stage3 mean_align ≈ 0.140`

说明对齐项确实在下降，不是完全无效。

但看 step 级别进入总损失的量级：

- `loss_align_term ≈ 0.005 ~ 0.009`
- 同时 `loss_recon ≈ 0.43 ~ 0.53`

也就是说：

- align 有影响
- 但它对总训练目标的牵引力仍偏弱
- 不足以强迫不同层把同一概念收拢到共享的 feature id 上

## 当前状态应该怎么评价

如果目标是：

### A. 训出一个能重建、能支持概念擦除的 SharedSAE

那当前已经算**部分成功，而且是有实质进展的成功**。

### B. 训出一个“同一 feature id 跨层语义对齐明显”的 SharedSAE

那当前只能算：

- **部分达成**
- **但还没有真正达到想要的对齐强度**

## 这次判断里还有一个重要细节

`mid.0` 目前对 `car/dog` 这类物体概念本来就偏弱。

所以不要把：

- `mid.0` 表现差

直接等同于：

- 整个 Shared 对齐彻底失败

更合理的看法是：

- `down.2.1`
- `up.0.0`
- `up.0.1`

已经有可用信号；

- `mid.0`

暂时不适合作为“对齐是否成功”的主证据层。

## 后续建议

如果下一阶段主要目标是“把共享 feature id 对齐真正推起来”，比起继续加复杂擦除技巧，更优先的是：

1. 提高 align 的真实影响力  
   例如尝试更高的 `align_weight_target`

2. 做一个显式的“跨层 top-k overlap”评估  
   不要再只靠主观感觉判断 feature id 是否对齐

3. 先重点观察 `down.2.1` 与 `up.*` 的跨层一致性  
   暂时不要让 `mid.0` 主导结论

4. 把“能擦除”和“id 是否对齐”分成两个不同指标  
   两者相关，但不是同一件事

## 当前一句话总结

这版 SharedSAE：

- **已经足够说明共享训练不是失败的**
- **但还不足以说明“共享 feature id 跨层对齐”已经达标**

用户当前“感觉 feature id 不太对齐”的判断，是合理且有结果支持的。
