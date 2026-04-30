# 2026-04-24 Turbo exp54 `car` 擦除实验：SAE 特征激活检查

## 背景

这条记录对应旧 `SAE Turbo / per-block SAE` 路线下的 `exp54` 擦除检查，目标是回答：

- 当前 `car` 擦除实验里，目标 SAE 特征到底有没有被激活；
- 各个 block 的激活强弱是否一致；
- “效果不好”更像是没有激活，还是激活存在但不够因果/不够持续。

本次结论基于以下诊断文件：

- `/root/cce/image_output/diag_intervention_unet.down_blocks.2.attentions.1.csv`
- `/root/cce/image_output/diag_intervention_unet.mid_block.attentions.0.csv`
- `/root/cce/image_output/diag_intervention_unet.up_blocks.0.attentions.0.csv`
- `/root/cce/image_output/diag_intervention_unet.up_blocks.0.attentions.1.csv`

以及对应概念统计文件：

- `/root/cce/concept_dict/down.2.1/car/top_positive_features.csv`
- `/root/cce/concept_dict/mid.0/car/top_positive_features.csv`
- `/root/cce/concept_dict/up.0.0/car/top_positive_features.csv`
- `/root/cce/concept_dict/up.0.1/car/top_positive_features.csv`

## 结论

### 1. 目标 SAE 特征不是“完全没激活”

当前 `car` 擦除实验里，目标特征有明显激活，不是“找不到特征”或“hook 没生效”。

从诊断 CSV 看：

- `up_blocks.0.attentions.1` 激活最强
- `down_blocks.2.attentions.1` 和 `mid_block.attentions.0` 中等
- `up_blocks.0.attentions.0` 很弱

所以问题不是“完全没有 SAE 响应”，而是“不同 block 的响应质量差异很大”。

### 2. 最强激活来自 `up_blocks.0.attentions.1`

见：

- `/root/cce/image_output/diag_intervention_unet.up_blocks.0.attentions.1.csv`

观察：

- `mean_abs_c_base` 大约在 `0.92 -> 2.66`
- `delta_over_x` 最高达到约 `0.29`

这说明：

- 这层目标特征确实在当前图像中活跃
- 干预量也实际打进了表示

这是当前四层里最值得保留继续试的 block。

### 3. `down` 和 `mid` 有激活，但不算特别强

见：

- `/root/cce/image_output/diag_intervention_unet.down_blocks.2.attentions.1.csv`
- `/root/cce/image_output/diag_intervention_unet.mid_block.attentions.0.csv`

观察：

- `down.2.1` 的 `mean_abs_c_base` 约 `0.5 -> 1.2`
- `mid.0` 的 `mean_abs_c_base` 约 `0.36 -> 1.04`
- `delta_over_x` 大致在 `0.01 -> 0.06` 与 `0.014 -> 0.042`

这说明：

- 这两层不是无效层
- 但作用强度明显弱于 `up.0.1`

### 4. `up_blocks.0.attentions.0` 对当前 `car` 擦除基本没贡献

见：

- `/root/cce/image_output/diag_intervention_unet.up_blocks.0.attentions.0.csv`

观察：

- 首步 `mean_abs_c_base` 只有 `0.0064`
- 后续大致也只是 `0.15 -> 0.26`
- `delta_over_x` 基本只有 `0.0006 -> 0.0115`

结论：

- 这层的目标特征激活很弱
- 当前把它混进多 block 擦除，只会增加噪声和解释复杂度

### 5. 激活只出现在前半程，不是全程持续

四个诊断文件都显示：

- 在 `timestep=951 -> 551` 之间，`active=1`
- 从 `timestep=501` 开始，`active=0`

这说明：

- 当前这轮干预只覆盖了前半段
- 后半程模型没有继续对这组概念特征做改动

对“擦除 car”这种目标来说，这通常不够，因为后半程还可能把概念重新补回来。

## 解释

所以这轮“效果不好”的更合理解释是：

- 特征确实有激活
- 但激活强度在不同 block 上差异很大
- 有效干预主要集中在 `up.0.1`
- 干预只发生在前半程

因此问题更像：

- block 选择不够干净
- 时间窗不够长
- 某些 top-k 特征不够稳定或不够因果

而不是：

- SAE 完全没学到
- 或干预完全没生效

## 后续建议

如果继续沿旧 Turbo 路线排查，优先级应为：

1. 只保留强层，例如先只试 `unet.up_blocks.0.attentions.1`
2. 干预窗口拉满到全程，而不是只到 `t=550`
3. 先关闭时间权重，避免把本来就薄的特征再压小
4. 暂时去掉 `unet.up_blocks.0.attentions.0`

但从当前实验质量看，更合理的下一步仍然是切到 `SharedSAE` 路线继续验证。
