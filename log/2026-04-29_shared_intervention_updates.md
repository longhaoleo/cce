# 2026-04-29 Shared Intervention Updates

## 这次改了什么

围绕 Shared 主线最近的擦除实验，做了 3 个直接影响后续实验默认行为的修改：

1. `locator` 默认主分数从 `saeuron` 改为 `taris`
2. `erase / batch` 新增时间权重放大系数 `--int_time_weight_scale`
3. `erase / batch` 新增 `projected_ablation` 干预模式

目标很明确：

- 先减少 `saeuron` 爆分对概念定位的干扰
- 允许显式测试“把时间权重弄大一点会不会更有效”
- 给擦除侧补一个更接近“删概念子空间投影”的实验模式，看看能不能减轻副作用

## 代码位置

### 1. TARIS 作为默认主分数

- [runtime/shared/locator.py](/root/cce/runtime/shared/locator.py)

关键变化：

- `--taris_score_mode` 默认值从 `saeuron` 改成 `taris`

这样后面直接跑定位时，如果不显式传参，生成的：

- `top_positive_features.csv`
- `top_negative_features.csv`

都会优先来自 TARIS 排名。

## 为什么要改

前面 `car / dog` 的 Shared 概念表里，`taris@10` 和 `saeuron@10` 的重合几乎为 0。
同时 `saeuron` 仍然经常因为 `neg_sigma` 很小而给出几千到几万的分数，这不适合继续当默认主排序。

保留 `saeuron` 作为对比输出仍然有价值，但不应该继续做第一入口。

## 2. 时间权重放大系数

### 新参数

- `--int_time_weight_scale`

加入位置：

- [runtime/shared/erase.py](/root/cce/runtime/shared/erase.py)
- [runtime/shared/batch.py](/root/cce/runtime/shared/batch.py)

对应结构字段：

- `TimeInterventionConfig.weight_scale`
- `InterventionSpec.time_weight_scale`

## 为什么要加

此前 Shared 擦除里 `feature_time_scores.csv` 的 `diff` 量级通常很小，常见在：

- `1e-3`
- `1e-2`

而 hook 侧如果启用了 `from_csv`，会直接把当前 token 激活系数乘上这组时间权重。
结果就是：

- 打开时间权重以后，经常不是“更准”
- 而是“整体被压薄”

这次增加 `--int_time_weight_scale` 之后，可以显式试：

- `1.0`
- `2.0`
- `3.0`
- `5.0`

看时间门控到底是帮助概念擦除，还是只是继续削弱信号。

## 设计细节

时间权重放大做了两层接入：

1. hook 实际干预时会乘上 `time_weight_scale`
2. 多 block 自动平衡时，也会基于放大后的权重强度估算 `block_scale_map`

这样不会出现：

- 你把时间权重放大了
- 但 block 自动平衡还是按旧强度算

## 3. projected_ablation

### 新模式

- `--int_mode projected_ablation`

加入位置：

- [runtime/shared/erase.py](/root/cce/runtime/shared/erase.py)
- [runtime/shared/batch.py](/root/cce/runtime/shared/batch.py)

同时新增：

- `--int_projection_ridge`

默认值：

- `1e-4`

## 它和普通 ablation 的区别

普通 `ablation` 做的是：

- 选中特征
- 用当前 SAE 系数解出概念重建
- 直接从表示里减掉这部分重建

`projected_ablation` 做的是：

- 取选中特征对应的 decoder 方向
- 构造这个特征集合张成的子空间
- 把当前 `x_norm` 在该子空间上的最小二乘投影分量解出来
- 再减掉这个投影

实现上用了带 ridge 的 Gram 逆：

- `P = D (D^T D + λI)^-1 D^T`

这样在特征方向彼此不够正交时，比“直接用当前 SAE 编码系数重建”更接近：

- 只删子空间分量
- 少删一点额外非概念成分

## 为什么值得试

当前 Shared 擦除已经开始有效，但仍然存在一个明显问题：

- 一旦把干预打得足够强，原图扰动也比较大

所以现在值得测试的不是“能不能擦掉”，而是：

- 能不能在保持擦除效果的同时，减少对非概念结构的破坏

`projected_ablation` 就是针对这个问题加的实验模式。

## 当前推荐对比方式

建议下一轮至少做这 3 组对照：

1. `ablation + no time weight`
2. `ablation + time_weight_scale=3`
3. `projected_ablation + time_weight_scale=3`

建议保持其他参数一致，例如：

- `blocks = down.2.1 + up.0.0 + up.0.1`
- `int_feature_top_k = 5`
- `int_scale = 10`

## 推荐命令

### 单图：普通 ablation，无时间权重

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --targetconcept car \
  --prompt "a photo of a car on a city street, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_car_ablate_no_time \
  --blocks \
    unet.down_blocks.2.attentions.1 \
    unet.up_blocks.0.attentions.0 \
    unet.up_blocks.0.attentions.1 \
  --int_mode ablation \
  --int_feature_top_k 5 \
  --int_scale 10 \
  --no-int_use_time_weight
```

### 单图：普通 ablation，放大时间权重

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --targetconcept car \
  --prompt "a photo of a car on a city street, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_car_ablate_time3 \
  --blocks \
    unet.down_blocks.2.attentions.1 \
    unet.up_blocks.0.attentions.0 \
    unet.up_blocks.0.attentions.1 \
  --int_mode ablation \
  --int_feature_top_k 5 \
  --int_scale 10 \
  --int_use_time_weight \
  --int_time_weight_scale 3.0
```

### 单图：projected_ablation，放大时间权重

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --targetconcept car \
  --prompt "a photo of a car on a city street, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_car_projected_time3 \
  --blocks \
    unet.down_blocks.2.attentions.1 \
    unet.up_blocks.0.attentions.0 \
    unet.up_blocks.0.attentions.1 \
  --int_mode projected_ablation \
  --int_feature_top_k 5 \
  --int_scale 10 \
  --int_use_time_weight \
  --int_time_weight_scale 3.0 \
  --int_projection_ridge 1e-4
```

## 当前状态判断

这次不是“重新定义主线”的大重构，而是一次很明确的实验能力增强：

- 默认概念定位更稳
- 时间权重现在可以系统地做强弱消融
- 擦除侧第一次有了“子空间投影式”备选模式

这 3 个改动应该足以支持下一轮更系统的 Shared 擦除对比。
