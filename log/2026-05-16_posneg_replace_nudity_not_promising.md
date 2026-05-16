# 2026-05-16 PosNeg Replace Nudity Not Promising

## 实验对象

测试一个最直接的同概念替换思路：

- 去掉 `top_positive_features.csv`
- 注入 `top_negative_features.csv`

脚本入口：

- `python -m runtime.shared.posneg_replace`

目标是验证：

- 同一概念内部的 `positive -> negative` 是否可以近似做“反向替换”

## 结果

当前结果不理想，不再继续作为主线方向推进。

### 1. `decorr03` 目录下的这次运行没有完整产出

目录：

- `image_output/decorr03/posneg_replace_nudity`

现状：

- 只有 `intervention_baseline.png`
- 没有 `intervention_steered.png`
- 没有 `run_manifest.json`

因此这次 `decorr03` 运行本身不能作为有效评估样本。

### 2. 完整跑通的是 `sae_x8_time` 版本，但结果明显崩坏

目录：

- `image_output/sae_x8_time/posneg_replace_nudity`

对应参数：

- `ckpt_dir = train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772`
- `int_scale = 3000`
- `int_inject_scale = 100`
- `mode = replace`

直接观察结果可以判断：

- 图像不是局部替换
- 也不是合理覆盖
- 而是明显出现整体纹理污染和语义崩坏

## 数值诊断

按 block 看，注入分支远大于正特征擦除分支：

- `down.2.1`
  - `mean_abs_delta_x ≈ 0.102`
  - `mean_abs_delta_x_inject ≈ 5.375`
- `mid.0`
  - `mean_abs_delta_x ≈ 0.0062`
  - `mean_abs_delta_x_inject ≈ 5.598`
- `up.0.0`
  - `mean_abs_delta_x ≈ 0.00225`
  - `mean_abs_delta_x_inject ≈ 1.317`
- `up.0.1`
  - `mean_abs_delta_x ≈ 0.0`
  - `mean_abs_delta_x_inject ≈ 0.124`

这说明：

- 即使 nominal `int_inject_scale` 只有 `100`
- 实际写回 latent 的负特征注入影响仍显著大于正特征擦除

所以问题不只是“强度稍微偏大”，而是方法方向本身不稳。

## 原因判断

当前更合理的解释是：

1. `top_negative_features` 不是安全的“反概念”
2. 它更像一组与目标概念打分相反、但语义上混杂的方向
3. 把这些方向直接注入，会把 latent 往很多不相关纹理/结构上推
4. 最终效果更像 adversarial-style texture rewrite，而不是可控替换

换句话说：

```text
positive ablation + negative injection
```

目前并不能成立为稳定的概念替换方案。

## 当前结论

这条路线先停止，不作为主线继续做。

保留的结论是：

- `posneg_replace` 可以留作一次失败尝试记录
- 但不再投入时间做系统调参

## 后续方向

后续替换实验仍以这两类为主：

1. 显式 replacement concept
   - 例如 `nudity -> cloth`
   - 后续再收窄成 `covered_torso / covered_chest`
2. 更局部的注入机制
   - 只在目标概念高激活区域附近注入
   - 不做全图负特征写入
