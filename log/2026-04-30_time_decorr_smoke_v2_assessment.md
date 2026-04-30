# 时间分支 + decoder decorrelation 第二轮 smoke 结果

## 实验对象

- 输出目录：`train/output_time_decorr_stage123_pilot_v2`
- 相对第一轮 `pilot` 的主要改动：
  - `decoder_decorr_weight`
    - 从 `3e-4`
    - 提高到 `1e-2`

## 结果摘要

### 1. time branch 正常参与训练

`lr_time` 为非零：

- `stage2`: `1e-4`
- `stage3`: `2e-5`

### 2. recon 略有改善

- `stage3 mean_recon`
  - `pilot`: `0.5333`
  - `pilot_v2`: `0.5317`

### 3. dead feature 略有下降，但仍然过高

- `stage3 tail100 dead_feature_frac`
  - `pilot`: `0.3918`
  - `pilot_v2`: `0.3759`

### 4. decoder decorrelation 开始产生作用，但权重仍偏弱

尾段 `loss_decoder_decorr_term`：

- `pilot`: `~1.58e-06`
- `pilot_v2`: `~5.10e-05`

## 结论

第二轮 smoke 比第一轮略好，但整体仍然是：

- 方向正确
- 问题未解决
- 暂不建议直接上 full
