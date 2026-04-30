# 时间分支 + decoder decorrelation 第三轮 smoke 结果

## 实验对象

- 输出目录：`train/output_time_decorr_stage123_pilot_v3`
- 相对第二轮 `pilot_v2` 的主要改动：
  - `decoder_decorr_weight`
    - 从 `1e-2`
    - 提高到 `5e-2`

## 结果摘要

### 1. decorrelation 项更强了，但结果没有变好

尾段 `loss_decoder_decorr_term`：

- `pilot_v2`: `~5.10e-05`
- `pilot_v3`: `~2.43e-04`

### 2. dead feature 反而回升

- `stage3 tail100 dead_feature_frac`
  - `pilot_v2`: `0.3759`
  - `pilot_v3`: `0.3961`

### 3. recon 和 align 没有改善

- `stage3 mean_recon`
  - `pilot_v2`: `0.5317`
  - `pilot_v3`: `0.5338`

## 路径对照补充

### Path1：降低 `time branch` 学习率

- 输出目录：`train/output_time_decorr_stage123_pilot_path1_low_lr_time`
- `stage3 tail100 dead_feature_frac ≈ 0.3766`

### Path2：去掉 `stage1`

- 输出目录：`train/output_time_decorr_stage23_pilot_path2_no_stage1`
- `stage3 tail100 dead_feature_frac ≈ 0.3421`

结论：

- `no stage1` 明显优于只降低 `lr_time`
- 主问题更像是 `stage1 + time_branch` 的组合过早把共享特征空间带偏

## 更新后的下一步

下一步优先沿 `path2` 继续：

- `no stage1`
- 保留 `stage2 + stage3`
- 多 block 联合训练
- 把训练规模从 smoke 放大到 `half`
