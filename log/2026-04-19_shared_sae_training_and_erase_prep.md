# 实验记录：SharedSAE 训练收尾与擦除实验准备

日期：2026-04-19  
负责人：Codex / 用户协作

## 1. 目标与假设

- 目标：完成 SharedSAE 简化主线 `exp_c_adapter_align` 的正式训练收尾检查，并整理后续概念查找与图片级擦除实验的可执行入口。
- 假设：当前这轮训练即使在最终保存时因磁盘空间不足报错，仍然已经产出可用于后续 Shared 概念定位与擦除测试的有效 checkpoint。

## 2. 环境与版本

- 代码版本：`167bdb1`
- 训练输出目录：`train/output_exp_c_adapter_align`
- 有效 checkpoint：`train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400`
- 当前输出体积：`train/output_exp_c_adapter_align` 约 `5.8G`

## 3. 实验配置

- 训练主线：`exp_c_adapter_align`
- 结构重点：
  - 共享字典 `shared_core`
  - 启用 `in_adapter`
  - 启用对齐损失 `align`
  - 未启用 `time_branch`
  - 未启用 `spatial_branch`
  - `out_adapter` 不是当前主线依赖模块
- 当前正式入口命令：

```bash
python train/run_train.py \
  --output_root train/output_exp_c_adapter_align \
  --local_files_only \
  --steps 50 \
  --validation_prompts 1000 \
  --stage2_train_prompts 20000 \
  --stage1_train_prompts 5000 \
  --calibration_prompts 1000 \
  --num_step_buckets 5 \
  --shard_prompts 250 \
  --experiment_preset exp_c_adapter_align
```

## 4. 结果

- `stage2` 汇总：
  - `steps=25040`
  - `mean_total=0.5183`
  - `mean_recon=0.4862`
  - `mean_auxk=0.3615`
  - `mean_align=0.6554`
- `stage3` 汇总：
  - `steps=2504`
  - `global_step=27544`
  - `mean_total=0.4679`
  - `mean_recon=0.4475`
  - `mean_auxk=0.4263`
  - `mean_align=0.1401`
- 训练在 `stage3` 结束后的最终保存阶段报错，错误来自 `torch.save(...)` 写盘失败；直接原因是磁盘空间不够，而不是训练过程中的数值爆炸或模型结构错误。
- 虽然最后一次保存失败，但 `stage3` 指标已经写入 `metrics/stage_metrics.jsonl`，说明该阶段训练本身已完成。
- 之后用于 Shared 概念查找和擦除实验时，不应再使用旧路径：
  - `train/output_pilot_exp_b_shared_align`
  - `train/output_pilot_exp_c_adapter_align`
- 当前应统一改用：
  - `train/output_exp_c_adapter_align`
  - 且优先显式指定 `--ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400`

## 5. 关键判断

- 这轮模型没有“训没了”。可用模型权重已经存在，且最新可靠版本就是 `stage3_step_0027400`。
- 当前这轮训练能支持的结论是：
  - `shared_core + in_adapter + align` 这条简化线已经完成训练，可进入概念定位与擦除验证。
  - 它还不能证明时间分支、空间分支或更复杂干预策略已经成立，因为这轮并没有训练这些模块。
- `AuxK` 在当前训练里不是主判断指标。早期或小规模实验里它可能受 dead-feature 触发条件影响，不适合作为当前主线是否成功的首要依据。

## 6. 擦除实验建议入口

- 第一步先重新做 Shared 概念查找，保证概念统计和当前 checkpoint 对齐：

```bash
python scripts/run_shared_exp53.py \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --only dog \
  --taris_top_k 10 \
  --taris_score_mode saeuron
```

- 第二步再做图片级概念擦除测试，先用最保守配置验证“能不能打到表示”：

```bash
python scripts/run_shared_concept_erase.py \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --targetconcept dog \
  --prompt "a dog on the grass, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_dog \
  --int_feature_top_k 1 \
  --int_scale 10
```

- 初次测试不建议一上来就开：
  - `--int_use_time_weight`
  - `--int_use_spatial_weight`
  - `--int_spatial_mask`
  - `--use_out_adapter_for_decode`
- 原因：当前训练主线本身没有时间/空间分支，先验证单点 Shared feature 干预是否有可见效果更稳。

## 7. 结论

- 是否支持假设：支持。
- 原因：
  - 训练在保存时因磁盘写入失败中断，但在此之前已经完成 `stage3` 并落盘多个 checkpoint。
  - `stage3_step_0027400` 可作为当前最可靠的正式实验入口。
  - 当前工作重点应从“继续怀疑训练是否完全失败”切换为“基于有效 checkpoint 做概念定位与擦除因果验证”。

## 8. 下一步

- 用 `stage3_step_0027400` 重跑一次 `run_shared_exp53.py`，生成与当前模型一致的 `concept_dict`。
- 基于新的 `concept_dict` 做小规模单概念图片擦除，并检查干预日志里 `delta_over_x` 是否明显大于接近零的无效干预水平。
