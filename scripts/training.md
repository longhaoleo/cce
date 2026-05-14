# Training Experiments

当前训练主线只保留 `stage2 -> stage3`：

- `stage2`：四个 block 联合训练。
- `stage3`：同一批训练数据上低学习率微调。
- 不再保留 `stage1` / `stage4` 入口。
- 不再保留 `decoder_decorr_weight`；独立性实验只使用 latent decorrelation。

## Smoke

```bash
cd /root/cce

PRESET=exp_a_shared_recon
OUT=train/output_smoke_${PRESET}
python train/run_smoke_train.py \
  --output_root "$OUT" \
  --local_files_only \
  --validation_prompts 50 \
  --stage2_train_prompts 500 \
  --calibration_prompts 100 \
  --experiment_preset "$PRESET"
```

## Run Now: Strong Independence Ablation

这条是当前要跑的新实验命令。它不覆盖当前论文主模型，而是单独验证更强 latent decorrelation 是否让 feature 更干净。

```bash
cd /root/cce

python train/run_train.py \
  --experiment_preset custom \
  --output_root train/output_time_latentdecorr_x8_top20_decorr03 \
  --local_files_only \
  --steps 50 \
  --validation_prompts 500 \
  --stage2_train_prompts 10000 \
  --calibration_prompts 500 \
  --num_step_buckets 5 \
  --shard_prompts 250 \
  --expansion_factor 8 \
  --top_k 20 \
  --auxk 512 \
  --use_time_branch \
  --time_branch_mode sincos_linear \
  --time_branch_warmup_start_ratio 0.3 \
  --time_branch_warmup_ratio 0.3 \
  --no-use_spatial_branch \
  --run_stage3 \
  --use_block_in_adapter \
  --align_weight_target 0.1 \
  --latent_decorr_weight 0.3 \
  --latent_decorr_top_k 512 \
  --group_bs 0 \
  --save_every_steps 0
```

训练中重点看：

- `loss_latent_decorr_term` 是否从 `~1e-4` 提升到 `~1e-3` 或更高。
- `val_recon` 是否明显变差。
- `dead_feature_frac` 是否明显升高。

## Pilot

```bash
cd /root/cce

PRESET=exp_b_shared_align
OUT=train/output_pilot_${PRESET}
python train/run_train.py \
  --output_root "$OUT" \
  --local_files_only \
  --validation_prompts 200 \
  --stage2_train_prompts 2000 \
  --calibration_prompts 200 \
  --num_step_buckets 5 \
  --shard_prompts 100 \
  --experiment_preset "$PRESET"
```

## Mainline: Larger SAE + Delayed Time Branch

当前论文主模型对应这组配置。

```bash
cd /root/cce

python train/run_train.py \
  --experiment_preset custom \
  --output_root train/output_time_latentdecorr_x8_top20_half \
  --local_files_only \
  --steps 50 \
  --validation_prompts 500 \
  --stage2_train_prompts 10000 \
  --calibration_prompts 500 \
  --num_step_buckets 5 \
  --shard_prompts 250 \
  --expansion_factor 8 \
  --top_k 20 \
  --auxk 512 \
  --use_time_branch \
  --time_branch_mode sincos_linear \
  --time_branch_warmup_start_ratio 0.3 \
  --time_branch_warmup_ratio 0.3 \
  --no-use_spatial_branch \
  --run_stage3 \
  --use_block_in_adapter \
  --align_weight_target 0.1 \
  --latent_decorr_weight 0.01 \
  --latent_decorr_top_k 256 \
  --save_every_steps 0
```

## Reference: Strong Independence Ablation Details

这轮不替换当前主模型，目标是验证更强独立性正则是否能让 feature 更干净、减少擦除副作用。

与当前主模型相比，只改两点：

- `latent_decorr_weight: 0.01 -> 0.3`
- `latent_decorr_top_k: 256 -> 512`

观察重点：

- `loss_latent_decorr_term` 是否从 `~1e-4` 提升到 `~1e-3` 或更高。
- `val_recon` 是否明显变差。
- `dead_feature_frac` 是否明显升高。
- `car / dog / nudity` 的擦除是否更干净，尤其 dog 的头部/身体覆盖是否改善。

```bash
cd /root/cce

python train/run_train.py \
  --experiment_preset custom \
  --output_root train/output_time_latentdecorr_x8_top20_decorr03 \
  --local_files_only \
  --steps 50 \
  --validation_prompts 500 \
  --stage2_train_prompts 10000 \
  --calibration_prompts 500 \
  --num_step_buckets 5 \
  --shard_prompts 250 \
  --expansion_factor 8 \
  --top_k 20 \
  --auxk 512 \
  --use_time_branch \
  --time_branch_mode sincos_linear \
  --time_branch_warmup_start_ratio 0.3 \
  --time_branch_warmup_ratio 0.3 \
  --no-use_spatial_branch \
  --run_stage3 \
  --use_block_in_adapter \
  --align_weight_target 0.1 \
  --latent_decorr_weight 0.3 \
  --latent_decorr_top_k 512 \
  --group_bs 0 \
  --save_every_steps 0
```
