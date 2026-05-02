# Training Experiments

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
  --stage1_train_prompts 100 \
  --calibration_prompts 100 \
  --experiment_preset "$PRESET"
```

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
  --stage1_train_prompts 500 \
  --calibration_prompts 200 \
  --num_step_buckets 5 \
  --shard_prompts 100 \
  --experiment_preset "$PRESET"
```

## Full

```bash
cd /root/cce

PRESET=exp_d_full
OUT=train/output_${PRESET}
python train/run_train.py \
  --output_root "$OUT" \
  --local_files_only \
  --steps 50 \
  --validation_prompts 1000 \
  --stage2_train_prompts 20000 \
  --stage1_train_prompts 5000 \
  --calibration_prompts 1000 \
  --num_step_buckets 5 \
  --shard_prompts 250 \
  --experiment_preset "$PRESET"
```

## Step-3 Time Branch + Decorrelation

### Full

```bash
cd /root/cce

python train/run_train.py \
  --experiment_preset custom \
  --output_root train/output_time_decorr_stage123 \
  --local_files_only \
  --steps 50 \
  --validation_prompts 1000 \
  --stage2_train_prompts 20000 \
  --stage1_train_prompts 5000 \
  --calibration_prompts 1000 \
  --num_step_buckets 5 \
  --shard_prompts 250 \
  --use_time_branch \
  --time_branch_mode sincos_linear \
  --no-use_spatial_branch \
  --run_stage1 \
  --run_stage3 \
  --no-run_stage4 \
  --use_block_in_adapter \
  --no-use_block_out_adapter \
  --align_weight_target 0.1 \
  --decoder_decorr_weight 3e-4 \
  --auxk 256 \
  --save_every_steps 0
```

### Half

```bash
cd /root/cce

python train/run_train.py \
  --experiment_preset custom \
  --output_root train/output_time_decorr_stage123_half \
  --local_files_only \
  --steps 50 \
  --validation_prompts 500 \
  --stage2_train_prompts 10000 \
  --stage1_train_prompts 2500 \
  --calibration_prompts 500 \
  --num_step_buckets 5 \
  --shard_prompts 250 \
  --use_time_branch \
  --time_branch_mode sincos_linear \
  --no-use_spatial_branch \
  --run_stage1 \
  --run_stage3 \
  --no-run_stage4 \
  --use_block_in_adapter \
  --no-use_block_out_adapter \
  --align_weight_target 0.1 \
  --decoder_decorr_weight 3e-4 \
  --auxk 256 \
  --save_every_steps 0
```

## Current Mainline

当前主线是 `no stage1` 的 `stage2 + stage3` 联合训练。

```bash
cd /root/cce

python train/run_train.py \
  --experiment_preset custom \
  --output_root train/output_time_decorr_stage23_half_no_stage1 \
  --local_files_only \
  --steps 50 \
  --validation_prompts 500 \
  --stage2_train_prompts 10000 \
  --stage1_train_prompts 2500 \
  --calibration_prompts 500 \
  --num_step_buckets 5 \
  --shard_prompts 250 \
  --use_time_branch \
  --time_branch_mode sincos_linear \
  --no-use_spatial_branch \
  --no-run_stage1 \
  --run_stage3 \
  --no-run_stage4 \
  --use_block_in_adapter \
  --no-use_block_out_adapter \
  --align_weight_target 0.1 \
  --decoder_decorr_weight 1e-2 \
  --auxk 256 \
  --save_every_steps 0
```

## Next Mainline: Larger SAE + Delayed Time Branch

这轮用于验证两个假设：

- 扩大字典和 `top_k`，减少概念被压进少数混合 feature。
- time branch 在 stage1 固定关闭，stage2 训练到中段再线性打开，stage3 固定全开，避免一开始把共享 feature 空间按 timestep 切碎。
- 用 latent covariance decorrelation 代替 decoder decorrelation，直接压制 batch 内 feature 共激活。

```bash
cd /root/cce

python train/run_train.py \
  --experiment_preset custom \
  --output_root train/output_time_latentdecorr_x8_top20_half \
  --local_files_only \
  --steps 50 \
  --validation_prompts 500 \
  --stage2_train_prompts 10000 \
  --stage1_train_prompts 2500 \
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
  --no-run_stage1 \
  --run_stage3 \
  --no-run_stage4 \
  --use_block_in_adapter \
  --no-use_block_out_adapter \
  --align_weight_target 0.1 \
  --decoder_decorr_weight 0.0 \
  --latent_decorr_weight 0.01 \
  --latent_decorr_top_k 256 \
  --save_every_steps 0
```
