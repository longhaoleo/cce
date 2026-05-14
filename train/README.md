# Train

这里是 SharedSAE 训练模块。正式入口只有两个：

- `run_smoke_train.py`
- `run_train.py`

模型定义、训练配置、checkpoint、编码和 block 归一化都已经统一收进顶层的 [SAE/README.md](../SAE/README.md)。
`train/` 现在只负责训练流程、loss、采样器和 trainer。

补充说明保留在：

- `PLAN.md`
- [`../scripts/training.md`](../scripts/training.md)

## 常用命令

### Smoke

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

### Pilot

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

### Full

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
  --calibration_prompts 1000 \
  --num_step_buckets 5 \
  --shard_prompts 250 \
  --experiment_preset "$PRESET"
```

## 输出

- `output_*/checkpoints/`
- `output_*/metrics/`
- `output_*/plots/`
- `output_*/run_manifest.json`

## 说明

- 默认训练空间是 `512` 分辨率，对应当前 Shared 主线的 `16x16` token 空间。
- 训练输出的 checkpoint 会被 `runtime/shared/` 和 `tools/` 直接复用。
