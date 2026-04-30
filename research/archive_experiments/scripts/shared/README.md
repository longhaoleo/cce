# Shared Inference

这个目录只保留当前在用的 SharedSAE 测试入口，不再混入旧 turbo/legacy 脚本。

## 文件

- `run_exp53.py`
  - 基于 `target_concept_dict/<concept>.json` 重新生成 `concept_dict/<block>/<concept>/...`
- `run_concept_erase.py`
  - 对单条 prompt 做 Shared 概念擦除
- `run_batch_concept_erase.py`
  - 对一批 prompt 做 Shared 概念擦除

## 常用命令

### 1. 概念定位

```bash
cd /root/cce

python scripts/shared/run_exp53.py \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --only car \
  --taris_top_k 10 \
  --taris_score_mode taris
```

### 2. 单条擦除

```bash
cd /root/cce

python scripts/shared/run_concept_erase.py \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --targetconcept car \
  --prompt "a photo of a car on a city street, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_car \
  --blocks \
    unet.down_blocks.2.attentions.1 \
    unet.up_blocks.0.attentions.0 \
    unet.up_blocks.0.attentions.1 \
  --int_feature_top_k 5 \
  --int_scale 20 \
  --no-int_use_time_weight
```

### 3. 批量擦除

```bash
cd /root/cce

python scripts/shared/run_batch_concept_erase.py \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --prompts_path batch_test_prompt/car.csv \
  --concepts car \
  --output_dir image_output/batch_shared_concept_erase_car
```

## 输入输出

- 输入概念：`target_concept_dict/<concept>.json`
- 概念统计输出：`concept_dict/<block_short>/<concept>/`
- 图片输出：`image_output/...`
- 诊断 CSV：每个 case 目录下的 `diag_shared_intervention_*.csv`
