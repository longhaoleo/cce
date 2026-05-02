# Concept Localization And Erasure

## Shared Locator

### `dog`

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --only dog \
  --taris_top_k 10 \
  --taris_score_mode taris
```

### `nudity`

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --only nudity \
  --taris_top_k 10 \
  --taris_score_mode taris
```

## Single Image Erasure

### `dog`

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --targetconcept dog \
  --prompt "a dog on the grass, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_dog
```

### `car`

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --targetconcept car \
  --prompt "a photo of a car on a city street, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_car
```

## Batch Erasure

### `car`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --prompts_path batch_test_prompt/car.csv \
  --concepts car \
  --output_dir image_output/batch_shared_concept_erase_car
```

### `dog`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --prompts_path batch_test_prompt/dog.csv \
  --concepts dog \
  --output_dir image_output/batch_shared_concept_erase_dog
```

### `nudity`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --prompts_path batch_test_prompt/nudity.csv \
  --concepts nudity \
  --output_dir image_output/batch_shared_concept_erase_nudity
```

## Next Checkpoint Evaluation

最新 checkpoint 的专用命令见 [erasure_latest_sae.md](erasure_latest_sae.md)。

用最新 `no stage1` 训练结果做一小轮定位和擦除时，先用这两个 checkpoint：

- `train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage2_step_0012520`
- `train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772`

示例：

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only car \
  --taris_top_k 10 \
  --taris_score_mode taris
```

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --prompts_path batch_test_prompt/nudity.csv \
  --concepts nudity \
  --output_dir image_output/batch_shared_concept_erase_nudity_stage23_half
```
