# Erasure With Latest SAE

当前最新训练结果：

- `train/output_time_latentdecorr_x8_top20_half/checkpoints/stage2_step_0012520`
- `train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772`

优先用 `stage3_step_0013772` 做定位和擦除验证。

## 1. 概念定位

### `car`

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only car \
  --taris_t_start 1000 \
  --taris_t_end 0 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

### `dog`

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only dog \
  --taris_t_start 1000 \
  --taris_t_end 0 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

### `nudity`

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only nudity \
  --taris_t_start 1000 \
  --taris_t_end 0 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

## 2. 单图擦除

### `car`

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --targetconcept car \
  --prompt "a photo of a car on a city street, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_car_stage23_half
```

### `dog`

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --targetconcept dog \
  --prompt "a dog on the grass, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_dog_stage23_half
```

### `nudity`

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --targetconcept nudity \
  --prompt "an unclothed adult portrait in soft natural light" \
  --output_dir image_output/shared_concept_erase_nudity_stage23_half
```

## 3. Batch 擦除

### `car`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --prompts_path batch_test_prompt/car.csv \
  --concepts car \
  --output_dir image_output/batch_shared_concept_erase_car
```

### `dog`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --prompts_path batch_test_prompt/dog.csv \
  --concepts dog \
  --output_dir image_output/batch_shared_concept_erase_dog
```

### `nudity`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --prompts_path batch_test_prompt/nudity.csv \
  --concepts nudity \
  --output_dir image_output/batch_shared_concept_erase_nudity
```

### `e nudity g d`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --prompts_path batch_test_prompt/dog.csv \
  --concepts nudity \
  --output_dir image_output/batch_shared_concept_erase_nudity_but_dog
```
