# Erasure With Latest SAE

当前最新训练结果：

- `train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage2_step_0012520`
- `train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772`

优先用 `stage3_step_0013772` 做定位和擦除验证。

## 1. 概念定位

### `car`

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only car \
  --taris_t_start 900 \
  --taris_t_end 100 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

### `dog`

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only dog \
  --taris_t_start 900 \
  --taris_t_end 100 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

### `nudity`

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only nudity \
  --taris_t_start 900 \
  --taris_t_end 100 \
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
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --prompts_path batch_test_prompt/car.csv \
  --concepts car \
  --output_dir image_output/batch_shared_concept_erase_car_stage23_half
```

### `dog`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --prompts_path batch_test_prompt/dog.csv \
  --concepts dog \
  --output_dir image_output/batch_shared_concept_erase_dog_stage23_half
```

### `nudity`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --prompts_path batch_test_prompt/nudity.csv \
  --concepts nudity \
  --output_dir image_output/batch_shared_concept_erase_nudity_stage23_half
```

## 4. 低副作用重跑建议

如果当前结果“擦得很彻底，但画面被重写太多”，先不要继续加大强度。优先跑下面两组：

注意：现在对外只保留一个强度参数 `--int_scale`。`feature_time_scores.csv` 只提供相对时间曲线，不再单独乘 `--int_time_weight_scale`。空间归一化默认开启；从实验反馈看，关掉后几乎没有擦除效果。

先用严格高频 blacklist 重新定位 `nudity`：

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only nudity \
  --concept_dict_freq_root concept_dict_freq_strict \
  --taris_t_start 900 \
  --taris_t_end 100 \
  --taris_num_steps 5 \
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
  --output_dir image_output/batch_shared_concept_erase_nudity_stage23_half_k3_s20 \
  --concept_dict_freq_root concept_dict_freq_strict \
  --int_feature_top_k 3 \
  --int_scale 20 \
  --int_t_start 900 \
  --int_t_end 100
```

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --prompts_path batch_test_prompt/nudity.csv \
  --concepts nudity \
  --output_dir image_output/batch_shared_concept_erase_nudity_stage23_half_k5_s40 \
  --concept_dict_freq_root concept_dict_freq_strict \
  --int_feature_top_k 5 \
  --int_scale 40 \
  --int_t_start 900 \
  --int_t_end 100
```

## 5. 对照建议

建议优先做这三组：

1. `car`
2. `dog`
3. `nudity`

重点看：

- 时间权重是否更自然
- 是否还需要特别大的时间倍率
- 副作用是否比旧基线更小
