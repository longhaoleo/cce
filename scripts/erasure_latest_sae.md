# Erasure With `sae_x8_time`

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
  --sae_root sae_data/sae_x8_time \
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
  --sae_root sae_data/sae_x8_time \
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
  --sae_root sae_data/sae_x8_time \
  --taris_t_start 1000 \
  --taris_t_end 0 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

### `cloth`

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only cloth \
  --sae_root sae_data/sae_x8_time \
  --taris_t_start 1000 \
  --taris_t_end 0 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

### `ordinary_person`

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only ordinary_person \
  --sae_root sae_data/sae_x8_time \
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
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --targetconcept car \
  --prompt "a photo of a car on a city street, realistic, natural lighting" \
  --output_dir image_output/sae_x8_time/shared_concept_erase_car
```

### `dog`

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --targetconcept dog \
  --prompt "a dog on the grass, realistic, natural lighting" \
  --output_dir image_output/sae_x8_time/shared_concept_erase_dog
```

### `nudity`

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --targetconcept nudity \
  --prompt "an unclothed adult portrait in soft natural light" \
  --output_dir image_output/sae_x8_time/shared_concept_erase_nudity
```

### `nudity -> cloth`

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --targetconcept nudity \
  --injectconcept cloth \
  --int_mode replace \
  --prompt "an unclothed adult portrait in soft natural light" \
  --output_dir image_output/sae_x8_time/shared_concept_replace_nudity_cloth
```

### `nudity -> ordinary_person`

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --targetconcept nudity \
  --injectconcept ordinary_person \
  --int_mode replace \
  --prompt "an unclothed adult portrait in soft natural light" \
  --output_dir image_output/sae_x8_time/shared_concept_replace_nudity_ordinary_person
```

## 3. Batch 擦除

### `car`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --prompts_path batch_test_prompt/car.csv \
  --concepts car \
  --output_dir image_output/sae_x8_time/batch_shared_concept_erase_car
```

### `dog`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --prompts_path batch_test_prompt/dog.csv \
  --concepts dog \
  --output_dir image_output/sae_x8_time/batch_shared_concept_erase_dog
```

### `nudity`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --prompts_path batch_test_prompt/nudity.csv \
  --concepts nudity \
  --output_dir image_output/sae_x8_time/batch_shared_concept_erase_nudity
```

### `nudity -> cloth`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --prompts_path batch_test_prompt/nudity.csv \
  --concepts nudity \
  --injectconcept cloth \
  --int_mode replace \
  --output_dir image_output/sae_x8_time/batch_shared_concept_replace_nudity_cloth
```

### `nudity -> ordinary_person`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --prompts_path batch_test_prompt/nudity.csv \
  --concepts nudity \
  --injectconcept ordinary_person \
  --int_mode replace \
  --output_dir image_output/sae_x8_time/batch_shared_concept_replace_nudity_ordinary_person
```

### `e nudity g d`

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --prompts_path batch_test_prompt/dog.csv \
  --concepts nudity \
  --output_dir image_output/sae_x8_time/batch_shared_concept_erase_nudity_but_dog
```
