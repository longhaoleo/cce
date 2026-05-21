# More Concept Erasure Candidates

这组概念用于快速扩展 `sae_x8_time_decorr03` 的擦除观察面。新增概念覆盖动物、自然物、常见物体、视觉风格和文字。

当前默认测试 `latent_decorr=0.3` 分支；如果要和 `latent_decorr=0.01` 对照，见 [sae_baseline_commands.md](sae_baseline_commands.md)。

- `cat`
- `bird`
- `flower`
- `bicycle`
- `chair`
- `anime_style`
- `text`

默认 checkpoint：

```text
train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772
```

默认 SAE 产物根目录：

```text
sae_data/sae_x8_time_decorr03
```

## 1. 概念定位

一次性跑全部新增概念：

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --only cat bird flower bicycle chair anime_style text \
  --taris_t_start 1000 \
  --taris_t_end 0 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

快速检查时可以先限制每边 prompt 数：

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --only cat bird flower bicycle chair anime_style text \
  --max_prompts_per_side 8 \
  --taris_t_start 1000 \
  --taris_t_end 0 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

## 2. 单图擦除

先用代表性单图看每个概念是否有明显擦除趋势：

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --targetconcept cat \
  --prompt "a tabby cat sitting on a windowsill in soft daylight" \
  --output_dir image_output/sae_x8_time_decorr03/shared_concept_erase_cat

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --targetconcept bird \
  --prompt "a colorful parrot sitting on a wooden stand" \
  --output_dir image_output/sae_x8_time_decorr03/shared_concept_erase_bird

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --targetconcept flower \
  --prompt "a bouquet of wildflowers on a wooden table" \
  --output_dir image_output/sae_x8_time_decorr03/shared_concept_erase_flower

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --targetconcept bicycle \
  --prompt "a bicycle parked beside a brick wall" \
  --output_dir image_output/sae_x8_time_decorr03/shared_concept_erase_bicycle

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --targetconcept chair \
  --prompt "a modern lounge chair in a minimalist room" \
  --output_dir image_output/sae_x8_time_decorr03/shared_concept_erase_chair

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --targetconcept anime_style \
  --prompt "an anime style city street at night" \
  --output_dir image_output/sae_x8_time_decorr03/shared_concept_erase_anime_style

python -m runtime.shared.erase \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --targetconcept text \
  --prompt "a poster with large readable text in the center" \
  --output_dir image_output/sae_x8_time_decorr03/shared_concept_erase_text
```

## 3. Batch 擦除

每个概念跑 20 条 prompt：

```bash
cd /root/cce

for concept in cat bird flower bicycle chair anime_style text; do
  python -m runtime.shared.batch \
    --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
    --local_files_only \
    --sae_root sae_data/sae_x8_time_decorr03 \
    --prompts_path "batch_test_prompt/${concept}.csv" \
    --concepts "${concept}" \
    --output_dir "image_output/sae_x8_time_decorr03/batch_shared_concept_erase_${concept}"
done
```

只想先看每个概念前 4 条：

```bash
cd /root/cce

for concept in cat bird flower bicycle chair anime_style text; do
  python -m runtime.shared.batch \
    --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
    --local_files_only \
    --sae_root sae_data/sae_x8_time_decorr03 \
    --prompts_path "batch_test_prompt/${concept}.csv" \
    --concepts "${concept}" \
    --max_prompts 4 \
    --output_dir "image_output/sae_x8_time_decorr03/batch_shared_concept_erase_${concept}_quick4"
done
```

## 4. Cross-Concept Stress Tests

用已有 prompt 擦除另一个概念，观察是否误伤相邻语义或风格：

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --prompts_path batch_test_prompt/cat.csv \
  --concepts dog \
  --output_dir image_output/sae_x8_time_decorr03/batch_shared_concept_erase_dog_but_cat

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --prompts_path batch_test_prompt/bird.csv \
  --concepts dog \
  --output_dir image_output/sae_x8_time_decorr03/batch_shared_concept_erase_dog_but_bird

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --prompts_path batch_test_prompt/anime_style.csv \
  --concepts van_gogh \
  --output_dir image_output/sae_x8_time_decorr03/batch_shared_concept_erase_van_gogh_but_anime_style

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time_decorr03 \
  --prompts_path batch_test_prompt/text.csv \
  --concepts glasses \
  --output_dir image_output/sae_x8_time_decorr03/batch_shared_concept_erase_glasses_but_text
```

建议优先观察：

- `cat` / `dog`：动物概念间是否共享或混淆。
- `bird`：细长结构、羽毛、天空背景是否被误当成概念。
- `flower` / `red`：颜色和对象是否解耦。
- `anime_style` / `van_gogh`：风格概念之间的边界。
- `text`：是否能擦掉文字区域，还是只破坏全局构图。
