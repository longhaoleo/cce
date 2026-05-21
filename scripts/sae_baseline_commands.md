# SAE Baseline Commands

当前有两套 SAE 分支：

| variant | tag | checkpoint | 说明 |
| --- | --- | --- | --- |
| `decorr03` | `sae_x8_time_decorr03` | `train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772` | `latent_decorr=0.3`，现在默认测试这套 |
| `decorr01` | `sae_x8_time` | `train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772` | `latent_decorr=0.01`，前一套主线基准 |

一键脚本：

```bash
cd /root/cce

./scripts/run_current_sae_baseline.sh
```

默认等价于：

```bash
cd /root/cce

SAE_VARIANT=decorr03 ./scripts/run_current_sae_baseline.sh
```

## 只跑 `latent_decorr=0.3`

快速测试，每个概念只跑前 4 条 batch prompt，locator 每边只用前 8 条 prompt：

```bash
cd /root/cce

MODE=quick ./scripts/run_current_sae_baseline.sh
```

全量定位 + 全量 batch：

```bash
cd /root/cce

MODE=all ./scripts/run_current_sae_baseline.sh
```

只跑新增概念：

```bash
cd /root/cce

CONCEPTS="cat bird flower bicycle chair anime_style text" ./scripts/run_current_sae_baseline.sh
```

只重新定位，不生成图：

```bash
cd /root/cce

MODE=locate CONCEPTS="cat bird flower bicycle chair anime_style text" ./scripts/run_current_sae_baseline.sh
```

只 batch，不重新定位：

```bash
cd /root/cce

MODE=batch CONCEPTS="cat bird flower bicycle chair anime_style text" ./scripts/run_current_sae_baseline.sh
```

## 对照跑 `latent_decorr=0.01`

```bash
cd /root/cce

SAE_VARIANT=decorr01 MODE=quick ./scripts/run_current_sae_baseline.sh
```

```bash
cd /root/cce

SAE_VARIANT=decorr01 MODE=all ./scripts/run_current_sae_baseline.sh
```

## 单条命令展开

`latent_decorr=0.3` 定位：

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

`latent_decorr=0.3` batch：

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

`latent_decorr=0.01` 定位：

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --only cat bird flower bicycle chair anime_style text \
  --taris_t_start 1000 \
  --taris_t_end 0 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

`latent_decorr=0.01` batch：

```bash
cd /root/cce

for concept in cat bird flower bicycle chair anime_style text; do
  python -m runtime.shared.batch \
    --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
    --local_files_only \
    --sae_root sae_data/sae_x8_time \
    --prompts_path "batch_test_prompt/${concept}.csv" \
    --concepts "${concept}" \
    --output_dir "image_output/sae_x8_time/batch_shared_concept_erase_${concept}"
done
```
