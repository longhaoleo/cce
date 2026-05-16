# Feature Frequency For `sae_x8_time`

当前对应训练分支：

- `sae_x8_time`
- `train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772`

这份文档只用于 `sae_x8_time`。当前已有统计目录：

```text
sae_data/sae_x8_time/feature-freq/coco30k/
```

## First Pass: Collect Stats

```bash
cd /root/cce

python tools/feature_frequency/run_collect_shared_stats.py \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --sae_root sae_data/sae_x8_time \
  --prompts_path data/coco_30k.csv \
  --blocks \
    unet.down_blocks.2.attentions.1 \
    unet.mid_block.attentions.0 \
    unet.up_blocks.0.attentions.0 \
    unet.up_blocks.0.attentions.1 \
  --max_prompts 1000 \
  --steps 50 \
  --guidance_scale 8.0 \
  --resolution 512 \
  --taris_t_start 1000 \
  --taris_t_end 0 \
  --taris_num_steps 5 \
  --aggregate max \
  --feature_top_k 500 \
  --run_name coco30k
```

## Second Pass: Build Blacklist

默认版本：

```bash
cd /root/cce

python tools/feature_frequency/run_build_blacklist.py \
  --stats_dir sae_data/sae_x8_time/feature-freq/coco30k \
  --sae_root sae_data/sae_x8_time \
  --feature_top_k 200 \
  --blacklist_freq_threshold 0.0 \
  --blacklist_active_ratio_min 0.95 \
  --blacklist_mean_min 0.0 \
  --blacklist_max_features 0
```


## Follow-Up

生成新 blacklist 后，重新跑对应概念定位，再做擦除：

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
