# Feature Frequency Experiments

## First Pass: Collect Stats

```bash
cd /root/cce

python tools/feature_frequency/run_collect_shared_stats.py \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --sae_root sae_data/exp_c_adapter_align \
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
  --taris_t_start 900 \
  --taris_t_end 100 \
  --taris_num_steps 5 \
  --aggregate max \
  --feature_top_k 200 \
  --run_name coco30k
```

## Second Pass: Build Blacklist

```bash
cd /root/cce

python tools/feature_frequency/run_build_blacklist.py \
  --stats_dir sae_data/exp_c_adapter_align/feature-freq/coco30k \
  --sae_root sae_data/exp_c_adapter_align \
  --feature_top_k 200 \
  --blacklist_freq_threshold 0.99 \
  --blacklist_active_ratio_min 0.3 \
  --blacklist_mean_min 0.0 \
  --blacklist_max_features 50
```
