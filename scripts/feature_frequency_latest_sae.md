# Feature Frequency For Latest SAE

当前最新训练结果：

- `train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772`

这份文档只用于最新 `no stage1` checkpoint。不要和旧基线的 `coco30k_stats_v1` 混用。

## First Pass: Collect Stats

```bash
cd /root/cce

python tools/feature_frequency/run_collect_shared_stats.py \
  --ckpt_dir train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772 \
  --local_files_only \
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

默认版本：

```bash
cd /root/cce

python tools/feature_frequency/run_build_blacklist.py \
  --stats_dir feature_frequency/coco30k \
  --feature_top_k 200 \
  --blacklist_freq_threshold 0.99 \
  --blacklist_active_ratio_min 0.3 \
  --blacklist_mean_min 0.0 \
  --blacklist_max_features 50
```

更严格的低副作用试验版本。这个版本会把 COCO 上 `active_ratio >= 0.90` 的常见特征都过滤掉，适合检查 nudity 这类容易选到“人体/画面通用结构”的概念：

```bash
cd /root/cce

python tools/feature_frequency/run_build_blacklist.py \
  --stats_dir feature_frequency/coco30k \
  --feature_top_k 200 \
  --blacklist_freq_threshold 0.0 \
  --blacklist_active_ratio_min 0.90 \
  --blacklist_mean_min 0.0 \
  --blacklist_max_features 500 \
  --concept_dict_freq_root concept_dict_freq_strict
```

## Follow-Up

生成新 blacklist 后，重新跑对应概念定位，再做擦除：

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
