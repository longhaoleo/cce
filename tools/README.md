# Tools

这个目录现在只保留 Shared 主线需要的辅助工具。

注意区分两类路径：

- `tools/feature_frequency/`
  - 统计脚本源码
- `feature_frequency/`
  - 旧平铺结果目录；新推荐迁到 `sae_data/<sae_tag>/feature-freq`

## 文件

- `feature_frequency/`
  - 旧版高频特征统计目录
  - 新版统一为 `sae_data/<sae_tag>/feature-freq`

## 推荐命令

完整命令索引见 [`../scripts/feature_frequency.md`](../scripts/feature_frequency.md)。

```bash
cd /root/cce

python tools/feature_frequency/run_collect_shared_stats.py \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --sae_root sae_data/<sae_tag> \
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
  --taris_t_end 200 \
  --taris_num_steps 5 \
  --aggregate max \
  --feature_top_k 200 \
  --run_name coco30k_stats_v1
```

```bash
cd /root/cce

python tools/feature_frequency/run_build_blacklist.py \
  --stats_dir sae_data/<sae_tag>/feature-freq/coco30k_stats_v1 \
  --sae_root sae_data/<sae_tag> \
  --feature_top_k 200 \
  --blacklist_freq_threshold 0.99 \
  --blacklist_active_ratio_min 0.3 \
  --blacklist_max_features 50
```

## 输出

- `sae_data/<sae_tag>/feature-freq/<run_name>/<block_short>/dataset_feature_stats.pt`
- `sae_data/<sae_tag>/feature-freq/<run_name>/<block_short>/all_feature_frequency_ranked.csv`
- `sae_data/<sae_tag>/feature-freq/<run_name>/<block_short>/top_feature_frequency.csv`
- `sae_data/<sae_tag>/concept-dig-freq/<block_short>/all_feature_frequency_ranked.csv`
- `sae_data/<sae_tag>/concept-dig-freq/<block_short>/top_feature_frequency.csv`
- `sae_data/<sae_tag>/blacklist/<block_short>/feature_blacklist.txt`

## 输入约定

- 推荐直接用 `data/coco_30k.csv`
- blacklist 的目标是找“跨概念通用、过于常见”的特征，不是找某个目标概念的特征
