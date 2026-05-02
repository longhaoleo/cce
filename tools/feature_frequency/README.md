# Feature Frequency

这里专门放 Shared 主线的高频特征统计工具。

## 两遍流程

完整命令索引见 [`../../scripts/feature_frequency.md`](../../scripts/feature_frequency.md)。

### 第一遍：只收集基础统计

这一步会跑 prompt-conditioned 轨迹，保存每个 block 的：

- `active_ratio`
- `mean_activation`
- `std_activation`
- 排序表
- `dataset_feature_stats.pt`

推荐命令：

```bash
cd /root/cce

python tools/feature_frequency/run_collect_shared_stats.py \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
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
  --taris_t_end 200 \
  --taris_num_steps 5 \
  --aggregate max \
  --feature_top_k 200 \
  --run_name coco30k_stats_v1
```

输出目录示例：

```text
feature_frequency/coco30k_stats_v1/
```

### 第二遍：按筛选规则生成 blacklist

这一步不再重跑轨迹，只读取第一遍的统计结果，然后按你当前想试的条件生成：

- `concept_dict_freq/<block>/feature_blacklist.txt`
- `concept_dict_freq/<block>/all_feature_frequency_ranked.csv`
- `concept_dict_freq/<block>/top_feature_frequency.csv`

推荐命令：

```bash
cd /root/cce

python tools/feature_frequency/run_build_blacklist.py \
  --stats_dir feature_frequency/coco30k_stats_v1 \
  --feature_top_k 200 \
  --blacklist_freq_threshold 0.99 \
  --blacklist_active_ratio_min 0.3 \
  --blacklist_mean_min 0.0 \
  --blacklist_max_features 50
```

## 为什么这样拆

- 第一遍最贵，因为要重新跑生成轨迹
- 第二遍很便宜，因为只是在已有统计上换筛选条件
- 后面你想比较不同 blacklist 规则时，只需要重跑第二遍
