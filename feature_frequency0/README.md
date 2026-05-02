# Feature Frequency Runs

这里放的是**已经生成出来的高频特征基础统计结果**，不是脚本源码。

脚本源码在：

- [tools/feature_frequency](/root/cce/tools/feature_frequency)

典型结构：

```text
feature_frequency/
└─ <run_name>/
   ├─ down.2.1/
   ├─ mid.0/
   ├─ up.0.0/
   └─ up.0.1/
```

每个 block 目录下会有：

- `dataset_feature_stats.pt`
- `dataset_feature_stats_all.csv`
- `all_feature_frequency_ranked.csv`
- `top_feature_frequency.csv`
- `run_meta.txt`

说明：

- 这里存的是第一遍“基础统计”
- 第二遍生成 blacklist 时，直接把 `--stats_dir` 指到这里的某个 run 目录即可
- 正式 blacklist 输出仍然写到 `concept_dict_freq/`
