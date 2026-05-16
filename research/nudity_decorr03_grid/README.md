# Nudity Decorr03 Grid

状态：诊断实验，已完成一轮，不属于当前正式运行入口。

## 目标

验证 `decorr03` checkpoint 上，`nudity` 擦除变差是否只是 `blacklist / top_k / scale` 没调好。

## 入口

- [run_grid.sh](run_grid.sh)
  - 对 `nudity` 跑三套 blacklist 和六组 `top_k / scale` 组合。

## 输入与输出

- 汇总表：
  - [summary_metrics.csv](summary_metrics.csv)
- checkpoint：
  - `train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772`
- 基础统计：
  - `feature_frequency/sae_x8_time_decorr03/coco30k/`
- blacklist 变体：
  - `concept_dict_freq/sae_x8_time_decorr03/q99_50/`
  - `concept_dict_freq/sae_x8_time_decorr03/ar95_all/`
  - `concept_dict_freq/sae_x8_time_decorr03/ar90_all/`
- 概念字典快照：
  - `concept_dict_grid/sae_x8_time_decorr03/nudity_q99_50/`
  - `concept_dict_grid/sae_x8_time_decorr03/nudity_ar95_all/`
  - `concept_dict_grid/sae_x8_time_decorr03/nudity_ar90_all/`
- 图片结果：
  - `image_output/sae_x8_time_decorr03/nudity_grid/`

## 结果

18 组全部完成，每组 20 个 case。

| 设置 | 总 `delta_over_x` 均值 |
| --- | ---: |
| `ar90_all_top10_scale3000` | `0.174` |
| `ar95_all_top10_scale3000` | `0.241` |
| `q99_50_top10_scale3000` | `0.324` |
| `q99_50_top20_scale5000` | `0.746` |

主要观察：

- `q99_50` 最激进，`ar95_all` 居中，`ar90_all` 最保守；`top_k` 和 `scale` 都能稳定放大干预。
- 但“更强”没有转化成“更准”：`q99_50` 提到 `top15/20 + scale5000` 后，画面已经明显改写，仍有若干 case 保留 nudity 语义。
- `ar95_all / ar90_all` 保图更好，但多数 case 擦除不够。
- `q99_50` top10 里 `27/40` 个特征的 `active_ratio >= 0.9`，说明当前 `nudity` 仍主要依赖高频泛化特征。
- blacklist 越严格，图像副作用越小；但 `ar90_all` 已经有过严迹象，当前更值得继续作为主对照的是 `ar95_all`。

当前默认 blacklist 已据此改成 `ar95_all`：

```text
blacklist_freq_threshold = 0.0
blacklist_active_ratio_min = 0.95
blacklist_max_features = 0
```

## 结论

这轮结果不支持继续只调 `blacklist / top_k / scale`。当前瓶颈更像是 `nudity` 语义过宽，单组 prompt 定位把多个子语义混到了一起。下一步应先拆细概念语义，再让 locator 对子概念逐组定位并聚合。
