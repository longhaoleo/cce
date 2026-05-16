# SAE Branch Registry

日期：2026-05-15

不同训练分支必须在整条擦除链上使用同一个 tag，避免 feature frequency、concept dict、batch output 互相覆盖。

## 当前分支

| tag | 训练目录 | 状态 | 说明 |
| --- | --- | --- | --- |
| `sae_align_v1` | `train/output_exp_c_adapter_align/` | 历史基线 | 早期 shared alignment 版本 |
| `sae_x8_time` | `train/output_time_latentdecorr_x8_top20_half/` | 当前主线 | `x8/top20 + time warmup + latent_decorr=0.01`，当前擦除效果最好 |
| `sae_x8_time_decorr03` | `train/output_time_latentdecorr_x8_top20_decorr03/` | 对照分支 | `latent_decorr=0.3` 强去相关 ablation |

## 目录命名规则

| 产物 | 规则 |
| --- | --- |
| 高频统计 | `feature_frequency/<tag>/<run_name>/` |
| blacklist | `concept_dict_freq/<tag>/` 或 `concept_dict_freq/<tag>/<variant>/` |
| 概念定位 | `concept_dict/<tag>/` |
| batch 擦除 | `image_output/<tag>/batch_shared_concept_<mode>_<concept>/` |
| sweep / grid | `image_output/<tag>/<sweep_name>/` |

## 当前目录映射

### `sae_x8_time`

```text
feature_frequency/sae_x8_time/coco30k/
concept_dict/sae_x8_time/
concept_dict_freq/sae_x8_time/
image_output/sae_x8_time/batch_shared_concept_erase_car/
image_output/sae_x8_time/batch_shared_concept_erase_dog/
image_output/sae_x8_time/batch_shared_concept_replace_nudity_cloth/
```

### `sae_x8_time_decorr03`

```text
feature_frequency/sae_x8_time_decorr03/coco30k/
concept_dict/sae_x8_time_decorr03/
concept_dict_freq/sae_x8_time_decorr03/q99_50_initial/
concept_dict_freq/sae_x8_time_decorr03/q99_50/
concept_dict_freq/sae_x8_time_decorr03/ar95_all/
concept_dict_freq/sae_x8_time_decorr03/ar90_all/
concept_dict_grid/sae_x8_time_decorr03/nudity_*/
image_output/sae_x8_time_decorr03/batch_shared_concept_erase_car/
image_output/sae_x8_time_decorr03/batch_shared_concept_erase_dog/
image_output/sae_x8_time_decorr03/batch_shared_concept_erase_nudity/
image_output/sae_x8_time_decorr03/nudity_grid/
```

## 命令约定

`locator` 现在支持：

```bash
--concept_output_root concept_dict/<tag>
```

后续每次定位都必须显式传：

```bash
python -m runtime.shared.locator \
  ... \
  --concept_output_root concept_dict/sae_x8_time
```

擦除时再配套传：

```bash
--concept_root concept_dict/sae_x8_time
--concept_dict_freq_root concept_dict_freq/sae_x8_time
```

这样一条 SAE 的训练、blacklist、概念定位和图片输出会保持同一 tag，不再混用。
