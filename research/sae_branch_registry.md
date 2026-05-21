# SAE Branch Registry

日期：2026-05-21

不同训练分支必须在整条擦除链上使用同一个 tag，避免 feature frequency、concept dict、batch output 互相覆盖。

## 当前分支

| tag | 训练目录 | 状态 | 说明 |
| --- | --- | --- | --- |
| `sae_align_v1` | `train/output_exp_c_adapter_align/` | 历史基线 | 早期 shared alignment 版本 |
| `sae_x8_time_decorr03` | `train/output_time_latentdecorr_x8_top20_decorr03/` | 当前默认测试分支 | `x8/top20 + time warmup + latent_decorr=0.3` |
| `sae_x8_time` | `train/output_time_latentdecorr_x8_top20_half/` | 可比基线 | `x8/top20 + time warmup + latent_decorr=0.01` |

## 目录命名规则

| 产物 | 规则 |
| --- | --- |
| 高频统计 | `sae_data/<tag>/feature-freq/<run_name>/` |
| blacklist | `sae_data/<tag>/blacklist/` 或 `sae_data/<tag>/blacklist/<variant>/` |
| blacklist 排序表 | `sae_data/<tag>/concept-dig-freq/` 或 `sae_data/<tag>/concept-dig-freq/<variant>/` |
| 概念定位 | `sae_data/<tag>/concept-dig/` |
| batch 擦除 | `image_output/<tag>/batch_shared_concept_<mode>_<concept>/` |
| sweep / grid | `image_output/<tag>/<sweep_name>/` |

## 当前目录映射

### `sae_x8_time`

```text
sae_data/sae_x8_time/feature-freq/coco30k/
sae_data/sae_x8_time/concept-dig/
sae_data/sae_x8_time/concept-dig-freq/
sae_data/sae_x8_time/blacklist/
image_output/sae_x8_time/batch_shared_concept_erase_car/
image_output/sae_x8_time/batch_shared_concept_erase_dog/
image_output/sae_x8_time/batch_shared_concept_replace_nudity_cloth/
```

### `sae_x8_time_decorr03`

```text
sae_data/sae_x8_time_decorr03/feature-freq/coco30k/
sae_data/sae_x8_time_decorr03/concept-dig/
sae_data/sae_x8_time_decorr03/concept-dig-freq/q99_50_initial/
sae_data/sae_x8_time_decorr03/concept-dig-freq/q99_50/
sae_data/sae_x8_time_decorr03/concept-dig-freq/ar95_all/
sae_data/sae_x8_time_decorr03/concept-dig-freq/ar90_all/
sae_data/sae_x8_time_decorr03/blacklist/q99_50_initial/
sae_data/sae_x8_time_decorr03/blacklist/q99_50/
sae_data/sae_x8_time_decorr03/blacklist/ar95_all/
sae_data/sae_x8_time_decorr03/blacklist/ar90_all/
image_output/sae_x8_time_decorr03/batch_shared_concept_erase_car/
image_output/sae_x8_time_decorr03/batch_shared_concept_erase_dog/
image_output/sae_x8_time_decorr03/batch_shared_concept_erase_nudity/
image_output/sae_x8_time_decorr03/nudity_grid/
```

## 命令约定

优先使用统一的 SAE 根目录参数：

```bash
--sae_root sae_data/<tag>
```

定位、blacklist、擦除都应使用同一个 `sae_root`：

```bash
python -m runtime.shared.locator \
  ... \
  --sae_root sae_data/sae_x8_time_decorr03
```

```bash
python tools/feature_frequency/run_build_blacklist.py \
  ... \
  --sae_root sae_data/sae_x8_time_decorr03
```

```bash
python -m runtime.shared.batch \
  ... \
  --sae_root sae_data/sae_x8_time_decorr03
```

这样一条 SAE 的训练、blacklist、概念定位和图片输出会保持同一 tag，不再混用。
