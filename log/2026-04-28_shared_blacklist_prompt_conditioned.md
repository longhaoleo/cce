# 2026-04-28 Shared blacklist（prompt-conditioned）记录

## 背景

这次将 `SharedSAE` 的全局高频特征统计从“真实图片 + empty condition”切换为：

- `prompt-conditioned`
- 与后续 `Shared exp53 / erase` 保持同分布
- 直接使用 `data/coco_30k.csv`

目标是生成更可信的全局 blacklist，避免之前出现：

- `active_ratio` 全 0
- `threshold = 0`
- 整层特征全部被拉黑
- `mid.0` 的 `top_positive_features.csv` 出现 `-inf`

## 本次运行配置

- checkpoint: `train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400`
- prompts: `data/coco_30k.csv`
- num_prompts_used: `1000`
- steps: `50`
- guidance_scale: `8.0`
- resolution: `512`
- selected_timesteps: `881 721 541 381 201`
- aggregate: `max`
- feature_top_k: `200`
- blacklist_quantile: `0.99`
- blacklist_active_ratio_min: `0.3`
- blacklist_mean_min: `0.0`
- blacklist_max_features: `50`
- feature_activation_eps: `1e-6`

## 结果摘要

本次 Shared prompt-conditioned blacklist 已正常生成，不再是“整层全黑”：

- `down.2.1`
  - threshold: `0.94258004`
  - blacklist_size: `34`
- `mid.0`
  - threshold: `0.99770999`
  - blacklist_size: `26`
- `up.0.0`
  - threshold: `0.99800003`
  - blacklist_size: `32`
- `up.0.1`
  - threshold: `0.99900001`
  - blacklist_size: `18`

对应输出目录：

- `concept_dict_freq/down.2.1/`
- `concept_dict_freq/mid.0/`
- `concept_dict_freq/up.0.0/`
- `concept_dict_freq/up.0.1/`

详细运行目录：

- `image_output/shared_feature_frequency/shared_prompt_freq_coco_30k_t201-881_n5/`

## 结论

这次统计链路已经恢复到可用状态：

- blacklist 现在来自 prompt-conditioned 轨迹
- 不再依赖图片集
- 与 Shared `exp53 / erase` 的分布更一致
- 当前每层都只屏蔽一小部分高频特征，没有再出现“全层过滤”

但这还不是擦除结果本身，只是把全局高频特征过滤准备好了。

## 下一步

建议按这个顺序继续：

1. 先重跑 Shared `exp53`
   - 让 `concept_dict/<block>/<concept>/top_positive_features.csv` 在新的全局 blacklist 下重新生成
   - 重点检查 `mid.0` 这层是否还出现 `-inf`

2. 再跑 Shared 擦除
   - 先只测一个概念，例如 `car`
   - 先看 baseline 与 intervention 的单图差异
   - 再看 `diag_shared_intervention_*.csv`

3. 如果 `mid.0` 仍然异常
   - 先临时把 `mid_block.attentions.0` 从 Shared 擦除里拿掉
   - 只用 `down.2.1 / up.0.0 / up.0.1` 验证方向

## 建议命令

先重跑 Shared `exp53`：

```bash
cd /root/cce

python scripts/run_shared_exp53.py \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --only car \
  --taris_top_k 10 \
  --taris_score_mode taris
```

然后跑 Shared 擦除：

```bash
cd /root/cce

python scripts/run_shared_concept_erase.py \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --targetconcept car \
  --prompt "a photo of a car on a city street, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_car_v2 \
  --int_feature_top_k 5 \
  --int_scale 10
```

## 备注

当前 `tools/run_shared_feature_frequency.py` 已改成：

- 读取 `data/*.csv` / `data/*.txt`
- prompt-conditioned 特征频率统计
- 每层 blacklist 单独产出

并且 `tools/feature_frequency_commands.md` 与 `tools/download_coco_images.md` 已同步更新为新流程说明。
