# Shared Runtime

这里是当前 Shared 主线真正的运行时实现，不再把实现代码挂在 `scripts/` 目录下面。

## 模块

- `pipeline.py`
  - checkpoint 解析、SDXL pipeline 加载、采样参数继承
- `locator.py`
  - Shared 概念定位主逻辑
- `erase.py`
  - Shared 单图概念擦除 / 注入 / 替换主逻辑
- `batch.py`
  - Shared 批量概念擦除 / 注入 / 替换主逻辑
- `io_utils.py`
  - 路径/文件名/图片提取等公共小工具
- `features/`
  - `delta.py`：轨迹 delta 提取
  - `scoring.py`：TARIS / SAeUron 打分与导出
  - `intervention.py`：block scale / coeff / 调试 CSV
  - `hook_ops.py`：hook 基础张量变换与时空窗口判断

## 推荐命令

完整命令索引见 [`../../scripts/concept_erasure.md`](../../scripts/concept_erasure.md)。

推荐统一使用 SAE 根目录布局：

```text
sae_data/<sae_tag>/
  concept-dig/
  concept-dig-freq/
  blacklist/
  feature-freq/
```

### 概念定位

`target_concept_dict/<concept>.json` 既支持普通 `pos_prompts / neg_prompts`，也支持额外的 `subconcepts`。
如果存在 `subconcepts`，`locator` 会先分别输出 `<concept>__<subconcept>`，再按 feature-wise max 聚合回父概念 `<concept>`，因此后续擦除命令仍然只需要写父概念名。

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --sae_root sae_data/<sae_tag> \
  --only car \
  --taris_top_k 10 \
  --taris_score_mode taris
```

`nudity` 示例：

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --sae_root sae_data/<sae_tag> \
  --only nudity \
  --taris_top_k 10 \
  --taris_score_mode taris
```

### 单图擦除

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --sae_root sae_data/<sae_tag> \
  --targetconcept car \
  --prompt "a photo of a car on a city street, realistic, natural lighting" \
  --output_dir image_output/shared_concept_erase_car \
  --blocks \
    unet.down_blocks.2.attentions.1 \
    unet.up_blocks.0.attentions.0 \
    unet.up_blocks.0.attentions.1 \
  --int_feature_top_k 5 \
  --int_scale 20 \
  --no-int_use_time_weight
```

### 单图替换

先定位 `cloth`，再做 `nudity -> cloth`：

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --sae_root sae_data/<sae_tag> \
  --targetconcept nudity \
  --injectconcept cloth \
  --int_mode replace \
  --prompt "an unclothed adult portrait in soft natural light" \
  --output_dir image_output/shared_concept_replace_nudity_cloth
```

### 批量擦除

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --sae_root sae_data/<sae_tag> \
  --prompts_path batch_test_prompt/car.csv \
  --concepts car \
  --output_dir image_output/batch_shared_concept_erase_car
```

`nudity` 示例：

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --sae_root sae_data/<sae_tag> \
  --prompts_path batch_test_prompt/nudity.csv \
  --concepts nudity \
  --output_dir image_output/batch_shared_concept_erase_nudity
```
