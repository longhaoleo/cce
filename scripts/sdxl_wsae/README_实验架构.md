# SDXL SAE 实验架构（模块化）

## 1. 目录分层

- `SAE/sae.py`
  - SAE 官方实现，直接调用 `SparseAutoencoder.load_from_disk(...)` 读取权重。

- `scripts/sdxl_wsae/core/`
  - `session.py`: 会话层（ `sdxl.py` 模型加载 + SAE 加载 + 轨迹采样）。
  - `intervention.py`: 干预 hook（Injection / Ablation）。

- `scripts/sdxl_wsae/experiments/`
  - `exp51_feature_dynamics_topk.py`: 特征动力学 Top-K 热图叠加（实验 51）。
  - `exp52_feature_dynamics_waterfall.py`: 特征动力学瀑布图 Money Plot（实验 52）。
  - `exp53_concept_locator_taris.py`: 概念定位（TARIS 时域平均相对重要性，实验 53）。
  - `shared_prepare.py`: 实验共享的“采样+缓存+delta 提取”模块。
  - `exp54_intervention_suite.py`: 干预统一入口（baseline + main + early/late），并包含单窗口干预函数供复用。
  - `exp05_structure_aspect.py`: 结构与画幅控制（复用 exp54_intervention_suite.py 的单窗口干预）。
  - `exp06_dual_encoder.py`: 双编码器解耦（复用 exp54_intervention_suite.py 的单窗口干预）。
  - `exp07_clip_alignment.py`: CLIP 定量评估（复用 exp54_intervention_suite.py 的单窗口干预输出）。
  - `registry.py`: 实验注册和分发。

- `scripts/sdxl_wsae/cli.py`
  - 统一参数入口，按 `--experiment` 分发到对应实验。

## 2. 入口脚本

- 唯一入口：`scripts/vslz_wsae_res_sdxl.py`

该入口会转调 `scripts/sdxl_wsae/cli.py`。

## 3. 常用命令

### 实验 52：瀑布图（核心图）

```bash
python scripts/vslz_wsae_res_sdxl.py \
  --experiment exp52 \
  --steps 50 \
  --seed 42
```

### 实验 51：Top-K 热图叠加

```bash
python scripts/vslz_wsae_res_sdxl.py \
  --experiment exp51 \
  --sae_top_k 10 \
  --delta_stride 2 \
  --steps 50 \
  --seed 42
```

### 实验 51：可视化“指定特征集合”（来自 exp53 的 csv）

```bash
python scripts/vslz_wsae_res_sdxl.py \
  --experiment exp51 \
  --blocks unet.mid_block.attentions.0 \
  --exp51_feature_csv out_concept_dict/red/top_positive_features.csv \
  --exp51_feature_k 20 \
  --delta_stride 2 \
  --steps 50 \
  --seed 42
```

### 实验 54：单窗口注入（原 exp04 的统一入口）

```bash
python scripts/vslz_wsae_res_sdxl.py \
  --experiment exp54 \
  --int_block unet.mid_block.attentions.0 \
  --int_feature_ids 123 \
  --int_mode injection \
  --int_scale 1.5 \
  --int_t_start 600 \
  --int_t_end 200
```

### 实验 54：早晚注入对比（蝴蝶效应）

```bash
python scripts/vslz_wsae_res_sdxl.py \
  --experiment exp54 \
  --int_feature_ids 123 \
  --early_start 1000 --early_end 800 \
  --late_start 200 --late_end 0
```

### 实验 7：CLIP 定量

```bash
python scripts/vslz_wsae_res_sdxl.py \
  --experiment exp07 \
  --int_feature_ids 123 \
  --clip_target_text red \
  --clip_preserve_text car
```

### 实验 53：概念定位（TARIS）

```bash
python scripts/vslz_wsae_res_sdxl.py \
  --experiment exp53 \
  --loc_block unet.mid_block.attentions.0 \
  --concept_name red_vs_blue \
  --taris_t_start 800 --taris_t_end 200 \
  --taris_num_steps 10 \
  --taris_top_k 20
```

exp53 的 prompts 从本地 json 读取：`target_concept_dict/{concept_name}.json`，示例：

```json
{
  "pos_prompts": ["a red car", "a red apple"],
  "neg_prompts": ["a blue car", "a blue apple"]
}
```

输出固定到：`out_concept_dict/{concept_name}/`
