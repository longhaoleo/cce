# CCE: SharedSAE Concept Erasure on SDXL Base

这个仓库现在只保留一条正式主线：

- 训练 `SharedSAE`
- 用 `SharedSAE` 做概念定位
- 用 `SharedSAE` 做单图/批量概念擦除
- 用 prompt-conditioned 高频统计生成全局 blacklist

旧 turbo 路线、旧统一实验入口、旧图像反推式 blacklist 统计已经移除，不再作为默认工作流的一部分。
顶层的 `SAE/` 目录现在也已经替换成当前项目正在使用的 SharedSAE 模型包，而不是旧单块 SAE 实现。

## 模块导航

- [scripts/README.md](scripts/README.md)
  - 当前主线实验命令索引
- [scripts/erasure_latest_sae.md](scripts/erasure_latest_sae.md)
  - 最新 `no stage1` checkpoint 的定位与擦除命令
- [train/README.md](train/README.md)
  - 训练、smoke、pilot、full run
- [runtime/shared/README.md](runtime/shared/README.md)
  - Shared 概念定位、单图擦除、batch 擦除的正式运行时
- [tools/README.md](tools/README.md)
  - Shared prompt-conditioned 高频特征统计 / blacklist
- [research/archive_experiments/README.md](research/archive_experiments/README.md)
  - 旧实验脚本归档，只做参考不参与当前主线

## 目录结构

```text
cce/
├─ SAE/                   # 当前 SharedSAE 模型包入口
├─ runtime/shared/        # Shared 主线运行时实现
├─ SDLens/                # Hooked pipeline 基础设施（当前保留）
├─ research/archive_experiments/  # 旧实验脚本归档
├─ feature_frequency/     # 已生成的基础统计 run 目录
├─ train/                 # SharedSAE 训练
├─ tools/                 # Shared 辅助工具与脚本源码
├─ target_concept_dict/   # 概念定义输入
├─ concept_dict/          # 概念定位输出
├─ concept_dict_freq/     # 全局 blacklist 输出
├─ batch_test_prompt/     # batch prompt 集合
├─ image_output/          # 图片输出
└─ log/                   # 实验记录
```

## 最短流程

### 1. 训练或指定一个 Shared checkpoint

训练命令见 [train/README.md](train/README.md)。

### 2. 先收集基础统计，再生成全局 blacklist（可选但推荐）

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
  --run_name coco30k_stats_v1
```

```bash
cd /root/cce

python tools/feature_frequency/run_build_blacklist.py \
  --stats_dir feature_frequency/coco30k_stats_v1 \
  --feature_top_k 200 \
  --blacklist_freq_threshold 0.99 \
  --blacklist_active_ratio_min 0.3 \
  --blacklist_max_features 50
```

### 3. 做概念定位

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --only car \
  --taris_top_k 10 \
  --taris_score_mode taris
```

### 4. 做单图擦除

```bash
cd /root/cce

python -m runtime.shared.erase \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
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

### 5. 做 batch 擦除

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400 \
  --local_files_only \
  --prompts_path batch_test_prompt/car.csv \
  --concepts car \
  --output_dir image_output/batch_shared_concept_erase_car
```

## 输入输出约定

- 概念定义：`target_concept_dict/<concept>.json`
- 概念定位输出：`concept_dict/<block_short>/<concept>/`
- 高频特征统计输出：`concept_dict_freq/<block_short>/`
- 图片输出：`image_output/...`
- 诊断 CSV：每个 case 目录内的 `diag_shared_intervention_*.csv`

## 依赖

```bash
pip install -r requirements.txt
```
