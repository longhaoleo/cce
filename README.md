# CCE: SharedSAE Concept Erasure on SDXL Base

这个仓库现在只保留一条正式主线：

- 训练 `SharedSAE`
- 用 `SharedSAE` 做概念定位
- 用 `SharedSAE` 做单图/批量概念擦除
- 用 prompt-conditioned 高频统计生成全局 blacklist

旧 turbo 路线、旧统一实验入口、旧图像反推式 blacklist 统计已经移除，不再作为默认工作流的一部分。
顶层的 `SAE/` 目录现在也已经替换成当前项目正在使用的 SharedSAE 模型包，而不是旧单块 SAE 实现。

## 当前默认分支

当前建议优先用 `latent_decorr=0.3` 分支做新概念擦除基准：

```text
tag:        sae_x8_time_decorr03
checkpoint: train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772
sae_root:   sae_data/sae_x8_time_decorr03
```

前一套可比基线是：

```text
tag:        sae_x8_time
checkpoint: train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772
sae_root:   sae_data/sae_x8_time
```

最短入口：

```bash
cd /root/cce

MODE=quick ./scripts/run_current_sae_baseline.sh
```

## 模块导航

- [scripts/README.md](scripts/README.md)
  - 当前主线实验命令索引
- [scripts/erasure_latest_sae.md](scripts/erasure_latest_sae.md)
  - 最新 `no stage1` checkpoint 的定位与擦除命令
- [scripts/sae_baseline_commands.md](scripts/sae_baseline_commands.md)
  - 当前 `decorr03` / `decorr01` 两套 SAE 分支的快速基准入口
- [scripts/more_concept_erasure.md](scripts/more_concept_erasure.md)
  - 新增概念的定位与擦除命令
- [train/README.md](train/README.md)
  - 训练、smoke、pilot、full run
- [runtime/shared/README.md](runtime/shared/README.md)
  - Shared 概念定位、单图擦除、batch 擦除的正式运行时
- [tools/README.md](tools/README.md)
  - Shared prompt-conditioned 高频特征统计 / blacklist
- [evaluation/README.md](evaluation/README.md)
  - Shared batch 擦除结果的 pixel / diag / CLIP / LPIPS / DreamSim / NSFW 量化评测
- [research/archive_experiments/README.md](research/archive_experiments/README.md)
  - 旧实验脚本归档，只做参考不参与当前主线
- [research/nsfw_selective_erasure_plan.md](research/nsfw_selective_erasure_plan.md)
  - 下一阶段 `SharedSAE + NSFW selective erasure` 论文级研究路线图
- [research/sae_branch_registry.md](research/sae_branch_registry.md)
  - 不同 SAE 训练分支与下游产物的命名注册表

## 目录结构

```text
cce/
├─ SAE/                   # 当前 SharedSAE 模型包入口
├─ runtime/shared/        # Shared 主线运行时实现
├─ SDLens/                # Hooked pipeline 基础设施（当前保留）
├─ research/archive_experiments/  # 旧实验脚本归档
├─ train/                 # SharedSAE 训练
├─ tools/                 # Shared 辅助工具与脚本源码
├─ evaluation/            # Shared batch 擦除量化评测
├─ target_concept_dict/   # 概念定义输入
├─ batch_test_prompt/     # batch prompt 集合
├─ sae_data/<sae_tag>/    # SAE 强绑定产物：定位、blacklist、频率统计
├─ image_output/          # 图片输出；正式实验按 SAE 分支分组
└─ log/                   # 实验记录
```

## 最短流程

### 1. 训练或指定一个 Shared checkpoint

训练命令见 [train/README.md](train/README.md)。

### 1.5. 下载评测权重（可选）

如果要跑 `evaluation.eval_clip`，先把 CLIP 权重下载到统一本地目录：

```bash
cd /root/cce

python download_clip.py
```

默认保存到：

```text
/root/autodl-tmp/models/clip-vit-base-patch32
```

评测脚本会按统一规则读取 CLIP：

```text
1. --clip_model 显式传入的路径或 HuggingFace id
2. CCE_CLIP_MODEL 环境变量
3. /root/autodl-tmp/models/clip-vit-base-patch32
4. openai/clip-vit-base-patch32
```

如果要跑 `evaluation.eval_lpips`，预先下载 LPIPS/AlexNet 权重：

```bash
cd /root/cce

python download_lpips.py
```

默认使用 Torch 缓存目录：

```text
/root/autodl-tmp/models/torch
```

评测脚本会按统一规则读取 Torch 缓存：

```text
1. --torch_home 显式传入的路径
2. CCE_TORCH_HOME 环境变量
3. TORCH_HOME 环境变量
4. /root/autodl-tmp/models/torch
```

### 2. 先收集基础统计，再生成全局 blacklist（可选但推荐）

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
  --run_name coco30k_stats_v1
```

```bash
cd /root/cce

python tools/feature_frequency/run_build_blacklist.py \
  --sae_root sae_data/<sae_tag> \
  --stats_dir sae_data/<sae_tag>/feature-freq/coco30k_stats_v1 \
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
- 概念定位输出：`sae_data/<sae_tag>/concept-dig/<block_short>/<concept>/`
- 高频特征统计输出：`sae_data/<sae_tag>/feature-freq/<run_name>/<block_short>/`
- blacklist 输出：`sae_data/<sae_tag>/blacklist/<block_short>/feature_blacklist.txt`
- 图片输出：`image_output/<sae_tag>/...`
- 分支命名注册表：`research/sae_branch_registry.md`
- 诊断 CSV：每个 case 目录内的 `diag_shared_intervention_*.csv`

## 依赖

```bash
pip install -r requirements.txt
```

如果要跑 LPIPS / DreamSim 等可选评估指标：

```bash
pip install -r requirements-eval.txt
```

旧实验脚本才需要：

```bash
pip install -r requirements-legacy.txt
```
