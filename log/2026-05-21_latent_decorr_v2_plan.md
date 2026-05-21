# 实验记录：latent decorrelation v2 修改计划

日期：2026-05-21  
负责人：Leo

## 1. 目标与假设

- 目标：下一步修改 SharedSAE 训练中的 `latent_decorrelation_loss`，先改成 offdiag-only，再增加可选的 block-pooled decorrelation。
- 假设：当前 token-level 全局拼接 decorrelation 对 `nudity` 这类局部组合概念过强，可能把皮肤、姿态、身体区域、局部裸露、光照和构图等协同子特征打散，导致 top-k 擦除预算不够或副作用变大。

## 2. 环境与版本

- 代码版本：待实现前记录 `git rev-parse --short HEAD`
- 模型路径：
  - 当前默认对照：`train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772`
  - 前一套基线：`train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772`
- SAE 路径：
  - `sae_data/sae_x8_time_decorr03`
  - `sae_data/sae_x8_time`

## 3. 实验配置

- 计划文档：`train/PLAN2.md`
- 核心修改：
  - `offdiag_only=True` 时只惩罚 `corr_ij^2, i != j`
  - 保留 `mode="token"` 作为 ablation
  - 新增 `mode="block_pooled"`，先对每个 block 的 token latent 做 mean pooling，再在 pooled latent 的 feature 维度做 offdiag decorrelation
- 预计新增参数：
  - `latent_decorr_mode`
  - `latent_decorr_pool`
  - `latent_decorr_eps`

## 4. 预期观察

- `token + offdiag-only`：保留去冗余作用，但减少对 feature 自相关尺度的隐性约束。
- `block_pooled + offdiag-only`：更偏向跨 block 语义去冗余，减少对局部 token 协同特征的破坏。
- 对 `nudity` 的关键预期：
  - 在 target suppression 不明显下降的前提下，LPIPS / DreamSim / CLIP 保真曲线改善。

## 5. 结论

- 已实现训练侧 latent decorrelation v2：`token/block_pooled`、`mean/topq/hybrid` 与固定 `offdiag-only`。
- 已实现 runtime 侧时间权重消融接口：统计时间权重、learned time weight、兼容 fuse mode，并默认保持旧统计时间权重路径。
- 已恢复并固定擦除输出诊断：baseline / steered / compare / eval pair，以及 `diag_time_weights_long.csv` 和 `diag_time_weights_summary.csv`。
- 当前本机 Python 环境缺少 `torch`，单元测试入口无法执行；已完成 `compileall` 语法检查，完整数值测试需在训练环境中运行。

## 6. 下一步

- 在有 `torch` 的训练环境运行 `python -m unittest tests/test_latent_decorr_and_time_weight.py`。
- 先校准时间权重量纲，再用 `car / dog / nudity` 跑 `A_no_time / B_stat_time / C_learned_time`，对比 target suppression 与 LPIPS / DreamSim / CLIP。

## 7. 2026-05-21 更新：learned time 量纲修正

- 观察结果：`stat_time` 的 `final_time_weight` 约为 `1e-3`，而旧 `learned_time = 2 * sigmoid(raw)` 约为 `1`，在 `int_scale=5000` 下会把干预放大数百倍，表现接近 `no_time` 并导致黑图。
- 代码修正：
  - 新增 `--int_learned_time_weight_mode {relative_window,absolute}`，默认使用 `relative_window`。
  - `relative_window` 会先在全部 denoising step 上计算 `time_branch_raw`，做 moving average、按时间维 z-score，再用 sigmoid 形成相对时间窗口，并按 feature 的时间均值归一到约 1。
  - `learned_only` 时使用 `learned_rel * --int_learned_time_weight_target_mean`，默认 `0.001`。
  - `product` 时使用 `stat_weight * learned_rel`，让 learned branch 只改变统计权重的时间形状。
  - 保留 `absolute` 作为兼容旧行为。
- 新增安全阈值：`--int_max_delta_over_x`，建议大批量消融先用 `0.1` 或 `0.2`，防止单组参数直接生成黑图。
- 新增诊断字段：
  - `int_scale`
  - `effective_gain_mean`
  - `effective_gain_max`
  - `learned_temporal_cv`
  - `learned_temporal_range`
  - `delta_safety_scale`

## 8. 2026-05-21 更新：最近实现事件

- 训练入口整理：
  - 新增 `scripts/run_latent_decorr_v2_train.sh`，用 `VARIANT` 选择 `token / block_pooled_mean / block_pooled_topq / block_pooled_hybrid`。
  - 当前推荐训练命令：`VARIANT=block_pooled_mean ./scripts/run_latent_decorr_v2_train.sh`。
  - 脚本默认使用 `x8/top20 + time_branch + latent_decorr_weight=0.3 + latent_decorr_top_k=512`。
- 训练预统计 cache：
  - `norm_scale_by_block` 现在会先检查 cache，不再每次重复跑 calibration 预统计。
  - 默认 cache 路径：`train/cache/norm_scale_by_block/<config_fingerprint>.json`。
  - 可用 `--norm_scale_cache_path` 或脚本环境变量 `NORM_SCALE_CACHE_PATH=...` 指定固定 cache 文件。
  - `run_manifest.json` 会记录 `norm_scale_cache_path` 和 `norm_scale_cache_fingerprint`。
- 擦除输出诊断：
  - 每个 single erase / batch case 保留 `intervention_baseline.png`、`intervention_steered.png`、`intervention_compare.png`、`eval_original/`、`eval_erased/`、`run_manifest.json`。
  - 时间权重 long/summary 表继续输出，并新增 `effective_gain_mean/max`，用于直接观察 `int_scale * final_time_weight` 是否过大。
  - 新增 `diag_time_weights_heatmap.png`，展示不同 timestep、不同 feature 的最终时间权重。
  - 新增 `diag_top_feature_final_activation.png`，展示经过时间系数处理后的 top feature 平均激活。
- 下一轮建议：
  - 先跑小批量 `A_no_time / B_stat_time / C_learned_time / D_stat_x_learned_rel`，重点检查是否正常出图。
  - 重点看 `diag_time_weights_summary.csv` 中的 `effective_gain_mean`、`delta_over_x`、`delta_safety_scale`。
  - 若 `D_stat_x_learned_rel` 能保持 `B_stat_time` 的正常出图，并让 learned 权重在 timestep 上出现结构，就继续扩大到 `car / dog / nudity`。
- 已完成的本地验证：
  - `python -m compileall -q SAE train runtime tests`
  - `bash -n scripts/run_latent_decorr_v2_train.sh`
- 未完成验证：
  - 当前本机环境缺少 `torch`，尚未运行训练 smoke 和单元测试；需在训练环境中执行。

## 9. 2026-05-21 观察：flower 擦除效果较差

- 现象：
  - 在当前批量擦除配置下，`flower` 的擦除效果不好。
  - 当前配置大致为：`int_scale=5000`、`int_feature_top_k=2`、`stat_time`、`--int_timestep_window 1000 300`、`--int_max_delta_over_x 0.2`。
- 初步记录：
  - `flower` 可能不是由少数 top feature 覆盖的单一物体概念，top-2 特征预算可能不够。
  - 也可能是当前时间窗只覆盖高/中噪声阶段，未覆盖足够的细节形成阶段。
  - 后续应对比 `top_k=5/10`、不同 `int_timestep_window`，以及 `stat_x_learned_rel` 是否能改善。
- 下一步建议：
  - 先保留当前结果目录作为失败样本。
  - 对 `flower` 单独跑一组小消融：
    - `int_feature_top_k = 2 / 5 / 10`
    - `int_timestep_window = 1000 300 / 800 100 / 1000 0`
    - `stat_only` 对比 `stat_x_learned_rel`

## 10. 2026-05-21 更新：组合概念实验 v0

- 新增组合概念：
  - `dog_glasses = dog with glasses`
  - `red_car = red + car`
  - `flower_van_gogh = flower + Van Gogh style`
- 新增文件：
  - `target_concept_dict/dog_glasses.json`
  - `target_concept_dict/red_car.json`
  - `target_concept_dict/flower_van_gogh.json`
  - `batch_test_prompt/dog_glasses.csv`
  - `batch_test_prompt/red_car.csv`
  - `batch_test_prompt/flower_van_gogh.csv`
  - `scripts/compositional_concept_erasure_v0.md`
  - `scripts/run_compositional_v0.sh`
  - `scripts/analyze_composition_overlap.py`
- v0 实验设计：
  - 先定位组合概念和对应原子概念。
  - 用 `feature_overlap.csv` 计算 `union_coverage` 与 `new_feature_ratio`。
  - 对每个组合跑五组：组合 target、A-only preservation、B-only preservation、atomic A on composite、atomic B on composite。
- 当前目标：
  - 判断组合概念更像原子 feature union，还是 emergent feature cluster。
  - 判断当前 flat feature erase 是否会误伤单独 A/B。
  - 特别观察 `flower_van_gogh` 是否比单独 `flower` 更容易或更难擦除。
