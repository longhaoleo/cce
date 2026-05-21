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
- 用 `car / dog / nudity` 跑 `A_no_time / B_stat_time / C_learned_time`，对比 target suppression 与 LPIPS / DreamSim / CLIP。
