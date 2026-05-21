# 实验日志（log）

这个目录用于记录项目试验阶段的过程、思路、失败与结论。

## 文件约定

- `YYYY-MM-DD_*.md`：某一天的实验记录
- `TEMPLATE.md`：实验记录模板（每次复制一份填写）

## 当前索引

- `2026-05-21_latent_decorr_v2_plan.md`：下一步训练正则修改计划；把 latent decorrelation 改成 offdiag-only，并增加可选 block-pooled 模式，用于验证 token-level 全局去相关是否过度打散 `nudity` 这类局部组合概念
- `2026-05-16_posneg_replace_nudity_not_promising.md`：`top_positive_features` 擦除 + `top_negative_features` 注入的同概念替换实验结果明显崩坏；负特征注入远大于正特征擦除，不再作为主线继续推进
- `2026-05-16_replace_cloth_range_issue.md`：`nudity -> cloth` 与 `nudity -> ordinary_person` 都说明 replacement 已具备可行性，但当前仍过于全局、容易改写人物结构；现阶段更适合作为扩展能力，主线仍以 erasure 为主
- `2026-05-15_sae_artifact_layout_unification.md`：统一 SAE 强绑定产物目录到 `sae_data/<sae_tag>/{concept-dig,concept-dig-freq,blacklist,feature-freq}`，并给 `locator / erase / batch / feature_frequency` 增加 `--sae_root`，避免 checkpoint 与 concept root 混用
- `2026-05-15_shared_replacement_restore.md`：恢复 Shared 主线里的 `injection / replace` 能力，新增 `cloth` 概念，并明确 replacement 必须保持可选以支持后续 `ablation vs replace vs replace+inject=0` 消融
- `2026-05-15_nudity_decorr03_grid_retrospective.md`：`decorr03` 上的 `nudity` blacklist/top-k/scale 小网格复盘；结论是当前问题不只是力度不足，而是单组 nudity 概念定位仍被高频泛化特征主导，下一步转向细粒度子概念定位
- `2026-05-06_latest_sae_car_dog_nudity_assessment.md`：最新 `x8/top20 + time warmup + latent decorrelation` SAE 的训练与 `car/dog/nudity` 擦除复查；结论是 SAE 本身是当前最好主线，`dog` 失败更像概念定位负样本设计问题，已覆盖 dog 的 concept JSON 和 batch CSV
- `2026-05-02_latest_sae_nudity_erasure_assessment.md`：最新 `no stage1` SAE 的 `nudity` 擦除结果记录；生成质量更稳，目标概念可彻底擦除，但副作用过大，图像语义被明显重写
- `2026-04-30_time_decorr_smoke_v3_assessment.md`：第三轮 smoke 结果，并补充 `path1(low_lr_time)` vs `path2(no_stage1)` 对比，结论是 `no_stage1` 更优，下一步沿 `stage2 + stage3` 多 block 联合训练继续
- `2026-04-30_time_decorr_smoke_v2_assessment.md`：第二轮 `time_branch + decoder decorrelation` smoke 结果，确认比第一轮略有改善，但 dead feature 仍然过高，暂不建议直接上 full
- `2026-04-29_combo_concepts_bat_man_batman_feature_assessment.md`：`bat / man / batman` 三个概念的定位强度、跨概念重合、与 `car / dog` 的对比，以及后续组合擦除对照建议
- `2026-04-29_shared_erase_batch_param_unification.md`：统一 `runtime/shared/erase.py` 与 `runtime/shared/batch.py` 的干预参数定义，并把默认强度口径收敛到同一套配置
- `2026-04-29_shared_nudity_timeweight_vs_projection.md`：`nudity` 概念实验中，时间权重放大到 `scale=80` 效果较好、更细粒度；最近结果复查确认主线仍应保留 `ablation + 强时间权重`，并下一轮单独测试 `mid.0`
- `2026-04-29_shared_alignment_status_assessment.md`：评估当前 SharedSAE 已达到重建与可擦除目标，但跨层共享 feature id 的语义对齐仍然不够强
- `2026-04-29_shared_intervention_updates.md`：Shared 主线把概念定位默认切到 TARIS，并新增时间权重放大系数与 `projected_ablation` 子空间投影擦除模式
- `2026-04-29_shared_mainline_runtime_refactor.md`：当前 Shared 主线已统一到 `SAE/ + runtime/shared/ + tools/feature_frequency/`，旧脚本迁出主线并归档到 `research/archive_experiments/`
- `2026-03-26_experiment_journal.md`：阶段摘要版记录
- `2026-03-31_detailed_change_process.md`：详细版代码改动过程与思路变化
- `2026-04-09_test_idea_attention_overlap_retrospective.md`：`test_idea` 注意力重叠原型的实现细节、方法分析与结论复盘
- `2026-04-19_shared_sae_training_and_erase_prep.md`：SharedSAE 训练收尾、checkpoint 可用性与后续概念定位/擦除准备
- `2026-04-22_turbo_exp53_car_restat_needed.md`：旧 Turbo 路线下 `car` 概念统计需要重做的原因与依据
- `2026-04-24_turbo_exp54_car_activation_check.md`：旧 Turbo `exp54` 擦除里各 block 的 SAE 特征激活强弱检查
- `2026-04-28_shared_blacklist_prompt_conditioned.md`：SharedSAE 全局高频特征 blacklist 切到 prompt-conditioned 统计后的参数、结果与下一步
- `2026-04-28_shared_erase_effective_but_destructive.md`：Shared 擦除已开始有效，但当前配置对原图扰动偏大，后续重点转向稳定性优化
- `2026-04-28_shared_512_baseline_quality_issue.md`：Shared 主线下 `512` 分辨率 baseline 画质本身较差，不能简单归因到 SAE 干预
- `../research/archive_experiments/README.md`：旧 `scripts/` 实验代码的归档说明，明确它与当前 `runtime/shared/` 主线的关系
- `../test_idea/README.md`：`test_idea` 历史原型目录的归档说明，说明它与当前主线的关系

## 建议原则

- 失败也记录（避免重复踩坑）
- 每条记录都包含可复现命令和输出目录
- 结论与下一步行动分开写
