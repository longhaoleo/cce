# 实验记录：Turbo 路线 `car` 概念定位需重统计

日期：2026-04-22  
负责人：Codex / 用户协作

## 1. 目标与背景

- 目标：检查旧 `SAE Turbo / per-block SAE` 路线下，`car` 概念定位结果是否足够可信，能否直接支持后续擦除实验。
- 背景：用户反馈当前 Turbo 路线“擦除几乎没有效果”，即使干预强度已经调得很大，图像层面仍看不到明显概念消除。

## 2. 检查对象

- 概念目录：`concept_dict/down.2.1/car/`
- 重点文件：
  - `top_positive_features.csv`
  - `top_negative_features.csv`
  - `feature_time_scores.csv`
  - `taris_meta.txt`

## 3. 关键观察

- 当前 `score_mode=saeuron`，见 `taris_meta.txt`。
- `top_positive_features.csv` 中 top feature 分数极大，例如：
  - `feature_id=2681, score=72493.609375`
  - `feature_id=1418, score=67838.328125`
- 但在 `feature_time_scores.csv` 中，对应 step 的真实归一化差分 `diff` 往往只有：
  - `1e-4 ~ 1e-3`
  - 少数接近 `1e-2`
- 多个 top feature 在负样本上满足：
  - `neg_mu_raw ≈ 0`
  - `neg_sigma_raw = 0`
- 在 SAeUron 公式
  - `score = (mu_pos - mu_neg) / (sigma_neg + eps)`
  下，这会导致分母接近 `eps=1e-6`，从而把轻微正激活放大成数万级分数。

## 4. 判断

- 当前这版 `car` 的 top feature 排名存在明显“分母爆分”风险。
- 也就是说，排名靠前的特征未必是真正最强的 `car` 因果特征，更可能是：
  - 在正样本里稍有激活
  - 在负样本里几乎静默
  - 因而被 `sigma_neg≈0` 数学放大
- 这会直接影响后续擦除：
  - 即使 `int_scale` 很大
  - 也可能只是强力干预了一批“高分但不关键”的特征
  - 因此图像层面看起来没有明显概念消失

## 5. 负样本设计问题

- 当前 `car` 的 `neg_prompts` 过于杂糅，既包含：
  - empty road / empty garage / empty parking lot
  - bicycle / motorcycle / bus / truck
  - flower / cat / portrait / bedroom / kitchen
  - solid black background / solid white background / noisy image
- 这会让负样本分布过散，进一步增加：
  - `neg_mu` 接近 0
  - `neg_sigma` 接近 0
  的概率，放大 SAeUron 分数不稳定性。

## 6. 代码相关问题

- `scripts/run_exp53.py`
  - 已修复 `--only` 的 argparse 默认值问题：
    - 之前 `nargs="*"` 配 `default="car"` 会被逐字符拆成 `{c, a, r}`
    - 导致 concept 过滤异常
  - 已改为支持 `--blocks`
  - 旧单个 `--block` 参数已移除
- `exp53` 路线当前仍建议显式传：
  - `--concept_dir`
  - `--model_id`
  - `--sae_root`
  以减少环境默认值引起的路径歧义

## 7. 结论

- 当前这版 `car` 统计结果不建议直接作为正式擦除依据。
- 更合理的判断是：
  - 需要重新统计
  - 而不是继续单纯增大擦除强度
- 当前“擦除没效果”的高概率原因是：
  - 特征选偏了
  - 而不是 hook 完全失效

## 8. 下一步建议

- 先重跑一次 `exp53`，主评分改为 `taris`，并导出对比分数。
- 对 `car` 的负样本集合做一次 hard-negative 重构，减少与目标概念无关的杂项场景。
- 重统计后优先对比：
  - `top_positive_features.csv`
  - `score_compare_taris_vs_saeuron.csv`
  看 top-k 是否显著洗牌。
- 只有在新一版概念特征更稳定后，再继续 Turbo 擦除验证。
