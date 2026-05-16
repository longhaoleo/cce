# 2026-05-15 SAE Artifact Layout Unification

## 背景

今天 `decorr03` 的 `nudity -> cloth` replacement 出现了一种非常一致的崩坏模式：

- 不同 prompt 都生成相似的青色纹理块
- 不是正常的“替换效果不好”
- 更像 feature id 与 checkpoint 对不上

复查后确认，问题不是 `replace` 模式没生效，而是：

- checkpoint 用的是 `train/output_time_latentdecorr_x8_top20_decorr03/...`
- 但命令默认读取的是仓库根下平铺的 `concept_dict/` 与 `concept_dict_freq/`
- 而不是当前 SAE 专属目录

因此这次暴露出的核心问题是：**SAE 强绑定产物目录没有统一，太容易混根目录。**

## 统一后的目录布局

以后统一按一个 SAE 根目录组织：

```text
sae_data/<sae_tag>/
  concept-dig/
  concept-dig-freq/
  blacklist/
  feature-freq/
```

语义约定：

- `concept-dig/`
  - 概念定位输出
  - 对应旧 `concept_dict/<sae_tag>/...`
- `concept-dig-freq/`
  - 第二遍 blacklist 生成时导出的排序表
  - 例如 `all_feature_frequency_ranked.csv`
  - 例如 `top_feature_frequency.csv`
- `blacklist/`
  - 正式用于 locator / erase / batch 的 `feature_blacklist.txt`
- `feature-freq/`
  - 第一遍 prompt-conditioned 基础统计
  - 包含 `dataset_feature_stats.pt`

## 代码改动

增加了统一 SAE 布局解析模块：

- `runtime/shared/sae_layout.py`

并在以下入口新增 `--sae_root`：

- `runtime.shared.locator`
- `runtime.shared.erase`
- `runtime.shared.batch`
- `tools/feature_frequency/run_collect_shared_stats.py`
- `tools/feature_frequency/run_build_blacklist.py`

解析规则：

- `locator`
  - `concept_output_root -> <sae_root>/concept-dig`
  - `concept_dict_freq_root -> <sae_root>/blacklist`
- `erase / batch`
  - `concept_root -> <sae_root>/concept-dig`
  - `concept_dict_freq_root -> <sae_root>/blacklist`
- `feature_frequency pass1`
  - `output_dir -> <sae_root>/feature-freq`
- `feature_frequency pass2`
  - `concept_dig_freq_root -> <sae_root>/concept-dig-freq`
  - `concept_dict_freq_root -> <sae_root>/blacklist`

仅当用户没有显式覆盖旧根目录参数时，`--sae_root` 才会接管默认值。

## 本次实际迁移

已将下列产物迁入 `sae_data/`：

- `concept_dict/sae_x8_time -> sae_data/sae_x8_time/concept-dig`
- `feature_frequency/sae_x8_time -> sae_data/sae_x8_time/feature-freq`
- `concept_dict_freq/sae_x8_time ->`
  - `sae_data/sae_x8_time/concept-dig-freq`
  - `sae_data/sae_x8_time/blacklist`

- `concept_dict/sae_x8_time_decorr03 -> sae_data/sae_x8_time_decorr03/concept-dig`
- `feature_frequency/sae_x8_time_decorr03 -> sae_data/sae_x8_time_decorr03/feature-freq`
- `concept_dict_freq/sae_x8_time_decorr03 ->`
  - `sae_data/sae_x8_time_decorr03/concept-dig-freq`
  - `sae_data/sae_x8_time_decorr03/blacklist`

其中 `sae_x8_time_decorr03` 额外保留了多个 blacklist 变体：

- `ar90_all`
- `ar95_all`
- `q99_50`
- `q99_50_initial`

并把根级默认口径铺成：

- `sae_data/sae_x8_time_decorr03/blacklist/<block>/`
- `sae_data/sae_x8_time_decorr03/concept-dig-freq/<block>/`

默认对应当前主线规则：`ar95_all`。

## 为什么这样可以避免再犯

现在只要命令里明确写：

```bash
--sae_root sae_data/sae_x8_time_decorr03
```

那么：

- locator 输出的概念特征
- erase / batch 读取的概念特征
- blacklist 的读取路径
- feature-frequency 的统计与第二遍结果

都会自动落到同一个 SAE 根目录下，不需要再分别手写：

- `concept_dict/...`
- `concept_dict_freq/...`
- `feature_frequency/...`

这能明显减少“checkpoint 对了，但 concept root 错了”的低级错误。

## 额外说明

- 这次只统一了**新命令和新输出**的目录组织。
- 旧目录没有自动迁移。
- 仓库里原有：
  - `concept_dict/`
  - `concept_dict_freq/`
  - `feature_frequency/`
  仍然保留，避免打断已有实验复现。

后续建议：

1. 新实验一律使用 `--sae_root`
2. 新日志与脚本命令一律写 `sae_data/<sae_tag>/...`
3. 如果某次结果异常，先检查 manifest 中：
   - `ckpt_dir`
   - `sae_root`
   - `concept_root`
   - `concept_dict_freq_root`
   是否同属一个 SAE

## 本次同步更新

- `runtime/shared/README.md`
- `tools/README.md`
- `tools/feature_frequency/README.md`
- `scripts/README.md`
- `scripts/feature_frequency_latest_sae.md`
- `scripts/erasure_latest_sae.md`
