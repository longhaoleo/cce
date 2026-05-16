# Experiment Scripts

这里集中放当前主线的实验命令，按实验类型拆分。

## 文件

- [training.md](training.md)
  - SharedSAE 训练、smoke、half、full
- [feature_frequency.md](feature_frequency.md)
  - 旧基线 checkpoint 的 prompt-conditioned 高频特征统计与 blacklist
- [feature_frequency_latest_sae.md](feature_frequency_latest_sae.md)
  - 最新 `no stage1` checkpoint 的高频特征统计与 blacklist
- [concept_erasure.md](concept_erasure.md)
  - 概念定位、单图擦除、batch 擦除
- [erasure_latest_sae.md](erasure_latest_sae.md)
  - 最新 `no stage1` checkpoint 的定位与擦除命令
## 约定

- 这些文档只放“怎么跑”的命令。
- SAE 强绑定产物统一建议放在：
  - `sae_data/<sae_tag>/concept-dig`
  - `sae_data/<sae_tag>/concept-dig-freq`
  - `sae_data/<sae_tag>/blacklist`
  - `sae_data/<sae_tag>/feature-freq`
- Python 实现仍然留在原位置：
  - `train/`
  - `runtime/shared/`
  - `tools/feature_frequency/`
