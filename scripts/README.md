# Experiment Scripts

这里集中放当前主线的实验命令，按实验类型拆分。

## 推荐入口

当前分支优先用这两个文件：

- [sae_baseline_commands.md](sae_baseline_commands.md)
  - `decorr03` / `decorr01` 两套 SAE 分支的 quick / locate / batch / all 命令。
- [run_current_sae_baseline.sh](run_current_sae_baseline.sh)
  - 一键跑当前 SAE 分支的定位与 batch 擦除。

最短 quick run：

```bash
cd /root/cce

MODE=quick ./scripts/run_current_sae_baseline.sh
```

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
- [more_concept_erasure.md](more_concept_erasure.md)
  - 新增 `cat` / `bird` / `flower` / `bicycle` / `chair` / `anime_style` / `text` 的定位与擦除命令
- [sae_baseline_commands.md](sae_baseline_commands.md)
  - 区分 `latent_decorr=0.3` 与 `latent_decorr=0.01` 两套 SAE 的一键命令
- [run_current_sae_baseline.sh](run_current_sae_baseline.sh)
  - 当前默认 `sae_x8_time_decorr03` 基准的一键定位与 batch 擦除脚本

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
