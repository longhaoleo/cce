# Latest SAE Car / Dog / Nudity Erasure Assessment

日期：2026-05-06

## 状态

当前主线继续使用最新训练的 SharedSAE：

```text
train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772
```

这轮模型是目前最值得继续推进的 SAE。`nudity` 擦除按肉眼观察是目前效果最好的一轮，`car` 擦除也明显有效；`dog` 擦除仍不理想。

## 训练配置

训练输出：

```text
train/output_time_latentdecorr_x8_top20_half/
```

关键配置：

- `expansion_factor=8`
- `n_dirs=10240`
- `top_k=20`
- `auxk=512`
- `use_time_branch=true`
- `time_branch_mode=sincos_linear`
- `time_branch_warmup_start_ratio=0.3`
- `time_branch_warmup_ratio=0.3`
- `run_stage1=false`
- `run_stage3=true`
- `decoder_decorr_weight=0.0`
- `latent_decorr_weight=0.01`
- `latent_decorr_top_k=256`

时间分支调度符合预期：

- stage2 前段 `time_branch_scale=0`
- stage2 中段线性升到 `1`
- stage3 全程 `time_branch_scale=1`

## 训练指标

来自：

```text
train/output_time_latentdecorr_x8_top20_half/run_manifest.json
train/output_time_latentdecorr_x8_top20_half/metrics/step_metrics.jsonl
```

stage2：

- `steps=12520`
- `mean_recon=0.4245`
- `val_recon=0.3834`
- `mean_align=0.2540`
- `val_align=0.0997`
- `mean_latent_decorr=0.01032`

stage3：

- `steps=1252`
- `mean_recon=0.3698`
- `val_recon=0.3798`
- `mean_align=0.0953`
- `val_align=0.0995`
- `mean_latent_decorr=0.00979`
- tail `dead_feature_frac≈0.0247`

稀疏率：

```text
latent_active_frac = 0.001953125 = 20 / 10240
```

这是由 `top_k=20` 和 `n_dirs=10240` 决定的正常结果，不是异常。

## 与旧模型对比

decoder 方向相关性抽样检查：

```text
old_x4_no_time:
  n_dirs=5120
  decoder_sample_offdiag_abs_mean≈0.0717
  decoder_sample_offdiag_sq_mean≈0.0090
  dead_feature_frac≈0.0146

new_x8_time_latentdecorr:
  n_dirs=10240
  decoder_sample_offdiag_abs_mean≈0.0499
  decoder_sample_offdiag_sq_mean≈0.0056
  dead_feature_frac≈0.0247
```

解释：

- 新模型重建更好。
- 新模型 decoder 方向相关性更低。
- dead feature 比旧模型略高，但仍在可接受范围。
- 字典容量翻倍后，概念擦除的图像连续性更好，说明这轮训练方向是有效的。

## latent decorrelation 的实际强度

虽然训练中加入了 latent covariance decorrelation，但当前权重下该项对总 loss 的贡献很小：

```text
stage3 loss_latent_decorr_term ≈ 7e-05 到 1.5e-04
```

判断：

- 该项已经进入训练和日志。
- 但它目前更像弱约束，不是强力解耦项。
- 如果后续继续追求更少副作用，可以考虑把 `latent_decorr_weight` 从 `0.01` 提高到 `0.03` 做小规模对照。

## Erasure 观察

本地可见 batch 输出：

```text
image_output/batch_shared_concept_erase_car/
image_output/batch_shared_concept_erase_dog/
```

`nudity` 结果本轮按肉眼观察是目前最好：

- 目标概念擦除更彻底。
- 生成质量比前一轮更稳定。
- 副作用比旧模型小，但仍需要继续观察是否过度重写图像语义。

`car` 结果较好：

- 当前 batch 使用 `scale=5000`。
- 多个 case 中右半边 erased image 能明显移除 car 主体。
- 诊断 CSV 显示干预进入了 down/mid/up 各层。

`dog` 结果不理想：

- 当前 batch 使用 `scale=9000`。
- 诊断 CSV 中 dog 的 `delta_over_x` 并不小，甚至整体强于 car。
- 图像中 dog 往往变成残缺动物结构，而不是稳定消失。

## Dog 失败原因判断

dog 不是“干预没打进去”。

证据：

- `dog` 的 `scale=9000` 高于 `car` 的 `scale=5000`。
- `dog` 的各层 `delta_over_x` 明显不低。
- erased 图像有强变化，但仍保留狗/动物形态。

更可能的问题是概念定位：

- 原 `target_concept_dict/dog.json` 的负样本包含 `cat / wolf / fox / rabbit / animal / plush toy dog / robotic dog / werewolf` 等近邻动物或 dog-like 内容。
- 这种对比会让 locator 找到“狗区别于其他动物”的残差特征，而不是“生成狗身体/四足动物/犬类主体”的核心因果特征。
- 结果是擦掉局部 dog feature，但 SDXL 仍能通过剩余动物/宠物结构继续生成残缺狗。

## 已做调整

直接覆盖了 dog 的输入文件：

```text
target_concept_dict/dog.json
batch_test_prompt/dog.csv
```

调整原则：

- `dog.json` 正样本保留多品种、多姿态 dog。
- `dog.json` 负样本改为“同场景但无主体”，去掉近邻动物负样本。
- `dog.csv` 改为更聚焦的 dog 全身、脸部、动作和场景测试样例。

校验结果：

```text
target_concept_dict/dog.json: 50 pos / 50 neg
batch_test_prompt/dog.csv: 20 cases
neg prompts: dog=0, puppy=0, cat=0, wolf=0, fox=0, animal=0
```

## 下一步

先不要改训练。下一步应重新定位 dog：

```bash
cd /root/cce

python -m runtime.shared.locator \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --only dog \
  --taris_t_start 1000 \
  --taris_t_end 0 \
  --taris_num_steps 5 \
  --taris_top_k 10 \
  --taris_score_mode taris
```

然后重新跑 dog batch：

```bash
cd /root/cce

python -m runtime.shared.batch \
  --ckpt_dir train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772 \
  --local_files_only \
  --prompts_path batch_test_prompt/dog.csv \
  --concepts dog \
  --output_dir image_output/batch_shared_concept_erase_dog
```

如果新 dog 定位仍然失败，再考虑两个方向：

- 增大 `dog` 的 `int_feature_top_k` 到 `10` 或 `15`。
- 增加一个更宽的 `animal_body` 辅助概念，与 `dog` 联合擦除。

## Follow-up: Dog 重新定位成功

重新覆盖 `target_concept_dict/dog.json` 和 `batch_test_prompt/dog.csv` 后，重新定位并跑 dog batch，dog 擦除成功。

代表性输出：

```text
image_output/batch_shared_concept_erase_dog/dog/case_000009/
```

对应 manifest：

```text
image_output/batch_shared_concept_erase_dog/dog/case_000009/run_manifest.json
```

关键参数：

- checkpoint：`train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772`
- concept：`dog`
- prompt：`a labrador swimming in a lake`
- seed：`3310`
- mode：`ablation`
- scale：`3000`
- feature_top_k：`10`
- time window：`1000 -> 0`
- spatial norm：enabled
- blocks：`down.2.1 / mid.0 / up.0.0 / up.0.1`

这个结果很关键：

- 旧 dog 配置下，`scale=9000`、`feature_top_k=5` 仍然擦除不稳定。
- 新 dog 定义下，`scale=3000`、`feature_top_k=10` 已经成功。
- 因此 dog 失败的主因基本确认是特征定位输入不好，而不是最新 SAE 训练失败。

这也反过来支持当前 locator 流程的敏感点：概念 JSON 的负样本不能随意包含近邻语义类别，否则 TARIS 更容易找出“区分类别的残差 feature”，而不是“生成目标主体的因果 feature”。

补充观察：

- 用更用心的 dog prompt 重新提取后，特征明显更有效，但它们更关注狗的头部、脸部和局部识别性特征。
- 对狗身体、四肢、躯干等完整主体结构的区分还不够强。
- 这可能是当前 dog 概念的真实特征分布：头部/脸部更容易被 SAE 和 TARIS 找到，身体结构更容易和一般动物、姿态、毛发、运动形状混在一起。
- 如果要覆盖更完整的 dog 概念，需要纳入更多 feature 或额外构造 `dog_body / animal_body` 这类辅助概念。
- 代价是误伤会增加，因为身体、四足姿态、毛发、动物轮廓这些 feature 更可能和其他动物或场景主体共享。

## 当前结论

这轮 SAE 不像训练坏了。相反，训练质量、图像连续性、`nudity` 和 `car` 擦除都说明它是当前最好主线。

`dog` 的问题已经通过重新设计 prompt 得到验证：它确实主要是概念定义和 locator 对比构造问题，不应归因到 SAE 训练失败。

## Follow-up: Stronger Independence Training Plan

用户希望尽量让每个 feature 都更接近干净语义。当前主模型的独立性正则确实偏弱：

```text
stage3 loss_latent_decorr_term mean ≈ 9.8e-05
stage3 loss_auxk_term mean ≈ 0.0106
stage3 loss_align_term mean ≈ 0.0095
```

也就是说，latent decorrelation 当前比 aux/align 小约两个数量级，更像轻微偏置，不是强约束。

因此下一轮安排为强独立性 ablation，而不是替换当前主模型：

```text
train/output_time_latentdecorr_x8_top20_decorr03
```

只改两个关键参数：

```text
latent_decorr_weight: 0.01 -> 0.3
latent_decorr_top_k: 256 -> 512
```

其他配置保持当前主线：

- `expansion_factor=8`
- `top_k=20`
- `auxk=512`
- `use_time_branch=true`
- `time_branch_warmup_start_ratio=0.3`
- `time_branch_warmup_ratio=0.3`
- `run_stage1=false`
- `run_stage3=true`
- `decoder_decorr_weight=0.0`

该实验的判断标准：

- 如果 `loss_latent_decorr_term` 提升到 `1e-3` 量级，同时 `val_recon/dead_feature_frac` 不明显恶化，则说明加强独立性是可行路线。
- 如果擦除更干净但图像质量下降，需要在 `0.1~0.3` 之间找折中。
- 如果 dog 身体覆盖改善但误伤其他动物/主体变多，应作为“coverage vs collateral damage” tradeoff 写入论文，而不是继续盲目加正则。

训练命令已写入：

```text
scripts/training.md
```

## Follow-up: Stronger Independence Training Result

强独立性 ablation 已完成：

```text
train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772
```

配置确认：

- `expansion_factor=8`
- `top_k=20`
- `auxk=512`
- `use_time_branch=true`
- `time_branch_warmup_start_ratio=0.3`
- `time_branch_warmup_ratio=0.3`
- `use_spatial_branch=false`
- `run_stage3=true`
- `latent_decorr_weight=0.3`
- `latent_decorr_top_k=512`

与主模型对比：

```text
main_001:
  output_root=train/output_time_latentdecorr_x8_top20_half
  latent_decorr_weight=0.01
  latent_decorr_top_k=256
  stage3 mean_recon=0.369844
  stage3 val_recon=0.379780
  stage3 mean_align=0.095272
  stage3 val_align=0.099513
  stage3 mean_latent_decorr=0.009788
  stage3 loss_latent_decorr_term mean≈0.000098
  stage3 dead_feature_frac last≈0.024707

decorr03:
  output_root=train/output_time_latentdecorr_x8_top20_decorr03
  latent_decorr_weight=0.3
  latent_decorr_top_k=512
  stage3 mean_recon=0.369714
  stage3 val_recon=0.379391
  stage3 mean_align=0.094896
  stage3 val_align=0.098965
  stage3 mean_latent_decorr=0.005398
  stage3 loss_latent_decorr_term mean≈0.001619
  stage3 dead_feature_frac last≈0.024805
```

判断：

- 强正则生效了。`mean_latent_decorr` 从 `0.009788` 降到 `0.005398`，约下降 45%。
- 正则项贡献从 `~1e-4` 提升到 `~1.6e-3`，达到预期的 `1e-3` 量级。
- 重建没有变差，反而略好：`val_recon 0.379780 -> 0.379391`。
- 对齐没有变差，略好：`val_align 0.099513 -> 0.098965`。
- dead feature 基本不变：`0.024707 -> 0.024805`。

decoder 方向相关性抽样：

```text
old_x4_no_time:
  n_dirs=5120
  decoder_sample_offdiag_abs_mean≈0.071654
  decoder_sample_offdiag_sq_mean≈0.008995
  dead_frac≈0.014648

main_001:
  n_dirs=10240
  decoder_sample_offdiag_abs_mean≈0.049868
  decoder_sample_offdiag_sq_mean≈0.005591
  dead_frac≈0.024707

decorr03:
  n_dirs=10240
  decoder_sample_offdiag_abs_mean≈0.051339
  decoder_sample_offdiag_sq_mean≈0.005933
  dead_frac≈0.024805
```

解释：

- `decorr03` 主要改善的是 latent 共激活独立性，而不是 decoder 方向正交性。
- decoder 方向相关性相比 `main_001` 没有进一步下降，甚至 sample abs mean 略高。
- 这符合当前 loss 设计：它惩罚 batch 内 feature 共激活，不直接惩罚 decoder 字典方向。

当前结论：

- `decorr03` 是一次成功的强独立性训练。
- 它没有破坏重建、对齐和 dead feature。
- 下一步值得用 `decorr03` 重新做 `car / dog / nudity` 定位与擦除对照。
- 判断重点不是训练 loss，而是擦除实验里是否出现更少副作用、更清晰的局部特征、更好的 dog head/body coverage。
