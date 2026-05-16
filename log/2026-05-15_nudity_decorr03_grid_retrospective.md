# Nudity Decorr03 Grid Retrospective

日期：2026-05-15

## 状态

这是一轮已完成的诊断实验，用来回答一个具体问题：

> `decorr03` 上 `nudity` 擦除变差，是否只是 blacklist、`top_k` 或 scale 没调好？

相关材料已整理到：

```text
research/nudity_decorr03_grid/
```

## 实验设置

checkpoint：

```text
train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772
```

网格：

- blacklist：
  - `q99_50`
  - `ar95_all`
  - `ar90_all`
- `top_k`：
  - `10 / 15 / 20`
- `scale`：
  - `3000 / 5000`

输出：

```text
image_output/sae_x8_time_decorr03/nudity_grid/
concept_dict_grid/sae_x8_time_decorr03/nudity_*/
concept_dict_freq/sae_x8_time_decorr03_*/
```

18 组全部完成，每组 20 个 case，共 360 个 manifest。

## 数值结果

代表性组合的总 `delta_over_x` 均值：

| 设置 | 均值 |
| --- | ---: |
| `ar90_all_top10_scale3000` | `0.174` |
| `ar95_all_top10_scale3000` | `0.241` |
| `q99_50_top10_scale3000` | `0.324` |
| `q99_50_top15_scale5000` | `0.663` |
| `q99_50_top20_scale5000` | `0.746` |

blacklist 强度变化也符合预期：

- `q99_50`：每层 52 个 blacklist feature
- `ar95_all`：约 `183 / 808 / 421 / 241`
- `ar90_all`：约 `277 / 981 / 522 / 311`

`nudity` top10 的全局活跃率：

- `q99_50`：平均 `0.896`，`27/40` 个 feature 的 `active_ratio >= 0.9`
- `ar95_all`：平均 `0.790`
- `ar90_all`：平均 `0.724`

## 视觉观察

- `q99_50` 干预最强，但加到 `top15/20 + scale5000` 后，很多 case 已经开始换构图、抽象化或大幅重写原图；同时仍不能保证稳定移除 nudity 语义。
- `ar95_all / ar90_all` 的图像保真度更好，但擦除明显偏弱。
- 所以这轮网格只证明“控制旋钮有效”，没有证明“现有特征集可通过调参修好”。

补充观察：

- blacklist 越严格，图像副作用越小。
- `q99_50` 保留了最多高频 feature，擦除最猛，但也最容易把整张图带乱。
- `ar95_all` 是目前更合理的中间档：相比 `q99_50`，保图更好；相比 `ar90_all`，又没有把擦除能力压得过低。
- `ar90_all` 已经开始表现出“过严”的迹象：图像最稳，但更容易把真正需要的 nudity feature 一并过滤掉。
- 因此下一轮更合理的主线不是继续无脑收紧 blacklist，而是先把概念拆细、把 feature 找准，再在较严格 blacklist 下验证是否能同时保留擦除能力与图像保真。

当前默认 blacklist 规则据此切到 `ar95_all`：

```text
blacklist_freq_threshold = 0.0
blacklist_active_ratio_min = 0.95
blacklist_max_features = 0
```

理由不是它最强，而是它在这轮实验里给出了当前最合理的折中：比 `q99_50` 更少副作用，又没有像 `ar90_all` 那样明显压低有效干预。

## 判断

当前瓶颈更像是概念定位，而不是干预强度：

- `nudity` 是复合属性，不像 `dog / car` 那样接近单一物体。
- 单组 prompt 把裸露皮肤、全身无衣、胸部暴露、古典裸体、剪影裸体等子语义混在了一起。
- raw TARIS 仍容易把高频泛化 feature 排到前面；这在 `q99_50` 下尤其明显。
- 因此，继续只扩大 `top_k` 或 scale，会在“擦不掉”和“把图打坏”之间移动，不能真正解决问题。

## 下一步

1. 把 `nudity` 拆成多个细粒度子概念。
2. 修改 `locator`，让一个父概念可以包含多个子概念：
   - 子概念分别定位；
   - 保留子概念输出；
   - 再按 feature-wise max 聚合回父概念，供现有擦除流程继续使用。
3. 重跑 `nudity` 定位与 batch 擦除，检查：
   - top feature 的 `active_ratio` 是否下降；
   - 多个子语义是否覆盖得更均衡；
   - 同等擦除效果下，是否能用更小的 `top_k / scale`。
