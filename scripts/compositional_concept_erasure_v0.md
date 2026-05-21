# Compositional Concept Erasure v0

目标不是先追求最好结果，而是先判断组合概念在 Shared SAE feature 空间里更像：

- 原子概念 feature 的 union
- 新出现的 emergent feature cluster
- 需要条件触发的 feature interaction

## 1. 组合概念

第一轮只跑三个组合，覆盖三种结构：

- `dog_glasses = dog with glasses`
- `red_car = red + car`
- `flower_van_gogh = flower + style`

对应原子概念：

```text
dog_glasses     dog       glasses
red_car         red       car
flower_van_gogh flower    van_gogh
```

## 2. 一键实验

默认每组只跑 5 条 prompt，先看当前方法是否有信号。

```bash
cd /root/cce

./scripts/run_compositional_v0.sh
```

常用覆盖项：

```bash
MAX_PROMPTS=20 \
INT_TOP_K=5 \
TIMESTEP_WINDOW_START=1000 \
TIMESTEP_WINDOW_END=300 \
./scripts/run_compositional_v0.sh
```

输出位置：

```text
image_output/sae_x8_time_decorr03/compositional_v0/
```

## 3. 这轮看什么

每个组合概念会跑五组：

- `target_<composite>`：擦组合概念，看目标压制。
- `preserve_<A>`：在 A-only prompt 上擦组合概念，看 A 是否被误伤。
- `preserve_<B>`：在 B-only prompt 上擦组合概念，看 B 是否被误伤。
- `atomic_<A>_on_<composite>`：只擦 A，看组合图被破坏到什么程度。
- `atomic_<B>_on_<composite>`：只擦 B，看组合图被破坏到什么程度。

如果 `target` 有效而 `preserve_A/B` 保留较好，说明组合 feature 可能有独立结构。

如果 `target` 无效但 atomic baseline 有效，说明组合定位不够强，或组合概念只是由原子 feature 支撑。

如果 `preserve_A/B` 被明显破坏，说明当前组合 feature 和原子 feature 仍高度纠缠。

## 4. Feature 结构分析

脚本会自动生成：

```text
image_output/sae_x8_time_decorr03/compositional_v0/feature_overlap.csv
```

字段重点看：

- `union_coverage`：组合 top features 被 A/B 原子 feature 覆盖的比例。
- `new_feature_ratio`：组合 top features 中不属于 A/B union 的比例。
- `atomic_overlap`：A 和 B 本身 feature 是否纠缠。

解释：

```text
new_feature_ratio 高：
组合更像 emergent feature cluster。

union_coverage 高：
组合更像原子 feature union。

atomic_overlap 高：
A/B 本身纠缠，条件擦除可能更必要。
```

## 5. 下一步判据

这轮只要回答三个问题：

1. 哪个组合最像独立概念？
2. 哪个组合最容易误伤单独 A/B？
3. `flower_van_gogh` 是否比单独 `flower` 更容易擦，还是更难？

如果 `dog_glasses` 或 `red_car` 出现“target 有效、A/B 保留好”，下一步做 gated erasure。

如果 `flower_van_gogh` 继续差，优先排查 feature budget：

```text
INT_TOP_K=2 / 5 / 10
TIMESTEP_WINDOW=1000..300 / 800..100 / 1000..0
```
