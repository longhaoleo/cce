# Shared Nudity 干预消融记录

日期：2026-04-29

## 背景

本轮在 Shared 主线下测试 `nudity` 概念的擦除效果，主要对比两类改动：

- 放大时间权重
- 使用 `projected_ablation` 子空间投影擦除

相关入口：

- `python -m runtime.shared.locator`
- `python -m runtime.shared.batch`

相关概念文件：

- `target_concept_dict/nudity.json`
- `batch_test_prompt/nudity.csv`

## 结论

### 1. 时间权重放大有效

经验结论：

- 在 `nudity` 概念上，开启时间权重并把 `int_time_weight_scale` 放大到 `80` 后，擦除效果明显更好。
- 相比不放大或较弱放大版本，这一版更像“细粒度抑制目标概念”，而不是粗暴破坏整体图像。

当前判断：

- 对 `nudity` 这类复合视觉概念，时间窗内的 step 权重是有用的。
- 合适地放大时间权重，可以让干预更加聚焦，减少“整图一起被拉坏”的感觉。

### 2. projected_ablation 当前效果不好

经验结论：

- 加入 `projected_ablation` 子空间投影后，当前结果不理想。
- 它会更明显地干扰原图生成，导致人物、构图或整体视觉质量被破坏。

当前判断：

- 对这版 SharedSAE 和当前 `nudity` 概念统计结果来说，子空间投影并没有带来更干净的概念擦除。
- 反而更像是在把原图表示的其他有用成分一起删掉。

## 当前阶段建议

- `nudity` 主线优先继续沿用普通 `ablation`
- 时间权重可以保留，并优先测试较大的 `int_time_weight_scale`
- `projected_ablation` 暂时不作为默认主线配置

## 最近结果检查补充

对 `image_output` 里最近三组 `nudity` batch 结果做了快速检查：

- `batch_shared_concept_erase_nudity`
- `batch_shared_concept_erase_nudity_tw`
- `batch_shared_concept_erase_nudity_proj_tw1000-0`

检查结论：

### 1. 最有参考价值的是时间权重版

- `batch_shared_concept_erase_nudity_tw` 是当前唯一完整跑完 `20` 条 case 的一组。
- 普通版 `batch_shared_concept_erase_nudity` 只完成了 `9` 条。
- `proj` 目录不是干净对照组，里面混入了 `ablation` 和 `projected_ablation` 两种模式，不能直接拿目录名当作真实配置。

### 2. 时间权重版总体判断仍然成立

- `ablation + int_time_weight_scale=80` 确实能更细粒度地压制 `nudity` 概念。
- 一部分 case 能做到“裸露概念减弱，同时人物或主体还保留一部分结构”。
- 但它仍然会在更强 prompt 上产生明显副作用，常见表现是：
  - 人体被重写
  - 构图被抽象化
  - 场景结构被大改

所以当前更准确的说法是：

- 概念抑制有效
- 但仍未达到低副作用、稳定保留原图的状态

### 3. projected_ablation 仍然不适合作为主线

- 当前真正成功写出 manifest 的 `projected_ablation` 样本很少。
- 从已完成的投影样本看，它会更明显地伤原图结构，不比普通 `ablation + 强时间权重` 更干净。
- 结论不变：暂时不把它当默认主线。

## 下一步

- 用新的干净输出目录，重新跑完整的 `nudity` batch 对照
- 主线继续保留：
  - `ablation`
  - 时间权重
  - `int_time_weight_scale=80`
- 下一轮把 `mid.0` 单独加回去测试，重点看：
  - 激活强弱
  - 对原图副作用是否上升
  - 是否真的提供额外概念信号

## 推荐解释口径

当前可以把这轮结果理解为：

- 时间权重放大：有效，而且能带来更细粒度的概念抑制
- 子空间投影：当前副作用较大，不适合作为默认擦除方式
