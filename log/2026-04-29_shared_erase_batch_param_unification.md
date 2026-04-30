# Shared `erase/batch` 参数统一

## 背景

`runtime/shared/erase.py` 和 `runtime/shared/batch.py` 之前各自维护一套干预参数定义，已经出现过默认值漂移：

- `int_scale`
- `int_time_weight_scale`
- 默认 `blocks`
- 时间窗

这会让单张实验和 batch 实验的行为不一致，后续很难复盘。

## 本次修改

### 1. 统一干预参数定义

把共享的 CLI 参数和配置构造逻辑收敛到 `runtime/shared/erase.py`：

- `DEFAULT_INTERVENTION_BLOCKS`
- `add_intervention_args(...)`
- `build_intervention_cfg_from_args(...)`

`runtime/shared/batch.py` 不再自己重复维护干预参数，而是直接复用这套定义。

### 2. 统一默认强度口径

默认值统一为：

- `int_scale = 150`
- `int_use_time_weight = true`
- `int_time_weight_scale = int_scale`（默认绑定）
- `int_t_start = 900`
- `int_t_end = 100`
- `int_feature_top_k = 5`
- `int_use_spatial_weight = true`
- 默认 `blocks`：
  - `unet.down_blocks.2.attentions.1`
  - `unet.mid_block.attentions.0`
  - `unet.up_blocks.0.attentions.0`
  - `unet.up_blocks.0.attentions.1`

### 3. 保留显式覆盖能力

虽然默认把 `int_time_weight_scale` 绑定到 `int_scale`，但 CLI 仍允许显式传 `--int_time_weight_scale` 覆盖，方便后续做消融。

## 结果

现在：

- 单张 `erase`
- 批量 `batch`

在干预参数层面使用的是同一套定义，不会再出现一边改了另一边忘记同步的问题。
