# 2026-04-29 Shared Mainline Runtime Refactor

## 当前状态

这是一次**主线仍在继续的整理性重构**，不是实验结束归档。

当前项目已经明确收成一条 Shared 主线：

- `SAE/`：模型定义、配置、checkpoint、编码、归一化
- `train/`：训练流程、loss、sampler、trainer
- `runtime/shared/`：概念定位、单图擦除、batch 擦除运行时
- `tools/feature_frequency/`：prompt-conditioned 高频特征统计与 blacklist 生成

旧 `scripts/` 下的 Shared 实现和旧 `sdxl_wsae` 目录已经退出主线，统一归档到：

- [research/archive_experiments/README.md](/root/cce/research/archive_experiments/README.md)

## 这次整理想解决什么

前一阶段主线虽然已经切到 Shared，但代码仍存在两个明显问题：

1. 模型相关代码同时散落在 `SAE/` 和 `train/`  
2. Shared 推理/擦除实现虽然已是正式代码，却仍挂在 `scripts/` 下，入口和实现混在一起

这次整理的目标就是：

- 让 `SAE/` 成为模型相关代码的唯一来源
- 让 `runtime/shared/` 成为 Shared 主线的唯一运行时实现
- 把旧脚本移出主线，但不直接销毁，方便后续复盘

## 关键文件与目录

### 当前主线

- [SAE/README.md](/root/cce/SAE/README.md)
- [SAE/sae.py](/root/cce/SAE/sae.py)
- [SAE/config.py](/root/cce/SAE/config.py)
- [SAE/checkpoint.py](/root/cce/SAE/checkpoint.py)
- [SAE/encoding.py](/root/cce/SAE/encoding.py)
- [SAE/normalization.py](/root/cce/SAE/normalization.py)
- [train/README.md](/root/cce/train/README.md)
- [train/run_train.py](/root/cce/train/run_train.py)
- [train/trainer.py](/root/cce/train/trainer.py)
- [runtime/shared/README.md](/root/cce/runtime/shared/README.md)
- [runtime/shared/pipeline.py](/root/cce/runtime/shared/pipeline.py)
- [runtime/shared/locator.py](/root/cce/runtime/shared/locator.py)
- [runtime/shared/erase.py](/root/cce/runtime/shared/erase.py)
- [runtime/shared/batch.py](/root/cce/runtime/shared/batch.py)
- [runtime/shared/io_utils.py](/root/cce/runtime/shared/io_utils.py)
- [runtime/shared/features/delta.py](/root/cce/runtime/shared/features/delta.py)
- [runtime/shared/features/scoring.py](/root/cce/runtime/shared/features/scoring.py)
- [runtime/shared/features/intervention.py](/root/cce/runtime/shared/features/intervention.py)
- [runtime/shared/features/hook_ops.py](/root/cce/runtime/shared/features/hook_ops.py)
- [tools/feature_frequency/README.md](/root/cce/tools/feature_frequency/README.md)

### 归档历史代码

- [research/archive_experiments/scripts/shared](/root/cce/research/archive_experiments/scripts/shared)
- [research/archive_experiments/scripts/sdxl_wsae/shared_sae](/root/cce/research/archive_experiments/scripts/sdxl_wsae/shared_sae)
- [research/archive_experiments/scripts/sdxl_wsae/core](/root/cce/research/archive_experiments/scripts/sdxl_wsae/core)

## 这次整理按什么顺序做的

### 1. 先统一模型代码所有权

把原来 `train/` 里和模型强相关的平行实现移除，只保留 `SAE/`：

- 删除了 `train/model.py`
- 删除了 `train/config.py`
- 删除了 `train/checkpoint.py`
- 删除了 `train/encoding.py`
- 删除了 `train/normalization.py`

训练层当前直接从 `SAE/` 取：

- `SharedSAE`
- `TrainConfig`
- `save_checkpoint / load_checkpoint`
- `build_coords_norm / check_expected_hw`
- `estimate_block_scales / apply_block_scale`

这样后面改模型时，不再需要在 `SAE/` 和 `train/` 里双改。

### 2. 再把 Shared 主线实现迁到 runtime

原先这些文件都在 `scripts/sdxl_wsae/shared_sae/`：

- `common.py`
- `locator.py`
- `erase.py`
- `batch_erase.py`
- `delta.py`
- `scoring.py`
- `intervention_utils.py`
- `defaults.py`

这批实现现在迁成：

- `runtime/shared/pipeline.py`
- `runtime/shared/locator.py`
- `runtime/shared/erase.py`
- `runtime/shared/batch.py`
- `runtime/shared/defaults.py`
- `runtime/shared/io_utils.py`
- `runtime/shared/features/*`

同时，旧 `scripts/shared/` 入口也从主线移除。

### 3. 处理 hook 基础函数

原本 Shared 擦除还借用了旧的：

- `scripts/sdxl_wsae/core/intervention.py`

为了让主线真正不再依赖 `scripts/`，这部分也一起平移进：

- `runtime/shared/features/hook_ops.py`

### 4. 保留 SDLens，不动底层基础设施

这次没有改：

- [SDLens/hooked_sd_pipeline.py](/root/cce/SDLens/hooked_sd_pipeline.py)
- [SDLens/hooked_scheduler.py](/root/cce/SDLens/hooked_scheduler.py)

原因是当前主线仍明确依赖它提供的 hook/cache 能力。  
本轮整理只做主线收口，不碰底层 hooked pipeline。

### 5. 把旧脚本整体归档而不是直接销毁

从主线视角看，旧 `scripts/` 实现已经移除；  
但为了后续对照和复盘，它们没有被彻底抹掉，而是恢复到：

- `research/archive_experiments/`

这样当前主线干净，同时历史证据还在。

## 现在该怎么用

当前 Shared 主线的命令入口统一成：

### 概念定位

```bash
python -m runtime.shared.locator ...
```

### 单图擦除

```bash
python -m runtime.shared.erase ...
```

### 批量擦除

```bash
python -m runtime.shared.batch ...
```

### 高频统计

```bash
python tools/feature_frequency/run_collect_shared_stats.py ...
python tools/feature_frequency/run_build_blacklist.py ...
```

## 当前结论

这次整理后，项目主线已经比之前清楚很多：

- `SAE/`：模型
- `train/`：训练
- `runtime/shared/`：运行时
- `tools/feature_frequency/`：统计工具
- `research/archive_experiments/`：历史代码

这是一次**结构整理成功，但还没彻底收尾**的中间 checkpoint。  
目前主线能导入、能跑，且旧代码已经退出主路径。

## 还没完全做完的点

1. `runtime/shared/` 内部还能继续细分职责  
   现在虽然已经比 `scripts/` 清楚，但 `pipeline / locator / erase / batch / features` 之间仍有一些私有函数互相调用，后面还可以进一步收紧边界。

2. 根目录产物仍然偏多  
   `concept_dict/`、`concept_dict_freq/`、`feature_frequency/`、`image_output/` 这些输出目录还都平铺在仓库根下，后面可以考虑进一步整理。

3. `research/archive_experiments/` 目前只是归档，没有再细分  
   如果后面历史代码继续增多，可以再按阶段或路线拆一层索引。

## 下一步建议

如果继续整理代码，最合理的下一步是：

1. 继续收 `runtime/shared/` 内部边界
2. 再考虑是否整理根目录输出结构
3. 暂时不要动 `SDLens/`

如果继续做实验，当前建议直接基于：

- [runtime/shared/README.md](/root/cce/runtime/shared/README.md)
- [tools/feature_frequency/README.md](/root/cce/tools/feature_frequency/README.md)

不要再回到旧 `scripts/` 路线。
