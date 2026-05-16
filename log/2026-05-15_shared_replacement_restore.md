# 2026-05-15 Shared Replacement Restore

## 背景

- 当前 `runtime/shared` 主线此前只保留了：
  - `ablation`
  - `projected_ablation`
- 原先实验分支里的 `concept injection` 已经不在主线运行时代码中。
- 当前需求不是只做 `nudity` 擦除，而是支持：
  - 去掉目标概念
  - 同时注入替换概念
  - 典型例子：`nudity -> cloth`
- 这个能力必须是可选的，后续还要做消融，不能把替换逻辑硬编码成默认路径。

## 本次代码改动

恢复并主线化了 replacement 能力，但保持最小接口：

- `runtime/shared/erase.py`
  - 新增 `--injectconcept`
  - `--int_mode` 扩展为：
    - `ablation`
    - `injection`
    - `replace`
    - `projected_ablation`
  - 新增 `--int_inject_scale`
  - `replace` 逻辑为：
    - 减去 `targetconcept` 的概念重建
    - 加上 `injectconcept` 的概念重建
- `runtime/shared/batch.py`
  - 复用同一套 replacement 参数
  - per-case manifest / jsonl 中记录 `injectconcept` 与 `int_inject_scale`
- `runtime/shared/features/hook_ops.py`
  - `InterventionSpec` 扩展为双分支：
    - source feature bundle
    - inject feature bundle
  - hook 内支持 `replace`
- `runtime/shared/features/intervention.py`
  - debug csv 增加注入分支诊断字段
- `target_concept_dict/cloth.json`
  - 新增替换概念 `cloth`

## 当前接口语义

### 1. 纯擦除

```bash
--int_mode ablation
```

- 只去掉 `targetconcept`
- 不需要 `injectconcept`

### 2. 纯注入

```bash
--int_mode injection --injectconcept cloth
```

- 不擦目标概念
- 只沿 `injectconcept` 方向加
- 当前主线暂未作为重点实验，但接口保留

### 3. 擦除 + 替换

```bash
--int_mode replace --injectconcept cloth
```

- 先减 `targetconcept`
- 再加 `injectconcept`

## 为什么说现在“可选”

默认行为仍然是：

```bash
int_mode = ablation
injectconcept = ""
int_inject_scale = -1
```

也就是说：

- 不传 replacement 参数时，主线仍然是原来的纯擦除
- replacement 只有显式启用时才生效

这满足后续消融的基本要求。

## 后续消融建议

建议固定同一 checkpoint、同一 prompt 集、同一 block、同一时间窗，只比较下面三组：

### A. 纯擦除

```bash
--int_mode ablation
```

用途：

- 评估“只删概念”时的移除能力和副作用

### B. 擦除 + 替换

```bash
--int_mode replace --injectconcept cloth
```

用途：

- 评估 replacement 是否比 pure ablation 更能保留画面结构

### C. 同路径但关注入分支

```bash
--int_mode replace --injectconcept cloth --int_inject_scale 0
```

用途：

- 代码路径与 replacement 相同
- 但显式关掉注入量
- 这个对照比直接拿 `ablation` 更严格，适合做实现层面消融

## 当前判断

- replacement 能力已经恢复到当前 Shared 主线
- 默认主线没有被破坏，仍然是 `ablation`
- 现在最值得做的是：
  - 先定位 `cloth`
  - 再跑 `nudity` 的 `ablation / replace / replace+inject=0` 三组对照
- 如果 replacement 有效，后面再扩展到：
  - `bare_torso -> shirt`
  - `full_body_unclothed -> dress`
  - `breast_exposure -> covered_upper_body`

## 验证

本次改动已做基础检查：

- `python -m py_compile runtime/shared/erase.py runtime/shared/batch.py runtime/shared/features/hook_ops.py runtime/shared/features/intervention.py`
- `python -m runtime.shared.erase --help`
- `python -m runtime.shared.batch --help`
- `python -m runtime.shared.batch --dry_run --prompts_path batch_test_prompt/nudity.csv --concepts nudity --injectconcept cloth --int_mode replace --max_prompts 1`
- `git diff --check`
