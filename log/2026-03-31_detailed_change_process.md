# 详细变更记录：代码改动过程与思路变化

日期：2026-03-31  
范围：`exp53 / exp54 / exp55 / batch / 文档`

---

## 0. 初始目标（阶段性）

- 让“概念提取 -> 概念擦除 -> 效果对比”成为一条可复现流水线。
- 支持多概念批量跑图，且能比较不同时间段干预效果。
- 降低运行阻塞（重复加载、路径不一致、参数混乱）。

---

## 1. 变更时间线（按思路演进）

### 阶段 A：先做可跑通（功能优先）

#### A1. 增加批量擦除框架

- 变更：
  - 新增 `scripts/run_batch_concept_erase.py` 批量脚本（读取 prompt 集合，按概念循环生成）。
- 直接原因：
  - 需要一次性跑多个概念 + 多个 prompt 的擦除效果。
- 初版取舍：
  - 先保证功能闭环（可跑、可出图），后续再优化性能与目录规范。

#### A2. 概念得分升级（TARIS -> TARIS/SAeUron 并行）

- 变更：
  - 在 `exp53` 中加入 SAeUron 风格分数（含负样本方差惩罚）。
  - 支持 TARIS 与 SAeUron 对比导出。
- 相关文件：
  - `scripts/sdxl_wsae/experiments/exp53_concept_locator_taris.py`
  - `scripts/run_exp53.py`
- 思路：
  - 不替换旧分数，而是并行保留，便于横向比较和回归。

---

### 阶段 B：针对运行体验做改造（效率 + 可靠性）

#### B1. 时间窗对比从“单窗口”升级为“多窗口”

- 触发问题：
  - 仅看单时间段干预，无法观察 early/mid/late 差异。
- 变更：
  - 批量脚本支持 `single/suite`；
  - `suite` 输出 `custom + early + mid + late` 对比结果。
- 结果：
  - 能直观看不同时段干预的影响边界，减少盲调。

#### B2. 重复加载 SDXL 导致耗时过大

- 触发问题：
  - 每条 case 反复初始化 SDXL/SAE，耗时明显。
- 变更：
  - 复用共享 session；同一批任务只加载一次模型。
- 思路变化：
  - 从“每次独立最安全”转向“共享会话优先，必要时再隔离”。

#### B3. 概念字典路径不统一导致读取失败

- 触发问题：
  - `out_concept_dict_*` 与新脚本目录约定不一致，出现 `top_positive_features.csv` 找不到。
- 变更：
  - 标准目录统一为 `concept_dict/<block_short>/<concept>/`。
  - 读取逻辑优先新目录，保留 legacy 回退。
  - 历史字典迁移到 `concept_dict/`。
- 相关文件：
  - `scripts/sdxl_wsae/experiments/exp53_concept_locator_taris.py`
  - `scripts/sdxl_wsae/experiments/exp54_intervention_suite.py`
  - `scripts/sdxl_wsae/cli.py`
  - `scripts/sdxl_wsae/experiments/exp55_noisy_latent_probe.py`

---

### 阶段 C：方法论反思（“补偿策略”讨论与回滚）

#### C1. 曾尝试：编辑量范数补偿

- 初衷：
  - 让不同概念/不同 block 的干预强度更稳定。
- 用户反馈核心疑问：
  - “是否只是把改变量变小，本质无差异？”
  - “去除部分后全局量不守恒，补偿意义不够直接。”

#### C2. 讨论后的判断

- 认同点：
  - 当前阶段优先保证机制直观、可解释、可对比。
- 决策：
  - 回滚补偿策略，移除旧参数接口，恢复纯干预路径。
- 最终表达：
  - `x_new = x ± scale * recon`

#### C3. 思路变化总结

- 从“工程稳态优先（加补偿）”
- 转为“实验可解释优先（先去掉补偿，保持机制简洁）”

---

## 2. 本阶段代码改动清单（主题级）

### 2.1 概念定位与导出

- `scripts/sdxl_wsae/experiments/exp53_concept_locator_taris.py`
  - 输出目录切换到 `concept_dict/...`
  - 保留 TARIS/SAeUron 双分数对比导出能力

- `scripts/run_exp53.py`
  - 批量运行说明与输出路径同步到 `concept_dict/...`

### 2.2 干预与读取路径

- `scripts/sdxl_wsae/experiments/exp54_intervention_suite.py`
  - 优先读取 `concept_dict/...`，并兼容 legacy 目录

- `scripts/sdxl_wsae/experiments/exp55_noisy_latent_probe.py`
  - 概念特征读取路径与新目录规范一致

### 2.3 CLI 与文档

- `scripts/sdxl_wsae/cli.py`
  - 帮助文本与目录约定同步（`concept_dict/...`）

- `README.md`
  - 重写为当前项目版本（去掉旧 UCE 通用描述）
  - 明确当前流程、输入格式、输出位置和常见报错

### 2.4 日志体系

- 新增 `log/` 目录与模板：
  - `log/README.md`
  - `log/TEMPLATE.md`
  - `log/2026-03-26_experiment_journal.md`
  - `log/2026-03-31_detailed_change_process.md`（本文件）

---

## 3. 关键“问题 -> 解决”映射

- 问题：`feature_rank_csv` 缺失  
  解决：先统一 block 与 `concept_dict/<block_short>/<concept>` 对齐，再跑 exp54。

- 问题：命令行换行导致 `--concepts: command not found`  
  解决：单行命令或规范续行，避免尾随空格/非法断行。

- 问题：模型路径加载失败（末尾空格）  
  解决：清理路径字符串，尤其是 `model_id` 末尾空白。

- 问题：`OMP_NUM_THREADS` 非法值  
  解决：`unset OMP_NUM_THREADS` 或设为合法整数。

---

## 4. 当前共识（阶段结论）

- 先保证流程一致性与可复现，优先减少“路径/参数/加载”类噪声。
- 方法上保持最小干预表达，后续再按对比结果决定是否引入更复杂约束。
- 所有思路变化都写入 `log/`，避免“口头决策”丢失。

---

## 5. 下一步建议（可执行）

- 固化 1 套标准回归命令（exp53 + exp54 + batch），每次改动必跑。
- 在日志中增加“对比图路径”字段，确保任何结论都可追溯到具体输出。
- 每次参数变更写“为何不用另一方案”，降低后续反复争论成本。

