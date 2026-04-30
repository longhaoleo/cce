# SDXL Base Shared SAE 训练计划 v1

## 摘要

目标是为 **SDXL Base** 训练一套可复用的 **Shared Sparse Autoencoder (Shared SAE)**，供后续概念定位与概念擦除研究使用。  
本计划只覆盖**训练模型本身**，不绑定具体擦除方法；但模型接口、配置和 checkpoint 必须为后续擦除与继续训练预留能力。

本方案的核心设计如下：

- 训练对象是 SDXL Base 的 4 个 U-Net attention block：
  - `unet.down_blocks.2.attentions.1`
  - `unet.mid_block.attentions.0`
  - `unet.up_blocks.0.attentions.0`
  - `unet.up_blocks.0.attentions.1`
- 默认空间基线对齐当前论文复现实验使用的 `512-space` 设置：
  - `resolution = 512`
  - 以上 4 个目标 block 默认落在 `16 x 16`
  - 因此默认 `N_tok = 256`
- 4 个 block 共用 **一个 latent dictionary**，因此 `feature_id` 在 4 层之间**全局共享**。
- 输入表征固定为 **delta 更新量**：
  - `delta = h_out - h_in`
  - 只使用 conditional branch
- 采用 **共享 SAE 主干 + block-specific 输入适配器** 的结构：
  - `block_in_adapter` 使用 **LoRA**
  - `block_out_adapter` 第一版仅保留接口，默认不进入正式训练主线
- 加入 **时间感知** 与 **空间感知**：
  - 时间和空间都作用在 encoder pre-activation
  - 第一版默认时间与空间**分开建模**
  - 时间分支预设 3 种模式，默认使用 `sincos_linear`
  - 空间位置编码基底固定为 **2D 正余弦编码**
- 损失函数固定包含：
  - `MSE reconstruction loss`
  - `AuxK loss`
  - `alignment loss`
- 训练数据固定来自 **LAION-COCO prompts**
- 文档目标路径：`train/PLAN.md`

本计划中，凡是“固定设计决策”的内容都不再留给实现者选择；凡是“实验超参数”的内容都列出默认值，并要求保留配置接口以支持消融。

---

## 一、模型目标与固定设计决策

### 1. 训练目标

训练一套 Shared SAE，使其能够：

- 在 4 个目标 block 上共享一套稀疏 feature dictionary
- 让同一个 `feature_id` 在不同 block 中尽量对应相近语义
- 保留 block 间必要差异，避免被硬共享压平
- 为后续概念定位、特征追踪和概念擦除提供统一 latent 空间

### 2. 本轮明确不做的事情

以下内容不在 v1 训练计划范围内：

- 不实现最终概念擦除策略
- 不设计 prompt-conditioned masking
- 不实现 block_out_adapter 的正式训练方案
- 不引入卷积式 patch mixer 或邻域聚合器
- 不改动现有老 SAE checkpoint 的兼容逻辑

### 3. 共享方式

固定采用以下结构，不再保留别的共享方案：

- 共享：
  - `shared_encoder`
  - `shared_decoder`
  - `pre_bias`
  - `latent_bias`
  - `time_branch`
  - `spatial_branch`
- block-specific：
  - `block_in_adapter`
  - `block_out_adapter` 预留接口

不采用“4 层完全硬共享且无 adapter”的方案。

### 4. 输入表示

每个样本的核心输入为：

- `x = delta = h_out - h_in`
- 只取 conditional branch
- 每个 timestep 下，每个 block 的特征图都展平为 token 序列
- token 数固定写作 `N_tok = H × W`，其中 `(H, W)` 由实际采样结果决定
- 默认对齐当前 `512-space` 基线，目标 block 应为 `16 x 16`，即每步 `256` 个 token
- 若显式切换到更高分辨率（例如 `1024`），这些 block 才会扩展到更大的网格，例如 `32 x 32`；这种运行应视为偏离默认 baseline 的扩展实验

实现要求：

- 默认配置写死为：
  - `resolution = 512`
  - `expected_h = 16`
  - `expected_w = 16`
- 训练日志必须输出首次观测到的 `(H, W)` 与 `tokens_per_group`
- 若用户显式配置 `expected_h/expected_w`，则采样结果与配置不一致时必须直接报错并停止
- 若用户把 `expected_h/expected_w = 0`，则表示按真实采样自动探测；这属于偏离默认 baseline 的显式选择，而不是默认行为

---

## 二、模型结构与公式

### 1. 基本符号

对单个 token 定义：

- `x ∈ R^d`：输入 token 的 delta 向量
- `d = d_model = 1280`
- `m = n_dirs`：latent feature 数
- `b`：block id
- `t`：scheduler timestep
- `(u, v)`：该 token 的二维归一化坐标
- `k = top_k`：每个 token 保留的稀疏特征数

### 2. block 输入适配器

每个 block 有一个 LoRA 输入适配器：

```text
x' = x_norm + (alpha_in / rank_in) * B_b(A_b(x_norm))
```

其中：

- `x_norm` 是 block 归一化后的输入
- `A_b: R^1280 -> R^rank_in`
- `B_b: R^rank_in -> R^1280`
- `rank_in = block_in_rank`
- `alpha_in = block_in_alpha`

要求：

- LoRA 分支应保持**整体残差零初始化**
- 推荐仅对 `B_b`（up/out 投影）做零初始化，`A_b`（down/in 投影）保留正常初始化
- 初始时整体行为尽量接近 identity
- 必须保证 adapter 从第一个优化 step 起就能获得非零梯度，不能出现“初始化后长期学不动”的情况

### 3. 时间分支

时间输入使用 scheduler 的真实 timestep，而不是 step index。

归一化：

```text
t_norm = t / 1000
```

时间编码：

```text
e_t = PE_1D_sincos(t_norm)
```

三种时间分支模式都要实现，但 v1 默认启用 `sincos_linear`：

1. `sincos_linear`
```text
b_t = Linear(e_t)
```

2. `sincos_mlp`
```text
b_t = MLP_t(e_t)
```

3. `sincos_film`
```text
(γ_t, β_t) = MLP_t_film(e_t)
```

### 4. 空间分支

空间位置以实际 `(H, W)` grid 的 patch 中心表示：

```text
u = 2 * (row + 0.5) / H - 1
v = 2 * (col + 0.5) / W - 1
```

二维位置编码：

```text
e_p = PE_2D_sincos(u, v)
```

三种空间分支模式都要实现，但 v1 默认启用 `sincos_linear`：

1. `sincos_linear`
```text
b_p = Linear(e_p)
```

2. `sincos_mlp`
```text
b_p = MLP_p(e_p)
```

3. `sincos_film`
```text
(γ_p, β_p) = MLP_film(e_p)
```

### 5. Shared SAE 主干

先计算共享 encoder pre-activation：

```text
base = W_enc(x' - c_pre) + b_lat
```

其中：

- `W_enc ∈ R^(m × d)`
- `W_dec ∈ R^(d × m)`
- `c_pre ∈ R^d`
- `b_lat ∈ R^m`

#### 默认模式：时间和空间分开、时间/空间都为 bias 型

当时间分支为 `sincos_linear` 或 `sincos_mlp`，且空间分支为 `sincos_linear` 或 `sincos_mlp` 时：

```text
p = base + b_t + b_p
z_pre = ReLU(p)
z = TopKKeep(z_pre, top_k)
x_hat = W_dec z + c_pre
```

#### 备用模式 1：时间为 FiLM，空间为 bias 型

当时间分支为 `sincos_film`，且空间分支为 `sincos_linear` 或 `sincos_mlp` 时：

```text
p = (1 + γ_t) · base + β_t + b_p
z_pre = ReLU(p)
z = TopKKeep(z_pre, top_k)
x_hat = W_dec z + c_pre
```

#### 备用模式 2：空间为 FiLM，时间为 bias 型

当时间分支为 `sincos_linear` 或 `sincos_mlp`，且空间分支为 `sincos_film` 时：

```text
p = (1 + γ_p) · (base + b_t) + β_p
z_pre = ReLU(p)
z = TopKKeep(z_pre, top_k)
x_hat = W_dec z + c_pre
```

#### 备用模式 3：时间与空间都为 FiLM

当时间分支和空间分支都为 `sincos_film` 时：

```text
p = (1 + γ_p) · ((1 + γ_t) · base + β_t) + β_p
z_pre = ReLU(p)
z = TopKKeep(z_pre, top_k)
x_hat = W_dec z + c_pre
```

### 6. decoder 约束

`W_dec` 保持 unit-norm dictionary：

- 每个训练 step 前，对 decoder gradient 做平行分量投影消除
- optimizer step 后，对 `W_dec` 按列重新归一化

要求与当前项目已有 Top-K SAE 的 decoder 规范保持一致。

### 7. block_out_adapter

v1 中：

- 模型结构中必须定义 `block_out_adapter`
- 但默认不开启
- 前向行为固定为 identity
- 不参与 v1 正式训练
- 若要验证续训接口，应作为显式打开的可选探针，而不是默认训练阶段

这样可以保证未来继续训练时，直接加载 v1 checkpoint 并启用 out adapter。

---

## 三、损失函数

总损失固定为：

```text
L_total = L_recon + λ_aux * L_auxk + λ_align * L_align
```

训练日志要求同时记录：

- `loss_recon`
- `loss_auxk`
- `loss_align`
- `auxk_term = λ_aux * L_auxk`
- `align_term = λ_align * L_align`
- Stage2/Stage3 验证集上的 `val_recon`
- Stage2/Stage3 验证集上的 `val_align`

避免只看原始 loss 数值而误判各项对总目标的真实贡献。

### 1. 重建损失 `L_recon`

使用标准均方误差：

```text
L_recon = mean(||x_hat - x_norm||_2^2)
```

说明：

- 对 token 和 channel 全部取平均
- 不采用 Huber loss
- 不采用 cosine reconstruction

### 2. AuxK 损失 `L_auxk`

AuxK 目标是用额外的 dead-feature 通道去重建主残差，保留 dead feature 复活能力。

定义：

```text
r = stopgrad(x_norm - x_hat)
z_aux = TopKKeep(ReLU(p_dead), auxk)
x_aux = W_dec z_aux
L_auxk = mean(||x_aux - r||_2^2)
```

其中：

- `p_dead` 是对长期未激活特征施加 mask 后得到的 pre-activation
- dead-feature 统计语义沿用当前项目已有 SAE 机制
- AuxK 只能重建 `stopgrad residual`，不能直接替代主重建路径
- 若当前 batch 中没有满足阈值的 dead feature，则允许 `x_aux = None` 且 `L_auxk = 0`；此时日志必须如实记录为 0，不能退化成对主残差的伪重建项

### 3. 对齐损失 `L_align`

对齐目标是让 4 个 block 在同一 `prompt + timestep` 下的 pooled latent 更接近，但不强制 token 级一一对应。

对同一 `prompt_id + step_idx + timestep`，四个 block 的 latent 分别为：

```text
z_down, z_mid, z_up0, z_up1 ∈ R^(N_tok × m)
```

先按 token 求平均：

```text
m_b = mean_tokens(z_b)
q_b = m_b / (||m_b||_2 + eps)
```

再以 `mid_block` 为锚点定义：

```text
L_align =
  ||q_down - q_mid||_2^2
+ ||q_up0  - q_mid||_2^2
+ ||q_up1  - q_mid||_2^2
```

说明：

- 只做 block-level pooled alignment
- 不做 token-wise alignment
- 不做 all-pairs alignment
- 锚点固定是 `mid_block`
- 这里的 `||·||_2^2` 指真实的平方 L2 范数，不应再按 latent 维度额外取 mean；否则对齐项会被错误缩小

---

## 四、数据管线与训练设置

### 1. 数据来源

训练数据固定为：

- 数据集：`LAION-COCO prompts`
- 模型：`stabilityai/stable-diffusion-xl-base-1.0`

每个 prompt 触发一次 SDXL Base 采样轨迹，并缓存 4 个目标 block 的输入输出。

### 2. 采样默认超参数

这些属于**实验超参数**，必须在配置中显式记录：

- `model_id = stabilityai/stable-diffusion-xl-base-1.0`
- `steps = 50`
- `guidance_scale = 8.0`
- `resolution = 512`
- `seed = base_seed + prompt_idx`
- `base_seed = 42`


### 3. step 采样策略

为了控制数据量，不保存每个 prompt 的全部 50 个 step，而是固定采用时间桶采样：

- 将 50 个 denoising steps 分成 5 个时间桶
- 每个桶随机采 1 个 step
- 因此每个 prompt 最多保留 5 个 step group

每个被保留的 group 包含：

- 同一个 `prompt_id`
- 同一个 `step_idx`
- 同一个 `timestep`
- 4 个 block 的 `[N_tok, 1280]` token 张量
- 一份共享 `coords_norm [N_tok, 2]`

### 4. 数据归一化

为减小 4 个 block 在数值尺度上的差异、避免共享 latent 被某一层的高幅值输入主导，对每个 block 使用一个**离线预统计的固定标量缩放系数** s_b。

对 block b，先在 calibration 子集上统计单个 token 的 L2 norm 平均值：

```
μ_b = (1 / N_b) * Σ_i ||x_i^(b)||_2 
```

其中：

- x_i^(b) 是 block b 上的原始 delta token
- N_b 是该 block 在 calibration 子集上的 token 总数
- 不做逐维标准化，不做 whitening，不做去均值，仅统计原始 token 向量的 L2 norm

然后定义 block 级缩放系数：

```
s_b = sqrt(d_model) / μ_b 
```

归一化后的输入为：

```
x_norm = s_b * x 
```

其目标是让各 block 满足：

```
E[||x_norm||_2] ≈ sqrt(d_model) 
即：
x_norm = s_b * x
s_b = sqrt(d_model) / E[||x||_2]
```

这样可以在尽量不破坏 delta 方向语义的前提下，将 4 个 block 的输入尺度拉到同一量级，便于训练共享 dictionary。

#### Calibration 子集默认设置

s_b 的统计使用固定 calibration 子集完成，默认配置为：

- 1000 个 prompts
- 每个 prompt 采 5 个时间桶，每桶随机保留 1 个 step
- 每个被保留 step 使用该 block 的全部 `N_tok = H × W` 个 token

因此每个 block 默认大约统计：

```
1000 × 5 × N_tok tokens
```

这个规模足以稳定估计 μ_b。

默认 `512-space` baseline 下：

```text
N_tok = 256
1000 × 5 × 256 = 1,280,000 tokens
```

若后续显式切到 `1024` 分辨率，目标 block 可能扩展为 `32 x 32`，则：

```text
N_tok = 1024
1000 × 5 × 1024 = 5,120,000 tokens
```

#### 实现要求

- s_b 必须在训练前离线统计完成，不能在训练中动态更新
- 每个 block 单独统计一个固定 s_b
- 训练和未来推理必须使用同一组 s_b
- norm_scale_by_block 必须写入训练配置和 checkpoint 元数据
- 统计脚本应额外记录：
  - mean_l2_norm
  - num_tokens
  - 可选诊断项 p95_l2_norm
- p95_l2_norm （ norm 的 95 分位数）仅用于诊断，不参与 s_b 的定义

### 5. 训练 batch 组织

训练不采用“完全随机打散 token”的方式，而采用 **group batch**。  
这里的一个 `group`，指的是：

- 同一个 `prompt_id`
- 同一个 `step_idx / timestep`
- 在这个时刻对应的一个或多个 block 的全部 patch token

之所以这样组织，是因为本方案里有 `L_align`：  
它需要比较**同一个 prompt、同一个去噪时刻**下，不同 block 的 latent 表示。如果把 token 全部随机打散成普通 batch，就无法在同一个 batch 中方便地构造这个对齐项。

#### Stage 1 的 group 定义

Stage 1 只使用 `mid_block`，此时一个 group 包含：

- `1` 个 block
- `N_tok` 个 token
- 每个 token 维度为 `1280`

因此单个 group 的张量规模可以理解为：

```text
[1, N_tok, 1280]
```

#### Stage 2 / Stage 3 的 group 定义

Stage 2 和 Stage 3 同时使用 4 个 block。  
此时一个 group 包含：

- 同一个 `prompt_id`
- 同一个 `step_idx / timestep`
- 4 个 block
- 每个 block 各有 `N_tok` 个 token

因此单个 group 的张量规模可以理解为：

```text
[4, N_tok, 1280]
```

####  Stage 1 和 Stage 2 的 batch size 

 Stage 1 和 Stage 2 单个 group 的体积不同：

- Stage 1 的 group 只有 `1 × N_tok` tokens
- Stage 2/3 的 group 有 `4 × N_tok` tokens
- 设 tokens_per_step_target = 4096。（对齐其他论文）

  - 一般公式：

```text
group_bs_stage1 = floor(tokens_per_step_target / (1 × N_tok))
group_bs_stage2 = floor(tokens_per_step_target / (4 × N_tok))
```

  - 默认 `512-space` baseline 下，`N_tok = 256`：

```text
group_bs_stage1 = floor(4096 / 256) = 16
group_bs_stage2 = floor(4096 / 1024) = 4
```

  - 若后续显式切到 `32 x 32`，则 `N_tok = 1024`：

```text
group_bs_stage1 = floor(4096 / 1024) = 4
group_bs_stage2 = floor(4096 / 4096) = 1
```

#### 实现要求

- dataloader 的最小采样单位应当是 `group`，不是单独 token
- Stage 2/3 必须保证同一 group 内 4 个 block 的样本严格来自同一个 `prompt_id + step_idx + timestep`
- `group_bs_stage1` 和 `group_bs_stage2` 都必须作为显式配置项保留，便于后续消融或按显存调整
- 训练开始时必须打印一次实际 `hw`、`tokens_per_group`、`tokens_per_step` 与按 `tokens_per_step_target` 推导出的建议 `group_bs`

### 6. 数据规模默认值

v1 默认训练语料来自 data/coco_30k.csv，总规模约 30,000 条 prompt。

数据划分固定如下：

- validation_prompts = 1000
- stage2_train_prompts = 20000
- stage1_train_prompts = 5000
- calibration_prompts = 1000
- num_step_buckets = 5
- shard_prompts = 250

划分规则为：

1. 先从全部 30,000 条 prompt 中固定切出 1000 条作为 validation split。
2. validation split 必须与所有训练数据和 calibration 数据**严格不重叠**。
3. 剩余 29,000 条构成 train pool。
4. 从 train pool 中抽取 20,000 条作为 Stage 2 / Stage 3 的训练集。
5. Stage 1 的 5,000 条 prompt 直接取自 Stage 2 / Stage 3 训练集的子集。
6. calibration 的 1,000 条 prompt 也从 train pool 中抽取，允许与 Stage 1 / Stage 2 / Stage 3 训练集重叠。

因此默认关系为：

```
30,000 total prompts 
├─ 1,000 validation (strictly disjoint) 
└─ 29,000 train_pool   
	├─ 20,000 stage2/stage3 train   
	│  └─ 5,000 stage1 warmup subset   
	└─ 1,000 calibration subset 
```

#### 设计说明

- validation 必须严格独立，用于 held-out reconstruction 和训练后评估。
- Stage 1 是 shared trunk 的预热阶段，不需要与 Stage 2 / Stage 3 额外去重，因此直接作为其子集。
- calibration 仅用于估计 block 级归一化系数 s_b，不承担泛化评测职责，因此允许与训练集重叠。
- num_step_buckets = 5 表示每条 prompt 默认从 50 个 denoising steps 中按时间桶采样 5 个 step。
- shard_prompts = 250 表示每个 activation shard 默认包含 250 条 prompt，对应提取、训练和删除的最小数据分片单位。

---

## 五、训练阶段

### Stage 1：共享主干预热

目的：
- 先在 `mid_block` 上学到稳定的共享 dictionary
- 先让时间/空间分支收敛到合理范围

设置：

- block：只用 `unet.mid_block.attentions.0`
- 训练参数：
  - `shared_encoder`
  - `shared_decoder`
  - `pre_bias`
  - `latent_bias`
  - `time_branch`
  - `spatial_branch`
- 冻结：
  - `block_in_adapter`
  - `block_out_adapter`
- 损失权重：
  - `λ_aux = auxk_coef = 1/32`
  - `λ_align = 0`
- 训练轮数：
  - `epochs_stage1 = 1.0`

### Stage 2：四层联合训练

目的：
- 将 4 个 block 对齐到共享 latent 空间
- 学到 block-specific 输入修正

设置：

- block：4 层同时参与
- 解冻：
  - `block_in_adapter`
- 继续冻结：
  - `block_out_adapter`
- 训练参数：
  - shared 主干
  - `block_in_adapter`
  - `time_branch`
  - `spatial_branch`
- 损失权重：
  - `λ_aux = 1/32`
  - `λ_align` 从 `0` warmup 到 `5e-2`
- warmup：
  - `align_warmup_ratio = 0.1`
- 训练轮数：
  - `epochs_stage2 = 1.0`

### Stage 3：稳定化微调

目的：
- 降低 feature 漂移
- 稳定 block 间共享 feature 语义

设置：

- block：仍为 4 层联合
- 学习率下调：
  - `lr_shared_stage3 = 2e-5`
  - `lr_adapter_stage3 = 1e-4`
  - `lr_time_stage3 = 2e-5`
  - `lr_spatial_stage3 = 2e-5`
- 损失权重：
  - `λ_align = 5e-2`
  - `λ_aux = 1/32`
- 训练轮数：
  - `epochs_stage3 = 0.1`

### Stage 4：可选续训接口验证

目的：
- 验证 v1 checkpoint 可以无缝打开 `block_out_adapter`
- 验证未来继续训练不会受阻

设置：

- 默认主训练流程不执行本阶段
- 仅在用户显式打开探针开关时执行
- 从 v1 checkpoint 恢复
- 开启 `use_block_out_adapter = true`
- 冻结：
  - shared 主干
  - `block_in_adapter`
  - `time_branch`
  - `spatial_branch`
- 只训练：
  - `block_out_adapter`
- 小规模 smoke finetune 即可：
  - `epochs_stage4 = 0.02`

---

## 六、优化器与实验超参数

以下为 v1 默认实验超参数，必须全部记录到配置与 checkpoint：

### 1. 结构超参数

- `d_model = 1280`
- `expansion_factor = 4`
- `n_dirs = 5120`
- `top_k = 10`
- `auxk = 256`

### 2. Adapter 超参数

- `use_block_in_adapter = true`
- `use_block_out_adapter = false`
- `run_stage4 = false`
- `block_adapter_type = lora`
- `block_in_rank = 16`
- `block_in_alpha = 16`
- `block_out_rank = 16`
- `block_out_alpha = 16`

### 3. 时间分支超参数

- `time_branch_mode = sincos_linear`
- `time_pos_encoding = sincos_1d`
- `time_embed_dim = 32`
- `time_hidden_dim = 128`

备用消融模式也必须支持：

- `sincos_mlp`
- `sincos_film`

### 4. 空间分支超参数

- `spatial_pos_encoding = sincos_2d`
- `spatial_embed_dim = 64`
- `spatial_hidden_dim = 128`
- `spatial_branch_mode = sincos_linear`

备用消融模式也必须支持：

- `sincos_mlp`
- `sincos_film`

### 5. 优化超参数

- optimizer：`Adam`
- `beta1 = 0.9`
- `beta2 = 0.999`
- `eps = 6.25e-10`
- `weight_decay = 0.0`
- `lr_shared = 1e-4`
- `lr_adapter = 2e-4`
- `lr_time = 1e-4`
- `lr_spatial = 1e-4`
- `grad_clip = 1.0`
- `decoder_unit_norm = true`

### 6. 损失超参数

- `auxk_coef = 1/32`
- `align_weight_target = 5e-2`
- `align_warmup_ratio = 0.1`
- `dead_tokens_threshold = 10_000_000`

---

## 七、接口与配置要求

### 1. 模型 API

Shared SAE 的接口固定为：

```python
encode(x, *, block_id, timestep, coords_norm, hw)
forward(x, *, block_id, timestep, coords_norm, hw)
```

要求：

- `block_id` 使用稳定映射，不依赖列表顺序隐式推断
- `coords_norm` 必须由数据管线显式传入
- `hw` 必须进入接口，即使 v1 只用于校验

### 2. Checkpoint 配置字段

checkpoint `config.json` 至少要记录：

- `shared_sae = true`
- `global_feature_space = true`
- `blocks = [...]`
- `d_model`
- `expansion_factor`
- `n_dirs`
- `top_k`
- `auxk`
- `auxk_coef`
- `dead_tokens_threshold`
- `use_block_in_adapter`
- `use_block_out_adapter`
- `block_adapter_type`
- `block_in_rank`
- `block_in_alpha`
- `block_out_rank`
- `block_out_alpha`
- `time_pos_encoding`
- `time_branch_mode`
- `time_embed_dim`
- `time_hidden_dim`
- `spatial_pos_encoding`
- `spatial_embed_dim`
- `spatial_hidden_dim`
- `spatial_branch_mode`
- `time_space_interaction = false`
- `loss_recon = mse`
- `loss_auxk = true`
- `loss_align = mid_anchor_pooled_l2_sq`
- `align_weight_target`
- `align_warmup_ratio`
- `decoder_decorr_weight`
- `steps`
- `guidance_scale`
- `resolution`
- `expected_h`
- `expected_w`
- `run_stage4`
- `norm_scale_by_block`
- `num_step_buckets`
- `group_bs_stage1`
- `group_bs_stage2`

### 3. 续训开关

训练脚本必须支持：

- `--resume`
- `--train_stage stage1|stage2|stage3|stage4`
- `--freeze_shared_sae`
- `--freeze_block_in_adapter`
- `--freeze_block_out_adapter`
- `--freeze_time_branch`
- `--freeze_spatial_branch`
- `--use_block_out_adapter`
- `--run_stage4`
- `--time_branch_mode`
- `--spatial_branch_mode`
- `--expansion_factor`
- `--top_k`
- `--auxk`
- `--block_in_rank`
- `--block_in_alpha`

这些接口的目的不是让 v1 默认流程变复杂，而是确保后续做消融和继续训练时无需重构训练系统。

---

## 八、测试与验收标准

### 1. 单元与烟雾测试

必须覆盖：

- `coords_norm` 与当前 token flatten 顺序严格一致
- sampler 能正确对齐同一 `prompt + timestep` 的 4 个 block
- 若用户显式设置 `expected_h/expected_w` 且与真实采样不一致，训练流程会 fail fast
- 若用户显式把 `expected_h/expected_w = 0`，训练流程会记录首次观测到的真实 `hw`
- `sincos_linear / sincos_mlp / sincos_film` 三种时间分支都能独立前向
- `sincos_linear / sincos_mlp / sincos_film` 三种空间分支都能独立前向
- `use_block_out_adapter = false` 时输出与 identity 行为一致
- checkpoint 能保存、加载、resume
- `run_stage4 = false` 时主训练默认只执行 Stage1/2/3
- Stage 4 在显式打开时能单独启用并训练 `block_out_adapter`

### 2. 训练级验收

v1 至少满足以下条件：

- Stage 1 的 `mid_block` 预热能稳定收敛
- Stage 2 进入四层联合后，latent 不塌缩
- dead feature ratio 可控
- `L_align` 加入后不会明显破坏 `L_recon`
- 训练日志能区分 raw loss 与 weighted term，避免误判 `AuxK` 或 `align` 的真实贡献
- Stage2/Stage3 必须同时报告 `val_recon` 与 `val_align`，用于观察“重建”和“对齐”是否在一起改进
- 同一 `feature_id` 在四层上呈现“可比较但不完全相同”的激活模式
- 修改 `expansion_factor / top_k / auxk / time_branch_mode / spatial_branch_mode` 后，训练仍能正常启动，并在日志和 checkpoint 中正确记录参数

### 3. 继续训练验收

若显式启用 Stage4，则必须验证以下流程：

- 从 v1 checkpoint 恢复
- 打开 `block_out_adapter`
- 冻结 shared 主干
- 单独训练 out adapter 若干步
- 保存与再次恢复都正常

---

## 九、实现假设

- 本计划只针对训练 Shared SAE，不涉及最终擦除策略设计。
- 当前默认目标模型是 `stabilityai/stable-diffusion-xl-base-1.0`。
- 默认空间基线对齐当前 `512-space` 设置，因此 4 个目标 block 在训练时默认按 `16x16` 处理；若未来切到更高分辨率，应在实验记录里明确注明该运行已偏离默认 baseline。
- 4 个目标 block 在训练时默认都为 `1280` 通道；若用户显式覆盖 `resolution/expected_h/expected_w`，则空间网格以真实采样结果为准。
- 时间编码基底固定为 1D 正余弦编码；v1 默认时间分支使用 `sincos_linear`。
- 位置编码基底固定为 2D 正余弦编码；v1 默认空间分支使用 `sincos_linear`。
- 时间和空间在 v1 中固定分开建模，不引入时空交互项。
- v1 必须从第一天起支持未来继续训练 `block_out_adapter`，即使它当前不参与正式训练。

---

## 十、渐进实验修改计划

根据当前实现状态与实验结果，v1 的渐进主线需要进一步收敛，不再同时追多个复杂变量。当前最核心的问题已经从“能不能擦掉概念”转成“为什么擦除时原图容易一起被带坏”。因此第十节的阶段定义调整为：

- 先固定一个已经验证可用的 **四层 shared + adapter + align** 主线；
- 在这个主线上先补 **特征解耦/去相关能力**；
- 时间与空间分支放到下一阶段单独推进，而不是和解耦正则同时引入。

这样做的目的，是先回答现在最紧迫的问题：**Shared 字典是否能在保持重建/擦除能力的同时，学出更干净、独立的特征方向**。

### 1. 实验顺序

按以下顺序推进，每一步只增加一个关键复杂度：

1. `Step-1: 当前已完成基线（无时空分支）`
   - 对应上一轮主结果：`exp_c_adapter_align`
   - 关闭 `time_branch`
   - 关闭 `spatial_branch`
   - 打开 `block_in_adapter`
   - 保留 `alignment loss`
   - 主体训练流程为 `Stage2 -> Stage3`
   - 目的：
     - 建立当前 SharedSAE 的可用基线
     - 验证“共享字典 + adapter + align”是否已经足够支撑概念提取与擦除
   - 当前结论：
     - 重建可用
     - 概念擦除可用
     - 但特征解耦不足，擦除副作用仍然偏大

2. `Step-2: 在 Step-1 上加入 decoder 去相关正则`
   - 仍关闭 `time_branch`
   - 仍关闭 `spatial_branch`
   - 保持 `block_in_adapter` 与 `alignment loss`
   - 新增 `decoder decorrelation regularizer`
   - 推荐先从：
     - `decoder_decorr_weight = 3e-4`
     - `align_weight_target = 0.1`
   - 目的：
     - 让共享字典方向彼此更独立
     - 降低概念擦除时对非目标内容的连带破坏
     - 为后续再次评估 `projected_ablation` 创造更干净的子空间

3. `Step-3: 在 Step-2 稳定后，再引入时间分支`
   - 打开 `time_branch`
   - 先不要求空间分支同时进入
   - 当前推荐训练路径为 `Stage1 -> Stage2 -> Stage3`
   - `Stage4` 继续关闭，不把 `block_out_adapter` 混入这一步
   - 目标不是追求更复杂结构，而是让后续擦除具备更细粒度的时间感知能力
   - 目的：
     - 让 feature 对不同 diffusion step 有更自然的强弱偏好
     - 减少对外部 `feature_time_scores + 大倍率 scale` 的依赖

4. `Step-4: 最后再评估是否需要空间分支`
   - 只有在时间分支已经证明有价值后再做
   - 目的：
     - 判断空间条件是否真的能为概念擦除带来额外收益
     - 避免一次引入太多变量导致结论不清

`Stage4` 与 `block_out_adapter` 继续保持为显式开启的续训探针，不进入该渐进主线。

### 2. 每一步必须保留的能力

每一阶段都必须同时检查以下能力，而不是只看单一数值：

- `重建能力`
  - 训练内用 `val_recon`
  - 图像级用 `LPIPS / DreamSim`
- `语义保持能力`
  - 图像级用 `CLIP` image-text 相似度
- `跨层共享能力`
  - 训练内用 `val_align`
- `训练稳定性`
  - 用 `latent_active_frac / dead_feature_frac`
  - 同时做人工抽样看是否出现明显发灰、结构塌陷或语义漂移

### 3. 推荐评价规则

当前仓库已有图像级评估脚本，可直接作为主评价：

- `evalscripts/lpips_eval.py`
- `evalscripts/dreamsim_eval.py`
- `evalscripts/mean_clip.py`
- 训练日志中的 `val_recon / val_align`

建议每一步都使用同一批 prompts 和 seeds，对比：

- `reference`: 原始 SDXL 输出
- `reconstructed`: 激活经过 SAE 重建后再继续生成的输出

推荐判定逻辑：

- `Step-1` 作为当前对照基线：
  - `val_recon` 稳定
  - 可完成概念擦除
  - 但图像副作用仍偏大
- `Step-2` 过关：
  - `loss_decoder_decorr` 有效下降
  - `val_recon` 不明显崩坏
  - 同样概念抑制强度下，图像副作用降低
- `Step-3` 过关：
  - 引入时间分支后，擦除时间权重不再依赖很大的外部放大倍数
  - 同样效果下，`int_time_weight_scale` 可明显下降
- `Step-4` 过关：
  - 空间分支带来明确收益，而不只是结构更复杂

### 4. 代码支持要求

为配合这条渐进路线，训练代码应显式支持以下开关：

- `--no-run_stage1`
- `--no-run_stage3`
- `--no-use_block_in_adapter`
- `--no-use_time_branch`
- `--no-use_spatial_branch`
- `--align_weight_target 0`
- `--save_every_steps 0`

这样可以不用改训练主逻辑，只通过配置切出上述 4 类实验。

### 5. 当前 Step-3 的保存策略

当前主线在引入时间分支时，推荐关闭中间 checkpoint，仅保留阶段末尾保存：

- `save_every_steps = 0`

其语义固定为：

- 不在训练中途按 step 保存 checkpoint
- 仍然保留每个阶段结束时的 checkpoint
- 仍然保留完整的 `step_metrics / stage_metrics / loss curves`

这样可以显著减少磁盘占用，避免在 `Stage1/2/3` 连续训练时产生过多中间大权重文件。
