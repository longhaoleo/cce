# SDXL Base Shared SAE 训练计划 v2：Latent Decorrelation 修订

日期：2026-05-21

## 目标

本计划只处理 `latent_decorrelation_loss` 的训练正则修订，不改变 SharedSAE 主体结构、数据采样、checkpoint 布局或擦除运行时接口。

核心目标：

- 把当前 latent decorrelation 从 `(corr - I)^2` 改成固定 `offdiag-only`。
- 保留当前 token-level 拼接模式，作为可复现实验基线。
- 增加 `block_pooled` 模式，让 decorrelation 更偏向跨 block 语义冗余，而不是强行打散局部 token 组合特征。
- 用 `car / dog / nudity` 对比验证：`nudity` 是否因为 token-level decorrelation 过度打散局部组合概念而变差。

## 背景判断

当前实现位置：

- `train/losses.py`
  - `latent_covariance_decorrelation_loss`
  - 当前会把所有 block 的 token latent 拼接：

```python
z = torch.cat(z_items, dim=0)
```

  - 当前惩罚：

```python
torch.mean((corr - eye).pow(2))
```

这个做法有两个隐含效果：

- 对角项也被纳入损失，等价于额外要求每个 feature 的自相关接近某个固定尺度。
- 所有 block、所有 token 在同一个样本组里全局混合，可能把局部组合概念需要的协同子特征强行拆散。

对 `car / dog` 这种整体物体概念，这种强拆可能仍然可用；对 `nudity` 这种局部、组合、区域依赖强的概念，可能导致：

```text
概念分离更明显
但完整概念被拆成更多 feature
top-k 擦除预算不够
nudity 擦除变差或副作用变大
```

## 修改一：Offdiag-Only

### 设计

固定只惩罚非对角项：

```python
offdiag = corr - torch.diag_embed(torch.diagonal(corr))
loss = offdiag.pow(2).mean()
```

### 理由

offdiag-only 只表达“不同 feature 少共激活”，不再额外要求每个 feature 的自相关严格维持某个数值。当前标准化后 `corr` 对角线通常接近 1，但显式去掉对角项可以避免 batch 大小、数值尺度、`N` 与 `N-1` 分母差异带来的隐性约束。

### 实现点

- `train/trainer.py`
  - 把配置透传到 `group_forward_losses`。
- `train/losses.py`
  - `latent_covariance_decorrelation_loss` 增加参数：

```python
eps: float = 1e-4
```

## 修改二：Block-Pooled Mode

### 设计

新增开关：

```text
latent_decorr_mode: "token" | "block_pooled" = "token"
```

保留当前行为为 `token`：

```python
z = torch.cat([z for z in z_by_block.values() if z.numel() > 0], dim=0)
```

新增 `block_pooled`：

```python
pooled = []
for z in z_by_block.values():
    if z.numel() > 0:
        pooled.append(z.mean(dim=0, keepdim=True))
z = torch.cat(pooled, dim=0)
```

v2 第一版直接支持三种 pooling：

```text
latent_decorr_pool = "mean" | "topq" | "hybrid"
```

- `mean`：每个 block 对 token latent 做均值池化。
- `topq`：每个 block、每个 feature 取 token 维 top fraction 的激活均值。
- `hybrid`：固定使用 `0.5 * mean + 0.5 * topq`。

### 理由

`block_pooled` 惩罚的是跨 block pooled latent 中的 feature 冗余，更接近“跨层共享语义去冗余”。它不会把单个图像局部区域内部本来需要协同的子特征强行拆散，因此更适合验证 `nudity` 这类局部组合概念的失败来源。

### 实现点

- `SAE/config.py`
  - 新增：

```python
latent_decorr_mode: str = "token"
latent_decorr_pool: str = "mean"
latent_decorr_pool_topq: float = 0.1
latent_decorr_eps: float = 1e-4
```

  - 校验：

```text
latent_decorr_mode in {"token", "block_pooled"}
latent_decorr_pool in {"mean", "topq", "hybrid"}
0 < latent_decorr_pool_topq <= 1
latent_decorr_eps > 0
```

- `train/run_train.py`
  - 新增 CLI：

```bash
--latent_decorr_mode token
--latent_decorr_pool mean
--latent_decorr_pool_topq 0.1
--latent_decorr_eps 1e-4
```

- `train/losses.py`
  - 将当前函数扩展为：

```python
def latent_covariance_decorrelation_loss(
    z_by_block,
    *,
    top_k: int,
    mode: str = "token",
    pool: str = "mean",
    pool_topq: float = 0.1,
    eps: float = 1e-4,
):
    ...
```

## 推荐实现顺序

### Step 1：只改 Offdiag-Only

目标：

- 默认 token-level 模式不变。
- 只把 `(corr - I)^2` 改成 offdiag-only。

验证：

- 单元级：构造小张量，确认对角项不影响 offdiag-only loss。
- 训练级：`latent_decorr_weight > 0` 时 loss 有正常数值，不出现 NaN。

### Step 2：保留 Token Ablation

目标：

- 让现有 `sae_x8_time` / `sae_x8_time_decorr03` 训练逻辑仍可复现。
- 命令显式写出：

```bash
--latent_decorr_mode token
```

验证：

- 训练 manifest 保存新增字段。
- metrics CSV 继续记录 `loss_latent_decorr` 和 `loss_latent_decorr_term`。

### Step 3：增加 Block-Pooled

目标：

- 实现 `latent_decorr_mode=block_pooled`。
- 第一版同时支持 `latent_decorr_pool=mean/topq/hybrid`。
- 不改变 align loss 的 `_pooled_unit` 逻辑，避免两个损失互相耦合。

验证：

- 当 block 数小于 2 或有效 feature 小于 2 时返回 0，不报错。
- `block_pooled` 下 corr 的样本维度是有效 block 数，而不是 token 数。
- 对同一批 z，`token` 与 `block_pooled` 产生不同但稳定的 loss。

### Step 4：概念擦除对比

固定比较：

```text
concepts: car, dog, nudity
checkpoint groups:
  A. token + old diag formula
  B. token + offdiag-only
  C. block_pooled + offdiag-only
```

重点指标：

- `target suppression`
  - 概念目标是否被有效压制。
- `LPIPS / DreamSim`
  - baseline 与 erased 图像的保真差异。
- `CLIP`
  - prompt 语义是否被保留。
- 人工检查：
  - 人物结构是否被重写。
  - 背景、光照、姿态是否过度漂移。
  - `nudity` 是否需要显著更大的 top-k 才能压制。

## 预期结论判据

支持假设的结果：

```text
block_pooled 下 nudity 的 LPIPS/DreamSim/CLIP 保真明显改善
且 target suppression 没有明显下降
```

这说明之前主要问题来自 token-level decorrelation 过度打散局部组合概念。

不支持假设的结果：

```text
block_pooled 保真没有改善，或 target suppression 明显下降
```

这说明问题更可能来自概念 prompt、blacklist、top-k 预算、干预策略或 nudity 子概念定位，而不是 decorrelation 粒度本身。

## 推荐命令草案

### Token + Offdiag-Only

```bash
cd /root/cce

python train/run_train.py \
  --output_root train/output_time_latentdecorr_x8_top20_offdiag \
  --local_files_only \
  --latent_decorr_weight 0.3 \
  --latent_decorr_top_k 20 \
  --latent_decorr_mode token \
  --save_every_steps 0
```

### Block-Pooled + Offdiag-Only

```bash
cd /root/cce

python train/run_train.py \
  --output_root train/output_time_latentdecorr_x8_top20_block_pooled \
  --local_files_only \
  --latent_decorr_weight 0.3 \
  --latent_decorr_top_k 20 \
  --latent_decorr_mode block_pooled \
  --latent_decorr_pool mean \
  --save_every_steps 0
```

具体训练数据量、stage 开关和 preset 应沿用当前 `scripts/training.md` 的最新主线命令，以上只表达新增参数。

## 成功标准

- 代码层：
  - 新增配置能进入 checkpoint config 与 run manifest。
  - 默认行为变为 `token + offdiag-only`，且不保留旧公式开关。
  - `block_pooled` 不影响默认训练路径。
- 实验层：
  - `car / dog` 不明显退化。
  - `nudity` 在同等擦除强度下保真曲线改善。
  - 若 `nudity` target suppression 下降，需要记录 top-k / scale 是否可补偿。

## 暂不做

- 不在 v2 第一版为 `hybrid` 增加可调混合权重，固定使用 50/50。
- 不改 `align_loss_from_group_latents`。
- 不改 runtime 擦除策略。
- 不改概念 prompt 与 blacklist 逻辑。
