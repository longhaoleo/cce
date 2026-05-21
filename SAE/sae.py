"""
SharedSAE 核心模型定义。

这份实现承载了当前项目里最关键的几层设计：

1. 一个跨 block 共享的 SAE 特征空间
   - 所有目标 block 共用同一套 encoder / decoder
   - 这样同一个 feature_id 才有机会在不同层上表达“相近语义”

2. 轻量 block-specific adapter
   - Shared 主干负责“公共特征空间”
   - block adapter 负责吸收每层输入分布的小偏移

3. 时间 / 空间条件分支
   - 时间分支让同一 feature 可以随扩散阶段改变激活偏置
   - 空间分支让同一 feature 可以随 token 位置改变激活偏置

4. 稀疏激活 + AuxK 恢复 dead feature
   - 主分支用 top-k 稀疏化
   - 辅助分支只在 dead feature 上工作，帮助恢复长期不激活的方向

这份代码更偏“模型解释层”，因此注释会尽量说明：
- 每一段参数在整个训练/推理链路中的职责
- 为什么前向要按当前这种顺序组合
- 哪些行为是为了稳定训练，而不是数学上唯一正确的做法
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoding import ensure_mode, normalize_timestep, sincos_1d, sincos_2d


@dataclass
class ForwardCache:
    """一次 SharedSAE 前向的关键中间量。

    这些张量不会只在训练里用到：

    - `x_hat`
      主分支重建结果，直接进入重建损失。
    - `z`
      top-k 稀疏激活，既用于训练诊断，也用于后续概念定位/擦除。
    - `pre_act`
      稀疏化之前的激活，AuxK 分支与某些调试分析会用到。
    - `x_aux`, `z_aux`
      dead-feature 恢复分支输出；没有触发时为 `None`。
    """

    x_hat: torch.Tensor
    z: torch.Tensor
    pre_act: torch.Tensor
    x_aux: Optional[torch.Tensor]
    z_aux: Optional[torch.Tensor]


def _topk_keep(values: torch.Tensor, k: int) -> torch.Tensor:
    """仅保留每行 top-k 值，其余置零。

    这里返回的仍然是 dense tensor，而不是稀疏索引结构。
    原因很实际：

    - decoder 是标准线性层，直接吃 dense latent 最省心
    - 训练时很多统计量也更容易直接在 dense 形式上做

    所以这里的“稀疏”是值域上的稀疏，而不是存储格式上的稀疏。
    """
    if int(k) <= 0:
        raise ValueError("k 必须 > 0")
    if int(k) >= int(values.shape[-1]):
        return values
    top_vals, top_inds = torch.topk(values, k=int(k), dim=-1)
    out = torch.zeros_like(values)
    out.scatter_(-1, top_inds, top_vals)
    return out


class LoRAAdapter(nn.Module):
    """低秩残差适配器。

    设计目的：

    - SharedSAE 主干要跨层共享，因此不能为每个 block 完全单独建一套 encoder/decoder
    - 但不同 block 的输入分布确实有偏移，所以这里用很薄的一层 LoRA 吸收差异

    初始化策略：

    - `up.weight = 0`
      初始时 adapter 是严格 identity，不会扰动主干
    - 但 `down` 不是全零
      这样从第一步开始梯度链路就是通的，不会出现“分支学不动”
    """

    def __init__(self, d_model: int, rank: int, alpha: int):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.down = nn.Linear(self.d_model, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    @property
    def scale(self) -> float:
        """返回 LoRA 缩放系数。"""
        return float(self.alpha) / float(max(1, self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 LoRA 残差映射。"""
        return x + self.scale * self.up(self.down(x))


class TimeBranch(nn.Module):
    """时间条件分支。

    它不直接生成最终 latent，而是给 encoder 输出施加“时间条件修正”。

    三种模式的含义：

    - `sincos_linear`
      最轻量：把 timestep 的 sin/cos 编码线性投影成一个 bias
    - `sincos_mlp`
      同样是 bias，但允许非线性映射
    - `sincos_film`
      生成 `(gamma, beta)`，对 base activation 做 FiLM 调制

    直觉上可以把它理解成：
    “同一个 feature 在不同扩散阶段，默认阈值或放缩可以不一样”。
    """

    def __init__(self, *, mode: str, embed_dim: int, hidden_dim: int, n_dirs: int):
        super().__init__()
        self.mode = ensure_mode(mode, name="time_branch_mode")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_dirs = int(n_dirs)

        if self.mode == "sincos_linear":
            self.proj = nn.Linear(self.embed_dim, self.n_dirs)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif self.mode == "sincos_mlp":
            self.proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.n_dirs),
            )
            last = self.proj[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2 * self.n_dirs),
            )
            # FiLM 比单纯 bias 更容易在训练初期带来过强扰动，
            # 所以最后一层零初始化，让模型先从“近似不加时间条件”起步。
            last = self.proj[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        timestep: torch.Tensor,
        n_tokens: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """根据时间步生成偏置或 FiLM 参数。

        这里会把单个 timestep 扩展到 token 维：
        同一个 block、同一个 denoising step 下，所有 token 共用同一组时间条件。
        """
        t_norm = normalize_timestep(timestep).to(device=timestep.device, dtype=timestep.dtype)
        t_emb = sincos_1d(t_norm, self.embed_dim)
        raw = self.proj(t_emb)

        if raw.shape[0] == 1 and int(n_tokens) > 1:
            raw = raw.expand(int(n_tokens), -1)
        elif raw.shape[0] != int(n_tokens):
            raw = raw[:1].expand(int(n_tokens), -1)

        if self.mode in {"sincos_linear", "sincos_mlp"}:
            return raw, None, None
        gamma_t, beta_t = raw.chunk(2, dim=-1)
        return None, gamma_t, beta_t


class SpatialBranch(nn.Module):
    """空间条件分支。

    和 `TimeBranch` 类似，但输入从“扩散时间”换成了“token 坐标”。

    它表达的是：
    同一个共享 feature 在不同空间位置上，可以有不同的先验激活偏置。
    """

    def __init__(self, *, mode: str, embed_dim: int, hidden_dim: int, n_dirs: int):
        super().__init__()
        self.mode = ensure_mode(mode, name="spatial_branch_mode")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_dirs = int(n_dirs)

        if self.mode == "sincos_linear":
            self.proj = nn.Linear(self.embed_dim, self.n_dirs)
        elif self.mode == "sincos_mlp":
            self.proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.n_dirs),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2 * self.n_dirs),
            )
            # 与时间分支同理，FiLM 模式先零初始化，避免空间条件一上来就把
            # 共享特征空间拉歪。
            last = self.proj[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        coords_norm: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """根据坐标生成偏置或 FiLM 参数。"""
        p_emb = sincos_2d(coords_norm, self.embed_dim)
        raw = self.proj(p_emb)
        if self.mode in {"sincos_linear", "sincos_mlp"}:
            return raw, None, None
        gamma_p, beta_p = raw.chunk(2, dim=-1)
        return None, gamma_p, beta_p


class SharedSAE(nn.Module):
    """Shared SAE 主模型。

    整体数据流可以概括成：

    1. 输入 token 先做可选的 block-specific input adapter
    2. 送入共享 encoder，得到 base activation
    3. 叠加时间 / 空间条件，得到 `pre_act`
    4. `relu + top-k` 得到稀疏 latent `z`
    5. 共享 decoder 重建
    6. 可选的 block-specific output adapter 只在需要时再作用于重建结果

    这个顺序很重要：

    - 共享空间尽量放在中间
    - block-specific 模块尽量只在输入/输出边界修正
    - 时间/空间条件作用在 encoder 输出之后，而不是直接作用在原输入上
    """

    def __init__(
        self,
        *,
        blocks: Tuple[str, ...] | list[str],
        d_model: int,
        n_dirs: int,
        top_k: int,
        auxk: int,
        dead_tokens_threshold: int,
        use_block_in_adapter: bool,
        use_block_out_adapter: bool,
        block_in_rank: int,
        block_in_alpha: int,
        block_out_rank: int,
        block_out_alpha: int,
        use_time_branch: bool,
        time_branch_mode: str,
        time_embed_dim: int,
        time_hidden_dim: int,
        use_spatial_branch: bool,
        spatial_branch_mode: str,
        spatial_embed_dim: int,
        spatial_hidden_dim: int,
    ):
        super().__init__()
        self.blocks = tuple(str(b) for b in blocks)
        self.block_set = set(self.blocks)
        # ModuleDict 的 key 不能直接安全地使用 "a.b.c" 这种层路径，
        # 所以这里建立一个 block_name <-> safe_key 的双向映射。
        self.block_name_to_safe = {b: self._safe_block_key(b) for b in self.blocks}
        self.safe_to_block_name = {v: k for k, v in self.block_name_to_safe.items()}
        if len(self.block_name_to_safe) != len(self.safe_to_block_name):
            raise ValueError("block 安全键发生冲突，请检查 block 列表")
        self.d_model = int(d_model)
        self.n_dirs = int(n_dirs)
        self.top_k = int(top_k)
        self.auxk = int(auxk)
        self.dead_tokens_threshold = int(dead_tokens_threshold)
        self.use_block_in_adapter = bool(use_block_in_adapter)
        self.use_block_out_adapter = bool(use_block_out_adapter)
        self.use_time_branch = bool(use_time_branch)
        self.use_spatial_branch = bool(use_spatial_branch)

        # encoder / decoder 是真正共享的字典学习主体。
        self.encoder = nn.Linear(self.d_model, self.n_dirs, bias=False)
        self.decoder = nn.Linear(self.n_dirs, self.d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(self.d_model))
        self.latent_bias = nn.Parameter(torch.zeros(self.n_dirs))

        # 用 tied init 让 decoder 初始字典方向与 encoder 对齐。
        # 这不是严格 tied weights，因为训练中两者仍可各自更新，
        # 但它提供了一个更平滑的起点。
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T
        unit_norm_decoder_(self)

        self.in_adapters = nn.ModuleDict()
        self.out_adapters = nn.ModuleDict()
        for b in self.blocks:
            safe_key = self._safe_key(b)
            self.in_adapters[safe_key] = LoRAAdapter(self.d_model, int(block_in_rank), int(block_in_alpha))
            self.out_adapters[safe_key] = LoRAAdapter(self.d_model, int(block_out_rank), int(block_out_alpha))

        self.time_branch = TimeBranch(
            mode=time_branch_mode,
            embed_dim=int(time_embed_dim),
            hidden_dim=int(time_hidden_dim),
            n_dirs=self.n_dirs,
        )
        self.spatial_branch = SpatialBranch(
            mode=spatial_branch_mode,
            embed_dim=int(spatial_embed_dim),
            hidden_dim=int(spatial_hidden_dim),
            n_dirs=self.n_dirs,
        )

        # 记录每个 feature 距离“上次显著激活”已经过去了多少 token。
        # 训练时 AuxK 分支用它来判断哪些 feature 已经“近似死亡”。
        self.register_buffer("stats_last_nonzero", torch.zeros(self.n_dirs, dtype=torch.long))

    def _check_block(self, block_name: str) -> None:
        """校验 block 名称是否合法。"""
        if block_name not in self.block_set:
            raise KeyError(f"未知 block_name: {block_name}")

    @staticmethod
    def _safe_block_key(block_name: str) -> str:
        """将 block 路径转换为 ModuleDict 可用的安全键。"""
        return str(block_name).replace(".", "__")

    def _safe_key(self, block_name: str) -> str:
        """返回指定 block 对应的安全键。"""
        self._check_block(block_name)
        return self.block_name_to_safe[block_name]

    def _apply_input_adapter(self, x: torch.Tensor, block_name: str) -> torch.Tensor:
        """应用输入适配器。

        主干共享的前提下，这一步负责吸收 block 级输入分布偏移。
        关闭 `use_block_in_adapter` 时，这里严格退化为 identity。
        """
        if not self.use_block_in_adapter:
            return x
        return self.in_adapters[self._safe_key(block_name)](x)

    def _apply_output_adapter(self, x_hat: torch.Tensor, block_name: str, use_out_adapter: bool) -> torch.Tensor:
        """应用输出适配器。

        output adapter 当前更多是一个后续实验开关，而不是默认主线。
        所以这里分成两层条件：

        - 模型结构上是否启用 `use_block_out_adapter`
        - 本次前向是否要求 `use_out_adapter`
        """
        if not (bool(use_out_adapter) and self.use_block_out_adapter):
            return x_hat
        return self.out_adapters[self._safe_key(block_name)](x_hat)

    def _compose_pre_activation(
        self,
        *,
        base: torch.Tensor,
        timestep: torch.Tensor,
        coords_norm: torch.Tensor,
        time_branch_scale: float = 1.0,
    ) -> torch.Tensor:
        """组合时间/空间分支，得到最终 pre-activation。

        这里把“共享 encoder 输出”视为基础语义表示 `base`，
        时间和空间分支都只在这个基础上施加修正。

        组合策略支持四种主要情况：

        - 只有 bias
        - 时间 FiLM + 空间 bias
        - 时间 bias + 空间 FiLM
        - 时间 / 空间都用 FiLM

        这样做比把时空信息直接拼到输入里更可控，也更方便解释。
        """
        n_tokens = int(base.shape[0])
        if self.use_time_branch:
            b_t, gamma_t, beta_t = self.time_branch(timestep=timestep, n_tokens=n_tokens)
            scale_t = float(time_branch_scale)
            if b_t is not None:
                b_t = b_t * scale_t
            if gamma_t is not None:
                gamma_t = gamma_t * scale_t
            if beta_t is not None:
                beta_t = beta_t * scale_t
        else:
            b_t = gamma_t = beta_t = None
        if self.use_spatial_branch:
            b_p, gamma_p, beta_p = self.spatial_branch(coords_norm=coords_norm)
        else:
            b_p = gamma_p = beta_p = None

        if (not self.use_time_branch) and (not self.use_spatial_branch):
            return base
        if not self.use_time_branch:
            if b_p is not None:
                return base + b_p
            if gamma_p is not None:
                return (1.0 + gamma_p) * base + beta_p
        if not self.use_spatial_branch:
            if b_t is not None:
                return base + b_t
            if gamma_t is not None:
                return (1.0 + gamma_t) * base + beta_t

        if b_t is not None and b_p is not None:
            return base + b_t + b_p
        if gamma_t is not None and b_p is not None:
            return (1.0 + gamma_t) * base + beta_t + b_p
        if b_t is not None and gamma_p is not None:
            return (1.0 + gamma_p) * (base + b_t) + beta_p
        if gamma_t is not None and gamma_p is not None:
            return (1.0 + gamma_p) * ((1.0 + gamma_t) * base + beta_t) + beta_p
        raise RuntimeError("时间/空间分支组合状态非法")

    @torch.no_grad()
    def get_learned_time_weight(
        self,
        timestep: torch.Tensor,
        feature_ids: list[int],
        *,
        transform: str = "neutral_sigmoid",
        temperature: float = 1.0,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回当前 timestep 下，每个目标 feature 的 learned time weight。"""
        device = self.latent_bias.device
        dtype = self.latent_bias.dtype
        ids = torch.tensor([int(fid) for fid in feature_ids], device=device, dtype=torch.long)
        if int(ids.numel()) == 0:
            empty = torch.zeros(0, device=device, dtype=dtype)
            return empty, empty

        timestep_t = timestep.to(device=device, dtype=dtype).reshape(-1)
        if int(timestep_t.numel()) == 0:
            timestep_t = torch.tensor([0.0], device=device, dtype=dtype)

        if not bool(self.use_time_branch):
            raw = torch.zeros(int(ids.numel()), device=device, dtype=dtype)
            return raw, torch.ones_like(raw) * float(scale)

        b_t, gamma_t, beta_t = self.time_branch(timestep=timestep_t, n_tokens=1)
        if b_t is not None:
            raw_full = b_t[0]
        elif beta_t is not None:
            raw_full = beta_t[0]
        elif gamma_t is not None:
            raw_full = gamma_t[0]
        else:
            raw_full = torch.zeros_like(self.latent_bias)
        raw = raw_full.index_select(0, ids)

        transform_norm = str(transform).strip().lower()
        temp = float(temperature)
        if transform_norm == "neutral_sigmoid":
            weight = 2.0 * torch.sigmoid(temp * raw)
        elif transform_norm == "relu":
            weight = torch.relu(raw)
        elif transform_norm == "abs":
            weight = raw.abs()
        elif transform_norm == "sigmoid":
            weight = torch.sigmoid(temp * raw)
        else:
            raise ValueError(f"未知 learned time transform: {transform}")
        return raw, weight * float(scale)

    def _update_dead_stats(self, z: torch.Tensor) -> None:
        """更新 dead-feature 统计。

        规则很简单：

        - 先假设所有 feature 都又“老了一批 token”
        - 若某个 feature 在当前 batch 中出现显著激活，就把它的计数清零

        这样 `stats_last_nonzero` 越大，表示该 feature 越久没真正参与重建。
        """
        active = torch.any(z > 1e-3, dim=0)
        self.stats_last_nonzero += int(z.shape[0])
        self.stats_last_nonzero[active] = 0

    def _compute_aux_branch(self, pre_act: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """计算 AuxK 分支重建。

        AuxK 的目标不是提升主重建，而是“救活”长期不激活的 feature。

        这里的做法是：

        - 先找 dead feature mask
        - 只保留这些 feature 的 pre-activation
        - 再在 dead-feature 子空间里做一次 top-k

        如果当前没有 dead feature，这个分支会直接返回 `None`，
        这样训练日志里对应项也能真实反映“此时 AuxK 没有参与”。
        """
        if int(self.auxk) <= 0:
            return None, None
        dead_mask_bool = self.stats_last_nonzero > int(self.dead_tokens_threshold)
        if not torch.any(dead_mask_bool):
            return None, None
        dead_mask = dead_mask_bool.to(pre_act.dtype)
        masked = pre_act * dead_mask.unsqueeze(0)
        z_aux = _topk_keep(torch.relu(masked), int(self.auxk))
        x_aux = self.decoder(z_aux)
        return x_aux, z_aux

    def forward(
        self,
        x_norm: torch.Tensor,
        *,
        block_name: str,
        timestep: torch.Tensor | int | float,
        coords_norm: torch.Tensor,
        use_out_adapter: bool = False,
        update_dead_stats: bool = True,
        time_branch_scale: float = 1.0,
    ) -> ForwardCache:
        """执行一次 Shared SAE 前向。

        参数约定：

        - `x_norm`
          必须是已经做过 block scale 的输入 token
        - `block_name`
          决定使用哪一层 adapter，但不会改变共享 encoder/decoder 本身
        - `timestep` / `coords_norm`
          提供时空条件

        输出：
        - `ForwardCache`
          主训练和后续概念定位/擦除都依赖它
        """
        self._check_block(block_name)
        x = self._apply_input_adapter(x_norm, block_name)
        base = self.encoder(x - self.pre_bias) + self.latent_bias

        if isinstance(timestep, (int, float)):
            timestep_t = torch.tensor([float(timestep)], device=x.device, dtype=x.dtype)
        else:
            timestep_t = timestep.to(device=x.device, dtype=x.dtype).reshape(-1)
            if timestep_t.numel() == 0:
                timestep_t = torch.tensor([0.0], device=x.device, dtype=x.dtype)

        # `pre_act` = 共享 encoder 输出 + 时空条件修正。
        p = self._compose_pre_activation(
            base=base,
            timestep=timestep_t,
            coords_norm=coords_norm.to(device=x.device, dtype=x.dtype),
            time_branch_scale=float(time_branch_scale),
        )
        # 主稀疏化发生在这里：ReLU 负责非负激活，top-k 负责控制稀疏度。
        z = _topk_keep(torch.relu(p), int(self.top_k))
        # 先用共享 decoder 重建，再按需叠加 output adapter。
        x_hat_shared = self.decoder(z) + self.pre_bias
        x_hat = self._apply_output_adapter(x_hat_shared, block_name, use_out_adapter=bool(use_out_adapter))

        if bool(update_dead_stats) and self.training:
            self._update_dead_stats(z)
        x_aux, z_aux = self._compute_aux_branch(p)
        return ForwardCache(x_hat=x_hat, z=z, pre_act=p, x_aux=x_aux, z_aux=z_aux)


def unit_norm_decoder_(model: SharedSAE) -> None:
    """对 decoder 字典列向量做单位范数归一化。

    训练中维持 decoder 列向量单位范数，有两个主要作用：

    - 减少“靠把字典向量放大来偷懒”的自由度
    - 让 latent 系数的尺度更可比、更稳定
    """
    w = model.decoder.weight.data
    norm = w.norm(dim=0, keepdim=True).clamp_min(1e-12)
    model.decoder.weight.data = w / norm


def unit_norm_decoder_grad_adjustment_(model: SharedSAE) -> None:
    """将 decoder 梯度投影到与字典向量正交的子空间。

    如果我们同时要求 decoder 保持单位范数，那么梯度中沿着字典向量本身的分量
    实际上是“无效方向”。这里先把那部分投影掉，再做优化，会更稳定。
    """
    if model.decoder.weight.grad is None:
        return
    grad = model.decoder.weight.grad
    weight = model.decoder.weight.data
    parallel = torch.einsum("dn,dn->n", weight, grad).unsqueeze(0) * weight
    model.decoder.weight.grad = grad - parallel


def build_trainable_param_groups(model: SharedSAE) -> Dict[str, list[nn.Parameter]]:
    """按模块类别收集参数，用于构建多学习率优化器。

    之所以显式拆成 `shared / adapter / time / spatial` 四组，
    是因为这几类参数在实验里通常需要不同学习率：

    - shared 主干最敏感
    - adapter 常常可以更激进
    - time / spatial 分支介于两者之间
    """
    shared: list[nn.Parameter] = []
    adapter: list[nn.Parameter] = []
    time: list[nn.Parameter] = []
    spatial: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("in_adapters.") or name.startswith("out_adapters."):
            adapter.append(p)
        elif name.startswith("time_branch."):
            time.append(p)
        elif name.startswith("spatial_branch."):
            spatial.append(p)
        else:
            shared.append(p)
    return {"shared": shared, "adapter": adapter, "time": time, "spatial": spatial}


def set_stage_trainable(model: SharedSAE, stage: str) -> None:
    """按训练阶段切换参数可训练状态。

    当前阶段策略大意是：

    - `stage2`
      放开共享主干、时空分支和 input adapter，学习跨 block 对齐
    - `stage3`
      在 stage2 基础上低学习率联合微调

    这样做是为了避免一上来所有模块一起学，导致责任边界太混。
    """
    valid = {"stage2", "stage3"}
    if stage not in valid:
        raise ValueError(f"未知 stage: {stage}")

    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if name.startswith(("encoder.", "decoder.", "pre_bias", "latent_bias")):
            p.requires_grad = True

    for p in model.time_branch.parameters():
        p.requires_grad = bool(model.use_time_branch)
    for p in model.spatial_branch.parameters():
        p.requires_grad = bool(model.use_spatial_branch)

    for p in model.in_adapters.parameters():
        p.requires_grad = True

    for p in model.out_adapters.parameters():
        p.requires_grad = False
