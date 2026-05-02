"""
Shared SAE 训练器。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from tqdm.auto import tqdm

from .losses import LossBreakdown, decoder_decorrelation_loss, group_forward_losses
from .metrics import MetricsWriter, StepMetric
from .prompt_data import PromptRecord, iter_prompt_shards
from .sampler import GroupSample, SDXLGroupSampler
from SAE import (
    SharedSAE,
    TrainConfig,
    apply_block_scale,
    build_trainable_param_groups,
    save_checkpoint,
    set_stage_trainable,
    unit_norm_decoder_,
    unit_norm_decoder_grad_adjustment_,
)


@dataclass
class StageResult:
    """阶段训练结果。

    输入：
    - 在训练完成后由训练器填充。

    输出：
    - 结构化阶段结果，便于汇总日志。
    """

    stage: str
    steps: int
    mean_total: float
    mean_recon: float
    mean_auxk: float
    mean_align: float
    mean_decoder_decorr: float
    mean_latent_decorr: float
    elapsed_sec: float
    observed_hw: tuple[int, int] | None = None
    group_bs: int | None = None
    tokens_per_group: int | None = None
    tokens_per_step: int | None = None


@dataclass
class BatchDiagnostics:
    """单个优化 step 的诊断信息。"""

    loss_auxk_term: float
    loss_align_term: float
    loss_decoder_decorr_term: float
    loss_latent_decorr_term: float
    time_branch_scale: float
    latent_active_frac: float
    dead_feature_frac: float


@dataclass
class EvalMetrics:
    """验证阶段的关键指标。"""

    recon: float
    align: float
    groups: int


class SharedSAETrainer:
    """Shared SAE 分阶段训练器。"""

    def __init__(
        self,
        *,
        cfg: TrainConfig,
        model: SharedSAE,
        sampler: SDXLGroupSampler,
        norm_scale_by_block: Dict[str, float],
        metrics_writer: MetricsWriter | None = None,
    ):
        """初始化训练器。

        输入：
        - cfg: 训练配置。
        - model: Shared SAE 模型。
        - sampler: 激活采样器。
        - norm_scale_by_block: 每个 block 的归一化缩放系数。

        输出：
        - SharedSAETrainer 实例。
        """
        self.cfg = cfg
        self.model = model
        self.sampler = sampler
        self.norm_scale_by_block = dict(norm_scale_by_block)
        self.device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.global_step = 0
        self.history: List[StageResult] = []
        self.model = self.model.to(self.device)
        self.metrics_writer = metrics_writer if metrics_writer is not None else MetricsWriter(cfg.output_root)

    def _extract_group_lrs(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """读取当前优化器各参数组学习率。

        输入：
        - optimizer: 当前阶段优化器。

        输出：
        - Dict[str,float]：`shared/adapter/time/spatial` 学习率。
        """
        out = {"shared": 0.0, "adapter": 0.0, "time": 0.0, "spatial": 0.0}
        for g in optimizer.param_groups:
            name = str(g.get("name", ""))
            if name in out:
                out[name] = float(g.get("lr", 0.0))
        return out

    def _build_optimizer(self, stage: str) -> torch.optim.Optimizer:
        """按阶段构建优化器。

        输入：
        - stage: `stage1|stage2|stage3|stage4`。

        输出：
        - torch.optim.Optimizer：含多参数组学习率的 Adam 优化器。
        """
        groups = build_trainable_param_groups(self.model)

        lr_shared = float(self.cfg.lr_shared)
        lr_adapter = float(self.cfg.lr_adapter)
        lr_time = float(self.cfg.lr_time)
        lr_spatial = float(self.cfg.lr_spatial)

        if stage == "stage3":
            lr_shared = 2e-5
            lr_adapter = 1e-4
            lr_time = float(self.cfg.lr_time_stage3)
            lr_spatial = 2e-5
        if stage == "stage4":
            lr_shared = 0.0
            lr_time = 0.0
            lr_spatial = 0.0

        param_groups = []
        if groups["shared"]:
            param_groups.append({"params": groups["shared"], "lr": lr_shared, "name": "shared"})
        if groups["adapter"]:
            param_groups.append({"params": groups["adapter"], "lr": lr_adapter, "name": "adapter"})
        if groups["time"]:
            param_groups.append({"params": groups["time"], "lr": lr_time, "name": "time"})
        if groups["spatial"]:
            param_groups.append({"params": groups["spatial"], "lr": lr_spatial, "name": "spatial"})

        if not param_groups:
            raise RuntimeError(f"阶段 {stage} 无可训练参数")

        return torch.optim.Adam(
            param_groups,
            betas=(float(self.cfg.beta1), float(self.cfg.beta2)),
            eps=float(self.cfg.eps),
            weight_decay=float(self.cfg.weight_decay),
        )

    def _expand_stage_records(self, records: Sequence[PromptRecord], epochs: float) -> List[PromptRecord]:
        """按浮点 epoch 展开样本序列。

        输入：
        - records: 原始 prompt 列表。
        - epochs: 浮点 epoch，如 `1.0/0.1/0.02`。

        输出：
        - List[PromptRecord]：展开后的阶段训练样本序列。
        """
        if not records:
            return []
        e = float(epochs)
        whole = int(math.floor(e))
        frac = e - float(whole)
        out: List[PromptRecord] = []
        for _ in range(whole):
            out.extend(records)
        if frac > 1e-12:
            keep = max(1, int(math.ceil(frac * len(records))))
            out.extend(records[:keep])
        return out

    def _align_weight(self, *, stage: str, local_step: int, total_steps: int) -> float:
        """计算当前 step 的对齐权重。

        输入：
        - stage: 当前训练阶段。
        - local_step: 阶段内步数（从 1 开始）。
        - total_steps: 该阶段总步数估计。

        输出：
        - float：本 step 使用的 `align_weight`。
        """
        if stage == "stage1":
            return 0.0
        if stage == "stage2":
            target = float(self.cfg.align_weight_target)
            warm_ratio = max(1e-6, float(self.cfg.align_warmup_ratio))
            warm_steps = max(1, int(math.ceil(warm_ratio * max(1, total_steps))))
            if local_step >= warm_steps:
                return target
            return target * float(local_step) / float(warm_steps)
        if stage == "stage3":
            return float(self.cfg.align_weight_target)
        return 0.0

    def _time_branch_scale(self, *, stage: str, local_step: int, total_steps: int) -> float:
        """按阶段进度渐进打开 time branch。"""
        if not bool(self.cfg.use_time_branch):
            return 0.0
        if stage == "stage1":
            return 0.0
        if stage in ("stage3", "stage4"):
            return 1.0

        start_ratio = float(self.cfg.time_branch_warmup_start_ratio)
        warm_ratio = float(self.cfg.time_branch_warmup_ratio)
        if warm_ratio <= 0.0:
            return 1.0

        total = max(1, int(total_steps))
        start_step = int(math.floor(max(0.0, start_ratio) * total))
        warm_steps = max(1, int(math.ceil(warm_ratio * total)))
        progress = (int(local_step) - start_step) / float(warm_steps)
        return float(max(0.0, min(1.0, progress)))

    def _stage_blocks(self, stage: str) -> List[str]:
        """返回阶段使用的 block 列表。

        输入：
        - stage: 阶段名。

        输出：
        - List[str]：当前阶段参与训练的 block。
        """
        if stage == "stage1":
            return [self.cfg.mid_block]
        return list(self.cfg.blocks)

    def _extract_groups_for_shard(self, shard_records: Sequence[PromptRecord], blocks: Sequence[str]) -> List[GroupSample]:
        """从一个 prompt shard 采样 group 列表。

        输入：
        - shard_records: 当前分片 prompt。
        - blocks: 当前阶段 block 列表。

        输出：
        - List[GroupSample]：该分片内的所有 group 样本。
        """
        groups: List[GroupSample] = []
        for rec in shard_records:
            gs = self.sampler.sample_prompt_groups(
                prompt_id=int(rec.prompt_id),
                prompt=str(rec.prompt),
                seed=int(rec.seed),
                blocks=blocks,
            )
            groups.extend(gs)
        return groups

    def _run_group_batch(
        self,
        *,
        batch_groups: Sequence[GroupSample],
        blocks: Sequence[str],
        stage: str,
        optimizer: torch.optim.Optimizer,
        align_weight: float,
        time_branch_scale: float,
    ) -> tuple[LossBreakdown, BatchDiagnostics]:
        """执行一个 group batch 的前向、反向与参数更新。

        输入：
        - batch_groups: 当前批次 group 列表。
        - blocks: 参与训练的 block 列表。
        - stage: 当前阶段名。
        - optimizer: 优化器。
        - align_weight: 当前 step 对齐损失权重。

        输出：
        - LossBreakdown：该批次的平均损失拆解。
        """
        optimizer.zero_grad(set_to_none=True)
        group_losses: List[LossBreakdown] = []
        total_latents = 0
        total_active = 0
        use_out_adapter = stage == "stage4"

        for g in batch_groups:
            cache_by_block = {}
            x_norm_by_block = {}
            coords = g.coords_norm.to(self.device, dtype=torch.float32)
            timestep_t = torch.tensor([float(g.timestep)], device=self.device, dtype=torch.float32)

            for b in blocks:
                if b not in g.block_tokens:
                    continue
                x = g.block_tokens[b].to(self.device, dtype=torch.float32)
                s_b = float(self.norm_scale_by_block[b])
                x_norm = apply_block_scale(x, s_b)
                cache = self.model(
                    x_norm,
                    block_name=b,
                    timestep=timestep_t,
                    coords_norm=coords,
                    use_out_adapter=bool(use_out_adapter),
                    update_dead_stats=True,
                    time_branch_scale=float(time_branch_scale),
                )
                cache_by_block[b] = cache
                x_norm_by_block[b] = x_norm
                total_latents += int(cache.z.numel())
                total_active += int((cache.z > 0).sum().item())

            loss = group_forward_losses(
                forward_cache_by_block=cache_by_block,
                x_norm_by_block=x_norm_by_block,
                blocks=list(blocks),
                mid_block=self.cfg.mid_block,
                auxk_coef=float(self.cfg.auxk_coef),
                align_weight=float(align_weight),
                latent_decorr_weight=float(self.cfg.latent_decorr_weight),
                latent_decorr_top_k=int(self.cfg.latent_decorr_top_k),
            )
            group_losses.append(loss)

        total = torch.stack([x.total for x in group_losses]).mean()
        decoder_decorr = decoder_decorrelation_loss(self.model.decoder.weight)
        total = total + float(self.cfg.decoder_decorr_weight) * decoder_decorr
        total.backward()

        if bool(self.cfg.decoder_unit_norm):
            unit_norm_decoder_grad_adjustment_(self.model)
        if float(self.cfg.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.cfg.grad_clip))
        optimizer.step()
        if bool(self.cfg.decoder_unit_norm):
            unit_norm_decoder_(self.model)

        recon = torch.stack([x.recon for x in group_losses]).mean().detach()
        auxk = torch.stack([x.auxk for x in group_losses]).mean().detach()
        align = torch.stack([x.align for x in group_losses]).mean().detach()
        latent_decorr = torch.stack([x.latent_decorr for x in group_losses]).mean().detach()
        decoder_decorr_detached = decoder_decorr.detach()
        total_detached = total.detach()
        diagnostics = BatchDiagnostics(
            loss_auxk_term=float(float(self.cfg.auxk_coef) * float(auxk)),
            loss_align_term=float(float(align_weight) * float(align)),
            loss_decoder_decorr_term=float(float(self.cfg.decoder_decorr_weight) * float(decoder_decorr_detached)),
            loss_latent_decorr_term=float(float(self.cfg.latent_decorr_weight) * float(latent_decorr)),
            time_branch_scale=float(time_branch_scale),
            latent_active_frac=float(total_active / max(1, total_latents)),
            dead_feature_frac=float(
                (self.model.stats_last_nonzero > int(self.cfg.dead_tokens_threshold)).float().mean().item()
            ),
        )
        return LossBreakdown(
            total=total_detached,
            recon=recon,
            auxk=auxk,
            align=align,
            decoder_decorr=decoder_decorr_detached,
            latent_decorr=latent_decorr,
        ), diagnostics

    def _group_batch_size(self, stage: str, *, hw: tuple[int, int], n_blocks: int) -> int:
        """返回阶段对应的 group batch size。

        输入：
        - stage: 阶段名。
        - hw: 当前 group 的空间尺寸。
        - n_blocks: 当前阶段参与训练的 block 数。

        输出：
        - int：配置值，或按 `tokens_per_step_target` 自动推导的 batch size。
        """
        cfg_bs = int(self.cfg.group_bs_stage1) if stage == "stage1" else int(self.cfg.group_bs_stage2)
        if cfg_bs > 0:
            return cfg_bs
        tokens_per_group = int(n_blocks) * int(hw[0]) * int(hw[1])
        return max(1, int(self.cfg.tokens_per_step_target) // max(1, tokens_per_group))

    def _iter_group_batches(self, groups: Sequence[GroupSample], batch_size: int) -> Iterable[List[GroupSample]]:
        """将 group 列表切分为 mini-batch。

        输入：
        - groups: 全量 group。
        - batch_size: 每批 group 数。

        输出：
        - Iterable[List[GroupSample]]：逐批返回 group 列表。
        """
        n = len(groups)
        i = 0
        while i < n:
            j = min(i + int(batch_size), n)
            yield list(groups[i:j])
            i = j

    def run_stage(self, stage: str, stage_records: Sequence[PromptRecord]) -> StageResult:
        """执行单个训练阶段。

        输入：
        - stage: `stage1|stage2|stage3|stage4`。
        - stage_records: 该阶段要使用的 prompt 列表。

        输出：
        - StageResult：阶段指标汇总。
        """
        t0 = time.time()
        blocks = self._stage_blocks(stage)
        set_stage_trainable(self.model, stage)
        self.model.use_block_out_adapter = stage == "stage4"

        optimizer = self._build_optimizer(stage)
        self.model.train()

        expanded_records = self._expand_stage_records(
            stage_records,
            {
                "stage1": float(self.cfg.epochs_stage1),
                "stage2": float(self.cfg.epochs_stage2),
                "stage3": float(self.cfg.epochs_stage3),
                "stage4": float(self.cfg.epochs_stage4),
            }[stage],
        )

        all_losses: List[LossBreakdown] = []
        local_step = 0
        logged_group_budget = False
        observed_hw: tuple[int, int] | None = None
        observed_group_bs: int | None = None
        observed_tokens_per_group: int | None = None
        observed_tokens_per_step: int | None = None
        prompt_pbar = tqdm(total=len(expanded_records), desc=f"{stage} prompts", unit="prompt")
        try:
            for shard_idx, shard in enumerate(iter_prompt_shards(expanded_records, int(self.cfg.shard_prompts))):
                groups: List[GroupSample] = []
                for rec in shard:
                    gs = self.sampler.sample_prompt_groups(
                        prompt_id=int(rec.prompt_id),
                        prompt=str(rec.prompt),
                        seed=int(rec.seed),
                        blocks=blocks,
                    )
                    groups.extend(gs)
                    prompt_pbar.update(1)
                if not groups:
                    continue
                hw = (int(groups[0].hw[0]), int(groups[0].hw[1]))
                gbs = self._group_batch_size(stage, hw=hw, n_blocks=len(blocks))
                if not logged_group_budget:
                    tokens_per_group = int(len(blocks) * int(hw[0]) * int(hw[1]))
                    tokens_per_step = int(tokens_per_group * gbs)
                    cfg_bs = int(self.cfg.group_bs_stage1) if stage == "stage1" else int(self.cfg.group_bs_stage2)
                    auto_enabled = cfg_bs <= 0
                    suggested_gbs = max(1, int(self.cfg.tokens_per_step_target) // max(1, tokens_per_group))
                    print(
                        f"[{stage}] group_budget hw={hw} blocks={len(blocks)} "
                        f"tokens/group={tokens_per_group} group_bs={gbs} "
                        f"group_bs_mode={'auto' if auto_enabled else 'fixed'} "
                        f"tokens/step={tokens_per_step} target_tokens/step={int(self.cfg.tokens_per_step_target)} "
                        f"suggested_group_bs={suggested_gbs}"
                    )
                    observed_hw = hw
                    observed_group_bs = gbs
                    observed_tokens_per_group = tokens_per_group
                    observed_tokens_per_step = tokens_per_step
                    logged_group_budget = True
                approx_total_steps = max(1, math.ceil(len(expanded_records) / max(1, int(self.cfg.shard_prompts))))
                approx_total_steps *= max(1, math.ceil(len(groups) / max(1, gbs)))

                for batch in self._iter_group_batches(groups, gbs):
                    local_step += 1
                    self.global_step += 1
                    align_w = self._align_weight(stage=stage, local_step=local_step, total_steps=approx_total_steps)
                    time_scale = self._time_branch_scale(stage=stage, local_step=local_step, total_steps=approx_total_steps)
                    loss, diagnostics = self._run_group_batch(
                        batch_groups=batch,
                        blocks=blocks,
                        stage=stage,
                        optimizer=optimizer,
                        align_weight=align_w,
                        time_branch_scale=time_scale,
                    )
                    all_losses.append(loss)
                    lrs = self._extract_group_lrs(optimizer)
                    self.metrics_writer.write_step(
                        StepMetric(
                            ts_unix=float(time.time()),
                            stage=str(stage),
                            global_step=int(self.global_step),
                            local_step=int(local_step),
                            loss_total=float(loss.total),
                            loss_recon=float(loss.recon),
                            loss_auxk=float(loss.auxk),
                            loss_align=float(loss.align),
                            loss_decoder_decorr=float(loss.decoder_decorr),
                            loss_latent_decorr=float(loss.latent_decorr),
                            loss_auxk_term=float(diagnostics.loss_auxk_term),
                            loss_align_term=float(diagnostics.loss_align_term),
                            loss_decoder_decorr_term=float(diagnostics.loss_decoder_decorr_term),
                            loss_latent_decorr_term=float(diagnostics.loss_latent_decorr_term),
                            align_weight=float(align_w),
                            time_branch_scale=float(diagnostics.time_branch_scale),
                            latent_active_frac=float(diagnostics.latent_active_frac),
                            dead_feature_frac=float(diagnostics.dead_feature_frac),
                            lr_shared=float(lrs["shared"]),
                            lr_adapter=float(lrs["adapter"]),
                            lr_time=float(lrs["time"]),
                            lr_spatial=float(lrs["spatial"]),
                        )
                    )

                    if self.global_step % int(self.cfg.log_every_steps) == 0:
                        print(
                            f"[{stage}] step={self.global_step} "
                            f"loss={float(loss.total):.6f} "
                            f"recon={float(loss.recon):.6f} "
                            f"auxk={float(loss.auxk):.6f} "
                            f"align={float(loss.align):.6f} "
                            f"decorr={float(loss.decoder_decorr):.6f} "
                            f"latent_decorr={float(loss.latent_decorr):.6f} "
                            f"auxk_term={diagnostics.loss_auxk_term:.6f} "
                            f"align_term={diagnostics.loss_align_term:.6f} "
                            f"decorr_term={diagnostics.loss_decoder_decorr_term:.6f} "
                            f"latent_decorr_term={diagnostics.loss_latent_decorr_term:.6f} "
                            f"active={diagnostics.latent_active_frac:.4f} "
                            f"dead={diagnostics.dead_feature_frac:.4f} "
                            f"align_w={align_w:.6f} "
                            f"time_scale={diagnostics.time_branch_scale:.4f}"
                        )

                    if int(self.cfg.save_every_steps) > 0 and self.global_step % int(self.cfg.save_every_steps) == 0:
                        save_checkpoint(
                            output_root=self.cfg.output_root,
                            stage=stage,
                            global_step=self.global_step,
                            cfg=self.cfg,
                            model=self.model,
                            optimizer=optimizer,
                            norm_scale_by_block=self.norm_scale_by_block,
                            extra={"local_step": local_step, "shard_idx": shard_idx},
                        )
        finally:
            prompt_pbar.close()

        if all_losses:
            total = float(torch.stack([x.total for x in all_losses]).mean().item())
            recon = float(torch.stack([x.recon for x in all_losses]).mean().item())
            auxk = float(torch.stack([x.auxk for x in all_losses]).mean().item())
            align = float(torch.stack([x.align for x in all_losses]).mean().item())
            decoder_decorr = float(torch.stack([x.decoder_decorr for x in all_losses]).mean().item())
            latent_decorr = float(torch.stack([x.latent_decorr for x in all_losses]).mean().item())
        else:
            total = recon = auxk = align = decoder_decorr = latent_decorr = 0.0

        result = StageResult(
            stage=stage,
            steps=local_step,
            mean_total=total,
            mean_recon=recon,
            mean_auxk=auxk,
            mean_align=align,
            mean_decoder_decorr=decoder_decorr,
            mean_latent_decorr=latent_decorr,
            elapsed_sec=float(time.time() - t0),
            observed_hw=observed_hw,
            group_bs=observed_group_bs,
            tokens_per_group=observed_tokens_per_group,
            tokens_per_step=observed_tokens_per_step,
        )
        self.history.append(result)
        self.metrics_writer.write_stage_summary(
            {
                "stage": result.stage,
                "steps": int(result.steps),
                "mean_total": float(result.mean_total),
                "mean_recon": float(result.mean_recon),
                "mean_auxk": float(result.mean_auxk),
                "mean_align": float(result.mean_align),
                "mean_decoder_decorr": float(result.mean_decoder_decorr),
                "mean_latent_decorr": float(result.mean_latent_decorr),
                "elapsed_sec": float(result.elapsed_sec),
                "global_step": int(self.global_step),
            }
        )

        save_checkpoint(
            output_root=self.cfg.output_root,
            stage=stage,
            global_step=self.global_step,
            cfg=self.cfg,
            model=self.model,
            optimizer=optimizer,
            norm_scale_by_block=self.norm_scale_by_block,
            extra={"stage_result": result.__dict__},
        )
        return result

    @torch.no_grad()
    def evaluate_stage_metrics(self, records: Sequence[PromptRecord], *, stage: str, max_groups: int = 200) -> EvalMetrics:
        """按阶段设置在验证集上评估 recon / align。

        输入：
        - records: 验证集 prompt 列表。
        - stage: 当前评估阶段。
        - max_groups: 为控制开销，最多评估多少个 group。

        输出：
        - EvalMetrics：平均 recon / align 与实际评估的 group 数。
        """
        was_training = self.model.training
        self.model.eval()
        blocks = self._stage_blocks(stage)
        use_out_adapter = stage == "stage4"
        recon_items: List[torch.Tensor] = []
        align_items: List[torch.Tensor] = []
        seen = 0
        try:
            for rec in tqdm(records, total=len(records), desc="val prompts", unit="prompt"):
                groups = self._extract_groups_for_shard([rec], blocks=blocks)
                for g in groups:
                    cache_by_block = {}
                    x_norm_by_block = {}
                    coords = g.coords_norm.to(self.device, dtype=torch.float32)
                    t = torch.tensor([float(g.timestep)], device=self.device, dtype=torch.float32)
                    for b in blocks:
                        if b not in g.block_tokens:
                            continue
                        x = g.block_tokens[b].to(self.device, dtype=torch.float32)
                        x_norm = apply_block_scale(x, float(self.norm_scale_by_block[b]))
                        cache = self.model(
                            x_norm,
                            block_name=b,
                            timestep=t,
                            coords_norm=coords,
                            use_out_adapter=bool(use_out_adapter),
                            update_dead_stats=False,
                        )
                        cache_by_block[b] = cache
                        x_norm_by_block[b] = x_norm

                    if not cache_by_block:
                        continue

                    loss = group_forward_losses(
                        forward_cache_by_block=cache_by_block,
                        x_norm_by_block=x_norm_by_block,
                        blocks=list(blocks),
                        mid_block=self.cfg.mid_block,
                        auxk_coef=0.0,
                        align_weight=1.0,
                    )
                    recon_items.append(loss.recon.detach())
                    if len(blocks) > 1:
                        align_items.append(loss.align.detach())
                    seen += 1
                    if seen >= int(max_groups):
                        break
                if seen >= int(max_groups):
                    break
        finally:
            if was_training:
                self.model.train()

        if not recon_items:
            return EvalMetrics(recon=0.0, align=0.0, groups=0)
        recon = float(torch.stack(recon_items).mean().item())
        align = float(torch.stack(align_items).mean().item()) if align_items else 0.0
        return EvalMetrics(recon=recon, align=align, groups=seen)
