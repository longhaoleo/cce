"""
SharedSAE 训练配置定义。

这份配置的职责很单纯：

1. 把训练一个 SharedSAE 所需的全部超参数放到同一个结构里；
2. 让训练脚本、checkpoint、日志、统计脚本共享同一套字段名；
3. 在真正开始训练前，尽可能提前发现不合法的参数组合。

也就是说，这里定义的是“实验长什么样”，而不是“训练怎么执行”。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


def _default_blocks() -> List[str]:
    """返回当前主线默认使用的 4 个 block。

    这 4 个 block 对应项目里最常用的一组层位：
    - 一个较深的 down block
    - 一个 mid block
    - 两个较浅的 up block

    把它们集中在默认函数里，能避免训练、定位、擦除脚本各自写一份硬编码。
    """
    return [
        "unet.down_blocks.2.attentions.1",
        "unet.mid_block.attentions.0",
        "unet.up_blocks.0.attentions.0",
        "unet.up_blocks.0.attentions.1",
    ]


@dataclass
class TrainConfig:
    """Shared SAE 训练总配置。

    字段可以粗略分成几组：

    - 数据/模型路径：决定从哪里加载 base model、prompt 列表、输出目录；
    - 采样设置：扩散步数、CFG、分辨率、随机种子；
    - block 与空间规格：选择训练哪些 block，以及预期的特征图大小；
    - SAE 结构：字典规模、top-k、AuxK、adapter/time/spatial 支路；
    - 优化参数：学习率、梯度裁剪、权重衰减；
    - 训练阶段：stage1/2/3/4 的开关、epoch、token 预算。

    这样一来，一次实验的“结构”和“流程”都可以被完整记录到 checkpoint 里。
    """

    prompts_csv: str = "data/coco_30k.csv"
    output_root: str = "train/output_shared_sae"
    experiment_preset: str = "custom"

    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    model_local_dir: str = ""
    local_files_only: bool = False
    device: str = "cuda"
    dtype: str = "fp16"
    steps: int = 50
    guidance_scale: float = 8.0
    resolution: int = 512
    base_seed: int = 42

    blocks: List[str] = field(default_factory=_default_blocks)
    num_step_buckets: int = 5
    expected_h: int = 16
    expected_w: int = 16

    validation_prompts: int = 1000
    stage2_train_prompts: int = 20000
    stage1_train_prompts: int = 5000
    calibration_prompts: int = 1000
    shard_prompts: int = 250
    split_seed: int = 2026

    d_model: int = 1280
    expansion_factor: int = 4
    top_k: int = 10
    auxk: int = 256
    dead_tokens_threshold: int = 10_000_000

    use_block_in_adapter: bool = True
    use_block_out_adapter: bool = False
    block_in_rank: int = 16
    block_in_alpha: int = 16
    block_out_rank: int = 16
    block_out_alpha: int = 16

    use_time_branch: bool = True
    time_branch_mode: str = "sincos_linear"
    time_embed_dim: int = 32
    time_hidden_dim: int = 128

    use_spatial_branch: bool = True
    spatial_branch_mode: str = "sincos_linear"
    spatial_embed_dim: int = 64
    spatial_hidden_dim: int = 128

    lr_shared: float = 1e-4
    lr_adapter: float = 2e-4
    lr_time: float = 1e-4
    lr_time_stage3: float = 2e-5
    lr_spatial: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 6.25e-10
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    decoder_unit_norm: bool = True

    auxk_coef: float = 1.0 / 32.0
    align_weight_target: float = 5e-2
    align_warmup_ratio: float = 0.1
    decoder_decorr_weight: float = 0.0

    epochs_stage1: float = 1.0
    epochs_stage2: float = 1.0
    epochs_stage3: float = 0.1
    epochs_stage4: float = 0.02
    run_stage1: bool = True
    run_stage3: bool = True
    run_stage4: bool = False

    tokens_per_step_target: int = 4096
    group_bs_stage1: int = 0
    group_bs_stage2: int = 0

    max_prompts_debug: int = 0
    save_every_steps: int = 200
    log_every_steps: int = 20

    @property
    def n_dirs(self) -> int:
        """返回共享字典大小。

        当前实现按 `expansion_factor * d_model` 定义隐藏维度总数。
        这样做的直觉是：字典大小随 block 表示维度线性放大，而不是手写一个脱钩的常数。
        """
        return int(self.expansion_factor) * int(self.d_model)

    @property
    def mid_block(self) -> str:
        """返回作为对齐锚点的中间层 block 名称。

        当前主线把 mid block 当成跨层语义对齐的天然参考点，所以这里给一个固定属性，
        方便训练、日志和分析脚本统一使用。
        """
        return "unet.mid_block.attentions.0"

    def validate(self) -> None:
        """校验配置合法性并在异常时抛错。

        这里主要拦三类问题：

        - 明显不可能的结构参数：如 `steps <= 0`、`top_k > n_dirs`；
        - 会破坏训练设定的采样/切分参数：如 stage1 prompt 数大于 stage2；
        - 与当前主线假设冲突的 block / mode 设置。

        训练一开始就做这些检查，可以避免跑了很久才因为配置错误中断。
        """
        if int(self.steps) <= 0:
            raise ValueError("steps 必须 > 0")
        if int(self.num_step_buckets) <= 0:
            raise ValueError("num_step_buckets 必须 > 0")
        if int(self.resolution) <= 0:
            raise ValueError("resolution 必须 > 0")
        if int(self.expected_h) < 0 or int(self.expected_w) < 0:
            raise ValueError("expected_h/expected_w 不能为负数；0 表示自动推断")
        if int(self.top_k) <= 0:
            raise ValueError("top_k 必须 > 0")
        if int(self.auxk) <= 0:
            raise ValueError("auxk 必须 > 0")
        if int(self.top_k) > int(self.n_dirs):
            raise ValueError("top_k 不能大于 n_dirs")
        if int(self.auxk) > int(self.n_dirs):
            raise ValueError("auxk 不能大于 n_dirs")

        valid_modes = {"sincos_linear", "sincos_mlp", "sincos_film"}
        if self.time_branch_mode not in valid_modes:
            raise ValueError(f"time_branch_mode 非法: {self.time_branch_mode}")
        if self.spatial_branch_mode not in valid_modes:
            raise ValueError(f"spatial_branch_mode 非法: {self.spatial_branch_mode}")

        if len(self.blocks) != 4:
            raise ValueError("blocks 必须包含 4 个目标层")
        if self.mid_block not in self.blocks:
            raise ValueError("blocks 必须包含 mid_block 作为对齐锚点")

        if int(self.validation_prompts) <= 0:
            raise ValueError("validation_prompts 必须 > 0")
        if int(self.stage2_train_prompts) <= 0:
            raise ValueError("stage2_train_prompts 必须 > 0")
        if int(self.stage1_train_prompts) <= 0:
            raise ValueError("stage1_train_prompts 必须 > 0")
        if int(self.stage1_train_prompts) > int(self.stage2_train_prompts):
            raise ValueError("stage1_train_prompts 不能大于 stage2_train_prompts")

        if int(self.tokens_per_step_target) <= 0:
            raise ValueError("tokens_per_step_target 必须 > 0")
        if int(self.group_bs_stage1) < 0 or int(self.group_bs_stage2) < 0:
            raise ValueError("group_bs 不能为负数；0 表示按真实 hw 自动推导")
        if int(self.log_every_steps) <= 0:
            raise ValueError("log_every_steps 必须 > 0")
        if int(self.save_every_steps) < 0:
            raise ValueError("save_every_steps 不能为负数；0 表示关闭中间 checkpoint")

    def to_dict(self) -> Dict:
        """将配置转换为可序列化字典。

        这个输出会被写进：
        - checkpoint 对应的 `config.json`
        - 训练/评估 manifest
        - 各种实验日志

        这里顺手把一些路径字段规范成字符串，避免后续 JSON 序列化不一致。
        """
        data = asdict(self)
        data["n_dirs"] = int(self.n_dirs)
        data["prompts_csv"] = str(Path(self.prompts_csv))
        data["output_root"] = str(Path(self.output_root))
        return data

    def ensure_paths(self) -> None:
        """规范路径并确保训练输出目录存在。

        当前训练入口仍然依赖这个方法，因此这里保留一个最小实现：

        - 规范 `prompts_csv` / `output_root` / `model_local_dir` 为字符串路径；
        - 创建 `output_root`，保证后续日志、checkpoint、metrics 能直接写入。

        不在这里做额外副作用，例如检查 prompts 是否存在或预建所有子目录。
        """
        self.prompts_csv = str(Path(self.prompts_csv).expanduser())
        self.output_root = str(Path(self.output_root).expanduser())
        if str(self.model_local_dir).strip():
            self.model_local_dir = str(Path(self.model_local_dir).expanduser())
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
