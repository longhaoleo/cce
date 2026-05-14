"""Evaluate SharedSAE batch erasure outputs.

The script reads the current runtime/shared batch layout:

    image_output/<run>/<concept>/case_000000/
      eval_original/<case>_<concept>.png
      eval_erased/<case>_<concept>.png
      run_manifest.json
      diag_shared_intervention_*.csv

It writes case-level metrics and aggregate summaries into a separate output
directory. CLIP/LPIPS/DreamSim are optional so missing heavy dependencies do not
block basic pixel/diagnostic evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_CLIP_MODEL_DIR = "/root/autodl-tmp/models/clip-vit-base-patch32"
DEFAULT_TORCH_HOME = "/root/autodl-tmp/models/torch"


@dataclass
class CaseItem:
    case_dir: Path
    case_number: int
    concept: str
    prompt: str
    original_path: Path
    erased_path: Path
    manifest: Dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantitative metrics for SharedSAE batch erasure outputs.")
    parser.add_argument("--batch_root", type=str, required=True, help="Batch output root, e.g. image_output/batch_shared_concept_erase_dog")
    parser.add_argument("--concept", type=str, default="", help="Concept subdirectory. If omitted, infer from batch_root.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write metrics CSV/JSON files.")
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="",
        help="Optional evaluation CSV from data/ or batch_test_prompt/. Must contain case_number,prompt.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="pixel,diag,clip",
        help="Comma-separated metrics: pixel,diag,clip,lpips,dreamsim. Default: pixel,diag,clip.",
    )
    parser.add_argument("--from_case", type=int, default=0)
    parser.add_argument("--to_case", type=int, default=10**9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--clip_model",
        type=str,
        default="",
        help=(
            "CLIP model path or HF id. Default rule: CCE_CLIP_MODEL env, then "
            f"{DEFAULT_CLIP_MODEL_DIR} if it exists, otherwise {DEFAULT_CLIP_MODEL_ID}."
        ),
    )
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target_texts", type=str, default="", help="Comma-separated target CLIP labels. Default: a photo of <concept>.")
    parser.add_argument(
        "--negative_texts",
        type=str,
        default="an empty scene,a background without the target object",
        help="Comma-separated non-target CLIP labels for target probability.",
    )
    parser.add_argument("--lpips_resize", type=int, default=256)
    parser.add_argument(
        "--torch_home",
        type=str,
        default="",
        help=(
            "Torch model cache for LPIPS/TorchVision. Default rule: CCE_TORCH_HOME env, "
            f"then TORCH_HOME env, then {DEFAULT_TORCH_HOME}."
        ),
    )
    return parser.parse_args()


def _split_csv_texts(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _default_target_texts(concept: str) -> List[str]:
    name = str(concept).strip()
    article = "an" if name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    if " " in name:
        return [f"a photo of {name}"]
    return [f"a photo of {article} {name}", name]


def _resolve_clip_model_path(value: str) -> str:
    explicit = str(value).strip()
    if explicit:
        return explicit
    env_value = os.environ.get("CCE_CLIP_MODEL", "").strip()
    if env_value:
        return env_value
    local_dir = Path(DEFAULT_CLIP_MODEL_DIR)
    if local_dir.exists():
        return str(local_dir)
    return DEFAULT_CLIP_MODEL_ID


def _resolve_torch_home(value: str) -> str:
    explicit = str(value).strip()
    if explicit:
        return explicit
    env_value = os.environ.get("CCE_TORCH_HOME", "").strip()
    if env_value:
        return env_value
    env_value = os.environ.get("TORCH_HOME", "").strip()
    if env_value:
        return env_value
    return DEFAULT_TORCH_HOME


def _find_image(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    return images[0] if images else None


def _case_number_from_dir(path: Path) -> int:
    m = re.search(r"case_(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse case number from {path}")
    return int(m.group(1))


def _resolve_concept_root(batch_root: Path, concept: str) -> tuple[Path, str]:
    if concept:
        candidate = batch_root / concept
        if candidate.exists():
            return candidate, concept
        return batch_root, concept

    case_dirs = sorted(batch_root.glob("case_*"))
    if case_dirs:
        manifest_path = case_dirs[0] / "run_manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            return batch_root, str(manifest.get("targetconcept") or batch_root.name)
        return batch_root, batch_root.name

    subdirs = [p for p in sorted(batch_root.iterdir()) if p.is_dir()]
    for sub in subdirs:
        if list(sub.glob("case_*")):
            return sub, sub.name
    raise FileNotFoundError(f"No case_* directories found under {batch_root}")


def _load_cases(batch_root: Path, concept: str, from_case: int, to_case: int) -> List[CaseItem]:
    concept_root, concept_name = _resolve_concept_root(batch_root, concept)
    cases: List[CaseItem] = []
    for case_dir in sorted(concept_root.glob("case_*")):
        case_number = _case_number_from_dir(case_dir)
        if case_number < int(from_case) or case_number > int(to_case):
            continue
        manifest_path = case_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        original_path = _find_image(case_dir / "eval_original")
        erased_path = _find_image(case_dir / "eval_erased")
        if original_path is None or erased_path is None:
            continue
        cases.append(
            CaseItem(
                case_dir=case_dir,
                case_number=case_number,
                concept=str(manifest.get("targetconcept") or concept_name),
                prompt=str(manifest.get("prompt") or ""),
                original_path=original_path,
                erased_path=erased_path,
                manifest=manifest,
            )
        )
    if not cases:
        raise RuntimeError(f"No valid paired cases found under {concept_root}")
    return cases


def _load_prompt_rows(prompts_path: str) -> Dict[int, Dict[str, str]]:
    if not str(prompts_path).strip():
        return {}
    path = Path(prompts_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"prompts_path not found: {path}")
    out: Dict[int, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "case_number" not in reader.fieldnames or "prompt" not in reader.fieldnames:
            raise ValueError(f"prompts_path must contain case_number,prompt columns: {path}")
        for row in reader:
            case_number = int(row["case_number"])
            out[case_number] = {k: str(v) for k, v in row.items() if v is not None}
    return out


def _image_array(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0


def _pixel_metrics(original_path: Path, erased_path: Path) -> Dict[str, float]:
    a = _image_array(original_path)
    b = _image_array(erased_path)
    if a.shape != b.shape:
        b_img = Image.open(erased_path).convert("RGB").resize((a.shape[1], a.shape[0]), Image.Resampling.BICUBIC)
        b = np.asarray(b_img).astype(np.float32) / 255.0
    diff = a - b
    mse = float(np.mean(diff * diff))
    l1 = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))
    return {"pixel_l1": l1, "pixel_mse": mse, "pixel_psnr": psnr}


def _read_diag_metrics(case_dir: Path) -> tuple[Dict[str, float], List[Dict[str, object]]]:
    all_delta_over_x: List[float] = []
    all_delta_x: List[float] = []
    block_rows: List[Dict[str, object]] = []
    for path in sorted(case_dir.glob("diag_shared_intervention_*.csv")):
        block = path.name.removeprefix("diag_shared_intervention_").removesuffix(".csv")
        deltas: List[float] = []
        delta_xs: List[float] = []
        active_count = 0
        total = 0
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                total += 1
                active_count += int(float(row.get("active", 0) or 0) > 0)
                d_over = float(row.get("delta_over_x", 0) or 0)
                d_x = float(row.get("mean_abs_delta_x", 0) or 0)
                deltas.append(d_over)
                delta_xs.append(d_x)
                all_delta_over_x.append(d_over)
                all_delta_x.append(d_x)
        if deltas:
            block_rows.append(
                {
                    "block": block,
                    "diag_block_mean_delta_over_x": float(np.mean(deltas)),
                    "diag_block_max_delta_over_x": float(np.max(deltas)),
                    "diag_block_mean_abs_delta_x": float(np.mean(delta_xs)),
                    "diag_block_active_ratio": float(active_count / max(1, total)),
                }
            )
    if not all_delta_over_x:
        return {}, block_rows
    return (
        {
            "diag_mean_delta_over_x": float(np.mean(all_delta_over_x)),
            "diag_max_delta_over_x": float(np.max(all_delta_over_x)),
            "diag_mean_abs_delta_x": float(np.mean(all_delta_x)),
        },
        block_rows,
    )


class ClipScorer:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        local_files_only: bool,
        target_texts: Sequence[str],
        negative_texts: Sequence[str],
    ) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            raise RuntimeError("CLIP metric requires transformers. Install transformers or run without --metrics clip.") from exc

        self.device = torch.device(device)
        try:
            self.model = CLIPModel.from_pretrained(model_name, local_files_only=bool(local_files_only)).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=bool(local_files_only))
        except OSError as exc:
            mode = "offline/local-files-only" if bool(local_files_only) else "online"
            raise RuntimeError(
                "Failed to load CLIP model for evaluation.\n"
                f"model={model_name}\n"
                f"mode={mode}\n"
                "If this machine has no cached CLIP weights, use one of:\n"
                "  1. run pixel/diag only: --metrics pixel,diag\n"
                "  2. download once: python download_clip.py\n"
                "  3. allow one online download in this command: remove --local_files_only\n"
                "  4. pass a local CLIP directory: --clip_model /path/to/clip-vit-base-patch32 --local_files_only"
            ) from exc
        self.model.eval()
        self.target_texts = list(target_texts)
        self.negative_texts = list(negative_texts)
        self.class_texts = self.target_texts + self.negative_texts

    @torch.no_grad()
    def score(self, *, image_path: Path, prompt: str) -> Dict[str, float]:
        image = Image.open(image_path).convert("RGB")

        prompt_inputs = self.processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        prompt_outputs = self.model(**prompt_inputs)
        prompt_logit = float(prompt_outputs.logits_per_image[0, 0].detach().cpu())

        class_inputs = self.processor(text=self.class_texts, images=image, return_tensors="pt", padding=True).to(self.device)
        class_outputs = self.model(**class_inputs)
        logits = class_outputs.logits_per_image[0].detach().float().cpu()
        probs = torch.softmax(logits, dim=0)
        target_n = max(1, len(self.target_texts))
        target_prob = float(probs[:target_n].sum().item())
        target_logit = float(logits[:target_n].max().item())
        negative_logit = float(logits[target_n:].max().item()) if len(logits) > target_n else float("nan")
        return {
            "clip_prompt_logit": prompt_logit,
            "clip_target_prob": target_prob,
            "clip_target_logit": target_logit,
            "clip_negative_logit": negative_logit,
            "clip_target_margin": target_logit - negative_logit if not math.isnan(negative_logit) else float("nan"),
        }


class LpipsScorer:
    def __init__(self, *, device: str, resize: int, torch_home: str) -> None:
        os.environ.setdefault("TORCH_HOME", str(Path(torch_home).expanduser()))

        import lpips
        import torchvision.transforms as transforms

        self.device = torch.device(device)
        self.loss_fn = lpips.LPIPS(net="alex").to(self.device)
        self.loss_fn.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(resize)),
                transforms.CenterCrop(int(resize)),
                transforms.ToTensor(),
            ]
        )

    def _load(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        x = self.transform(image).unsqueeze(0)
        x = (x - 0.5) * 2.0
        return x.to(self.device)

    @torch.no_grad()
    def score(self, original_path: Path, erased_path: Path) -> Dict[str, float]:
        value = self.loss_fn(self._load(original_path), self._load(erased_path))
        return {"lpips_alex": float(value.item())}


class DreamSimScorer:
    def __init__(self, *, device: str) -> None:
        from dreamsim import dreamsim

        self.device = torch.device(device)
        self.model, self.preprocess = dreamsim(pretrained=True, device=str(self.device))

    @torch.no_grad()
    def score(self, original_path: Path, erased_path: Path) -> Dict[str, float]:
        img1 = self.preprocess(Image.open(original_path).convert("RGB")).to(self.device)
        img2 = self.preprocess(Image.open(erased_path).convert("RGB")).to(self.device)
        value = self.model(img1, img2)
        return {"dreamsim_distance": float(value.item())}


def _mean(values: Iterable[float]) -> float:
    xs = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"case_number"}
        }
    )
    summary: Dict[str, object] = {"num_cases": len(rows)}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
        vals = [x for x in vals if not math.isnan(x)]
        if not vals:
            continue
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_median"] = float(np.median(vals))
        summary[f"{key}_min"] = float(np.min(vals))
        summary[f"{key}_max"] = float(np.max(vals))
    return summary


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    metrics = {m.strip().lower() for m in str(args.metrics).split(",") if m.strip()}
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(
        Path(args.batch_root).expanduser().resolve(),
        concept=str(args.concept).strip(),
        from_case=int(args.from_case),
        to_case=int(args.to_case),
    )
    prompt_rows = _load_prompt_rows(str(args.prompts_path))
    concept = cases[0].concept
    target_texts = _split_csv_texts(args.target_texts) or _default_target_texts(concept)
    negative_texts = _split_csv_texts(args.negative_texts)

    clip_scorer: Optional[ClipScorer] = None
    lpips_scorer: Optional[LpipsScorer] = None
    dreamsim_scorer: Optional[DreamSimScorer] = None

    if "clip" in metrics:
        clip_model_resolved = _resolve_clip_model_path(str(args.clip_model))
        clip_scorer = ClipScorer(
            model_name=clip_model_resolved,
            device=str(args.device),
            local_files_only=bool(args.local_files_only),
            target_texts=target_texts,
            negative_texts=negative_texts,
        )
    if "lpips" in metrics:
        lpips_scorer = LpipsScorer(
            device=str(args.device),
            resize=int(args.lpips_resize),
            torch_home=_resolve_torch_home(str(args.torch_home)),
        )
    if "dreamsim" in metrics:
        dreamsim_scorer = DreamSimScorer(device=str(args.device))

    rows: List[Dict[str, object]] = []
    block_rows: List[Dict[str, object]] = []
    for item in cases:
        row: Dict[str, object] = {
            "case_number": int(item.case_number),
            "concept": item.concept,
            "prompt": item.prompt,
            "dataset_prompt": prompt_rows.get(int(item.case_number), {}).get("prompt", ""),
            "dataset_evaluation_seed": prompt_rows.get(int(item.case_number), {}).get("evaluation_seed", ""),
            "seed": item.manifest.get("seed", ""),
            "scale": item.manifest.get("intervention", {}).get("scale", ""),
            "feature_top_k": item.manifest.get("intervention", {}).get("feature_top_k", ""),
            "original_path": str(item.original_path),
            "erased_path": str(item.erased_path),
        }

        if "pixel" in metrics:
            row.update(_pixel_metrics(item.original_path, item.erased_path))
        if "diag" in metrics:
            diag, blocks = _read_diag_metrics(item.case_dir)
            row.update(diag)
            for block_row in blocks:
                block_rows.append({"case_number": int(item.case_number), "concept": item.concept, **block_row})
        if clip_scorer is not None:
            eval_prompt = prompt_rows.get(int(item.case_number), {}).get("prompt", "") or item.prompt
            original_clip = clip_scorer.score(image_path=item.original_path, prompt=eval_prompt)
            erased_clip = clip_scorer.score(image_path=item.erased_path, prompt=eval_prompt)
            for key, value in original_clip.items():
                row[f"original_{key}"] = value
            for key, value in erased_clip.items():
                row[f"erased_{key}"] = value
            row["delta_clip_prompt_logit"] = erased_clip["clip_prompt_logit"] - original_clip["clip_prompt_logit"]
            row["target_prob_drop"] = original_clip["clip_target_prob"] - erased_clip["clip_target_prob"]
            row["target_margin_drop"] = original_clip["clip_target_margin"] - erased_clip["clip_target_margin"]
        if lpips_scorer is not None:
            row.update(lpips_scorer.score(item.original_path, item.erased_path))
        if dreamsim_scorer is not None:
            row.update(dreamsim_scorer.score(item.original_path, item.erased_path))
        rows.append(row)
        print(f"[eval] case={item.case_number} concept={item.concept}")

    summary = _summarize(rows)
    summary.update(
        {
            "concept": concept,
            "batch_root": str(Path(args.batch_root).expanduser().resolve()),
            "prompts_path": str(args.prompts_path),
            "metrics": sorted(metrics),
            "clip_model": _resolve_clip_model_path(str(args.clip_model)) if "clip" in metrics else "",
            "torch_home": _resolve_torch_home(str(args.torch_home)) if "lpips" in metrics else "",
            "target_texts": target_texts,
            "negative_texts": negative_texts,
        }
    )

    _write_csv(output_dir / "case_metrics.csv", rows)
    _write_csv(output_dir / "block_diag_metrics.csv", block_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_csv(output_dir / "summary.csv", [summary])
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"[eval] wrote {output_dir / 'case_metrics.csv'}")
    print(f"[eval] wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
