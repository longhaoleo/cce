#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

VARIANT="${VARIANT:-block_pooled_mean}"
case "${VARIANT}" in
  token)
    DECORR_MODE="token"
    DECORR_POOL="mean"
    ;;
  block_pooled_mean)
    DECORR_MODE="block_pooled"
    DECORR_POOL="mean"
    ;;
  block_pooled_topq)
    DECORR_MODE="block_pooled"
    DECORR_POOL="topq"
    ;;
  block_pooled_hybrid)
    DECORR_MODE="block_pooled"
    DECORR_POOL="hybrid"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}. Use token, block_pooled_mean, block_pooled_topq, or block_pooled_hybrid." >&2
    exit 2
    ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-train/output_time_latentdecorr_v2_${VARIANT}}"
VALIDATION_PROMPTS="${VALIDATION_PROMPTS:-500}"
STAGE2_TRAIN_PROMPTS="${STAGE2_TRAIN_PROMPTS:-10000}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-500}"
SHARD_PROMPTS="${SHARD_PROMPTS:-250}"
LATENT_DECORR_WEIGHT="${LATENT_DECORR_WEIGHT:-0.3}"
LATENT_DECORR_TOP_K="${LATENT_DECORR_TOP_K:-512}"
LATENT_DECORR_POOL_TOPQ="${LATENT_DECORR_POOL_TOPQ:-0.1}"
NORM_SCALE_CACHE_PATH="${NORM_SCALE_CACHE_PATH:-}"
NORM_SCALE_CACHE_ARGS=()
if [[ -n "${NORM_SCALE_CACHE_PATH}" ]]; then
  NORM_SCALE_CACHE_ARGS=(--norm_scale_cache_path "${NORM_SCALE_CACHE_PATH}")
fi

echo "[latent-decorr-v2] variant=${VARIANT}"
echo "[latent-decorr-v2] output_root=${OUTPUT_ROOT}"
echo "[latent-decorr-v2] mode=${DECORR_MODE} pool=${DECORR_POOL} topq=${LATENT_DECORR_POOL_TOPQ}"
echo "[latent-decorr-v2] norm_scale_cache=${NORM_SCALE_CACHE_PATH:-auto}"

python train/run_train.py \
  --experiment_preset custom \
  --output_root "${OUTPUT_ROOT}" \
  --local_files_only \
  --steps 50 \
  --validation_prompts "${VALIDATION_PROMPTS}" \
  --stage2_train_prompts "${STAGE2_TRAIN_PROMPTS}" \
  --calibration_prompts "${CALIBRATION_PROMPTS}" \
  --num_step_buckets 5 \
  --shard_prompts "${SHARD_PROMPTS}" \
  --expansion_factor 8 \
  --top_k 20 \
  --auxk 512 \
  --use_time_branch \
  --time_branch_mode sincos_linear \
  --time_branch_warmup_start_ratio 0.3 \
  --time_branch_warmup_ratio 0.3 \
  --no-use_spatial_branch \
  --run_stage3 \
  --use_block_in_adapter \
  --align_weight_target 0.1 \
  --latent_decorr_weight "${LATENT_DECORR_WEIGHT}" \
  --latent_decorr_top_k "${LATENT_DECORR_TOP_K}" \
  --latent_decorr_mode "${DECORR_MODE}" \
  --latent_decorr_pool "${DECORR_POOL}" \
  --latent_decorr_pool_topq "${LATENT_DECORR_POOL_TOPQ}" \
  --latent_decorr_eps 1e-4 \
  "${NORM_SCALE_CACHE_ARGS[@]}" \
  --group_bs 0 \
  --save_every_steps 0
