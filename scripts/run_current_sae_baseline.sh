#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

SAE_VARIANT="${SAE_VARIANT:-decorr03}"
case "${SAE_VARIANT}" in
  decorr03|0.3|latent_decorr0.3)
    DEFAULT_SAE_TAG="sae_x8_time_decorr03"
    DEFAULT_CKPT_DIR="train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772"
    ;;
  decorr01|0.01|latent_decorr0.01|x8_time)
    DEFAULT_SAE_TAG="sae_x8_time"
    DEFAULT_CKPT_DIR="train/output_time_latentdecorr_x8_top20_half/checkpoints/stage3_step_0013772"
    ;;
  *)
    echo "Unknown SAE_VARIANT=${SAE_VARIANT}. Use decorr03 or decorr01." >&2
    exit 2
    ;;
esac

SAE_TAG="${SAE_TAG:-${DEFAULT_SAE_TAG}}"
CKPT_DIR="${CKPT_DIR:-${DEFAULT_CKPT_DIR}}"
SAE_ROOT="${SAE_ROOT:-sae_data/${SAE_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-image_output/${SAE_TAG}}"
MODE="${MODE:-all}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
LOCATOR_MAX_PROMPTS_PER_SIDE="${LOCATOR_MAX_PROMPTS_PER_SIDE:-0}"

DEFAULT_CONCEPTS=(
  car
  dog
  nudity
  cat
  bird
  flower
  bicycle
  chair
  anime_style
  text
)

if [[ "${CONCEPTS:-}" == "" ]]; then
  CONCEPT_LIST=("${DEFAULT_CONCEPTS[@]}")
else
  # shellcheck disable=SC2206
  CONCEPT_LIST=(${CONCEPTS})
fi

echo "[current-sae] variant=${SAE_VARIANT}"
echo "[current-sae] tag=${SAE_TAG}"
echo "[current-sae] ckpt=${CKPT_DIR}"
echo "[current-sae] sae_root=${SAE_ROOT}"
echo "[current-sae] mode=${MODE}"
echo "[current-sae] concepts=${CONCEPT_LIST[*]}"

run_locator() {
  local locator_limit_args=()
  if [[ "${LOCATOR_MAX_PROMPTS_PER_SIDE}" != "0" ]]; then
    locator_limit_args=(--max_prompts_per_side "${LOCATOR_MAX_PROMPTS_PER_SIDE}")
  fi

  python -m runtime.shared.locator \
    --ckpt_dir "${CKPT_DIR}" \
    --local_files_only \
    --sae_root "${SAE_ROOT}" \
    --only "${CONCEPT_LIST[@]}" \
    "${locator_limit_args[@]}" \
    --taris_t_start 1000 \
    --taris_t_end 0 \
    --taris_num_steps 5 \
    --taris_top_k 10 \
    --taris_score_mode taris
}

run_batch() {
  local concept
  local max_prompt_args=()
  if [[ "${MAX_PROMPTS}" != "0" ]]; then
    max_prompt_args=(--max_prompts "${MAX_PROMPTS}")
  fi

  for concept in "${CONCEPT_LIST[@]}"; do
    python -m runtime.shared.batch \
      --ckpt_dir "${CKPT_DIR}" \
      --local_files_only \
      --sae_root "${SAE_ROOT}" \
      --prompts_path "batch_test_prompt/${concept}.csv" \
      --concepts "${concept}" \
      "${max_prompt_args[@]}" \
      --output_dir "${OUTPUT_ROOT}/batch_shared_concept_erase_${concept}"
  done
}

case "${MODE}" in
  locate)
    run_locator
    ;;
  batch)
    run_batch
    ;;
  quick)
    MAX_PROMPTS="${MAX_PROMPTS:-4}"
    if [[ "${MAX_PROMPTS}" == "0" ]]; then
      MAX_PROMPTS=4
    fi
    LOCATOR_MAX_PROMPTS_PER_SIDE="${LOCATOR_MAX_PROMPTS_PER_SIDE:-8}"
    if [[ "${LOCATOR_MAX_PROMPTS_PER_SIDE}" == "0" ]]; then
      LOCATOR_MAX_PROMPTS_PER_SIDE=8
    fi
    run_locator
    run_batch
    ;;
  all)
    run_locator
    run_batch
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use MODE=locate, MODE=batch, MODE=quick, or MODE=all." >&2
    exit 2
    ;;
esac
