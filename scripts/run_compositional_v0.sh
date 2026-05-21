#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

CKPT_DIR="${CKPT_DIR:-train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772}"
SAE_ROOT="${SAE_ROOT:-sae_data/sae_x8_time_decorr03}"
OUT_ROOT="${OUT_ROOT:-image_output/sae_x8_time_decorr03/compositional_v0}"
MAX_PROMPTS="${MAX_PROMPTS:-5}"
INT_SCALE="${INT_SCALE:-5000}"
INT_TOP_K="${INT_TOP_K:-5}"
TIMESTEP_WINDOW_START="${TIMESTEP_WINDOW_START:-1000}"
TIMESTEP_WINDOW_END="${TIMESTEP_WINDOW_END:-300}"
MAX_DELTA_OVER_X="${MAX_DELTA_OVER_X:-0.2}"

COMPOSITES=("dog_glasses" "red_car" "flower_van_gogh")
ATOMIC_A=("dog" "red" "flower")
ATOMIC_B=("glasses" "car" "van_gogh")
LOCATE_CONCEPTS=("dog_glasses" "red_car" "flower_van_gogh" "dog" "red" "flower" "glasses" "car" "van_gogh")

BLOCK_SHORTS=("down.2.1" "mid.0" "up.0.0" "up.0.1")
NEED_LOCATE=()
for concept in "${LOCATE_CONCEPTS[@]}"; do
  missing=0
  for block in "${BLOCK_SHORTS[@]}"; do
    if [[ ! -s "${SAE_ROOT}/concept-dig/${block}/${concept}/top_positive_features.csv" ]]; then
      missing=1
      break
    fi
  done
  if [[ "${missing}" -eq 1 ]]; then
    NEED_LOCATE+=("${concept}")
  else
    echo "[compositional-v0] skip locator, found cached concept=${concept}"
  fi
done

echo "[compositional-v0] ckpt=${CKPT_DIR}"
echo "[compositional-v0] sae_root=${SAE_ROOT}"
echo "[compositional-v0] out=${OUT_ROOT}"
echo "[compositional-v0] max_prompts=${MAX_PROMPTS} top_k=${INT_TOP_K} timestep=${TIMESTEP_WINDOW_START}:${TIMESTEP_WINDOW_END}"

if [[ "${#NEED_LOCATE[@]}" -gt 0 ]]; then
  echo "[compositional-v0] locate missing concepts: ${NEED_LOCATE[*]}"
  python -m runtime.shared.locator \
    --ckpt_dir "${CKPT_DIR}" \
    --local_files_only \
    --sae_root "${SAE_ROOT}" \
    --only "${NEED_LOCATE[@]}" \
    --taris_t_start 1000 \
    --taris_t_end 0 \
    --taris_num_steps 5 \
    --taris_top_k 10 \
    --taris_score_mode taris
else
  echo "[compositional-v0] all locator outputs already exist; skip locator"
fi

python scripts/analyze_composition_overlap.py \
  --sae_root "${SAE_ROOT}" \
  --top_k 20 \
  --out_csv "${OUT_ROOT}/feature_overlap.csv"

for i in "${!COMPOSITES[@]}"; do
  comp="${COMPOSITES[$i]}"
  a="${ATOMIC_A[$i]}"
  b="${ATOMIC_B[$i]}"

  echo "[compositional-v0] target suppression: erase ${comp} on ${comp} prompts"
  python -m runtime.shared.batch \
    --ckpt_dir "${CKPT_DIR}" \
    --local_files_only \
    --sae_root "${SAE_ROOT}" \
    --prompts_path "batch_test_prompt/${comp}.csv" \
    --concepts "${comp}" \
    --max_prompts "${MAX_PROMPTS}" \
    --int_scale "${INT_SCALE}" \
    --int_inject_scale "${INT_SCALE}" \
    --int_feature_top_k "${INT_TOP_K}" \
    --int_use_stat_time_weight \
    --no-int_use_learned_time_weight \
    --int_time_fuse_mode stat_only \
    --int_max_delta_over_x "${MAX_DELTA_OVER_X}" \
    --int_timestep_window "${TIMESTEP_WINDOW_START}" "${TIMESTEP_WINDOW_END}" \
    --output_dir "${OUT_ROOT}/${comp}/target_${comp}"

  echo "[compositional-v0] preservation A: erase ${comp} on ${a} prompts"
  python -m runtime.shared.batch \
    --ckpt_dir "${CKPT_DIR}" \
    --local_files_only \
    --sae_root "${SAE_ROOT}" \
    --prompts_path "batch_test_prompt/${a}.csv" \
    --concepts "${comp}" \
    --max_prompts "${MAX_PROMPTS}" \
    --int_scale "${INT_SCALE}" \
    --int_inject_scale "${INT_SCALE}" \
    --int_feature_top_k "${INT_TOP_K}" \
    --int_use_stat_time_weight \
    --no-int_use_learned_time_weight \
    --int_time_fuse_mode stat_only \
    --int_max_delta_over_x "${MAX_DELTA_OVER_X}" \
    --int_timestep_window "${TIMESTEP_WINDOW_START}" "${TIMESTEP_WINDOW_END}" \
    --output_dir "${OUT_ROOT}/${comp}/preserve_${a}"

  echo "[compositional-v0] preservation B: erase ${comp} on ${b} prompts"
  python -m runtime.shared.batch \
    --ckpt_dir "${CKPT_DIR}" \
    --local_files_only \
    --sae_root "${SAE_ROOT}" \
    --prompts_path "batch_test_prompt/${b}.csv" \
    --concepts "${comp}" \
    --max_prompts "${MAX_PROMPTS}" \
    --int_scale "${INT_SCALE}" \
    --int_inject_scale "${INT_SCALE}" \
    --int_feature_top_k "${INT_TOP_K}" \
    --int_use_stat_time_weight \
    --no-int_use_learned_time_weight \
    --int_time_fuse_mode stat_only \
    --int_max_delta_over_x "${MAX_DELTA_OVER_X}" \
    --int_timestep_window "${TIMESTEP_WINDOW_START}" "${TIMESTEP_WINDOW_END}" \
    --output_dir "${OUT_ROOT}/${comp}/preserve_${b}"

  echo "[compositional-v0] atomic baseline: erase ${a} on ${comp} prompts"
  python -m runtime.shared.batch \
    --ckpt_dir "${CKPT_DIR}" \
    --local_files_only \
    --sae_root "${SAE_ROOT}" \
    --prompts_path "batch_test_prompt/${comp}.csv" \
    --concepts "${a}" \
    --max_prompts "${MAX_PROMPTS}" \
    --int_scale "${INT_SCALE}" \
    --int_inject_scale "${INT_SCALE}" \
    --int_feature_top_k "${INT_TOP_K}" \
    --int_use_stat_time_weight \
    --no-int_use_learned_time_weight \
    --int_time_fuse_mode stat_only \
    --int_max_delta_over_x "${MAX_DELTA_OVER_X}" \
    --int_timestep_window "${TIMESTEP_WINDOW_START}" "${TIMESTEP_WINDOW_END}" \
    --output_dir "${OUT_ROOT}/${comp}/atomic_${a}_on_${comp}"

  echo "[compositional-v0] atomic baseline: erase ${b} on ${comp} prompts"
  python -m runtime.shared.batch \
    --ckpt_dir "${CKPT_DIR}" \
    --local_files_only \
    --sae_root "${SAE_ROOT}" \
    --prompts_path "batch_test_prompt/${comp}.csv" \
    --concepts "${b}" \
    --max_prompts "${MAX_PROMPTS}" \
    --int_scale "${INT_SCALE}" \
    --int_inject_scale "${INT_SCALE}" \
    --int_feature_top_k "${INT_TOP_K}" \
    --int_use_stat_time_weight \
    --no-int_use_learned_time_weight \
    --int_time_fuse_mode stat_only \
    --int_max_delta_over_x "${MAX_DELTA_OVER_X}" \
    --int_timestep_window "${TIMESTEP_WINDOW_START}" "${TIMESTEP_WINDOW_END}" \
    --output_dir "${OUT_ROOT}/${comp}/atomic_${b}_on_${comp}"
done

echo "[compositional-v0] done"
