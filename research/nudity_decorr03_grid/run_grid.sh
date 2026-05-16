#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CKPT="${CKPT:-train/output_time_latentdecorr_x8_top20_decorr03/checkpoints/stage3_step_0013772}"
PROMPTS_PATH="${PROMPTS_PATH:-batch_test_prompt/nudity.csv}"
STATS_DIR="${STATS_DIR:-feature_frequency/sae_x8_time_decorr03/coco30k}"
GRID_ROOT="${GRID_ROOT:-image_output/sae_x8_time_decorr03/nudity_grid}"
CONCEPT_ARCHIVE_ROOT="${CONCEPT_ARCHIVE_ROOT:-concept_dict_grid}"
WORK_CONCEPT_ROOT="${WORK_CONCEPT_ROOT:-concept_dict/sae_x8_time_decorr03}"
MAX_PROMPTS="${MAX_PROMPTS:-5}"
FROM_CASE="${FROM_CASE:-0}"
TILL_CASE="${TILL_CASE:-1000000}"
RUN_BASELINE="${RUN_BASELINE:-1}"

BLOCKS=(
  "unet.down_blocks.2.attentions.1"
  "unet.mid_block.attentions.0"
  "unet.up_blocks.0.attentions.0"
  "unet.up_blocks.0.attentions.1"
)
BLOCK_DIRS=(down.2.1 mid.0 up.0.0 up.0.1)

BLACKLIST_VARIANTS=(
  "q99_50:0.99:0.30:50"
  "ar95_all:0.0:0.95:0"
  "ar90_all:0.0:0.90:0"
)

TOP_KS=(10 15 20)
SCALES=(3000 5000)

if [[ "$RUN_BASELINE" == "1" ]]; then
  BASELINE_ARGS=()
else
  BASELINE_ARGS=(--no_baseline)
fi

BACKUP_DIR="$(mktemp -d)"
restore_concept_dict() {
  for block_dir in "${BLOCK_DIRS[@]}"; do
    if [[ -d "$BACKUP_DIR/$block_dir/nudity" ]]; then
      mkdir -p "$WORK_CONCEPT_ROOT/$block_dir"
      rm -rf "$WORK_CONCEPT_ROOT/$block_dir/nudity"
      cp -a "$BACKUP_DIR/$block_dir/nudity" "$WORK_CONCEPT_ROOT/$block_dir/nudity"
    fi
  done
  rm -rf "$BACKUP_DIR"
}
trap restore_concept_dict EXIT

for block_dir in "${BLOCK_DIRS[@]}"; do
  if [[ -d "$WORK_CONCEPT_ROOT/$block_dir/nudity" ]]; then
    mkdir -p "$BACKUP_DIR/$block_dir"
    cp -a "$WORK_CONCEPT_ROOT/$block_dir/nudity" "$BACKUP_DIR/$block_dir/nudity"
  fi
done

echo "[nudity-grid] checkpoint=$CKPT"
echo "[nudity-grid] prompts=$PROMPTS_PATH max_prompts=$MAX_PROMPTS"
echo "[nudity-grid] stats_dir=$STATS_DIR"
echo "[nudity-grid] grid_root=$GRID_ROOT"

if [[ ! -f "$STATS_DIR/down.2.1/dataset_feature_stats.pt" ]]; then
  echo "[nudity-grid] missing cached stats, collecting feature frequency first"
  python tools/feature_frequency/run_collect_shared_stats.py \
    --ckpt_dir "$CKPT" \
    --local_files_only \
    --prompts_path data/coco_30k.csv \
    --blocks "${BLOCKS[@]}" \
    --max_prompts 1000 \
    --steps 50 \
    --guidance_scale 8.0 \
    --resolution 512 \
    --taris_t_start 1000 \
    --taris_t_end 0 \
    --taris_num_steps 5 \
    --aggregate max \
    --feature_top_k 200 \
    --output_dir "$(dirname "$STATS_DIR")" \
    --run_name "$(basename "$STATS_DIR")"
else
  echo "[nudity-grid] reusing cached stats"
fi

mkdir -p "$GRID_ROOT" "$CONCEPT_ARCHIVE_ROOT"

for spec in "${BLACKLIST_VARIANTS[@]}"; do
  IFS=":" read -r variant quantile active_min max_features <<< "$spec"
  blacklist_root="concept_dict_freq/sae_x8_time_decorr03/${variant}"
  concept_root="$CONCEPT_ARCHIVE_ROOT/sae_x8_time_decorr03/nudity_${variant}"

  echo "[nudity-grid] build blacklist variant=$variant quantile=$quantile active_min=$active_min max_features=$max_features"
  python tools/feature_frequency/run_build_blacklist.py \
    --stats_dir "$STATS_DIR" \
    --feature_top_k 200 \
    --blacklist_freq_threshold "$quantile" \
    --blacklist_active_ratio_min "$active_min" \
    --blacklist_mean_min 0.0 \
    --blacklist_max_features "$max_features" \
    --concept_dict_freq_root "$blacklist_root"

  echo "[nudity-grid] locate nudity variant=$variant"
  python -m runtime.shared.locator \
    --ckpt_dir "$CKPT" \
    --local_files_only \
    --only nudity \
    --concept_output_root "$WORK_CONCEPT_ROOT" \
    --blocks "${BLOCKS[@]}" \
    --taris_t_start 1000 \
    --taris_t_end 0 \
    --taris_num_steps 5 \
    --taris_top_k 30 \
    --concept_dict_freq_root "$blacklist_root" \
    --taris_score_mode taris

  echo "[nudity-grid] archive concept_dict -> $concept_root"
  for block_dir in "${BLOCK_DIRS[@]}"; do
    mkdir -p "$concept_root/$block_dir"
    rm -rf "$concept_root/$block_dir/nudity"
    cp -a "$WORK_CONCEPT_ROOT/$block_dir/nudity" "$concept_root/$block_dir/nudity"
  done

  for top_k in "${TOP_KS[@]}"; do
    for scale in "${SCALES[@]}"; do
      out_dir="$GRID_ROOT/${variant}_top${top_k}_scale${scale}"
      echo "[nudity-grid] batch variant=$variant top_k=$top_k scale=$scale -> $out_dir"
      python -m runtime.shared.batch \
        --ckpt_dir "$CKPT" \
        --local_files_only \
        --prompts_path "$PROMPTS_PATH" \
        --concepts nudity \
        --concept_root "$concept_root" \
        --concept_dict_freq_root "$blacklist_root" \
        --output_dir "$out_dir" \
        --from_case "$FROM_CASE" \
        --till_case "$TILL_CASE" \
        --max_prompts "$MAX_PROMPTS" \
        --int_scale "$scale" \
        --int_feature_top_k "$top_k" \
        "${BASELINE_ARGS[@]}"
    done
  done
done

echo "[nudity-grid] done"
echo "[nudity-grid] outputs: $GRID_ROOT"
