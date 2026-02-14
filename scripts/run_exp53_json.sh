#!/usr/bin/env bash
set -euo pipefail

# 最简单：bash + jq 解析 JSON，然后跑 exp53（TARIS）。
#
# 依赖：jq
# 用法：
#   bash scripts/run_exp53_json.sh scripts/targetconcept/exp53_red_vs_blue.json
#
# JSON 支持两种顶层：
# 1) 单概念 object：
#    { "concept_name": "red_vs_blue", "pos_prompts": [...], "neg_prompts": [...] }
# 2) 多概念 array：
#    [ { ... }, { ... } ]
#
# 可选字段（不写就用默认/环境变量）：
# - concept_name, loc_block, taris_t_start, taris_t_end, taris_num_steps, taris_delta, taris_top_k

if ! command -v jq >/dev/null 2>&1; then
  echo "missing dependency: jq" >&2
  echo "install jq first, then retry." >&2
  exit 1
fi

json_path="${1:-}"
if [[ -z "$json_path" ]]; then
  echo "usage: bash scripts/run_exp53_json.sh <concept.json>" >&2
  exit 1
fi
if [[ ! -f "$json_path" ]]; then
  echo "json not found: $json_path" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
json_path="$(realpath -m "$json_path")"

# 运行参数：优先环境变量，没设就用默认
OUTPUT_ROOT="${OUTPUT_ROOT:-./out_taris_json}"
SDXL_UNBOX_ROOT="${SDXL_UNBOX_ROOT:-~/sdxl-unbox}"
SAE_ROOT="${SAE_ROOT:-~/sdxl-saes}"
MODEL_ID="${MODEL_ID:-~/datasets/sd-xl/sdxl_diffusers_fp16}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
PREFER_K="${PREFER_K:-5}"
PREFER_HIDDEN="${PREFER_HIDDEN:-5120}"
STEPS="${STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-8.0}"
SEED="${SEED:-42}"
LOC_BLOCK_DEFAULT="${LOC_BLOCK:-unet.mid_block.attentions.0}"
TARIS_T_START_DEFAULT="${TARIS_T_START:-800}"
TARIS_T_END_DEFAULT="${TARIS_T_END:-200}"
TARIS_NUM_STEPS_DEFAULT="${TARIS_NUM_STEPS:-10}"
TARIS_DELTA_DEFAULT="${TARIS_DELTA:-1e-6}"
TARIS_TOP_K_DEFAULT="${TARIS_TOP_K:-50}"

mkdir -p "$OUTPUT_ROOT"

json_stem="$(basename "$json_path")"
json_stem="${json_stem%.json}"

is_array="$(jq -r 'if type=="array" then "yes" else "no" end' "$json_path")"
if [[ "$is_array" == "yes" ]]; then
  n="$(jq 'length' "$json_path")"
  indices="$(seq 0 $((n-1)))"
else
  indices="single"
fi

run_one() {
  local selector="$1" # "single" or numeric index
  local q=". "
  if [[ "$selector" != "single" ]]; then
    q=".[$selector]"
  fi

  local concept_name
  concept_name="$(jq -r "$q | (.concept_name // \"\")" "$json_path" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  if [[ -z "$concept_name" ]]; then
    if [[ "$selector" == "single" ]]; then
      concept_name="$json_stem"
    else
      concept_name="${json_stem}_$(printf '%02d' "$selector")"
    fi
  fi

  local loc_block taris_t_start taris_t_end taris_num_steps taris_delta taris_top_k
  loc_block="$(jq -r "$q | (.loc_block // \"\")" "$json_path")"
  [[ -z "$loc_block" || "$loc_block" == "null" ]] && loc_block="$LOC_BLOCK_DEFAULT"

  taris_t_start="$(jq -r "$q | (.taris_t_start // \"\")" "$json_path")"
  [[ -z "$taris_t_start" || "$taris_t_start" == "null" ]] && taris_t_start="$TARIS_T_START_DEFAULT"

  taris_t_end="$(jq -r "$q | (.taris_t_end // \"\")" "$json_path")"
  [[ -z "$taris_t_end" || "$taris_t_end" == "null" ]] && taris_t_end="$TARIS_T_END_DEFAULT"

  taris_num_steps="$(jq -r "$q | (.taris_num_steps // \"\")" "$json_path")"
  [[ -z "$taris_num_steps" || "$taris_num_steps" == "null" ]] && taris_num_steps="$TARIS_NUM_STEPS_DEFAULT"

  taris_delta="$(jq -r "$q | (.taris_delta // \"\")" "$json_path")"
  [[ -z "$taris_delta" || "$taris_delta" == "null" ]] && taris_delta="$TARIS_DELTA_DEFAULT"

  taris_top_k="$(jq -r "$q | (.taris_top_k // \"\")" "$json_path")"
  [[ -z "$taris_top_k" || "$taris_top_k" == "null" ]] && taris_top_k="$TARIS_TOP_K_DEFAULT"

  local -a pos_prompts=()
  local -a neg_prompts=()
  mapfile -t pos_prompts < <(jq -r "$q | .pos_prompts[]?" "$json_path")
  mapfile -t neg_prompts < <(jq -r "$q | .neg_prompts[]?" "$json_path")

  if [[ ${#pos_prompts[@]} -lt 1 || ${#neg_prompts[@]} -lt 1 ]]; then
    echo "[skip] $concept_name (missing pos_prompts/neg_prompts)" >&2
    return 0
  fi

  local out_dir="$OUTPUT_ROOT/$concept_name"
  mkdir -p "$out_dir"

  echo "[run] concept=$concept_name pos=${#pos_prompts[@]} neg=${#neg_prompts[@]} out=$out_dir"
  (
    set -x
    cd "$ROOT_DIR"
    python scripts/vslz_wsae_res_sdxl.py \
      --experiment exp53 \
      --output_dir "$out_dir" \
      --sdxl_unbox_root "$SDXL_UNBOX_ROOT" \
      --sae_root "$SAE_ROOT" \
      --model_id "$MODEL_ID" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --prefer_k "$PREFER_K" \
      --prefer_hidden "$PREFER_HIDDEN" \
      --steps "$STEPS" \
      --guidance_scale "$GUIDANCE_SCALE" \
      --seed "$SEED" \
      --loc_block "$loc_block" \
      --concept_name "$concept_name" \
      --pos_prompts "${pos_prompts[@]}" \
      --neg_prompts "${neg_prompts[@]}" \
      --taris_t_start "$taris_t_start" \
      --taris_t_end "$taris_t_end" \
      --taris_num_steps "$taris_num_steps" \
      --taris_delta "$taris_delta" \
      --taris_top_k "$taris_top_k"
  )
}

for idx in $indices; do
  run_one "$idx"
done

