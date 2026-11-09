#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 2 ]]; then
  echo 'Usage: bash scripts/run.sh <pytorch|legacy|infini> <new-output-directory>' >&2
  exit 2
fi
mode="$1"
case "$mode" in pytorch|legacy|infini) ;; *) echo "Unknown mode: $mode" >&2; exit 2 ;; esac
: "${MODEL_PATH:?Set MODEL_PATH to the official local Diffusers weights}"
: "${FIRST_FRAME:?Set FIRST_FRAME to the official Golde33443.jpg}"
: "${PYTHON:?Set PYTHON to the absolute interpreter path in the selected environment}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export INFINILM_SOURCE="${INFINILM_SOURCE:-$repo/sources/InfiniLM}"
if [[ -e "$2" ]]; then echo 'Use a new output directory to preserve prior results.' >&2; exit 2; fi
mkdir -p "$2"
out="$(cd "$2" && pwd)"
export MODEL_PATH FIRST_FRAME OUTPUT_VIDEO="$out/generated.mp4"
"$PYTHON" -I "$repo/scripts/check_environment.py" "$mode" > "$out/environment.json"
if [[ "$mode" == pytorch ]]; then
  "$PYTHON" -I -u "$repo/references/pytorch_fl2va_group_offload.py" 2>&1 | tee "$out/run.log"
else
  args=(fl2va --model-path "$MODEL_PATH" --text 'A dog falls asleep'
    --first-frame "$FIRST_FRAME" --gpus 0,1,2,3,4,5,6,7
    --num-frames 124 --steps 10 --seed 42 --height 384 --width 576 --output "$OUTPUT_VIDEO")
  if [[ "$mode" == legacy ]]; then
    "$PYTHON" -I -u "$INFINILM_SOURCE/examples/minimax_h3.py" "${args[@]}" \
      --legacy --legacy-text-gpus 0,1,2,3 --legacy-transformer-gpus 4,5,6,7 2>&1 | tee "$out/run.log"
  else
    "$PYTHON" -I -u "$repo/scripts/run_h3_bsa.py" "${args[@]}" 2>&1 | tee "$out/run.log"
  fi
fi
sha256sum "$OUTPUT_VIDEO" | tee "$out/video.sha256"
