#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 3 ]]; then
  echo 'Usage: bash scripts/compare.sh <reference.mp4> <candidate.mp4> <new-output-directory>' >&2
  exit 2
fi
: "${PYTHON:?Set the scoring environment interpreter}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source="${INFINILM_SOURCE:-$repo/sources/InfiniLM}"
if [[ -e "$3" ]]; then echo 'Comparison output already exists.' >&2; exit 2; fi
mkdir -p "$3"
"$PYTHON" -I -u "$source/examples/minimax_h3_compare.py" --model-path "${MODEL_PATH:-$repo}" --tasks fl2va \
  --reference-fl2va-video "$1" --infinilm-fl2va-video "$2" --output-dir "$3" 2>&1 | tee "$3/compare.log"
