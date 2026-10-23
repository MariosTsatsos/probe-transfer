#!/usr/bin/env bash
# Colab version: outputs go to Drive, model downloads are deleted after each extraction (Colab disk is ~80 GB).
# Usage: bash run_colab.sh <csv> <output dir>; the scripts are read from the folder this file is in
set -e
S="$(cd "$(dirname "$0")" && pwd)"; CSV=$1; OUT=$2; mkdir -p "$OUT"; cd "$OUT"
declare -A BASE=( [math-7b]=Qwen/Qwen2.5-Math-7B [llama-8b]=meta-llama/Llama-3.1-8B
                  [qwen-14b]=Qwen/Qwen2.5-14B [qwen-32b]=Qwen/Qwen2.5-32B )
declare -A ARM=(  [math-7b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B [llama-8b]=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
                  [qwen-14b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B [qwen-32b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B )
# Llama runs last: its base model is gated and needs approved access on huggingface.co.
# If approval does not come, set LLAMA_BASE to an ungated mirror of the same weights, e.g.
#   LLAMA_BASE=unsloth/Llama-3.1-8B bash run_colab.sh ...
[ -n "$LLAMA_BASE" ] && BASE[llama-8b]=$LLAMA_BASE
for f in math-7b qwen-14b qwen-32b llama-8b; do
  # A100 40 GB: everything in bfloat16 except the 32B pair (65 GB in bfloat16), which loads in 4-bit.
  Q=""; [ "$f" = qwen-32b ] && Q="--bits4"
  for side in base arm; do
    [ $side = base ] && M=${BASE[$f]} || M=${ARM[$f]}
    if [ ! -f acts_${f}_${side}.npz ]; then
      rm -rf ~/.cache/huggingface                    # clear cache, including partial downloads
      python "$S/extract.py" --model $M --csv "$CSV" --out acts_${f}_${side}.npz $Q
      rm -rf ~/.cache/huggingface
    fi
  done
  python "$S/analyze.py" --csv "$CSV" --base acts_${f}_base.npz --arm acts_${f}_arm.npz --name $f | tee ${f}_log.txt
done
