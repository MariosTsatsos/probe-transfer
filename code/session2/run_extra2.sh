#!/usr/bin/env bash
# Second compute session. Per family, three models are loaded once each:
#   base      -> ext_<f>_base.npz   (independent set, chosen layers)
#   distilled -> ext_<f>_arm.npz    (independent set, chosen layers)
#   Instruct  -> acts_<f>_inst.npz  (XSTest, all layers; same layout as acts_<f>_base.npz)
#             -> ext_<f>_inst.npz   (independent set, chosen layers)
# Usage: bash run_extra2.sh <xstest csv> <external csv> <output dir>
# Resumable: a model is skipped when all of its output files exist. Nothing existing is overwritten.
set -e
S="$(cd "$(dirname "$0")" && pwd)"; XS=$1; EXT=$2; OUT=$3; mkdir -p "$OUT"; cd "$OUT"
declare -A BASE=( [math-7b]=Qwen/Qwen2.5-Math-7B [llama-8b]=meta-llama/Llama-3.1-8B
                  [qwen-14b]=Qwen/Qwen2.5-14B [qwen-32b]=Qwen/Qwen2.5-32B )
declare -A ARM=(  [math-7b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B [llama-8b]=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
                  [qwen-14b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B [qwen-32b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B )
declare -A INST=( [math-7b]=Qwen/Qwen2.5-Math-7B-Instruct [llama-8b]=meta-llama/Llama-3.1-8B-Instruct
                  [qwen-14b]=Qwen/Qwen2.5-14B-Instruct [qwen-32b]=Qwen/Qwen2.5-32B-Instruct )
# Layers saved for the independent set: the paper's fixed layer plus every layer chosen in any XSTest fold.
declare -A LAY=( [math-7b]=17,18,19,27 [llama-8b]=12,13,17,18,21,23 [qwen-14b]=27,31,40,42,45 [qwen-32b]=29,43,47 )
[ -n "$LLAMA_BASE" ] && BASE[llama-8b]=$LLAMA_BASE      # ungated mirrors, if access to the gated repos fails
[ -n "$LLAMA_INST" ] && INST[llama-8b]=$LLAMA_INST

run () {  # run <model> <job>...   (clears the model cache before and after; Colab disk is small)
  rm -rf ~/.cache/huggingface
  python "$S/extract_subset.py" --model "$@"
  rm -rf ~/.cache/huggingface
}

for f in math-7b qwen-14b qwen-32b llama-8b; do
  Q=""; [ "$f" = qwen-32b ] && Q="--bits4"
  [ -f ext_${f}_base.npz ] || run ${BASE[$f]} --job "$EXT:ext_${f}_base.npz:${LAY[$f]}" $Q
  [ -f ext_${f}_arm.npz ]  || run ${ARM[$f]}  --job "$EXT:ext_${f}_arm.npz:${LAY[$f]}" $Q
  J=()   # jobs for the Instruct model (paths may contain spaces, so an array is used)
  [ -f acts_${f}_inst.npz ] || J+=(--job "$XS:acts_${f}_inst.npz:all")
  [ -f ext_${f}_inst.npz ]  || J+=(--job "$EXT:ext_${f}_inst.npz:${LAY[$f]}")
  [ ${#J[@]} -eq 0 ] || run ${INST[$f]} "${J[@]}" $Q
done
echo "all extractions done"
