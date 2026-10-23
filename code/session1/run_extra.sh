#!/usr/bin/env bash
# Second analysis pass on the saved activations: many control directions + vocabulary projection.
# Usage: bash run_extra.sh <csv> <output dir>    (run after run_colab.sh; needs HF_TOKEN in the environment)
set -e
S="$(cd "$(dirname "$0")" && pwd)"; CSV=$1; OUT=$2; cd "$OUT"
declare -A BASE=( [math-7b]=Qwen/Qwen2.5-Math-7B [llama-8b]=meta-llama/Llama-3.1-8B
                  [qwen-14b]=Qwen/Qwen2.5-14B [qwen-32b]=Qwen/Qwen2.5-32B )
declare -A ARM=(  [math-7b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B [llama-8b]=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
                  [qwen-14b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B [qwen-32b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B )
for f in math-7b llama-8b qwen-14b qwen-32b; do
  [ -f acts_${f}_arm.npz ] || { echo "skip $f (no activations yet)"; continue; }
  python "$S/controls.py" --csv "$CSV" --base acts_${f}_base.npz --arm acts_${f}_arm.npz --name $f | tee ${f}_controls_log.txt
  rm -rf ~/.cache/huggingface/hub/models--*
  python "$S/vocab.py" --csv "$CSV" --base_id ${BASE[$f]} --arm_id ${ARM[$f]} \
         --base acts_${f}_base.npz --arm acts_${f}_arm.npz --name $f | tee ${f}_vocab_log.txt
  rm -rf ~/.cache/huggingface/hub/models--*
done
