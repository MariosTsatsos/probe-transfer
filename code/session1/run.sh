#!/usr/bin/env bash
# Usage: bash run.sh /path/to/xstest_prompts.csv
set -e
CSV=$1
declare -A BASE=( [math-7b]=Qwen/Qwen2.5-Math-7B [llama-8b]=meta-llama/Llama-3.1-8B
                  [qwen-14b]=Qwen/Qwen2.5-14B [qwen-32b]=Qwen/Qwen2.5-32B )
declare -A ARM=(  [math-7b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B [llama-8b]=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
                  [qwen-14b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B [qwen-32b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B )
for f in math-7b llama-8b qwen-14b qwen-32b; do
  Q=""; [ "$f" = qwen-32b ] && Q="--bits4"
  [ -f acts_${f}_base.npz ] || python extract.py --model ${BASE[$f]} --csv $CSV --out acts_${f}_base.npz $Q
  [ -f acts_${f}_arm.npz ]  || python extract.py --model ${ARM[$f]}  --csv $CSV --out acts_${f}_arm.npz  $Q
  python analyze.py --csv $CSV --base acts_${f}_base.npz --arm acts_${f}_arm.npz --name $f | tee ${f}_log.txt
done
