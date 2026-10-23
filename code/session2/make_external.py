"""Build the independent evaluation set (never used for training).

Sources (downloaded from the HuggingFace Hub):
  bench-llms/or-bench, config or-bench-hard-1k : seemingly toxic but benign prompts (Cui et al., 2024)  -> label safe
  bench-llms/or-bench, config or-bench-toxic   : toxic prompts from the same benchmark                   -> label unsafe
  tatsu-lab/alpaca                             : ordinary instructions, no safety content, 500 sampled  -> label safe
Output: external_prompts.csv with columns prompt, source, category, label.
Usage: python make_external.py --out /path/external_prompts.csv [--n_alpaca 500] [--seed 0]
"""
import argparse
import pandas as pd
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--n_alpaca", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

hard = load_dataset("bench-llms/or-bench", "or-bench-hard-1k")["train"].to_pandas()
toxic = load_dataset("bench-llms/or-bench", "or-bench-toxic")["train"].to_pandas()
alp = load_dataset("tatsu-lab/alpaca")["train"].to_pandas()

hard = pd.DataFrame(dict(prompt=hard["prompt"], source="orbench_hard", category=hard["category"], label="safe"))
toxic = pd.DataFrame(dict(prompt=toxic["prompt"], source="orbench_toxic", category=toxic["category"], label="unsafe"))
alp = alp[(alp["input"].fillna("").str.strip() == "") & alp["instruction"].str.len().between(20, 300)]
alp = alp.sample(n=a.n_alpaca, random_state=a.seed)
alp = pd.DataFrame(dict(prompt=alp["instruction"], source="alpaca", category="none", label="safe"))

df = pd.concat([hard, toxic, alp], ignore_index=True)
df["prompt"] = df["prompt"].astype(str).str.strip()
df = df[df["prompt"].str.len() > 0].drop_duplicates("prompt").reset_index(drop=True)
df.to_csv(a.out, index=False)
print(df.groupby(["source", "label"]).size())
print("written", a.out, len(df), "prompts")
