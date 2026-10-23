"""Vocabulary projection: which words does a direction point at?

Downloads only the weight shard(s) holding the final norm and the unembedding (lm_head) of each
model, not the whole model. Builds directions from the saved activations at the chosen layer,
multiplies by the final-norm weight and the unembedding, and lists the top tokens each way.
Directions: safety (z-scored diff of means, mapped back to raw space), the logistic-probe weight
vector, and the first type-split control. Each model's direction goes through its own unembedding.
Usage: python vocab.py --csv xstest_prompts.csv --base_id Qwen/Qwen2.5-Math-7B --arm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                       --base acts_math-7b_base.npz --arm acts_math-7b_arm.npz --name math-7b [--layer L] [--topk 25]
"""
import argparse, json, os
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True); p.add_argument("--base_id", required=True); p.add_argument("--arm_id", required=True)
p.add_argument("--base", required=True); p.add_argument("--arm", required=True); p.add_argument("--name", required=True)
p.add_argument("--layer", type=int, default=None); p.add_argument("--topk", type=int, default=25)
p.add_argument("--C", type=float, default=1.0)
a = p.parse_args()

df = pd.read_csv(a.csv)
y = (df["label"].str.lower() == "unsafe").astype(int).values
g = df["type"].str.replace(r"^contrast_", "", regex=True)
g = g.where(~g.str.contains("privacy"), "privacy").where(~g.str.contains("discr"), "discr").values
groups = sorted(set(g)); y_ctrl = np.isin(g, groups[:4]).astype(int)
y_how = (df["prompt"].str.strip().str.split().str[0].str.lower() == "how").astype(int).values
y_len = (df["prompt"].str.len() > df["prompt"].str.len().median()).astype(int).values
if a.layer is None:
    a.layer = int(np.median(pd.read_csv(f"{a.name}_folds.csv")["layer"]))

def load_head(repo):
    """Return (unembedding [V, d] float32, final-norm weight [d] float32), downloading only the needed shards."""
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN is not set: run the login cell (HF_TOKEN_READ) first, then rerun this cell.")
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo, token=tok)   # fails here, clearly, if the token or access is wrong
    if "model.safetensors.index.json" in files:
        idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json", token=tok)))["weight_map"]
    else:                                   # single-file model
        idx = None
    head_key = "lm_head.weight"
    if idx is not None and head_key not in idx:
        head_key = "model.embed_tokens.weight"   # tied embeddings
    norm_key = "model.norm.weight"
    out = {}
    for key in (head_key, norm_key):
        fname = idx[key] if idx is not None else "model.safetensors"
        path = hf_hub_download(repo, fname, token=tok)
        with safe_open(path, framework="pt") as f:
            out[key] = f.get_tensor(key).float().numpy()
    return out[head_key], out[norm_key]

def directions(X, yy, yc):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    dm = lambda lab: Z[lab == 1].mean(0) - Z[lab == 0].mean(0)
    lr = LogisticRegression(C=a.C, max_iter=5000).fit(Z, yy)
    # a direction in z-space corresponds to direction/sd in raw activation space
    return {"safety_diffmeans": dm(yy) / sd, "safety_probe": lr.coef_.ravel() / sd,
            "control_typesplit": dm(yc) / sd, "control_how": dm(y_how) / sd, "control_length": dm(y_len) / sd}

def project(W, gnorm, d, tokzr, k):
    v = d / (np.linalg.norm(d) + 1e-12)
    logits = W @ (gnorm * v)
    order = np.argsort(logits)
    top = [tokzr.decode([int(i)]) for i in order[::-1][:k]]
    bottom = [tokzr.decode([int(i)]) for i in order[:k]]
    return {"toward_unsafe": top, "toward_safe": bottom}

from transformers import AutoTokenizer
result = {"layer": a.layer}
for side, repo, path in (("base", a.base_id, a.base), ("arm", a.arm_id, a.arm)):
    X = np.load(path)["acts"][a.layer].astype(np.float32)
    W, gnorm = load_head(repo)
    tokzr = AutoTokenizer.from_pretrained(repo, token=os.environ.get("HF_TOKEN"))
    result[side] = {name: project(W, gnorm, d, tokzr, a.topk) for name, d in directions(X, y, y_ctrl).items()}
    print(f"\n=== {a.name} {side} ({repo}), layer {a.layer} ===")
    for name, r in result[side].items():
        print(f"{name:20s} unsafe: {r['toward_unsafe'][:15]}")
        print(f"{'':20s} safe:   {r['toward_safe'][:15]}")
json.dump(result, open(f"{a.name}_vocab.json", "w"), indent=1, ensure_ascii=False)
