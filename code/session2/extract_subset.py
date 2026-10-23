"""Extract pooled residual-stream activations for one model on one or more prompt files, loading the model once.

Same pooling as extract.py (raw prompts, no chat template, no BOS, mean of the last K prompt tokens).
Each job is  <csv>:<out.npz>:<layers>  where <layers> is "all" or a comma-separated list of 0-based layer
indices (index j = entry j+1 of the HuggingFace hidden_states tuple, as in extract.py).
Output npz:  acts float16 [n_layers, N, d], layers int64 [n_layers], tokh int64 [N].
With layers=all the file has the same layout as extract.py's output (plus the 'layers' array).
Usage: python extract_subset.py --model M --job xstest.csv:acts_x.npz:all --job ext.csv:ext_x.npz:17,18,19 [--bits4]
"""
import argparse, hashlib
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--job", action="append", required=True)   # csv:out:layers, repeatable
p.add_argument("--bits4", action="store_true")
p.add_argument("--last_k", type=int, default=8)
p.add_argument("--bs", type=int, default=16)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
if a.bits4:
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
n_layers_total = model.config.num_hidden_layers

for job in a.job:
    csv, out, lay = job.split(":")
    layers = list(range(n_layers_total)) if lay == "all" else [int(x) for x in lay.split(",")]
    prompts = pd.read_csv(csv)["prompt"].astype(str).tolist()
    tokh = np.array([int(hashlib.md5(str(tok(t, add_special_tokens=False).input_ids).encode()).hexdigest()[:15], 16)
                     for t in prompts], dtype=np.int64)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(prompts), a.bs):
            enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            hs = model(**enc, output_hidden_states=True).hidden_states   # (L+1) x [B, T, d]
            H = torch.stack([hs[l + 1] for l in layers], 0)              # [n_layers, B, T, d]
            k = enc["attention_mask"].sum(1).clamp(max=a.last_k)         # left padding: real tokens are last
            pooled = [H[:, b, -int(k[b]):, :].float().mean(1) for b in range(H.shape[1])]
            chunks.append(torch.stack(pooled, 1).cpu().half())
            if (i // a.bs) % 10 == 0:
                print(f"{a.model} [{csv}]: {min(i + a.bs, len(prompts))}/{len(prompts)}", flush=True)
    acts = torch.cat(chunks, 1).numpy()
    np.savez(out, acts=acts, layers=np.array(layers, dtype=np.int64), tokh=tokh)
    print("saved", out, acts.shape, "layers", "all" if lay == "all" else layers, flush=True)
