"""Extract pooled residual-stream activations for XSTest prompts, all layers.

Output: <out>.npz with
  acts  float16 [L, N, d]   mean of the last K prompt tokens, every layer (embeddings excluded)
  tokh  int64   [N]         hash of each prompt's token ids (to verify identical tokenisation)
"""
import argparse, hashlib
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--csv", required=True)          # xstest_prompts.csv (columns: prompt, type, label)
p.add_argument("--out", required=True)
p.add_argument("--bits4", action="store_true")  # use for the 32B pair (both models)
p.add_argument("--last_k", type=int, default=8)
p.add_argument("--bs", type=int, default=16)
a = p.parse_args()

prompts = pd.read_csv(a.csv)["prompt"].tolist()
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
if a.bits4:
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()

# Raw prompts, no chat template, no BOS: identical text tokens across base and distilled models.
tokh = np.array([int(hashlib.md5(str(tok(t, add_special_tokens=False).input_ids).encode()).hexdigest()[:15], 16)
                 for t in prompts], dtype=np.int64)

chunks = []
with torch.no_grad():
    for i in range(0, len(prompts), a.bs):
        enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states   # (L+1) x [B, T, d]
        H = torch.stack(hs[1:], 0)                                   # [L, B, T, d]
        k = enc["attention_mask"].sum(1).clamp(max=a.last_k)         # left padding: real tokens are last
        pooled = [H[:, b, -int(k[b]):, :].float().mean(1) for b in range(H.shape[1])]
        chunks.append(torch.stack(pooled, 1).cpu().half())
        print(f"{a.model}: {min(i + a.bs, len(prompts))}/{len(prompts)}", flush=True)

acts = torch.cat(chunks, 1).numpy()
np.savez(a.out, acts=acts, tokh=tokh)
print("saved", a.out, acts.shape)
