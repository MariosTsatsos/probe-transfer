# Linear Probes Transfer from Base to Distilled Models despite Rotation

Code, result files and paper source for the study of whether a linear probe for harmful requests, trained on a base language model, still works on that base model's successors (the DeepSeek-R1 distilled model and the developer's instruction-tuned model), in four model families: Qwen2.5-Math-7B, Llama-3.1-8B, Qwen2.5-14B and Qwen2.5-32B.

Author: Marios Tsatsos (independent researcher). Preprint: arXiv (link to be added).

## Layout

- `code/session1/`: first compute session (base and distilled models on XSTest).
  - `extract.py`: pooled residual-stream activations (mean of the last 8 prompt tokens, raw prompts, no chat template, no BOS token), every layer, saved as `acts_<family>_<side>.npz`.
  - `analyze.py`: leave-one-group-out folds over XSTest's prompt types, inner layer choice, base, own and transferred probe AUC, writes `<family>_folds.csv`, `<family>_layers.csv`, `<family>_summary.json`.
  - `cosines.py`: safety-direction cosines in base-scaled, own-scaled and raw coordinates, per fold and per layer, and the fixed-layer check; writes `<family>_cosines_folds.csv`, `<family>_cosines_layers.csv`, `<family>_fixedlayer_folds.csv`.
  - `controls.py`, `controls_bs.py`: 38 control directions (35 prompt-group splits, question form, capitalised word, length); `controls_bs.py` is the base-scaled version used in the paper; writes `<family>_controls.csv`, `<family>_controls_bs.csv`.
  - `vocab.py`: exploratory vocabulary projection of the directions (Appendix B.3).
  - `run_colab.sh`, `run_extra.sh`, `run.sh`: driver scripts (`run.sh` is the non-Colab variant); `probe_transfer.ipynb`: the Colab notebook that ran them.
- `code/session2/`: second compute session (instruction-tuned models on XSTest; the independent set).
  - `make_external.py`: builds the independent set (OR-Bench hard-1k safe, OR-Bench toxic unsafe, 500 Alpaca instructions) from the Hugging Face datasets.
  - `extract_subset.py`: one model load, several extraction jobs (`--job csv:out:layers`).
  - `eval_external.py`: own and transferred probes on the independent set, text-only baselines, bootstrap intervals, flag rates at threshold 0 and at the XSTest 90%-recall threshold.
  - `run_extra2.sh`: resumable driver; `probe_session2.ipynb`: the Colab notebook that ran it.
- `results/session1/`, `results/session2/`: every result file the paper uses, as produced by the scripts above.
- `paper/`: LaTeX source (TMLR style), `gather.py` (reads `results/` and writes `numbers.json`, the single source of every number in the paper), `make_figures.py` and `make_schematic.py` (draw the figures from `numbers.json`), and the figures.

## Reproducing

Activations (needs a GPU; the 32B pair loads in 4-bit):

```
bash code/session1/run_colab.sh data/xstest_prompts.csv out/
bash code/session1/run_extra.sh data/xstest_prompts.csv out/
python code/session2/make_external.py --out data/external_prompts.csv
bash code/session2/run_extra2.sh data/xstest_prompts.csv data/external_prompts.csv out/
```

Numbers and figures from the result files (CPU, under a minute):

```
cd paper && python gather.py && python make_figures.py && python make_schematic.py && latexmk -pdf main.tex
```

`gather.py` needs `data/xstest_prompts.csv`, the official XSTest file (`xstest_prompts.csv` in https://github.com/paul-rottger/xstest), which is not redistributed here; the independent set is rebuilt by `make_external.py`.

Software: Python 3, PyTorch, transformers (5.20.1 as installed on Google Colab in October 2026), bitsandbytes, scikit-learn, scipy, pandas, datasets. Models are loaded from the Hugging Face Hub; `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct` are gated and need approved access.

## Licence

MIT (see `LICENSE`). The result files are released under the same terms. XSTest, OR-Bench and Alpaca are the property of their authors and are subject to their own licences.
