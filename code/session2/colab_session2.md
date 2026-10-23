# Second compute session: what to upload and what to run

Upload to `notebooks` (same folder as extract.py): `extract_subset.py`, `make_external.py`, `run_extra2.sh`, `eval_external.py`.
Nothing existing is overwritten; all new files have new names.

Runtime: A100. Run the existing mount, pip-install and HF-login cells of probe_transfer.ipynb first, then the three cells below.

## Cell 1: build the independent prompt set (CPU, about a minute)

```python
D = "/content/drive/MyDrive/Work/Mech Interp/BlueDot Technical Project - P4/notebooks"
O = D + "/output_analysis"
!python "$D/make_external.py" --out "$O/external_prompts.csv"
```

Expected: about 1,319 orbench_hard (safe), 655 orbench_toxic (unsafe), 500 alpaca (safe), minus any duplicates.

## Cell 2: extractions (GPU; the long one)

```python
!bash "$D/run_extra2.sh" "$O/xstest_prompts.csv" "$O/external_prompts.csv" "$D"
```

Order: math-7b, qwen-14b, qwen-32b, llama-8b. Twelve model downloads (base, distilled, Instruct per family), cache cleared between models as before. New files in `notebooks`: `ext_<f>_base.npz`, `ext_<f>_arm.npz`, `ext_<f>_inst.npz` (independent set, 3 to 6 layers each) and `acts_<f>_inst.npz` (XSTest, all layers). Total new Drive usage about 1.9 GB (0.7 GB for the four Instruct XSTest files, 1.2 GB for the twelve independent-set files); check you have that free before starting. If the session drops, rerun the same cell: finished files are skipped.

If `meta-llama/Llama-3.1-8B-Instruct` returns 403, your Llama access does not cover the Instruct repo; request it on its model page, or rerun with an ungated mirror:
`!LLAMA_INST=unsloth/Llama-3.1-8B-Instruct bash "$D/run_extra2.sh" ...`

## Cell 3: analysis (CPU, a few minutes)

```python
for f, L in [("math-7b", 18), ("llama-8b", 17), ("qwen-14b", 40), ("qwen-32b", 43)]:
    # Instruct counterpart on XSTest: same analysis as the distilled arm (transfer, ceiling, cosines, fixed layer)
    !python "$D/analyze.py" --csv "$O/xstest_prompts.csv" --base "$D/acts_{f}_base.npz" --arm "$D/acts_{f}_inst.npz" --name "$O/{f}_inst"
    !python "$D/cosines.py" --csv "$O/xstest_prompts.csv" --base "$D/acts_{f}_base.npz" --arm "$D/acts_{f}_inst.npz" --name "$O/{f}_inst"
    # independent set: XSTest-trained probes, own and transferred, plus text baselines
    !python "$D/eval_external.py" --xs "$O/xstest_prompts.csv" --ext "$O/external_prompts.csv" --layer {L} --name "$O/{f}" \
        --xs_base "$D/acts_{f}_base.npz" --xs_arm "$D/acts_{f}_arm.npz" --xs_inst "$D/acts_{f}_inst.npz" \
        --ext_base "$D/ext_{f}_base.npz" --ext_arm "$D/ext_{f}_arm.npz" --ext_inst "$D/ext_{f}_inst.npz"
```

Send me, from `output_analysis`: `<f>_inst_folds.csv`, `<f>_inst_summary.json`, `<f>_inst_layers.csv`, `<f>_inst_cosines_summary.json`, `<f>_inst_fixedlayer_folds.csv`, `<f>_external_table.csv`, `<f>_external.json` for all four families, plus the printed output of Cell 3.
