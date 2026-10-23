"""Collect every number used in the paper's figures and tables from the result files; write numbers.json.
Run from the paper/ directory. Needs data/xstest_prompts.csv (see README)."""
import os
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
U=os.path.join(ROOT,"results","session1")+"/"
Z=os.path.join(ROOT,"results","session2")+"/"
X=os.path.join(ROOT,"data","xstest_prompts.csv")
FAM=["math-7b","llama-8b","qwen-14b","qwen-32b"]






def ci(v):
    v=np.asarray(v,float); return [float(v.mean()), float(stats.t.ppf(0.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))]
out={}
# text baselines recomputed on the same 8 folds
df=pd.read_csv(X); y=(df.label=="unsafe").astype(int).values
g=df.type.str.replace(r"^contrast_","",regex=True); g=g.where(~g.str.contains("privacy"),"privacy").where(~g.str.contains("discr"),"discr").values
groups=sorted(set(g))
tb={}
for tag,kw in [("word1",dict(analyzer="word",ngram_range=(1,1))),("word12",dict(analyzer="word",ngram_range=(1,2))),("char35",dict(analyzer="char_wb",ngram_range=(3,5)))]:
    a={}
    for h in groups:
        te,tr=g==h,g!=h
        V=TfidfVectorizer(sublinear_tf=True,**kw).fit(df.prompt[tr]); lr=LogisticRegression(C=1,max_iter=5000).fit(V.transform(df.prompt[tr]),y[tr])
        a[h]=roc_auc_score(y[te],lr.decision_function(V.transform(df.prompt[te])))
    tb[tag]=a
L=df.prompt.str.len().values; tb["length"]={h:roc_auc_score(y[g==h],-L[g==h]) for h in groups}
out["text"]={k:{"mean_ci":ci(list(v.values())),"min":min(v.values()),"max":max(v.values()),"folds":v} for k,v in tb.items()}
for f in FAM:
    D=pd.read_csv(f"{U}{f}_folds.csv"); I=pd.read_csv(f"{Z}{f}_inst_folds.csv")
    C=pd.read_csv(f"{U}{f}_cosines_folds.csv"); CI_=pd.read_csv(f"{Z}{f}_inst_cosines_folds.csv")
    assert list(D.group)==list(I.group)==list(C.group)
    F=pd.read_csv(f"{U}{f}_fixedlayer_folds.csv"); FI=pd.read_csv(f"{Z}{f}_inst_fixedlayer_folds.csv")
    E=json.load(open(f"{Z}{f}_external.json"))["results"]
    marg=[D.auc_base[D.group==h].iloc[0]-tb["char35"][h] for h in D.group]
    r=dict(layers=D.layer.tolist(),
        base=ci(D.auc_base), dist_tr=ci(D.auc_tr_basescaler), dist_own=ci(D.auc_own), dist_change=ci(D.diff_tr),
        inst_tr=ci(I.auc_tr_basescaler), inst_own=ci(I.auc_own), inst_change=ci(I.diff_tr), inst_base_check=ci(I.auc_base),
        dist_cos_bs=ci(C.basescale), dist_cos_own=ci(C.ownscale), dist_cos_raw=ci(C.raw), dist_raw_range=[float(C.raw.min()),float(C.raw.max())], dist_probe_raw=ci(C.probe_raw), dist_probe_own=ci(C.probe_ownscale),
        inst_cos_bs=ci(CI_.basescale), inst_probe_raw=ci(CI_.probe_raw),
        dist_share=ci(D.eta_basescaler), dist_share_minus_cos=ci(D.eta_basescaler-C.basescale), dist_share_minus_probe=ci(D.eta_basescaler-C.probe_raw), dist_ceiling_minus_base=ci(D.auc_own-D.auc_base),
        inst_share=ci(I.eta_basescaler), inst_share_minus_cos=ci(I.eta_basescaler-CI_.basescale), inst_ceiling_minus_base=ci(I.auc_own-I.auc_base),
        fixed=dict(layer=None, base=float(F.auc_base.mean()), dist_tr=float(F.auc_tr.mean()), dist_own=float(F.auc_own.mean()), dist_change=ci(F.diff_tr), dist_range=[float(F.diff_tr.min()),float(F.diff_tr.max())],
                   inst_tr=float(FI.auc_tr.mean()), inst_own=float(FI.auc_own.mean()), inst_change=ci(FI.diff_tr), inst_range=[float(FI.diff_tr.min()),float(FI.diff_tr.max())]),
        margin_char=ci(marg), margin_min=float(min(marg)),
        ext={k:{m:E[k][m] for m in E[k]} for k in E})
    r["dist_angle"]=float(np.degrees(np.arccos(r["dist_cos_bs"][0]))); r["inst_angle"]=float(np.degrees(np.arccos(r["inst_cos_bs"][0])))
    Ld=pd.read_csv(f"{U}{f}_cosines_layers.csv"); Lo=pd.read_csv(f"{U}{f}_layers.csv")
    Li=pd.read_csv(f"{Z}{f}_inst_cosines_layers.csv"); Lio=pd.read_csv(f"{Z}{f}_inst_layers.csv")
    r["sweep"]=dict(dist_cos=Ld.basescale.tolist(), dist_change=(Lo.auc_dm_tr-Lo.auc_dm_base).tolist(), inst_cos=Li.basescale.tolist(), inst_change=(Lio.auc_dm_tr-Lio.auc_dm_base).tolist())
    R=pd.read_csv(f"{U}{f}_controls_bs.csv")
    r["controls"]=dict(label=float(R[R.direction=="safety"].cos_full_bs.iloc[0]), splits=R[R.direction.str.startswith("typesplit")].cos_full_bs.tolist(),
                       how=float(R[R.direction=="question_how"].cos_full_bs.iloc[0]), named=float(R[R.direction=="named_entity"].cos_full_bs.iloc[0]), length=float(R[R.direction=="length_long"].cos_full_bs.iloc[0]))
    out[f]=r
# family-level tests
fam=lambda key: [out[f][key][0] for f in FAM]
out["family_p"]={k:float(stats.ttest_1samp(fam(k),0).pvalue) for k in ["dist_share_minus_cos","dist_share_minus_probe","dist_ceiling_minus_base","inst_share_minus_cos","inst_ceiling_minus_base"]}
json.dump(out,open(os.path.join(HERE,"numbers.json"),"w"),indent=1)
p=lambda v: f"{v[0]:+.3f}±{v[1]:.3f}"
print("text:",{k:(round(v['mean_ci'][0],3),round(v['mean_ci'][1],3),round(v['min'],2),round(v['max'],2)) for k,v in out['text'].items()})
for f in FAM:
    r=out[f]; e=r["ext"]
    print(f, "base",round(r['base'][0],3),"dist",round(r['dist_tr'][0],3),p(r['dist_change']),"inst",round(r['inst_tr'][0],3),p(r['inst_change']),"instbase",round(r['inst_base_check'][0],3),
          "ang",round(r['dist_angle'],1),round(r['inst_angle'],1),"margin",p(r['margin_char']),"min",round(r['margin_min'],3))
    print("   ext own_base",round(e['own_base']['auc_hard_vs_toxic'],3),"tr_d",round(e['transfer_distilled']['diff_vs_base_auc_hard'],3),np.round(e['transfer_distilled']['diff_vs_base_auc_hard_ci'],3),"tr_i",round(e['transfer_instruct']['diff_vs_base_auc_hard'],3),np.round(e['transfer_instruct']['diff_vs_base_auc_hard_ci'],3),
          "fpr_hard@0",round(e['own_base']['fpr_hard@0'],3),"tr_d fpr_hard@0",round(e['transfer_distilled']['fpr_hard@0'],3),"tr_i",round(e['transfer_instruct']['fpr_hard@0'],3))
    print("   fixed", p(r['fixed']['dist_change']), p(r['fixed']['inst_change']), "controls label",round(r['controls']['label'],3), "split min/max", round(min(r['controls']['splits']),3), round(max(r['controls']['splits']),3))
print(out["family_p"])
print("text char35 OR:", round(out['math-7b']['ext']['text_char35']['auc_hard_vs_toxic'],3), np.round(out['math-7b']['ext']['text_char35']['auc_hard_vs_toxic_ci'],3))
