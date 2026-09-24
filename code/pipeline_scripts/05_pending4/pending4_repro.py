#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4 重跑包：FD + extensive/intensive + entropy + CRE fractional logit（Kimi 授权，2026-09-23）
合规：JSON+md5+seed+数据md5 申报；不改稿件；全部从面板真实计算。
"""
import json, hashlib, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BAL = "/Coze/Drive/扣子/所有对话/主对话/论文4_数据清洗/kuailive_panel_balanced.csv"
CLN = "/Coze/Drive/扣子/所有对话/主对话/论文4_数据清洗/kuailive_panel_cleaned.csv"
OUT_DIR = "/Coze/Drive/班主任/论文4_大修_20260922/PENDING4_重跑/out"
os.makedirs(OUT_DIR, exist_ok=True)

def md5_of(path):
    h = hashlib.md5()
    with open(path,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): h.update(c)
    return h.hexdigest()

results = {}
bal = pd.read_csv(BAL)
for df in (bal,):
    denom = df.click_count+df.like_count+df.comment_count
    df["DEP_full"] = np.where(denom>0, df.comment_count/denom, np.nan)

# ============ 1. FD（first-difference, V4.1 S26 锚）============
cln = pd.read_csv(CLN)
denom_c = cln.click_count+cln.like_count+cln.comment_count
cln["DEP_full"] = np.where(denom_c>0, cln.comment_count/denom_c, np.nan)
cln["ln_watch"] = np.log1p(cln.watch_time_total)
cln["ln_gift_amt"] = np.log1p(cln.gift_amount_total)
w1 = cln[cln.wave==1].set_index("user_id"); w3 = cln[cln.wave==3].set_index("user_id")
common = w1.index.intersection(w3.index)
fd = pd.DataFrame({
    "d_DEP": w3.loc[common,"DEP_full"] - w1.loc[common,"DEP_full"],
    "d_ln_gift": w3.loc[common,"ln_gift_amt"] - w1.loc[common,"ln_gift_amt"],
    "d_deep": w3.loc[common,"deep_count"] - w1.loc[common,"deep_count"],
    "d_psi": w3.loc[common,"psi"] - w1.loc[common,"psi"],
    "d_ln_watch": w3.loc[common,"ln_watch"] - w1.loc[common,"ln_watch"],
    "d_click": w3.loc[common,"click_count"] - w1.loc[common,"click_count"],
    "d_like": w3.loc[common,"like_count"] - w1.loc[common,"like_count"],
    "d_comment": w3.loc[common,"comment_count"] - w1.loc[common,"comment_count"],
}).dropna(subset=["d_psi"])
print(f"[FD] N={len(fd)} (V4.1=21,105)")
fd_res = {}
for y in ["d_DEP","d_ln_gift","d_deep"]:
    # simple: dy ~ d_psi
    m1 = smf.ols(f"{y} ~ d_psi", data=fd.dropna(subset=[y])).fit()
    # +controls: 加 Δwatch/Δclick/Δlike/Δcomment 中除 outcome 组成外项——V4.1 未明，采用 Δwatch+Δdeep 行为控制外的保守集
    if y=="d_DEP":
        m2 = smf.ols(f"{y} ~ d_psi + d_ln_watch", data=fd.dropna(subset=[y,"d_ln_watch"])).fit()
    elif y=="d_ln_gift":
        m2 = smf.ols(f"{y} ~ d_psi + d_ln_watch + d_deep", data=fd.dropna(subset=[y,"d_ln_watch","d_deep"])).fit()
    else:
        m2 = smf.ols(f"{y} ~ d_psi + d_ln_watch", data=fd.dropna(subset=[y,"d_ln_watch"])).fit()
    fd_res[y] = {"simple": {"coef": float(m1.params["d_psi"]), "se": float(m1.bse["d_psi"]), "p": float(m1.pvalues["d_psi"]), "n": int(m1.nobs), "r2": float(m1.rsquared)},
                 "controls": {"coef": float(m2.params["d_psi"]), "se": float(m2.bse["d_psi"]), "p": float(m2.pvalues["d_psi"]), "n": int(m2.nobs), "r2": float(m2.rsquared)}}
    print(f"  {y}: simple {m1.params['d_psi']:+.2e}(p={m1.pvalues['d_psi']:.3f}) | +ctrl {m2.params['d_psi']:+.2e}(p={m2.pvalues['d_psi']:.3f})")
results["FD"] = fd_res

# ============ 2. Extensive / Intensive（V4.1 S12 锚：+1.564 N=61,728 / −0.492 N=43,183）============
bal["ln_psi"] = np.log1p(bal.psi)
bal["comment_pos"] = (bal.comment_count>0).astype(int)
ext = smf.logit("comment_pos ~ ln_psi + C(wave)", data=bal).fit(disp=0)
inten_df = bal[bal.comment_count>0].dropna(subset=["DEP_full"])
inten = smf.ols("DEP_full ~ ln_psi + C(wave)", data=inten_df).fit()
results["extensive"] = {"coef": float(ext.params["ln_psi"]), "se": float(ext.bse["ln_psi"]), "p": float(ext.pvalues["ln_psi"]), "n": int(ext.nobs)}
results["intensive"] = {"coef": float(inten.params["ln_psi"]), "se": float(inten.bse["ln_psi"]), "p": float(inten.pvalues["ln_psi"]), "n": int(inten.nobs)}
print(f"[Margins] extensive logit β={ext.params['ln_psi']:+.3f} N={int(ext.nobs)} (V4.1=+1.564/61,728)")
print(f"[Margins] intensive OLS  β={inten.params['ln_psi']:+.3f} N={int(inten.nobs)} (V4.1=−0.492/43,183)")

# ============ 3. CRE fractional logit（V4.1 锚：+0.1682, SE=0.0097, N=61,728）============
cre = bal.dropna(subset=["DEP_full"]).copy()
cre["psi_between"] = cre.groupby("user_id")["psi"].transform("mean")
# Chamberlain-Mundlak: within psi + between mean psi
glm = smf.glm("DEP_full ~ psi + psi_between + C(wave)", data=cre,
              family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": cre.user_id})
results["CRE_fractional_logit"] = {"coef_psi": float(glm.params["psi"]), "se": float(glm.bse["psi"]),
    "p": float(glm.pvalues["psi"]), "coef_between": float(glm.params["psi_between"]),
    "n": int(glm.nobs)}
print(f"[CRE] fractional logit ψ_within={glm.params['psi']:+.6f}(SE={glm.bse['psi']:.6f}) ψ_between={glm.params['psi_between']:+.4f} (V4.1=+0.1682±0.0097)")

# ============ 4. Entropy 多元回归（V4.1 锚：β=−0.000345, z=−12.64, R²=10.4%, N=20,576）============
acts = bal.groupby("user_id")[["click_count","like_count","comment_count","gift_count"]].sum()
shares = acts.div(acts.sum(axis=1).replace(0,np.nan), axis=0)
H = -(shares.replace(0,np.nan)*np.log(shares.replace(0,np.nan))).sum(axis=1)
psi_mean = bal.groupby("user_id")["psi"].mean()
u = bal.drop_duplicates("user_id").set_index("user_id")
age_map = {"0-11":0,"12-17":1,"18-23":2,"24-30":3,"31-40":4,"41-49":5,"50+":6}
Xu = pd.DataFrame({"H": H, "psi_mean": psi_mean,
                   "age_num": u.age.map(age_map), "female": (u.gender=="F").astype(int)}).dropna()
mH = smf.ols("H ~ psi_mean + age_num + female", data=Xu).fit()
results["entropy"] = {"coef_psi": float(mH.params["psi_mean"]), "se": float(mH.bse["psi_mean"]),
    "z": float(mH.tvalues["psi_mean"]), "p": float(mH.pvalues["psi_mean"]),
    "r2": float(mH.rsquared), "n": int(mH.nobs)}
print(f"[Entropy] β_psi={mH.params['psi_mean']:+.6f} z={mH.tvalues['psi_mean']:+.2f} R²={mH.rsquared*100:.1f}% N={int(mH.nobs)} (V4.1=−0.000345/z=−12.64/R²=10.4%/20,576)")

# ============ 5. 落盘 + 申报 ============
results["provenance"] = {"script": "pending4_repro.py (Coze 2026-09-23, v1)",
    "data": {"balanced": BAL, "balanced_md5": md5_of(BAL), "cleaned": CLN, "cleaned_md5": md5_of(CLN)},
    "seed": None, "libs": f"statsmodels {sm.__version__}"}
out_json = os.path.join(OUT_DIR, "pending4_results.json")
with open(out_json,"w") as f: json.dump(results, f, indent=1, ensure_ascii=False)
print(f"\nJSON: {out_json}\nJSON md5: {md5_of(out_json)}")
