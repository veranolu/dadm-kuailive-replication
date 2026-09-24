#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4 DID 考古复原·正式固化脚本（三件套之①脚本）
==============================================
目的：固化 2026-09-23 考古结论——V4.1 DID 数字在以下口径下全部可复现：
  TWFE  : entity FE only（不含 time FE），PanelOLS cluster(user) SE
  平趋  : ΔY(W2-W1) ~ treated 一阶差分 OLS（普通 OLS SE）
分组  : PSI W1->W3 pct_change; treated<-30%, control ±10%; post=W3
输出  : did_archaeology_results.json（三件套之②输出）
认定书: P4_DID_PROVENANCE_复原认定书.md（三件套之③台账行=P4_MASTER_NUMERICAL_PROVENANCE.xlsx DID 4行）
作者: Coze | 2026-09-23 v1.0 | 无 seed（frequentist）
"""
import json, hashlib, os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

PANEL = "/Coze/Drive/扣子/所有对话/主对话/论文4_数据清洗/kuailive_panel_balanced.csv"
OUT_DIR = "/Coze/Drive/班主任/论文4_大修_20260922/DID_考古复原/out"
os.makedirs(OUT_DIR, exist_ok=True)

def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

# ---------- 数据 ----------
panel = pd.read_csv(PANEL)
denom = panel.click_count + panel.like_count + panel.comment_count
panel["DEP"] = np.where(denom > 0, panel.comment_count / denom, 0.0)
panel["ln_gift_amount"] = np.log1p(panel.gift_amount_total)

# ---------- 分组（与 V4.1 完全一致：6,690/1,828/12,058） ----------
w1 = panel.loc[panel.wave == 1, ["user_id", "psi"]].rename(columns={"psi": "psi_w1"})
w3 = panel.loc[panel.wave == 3, ["user_id", "psi"]].rename(columns={"psi": "psi_w3"})
chg = w1.merge(w3, on="user_id")
chg["pct"] = (chg.psi_w3 - chg.psi_w1) / chg.psi_w1.replace(0, np.nan)
chg["group"] = "excluded"
chg.loc[chg.pct < -0.30, "group"] = "treated"
chg.loc[chg.pct.between(-0.10, 0.10), "group"] = "control"
panel = panel.merge(chg[["user_id", "group"]], on="user_id")
d = panel[panel.group.isin(["treated", "control"])].copy()
d["post"] = (d.wave == 3).astype(int)
d["treated"] = (d.group == "treated").astype(int)
d["did"] = d.treated * d.post

n_t = panel.loc[(panel.wave == 1) & (panel.group == "treated"), "user_id"].nunique()
n_c = panel.loc[(panel.wave == 1) & (panel.group == "control"), "user_id"].nunique()
n_e = panel.loc[(panel.wave == 1) & (panel.group == "excluded"), "user_id"].nunique()

results = {"group_counts": {"treated": int(n_t), "control": int(n_c), "excluded": int(n_e)},
           "V4_1_anchors": {"treated": 6690, "control": 1828, "excluded": 12058, "N": 25554}}

# ---------- TWFE: entity FE only ----------
anchor_twfe = {"DEP": -0.0102, "ln_gift_amount": -0.600, "deep_count": -2.476}
results["TWFE_entity_FE_only"] = {}
for y, anchor in anchor_twfe.items():
    dy = d.dropna(subset=[y]).set_index(["user_id", "wave"])
    mod = PanelOLS(dy[y], dy[["did"]], entity_effects=True, time_effects=False,
                   check_rank=False, drop_absorbed=True)
    r = mod.fit(cov_type="clustered", cluster_entity=True)
    dev = abs(r.params["did"] - anchor) / abs(anchor)
    results["TWFE_entity_FE_only"][y] = {
        "coef": float(r.params["did"]), "se": float(r.std_errors["did"]),
        "pval": float(r.pvalues["did"]), "n_obs": int(r.nobs),
        "V4_1": anchor, "deviation_pct": float(dev * 100), "hit": bool(dev < 0.03)}
    print(f"TWFE {y:15s}: δ={r.params['did']:+.4f} vs V4.1={anchor:+.4f} ({dev*100:.2f}%)")

# ---------- 平趋: ΔY(W2-W1) ~ treated ----------
uids = d.user_id.unique()
u1 = panel[(panel.wave == 1) & panel.user_id.isin(uids)].set_index("user_id")
u2 = panel[(panel.wave == 2) & panel.user_id.isin(uids)].set_index("user_id")
delta = pd.DataFrame({"treated": (u1.group == "treated").astype(int)})
anchor_pt = {"DEP": (-0.0048, 0.140), "ln_gift_amount": (-0.376, 0.001), "deep_count": (-1.905, 0.001)}
results["PT_delta_OLS"] = {}
for y, (a_coef, a_p) in anchor_pt.items():
    delta["dY"] = u2[y] - u1[y]
    dd = delta.dropna(subset=["dY"])
    m = smf.ols("dY ~ treated", data=dd).fit()
    dev = abs(m.params["treated"] - a_coef) / max(abs(a_coef), 1e-9)
    concl_match = (m.pvalues["treated"] > 0.05) == (a_p > 0.05)
    results["PT_delta_OLS"][y] = {
        "coef": float(m.params["treated"]), "se": float(m.bse["treated"]),
        "pval": float(m.pvalues["treated"]), "n_obs": int(m.nobs),
        "V4_1_coef": a_coef, "V4_1_p": a_p,
        "deviation_pct": float(dev * 100), "conclusion_match": bool(concl_match)}
    print(f"PT Δ{y:15s}: δ1={m.params['treated']:+.4f} (SE={m.bse['treated']:.4f}, p={m.pvalues['treated']:.3f}) "
          f"vs V4.1={a_coef:+.4f}(p={a_p}) | 结论一致={concl_match}")

# ---------- 落盘 ----------
results["provenance"] = {
    "script": "did_archaeology_repro.py v1.0 (Coze 2026-09-23)",
    "data_path": PANEL, "data_md5": md5_of(PANEL),
    "spec_TWFE": "entity FE only (no time FE); PanelOLS cluster(user)",
    "spec_PT": "OLS: (Y_W2 - Y_W1) ~ treated; plain OLS SE",
    "seed": None, "engine": f"linearmodels PanelOLS / statsmodels OLS",
    "cert_doc": "P4_DID_PROVENANCE_复原认定书.md"}
out_json = os.path.join(OUT_DIR, "did_archaeology_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print(f"\nJSON: {out_json}\nJSON md5: {md5_of(out_json)}\n数据 md5: {results['provenance']['data_md5']}")
