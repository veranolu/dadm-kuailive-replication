#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEP 口径 DID 复跑（裁决 B 增量证据）
====================================
目的：V4.1 报 δ_DEP=−0.0102（TWFE）、DEP 平趋 δ₁=−0.0048 (p=0.140)，但 DEP 版 DID 脚本无存档。
本脚本完全复制 did_analysis.py 的分组与设定（PSI W1→W3 pct_change, <−30% treated, ±10% control,
post=W3, entity+time 双 FE PanelOLS, cluster SE），outcome 换为 DEP_full（面板构造），
并同步跑 idi 作对照，判定 V4.1 的 DEP 口径 DID 数字可否复现。
合规：纯计算审计，不改任何稿件；产物 JSON+md5+seed+数据 md5 申报（Kimi 铁律）。
作者: Coze  |  2026-09-23  |  seed: 无需（frequentist，无随机抽样）
"""
import json, hashlib, os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

PANEL_PATH = "/Coze/Drive/扣子/所有对话/主对话/论文4_数据清洗/kuailive_panel_balanced.csv"
OUT_DIR = "/Coze/Drive/班主任/论文4_大修_20260922/DID_DEP_复跑/out"
os.makedirs(OUT_DIR, exist_ok=True)

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

# ---------- 1. 数据 + DEP 构造 ----------
panel = pd.read_csv(PANEL_PATH)
denom = panel["click_count"] + panel["like_count"] + panel["comment_count"]
panel["DEP_full"] = np.where(denom > 0, panel["comment_count"] / denom, np.nan)
panel["ln_gift_amount"] = np.log1p(panel["gift_amount_total"])

# ---------- 2. 分组（与 did_analysis.py 完全一致） ----------
psi_w1 = panel.loc[panel.wave==1, ["user_id","psi"]].rename(columns={"psi":"psi_w1"})
psi_w3 = panel.loc[panel.wave==3, ["user_id","psi"]].rename(columns={"psi":"psi_w3"})
psi_chg = psi_w1.merge(psi_w3, on="user_id")
psi_chg["psi_pct_change"] = (psi_chg.psi_w3 - psi_chg.psi_w1) / psi_chg.psi_w1.replace(0, np.nan)
psi_chg["group_30"] = "excluded"
psi_chg.loc[psi_chg.psi_pct_change < -0.30, "group_30"] = "treated"
psi_chg.loc[psi_chg.psi_pct_change.between(-0.10, 0.10), "group_30"] = "control"
panel = panel.merge(psi_chg[["user_id","group_30"]], on="user_id", how="left")
panel["post"] = (panel.wave==3).astype(int)
panel["treated"] = (panel.group_30=="treated").astype(int)
panel["did"] = panel.treated * panel.post
panel["wave2"] = (panel.wave==2).astype(int)
panel["wave3d"] = (panel.wave==3).astype(int)

n_t = panel.loc[(panel.wave==1)&(panel.group_30=="treated"),"user_id"].nunique()
n_c = panel.loc[(panel.wave==1)&(panel.group_30=="control"),"user_id"].nunique()
n_e = panel.loc[(panel.wave==1)&(panel.group_30=="excluded"),"user_id"].nunique()
print(f"分组: treated={n_t}, control={n_c}, excluded={n_e}  (V4.1: 6,690/1,828/12,058)")

did_df = panel[panel.group_30.isin(["treated","control"])].copy()
did_df = did_df.set_index(["user_id","wave"])

results = {"group_counts": {"treated": int(n_t), "control": int(n_c), "excluded": int(n_e)}}

# ---------- 3. TWFE binary DID ----------
def twfe(df, y):
    d = df.dropna(subset=[y])
    mod = PanelOLS(d[y], d[["did"]], entity_effects=True, time_effects=True,
                   check_rank=False, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return dict(coef=float(res.params["did"]), se=float(res.std_errors["did"]),
                pval=float(res.pvalues["did"]), n_obs=int(res.nobs))

# ---------- 4. 事件研究平趋 ----------
def event_study(df, y):
    d = df.dropna(subset=[y]).copy()
    d["treated_wave2"] = d.treated * d.wave2
    d["treated_wave3"] = d.treated * d.wave3d
    mod = PanelOLS(d[y], d[["treated_wave2","treated_wave3"]], entity_effects=True,
                   time_effects=True, check_rank=False, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return dict(delta1=float(res.params["treated_wave2"]), se1=float(res.std_errors["treated_wave2"]),
                p1=float(res.pvalues["treated_wave2"]),
                delta2=float(res.params["treated_wave3"]), se2=float(res.std_errors["treated_wave3"]),
                p2=float(res.pvalues["treated_wave3"]),
                pt_held_p05=bool(res.pvalues["treated_wave2"] > 0.05))

for y in ["DEP_full", "idi", "ln_gift_amount", "deep_count"]:
    results[f"twfe_{y}"] = twfe(did_df, y)
    results[f"event_{y}"] = event_study(did_df, y)
    t = results[f"twfe_{y}"]; e = results[f"event_{y}"]
    print(f"{y:15s} TWFE δ={t['coef']:+.4f} (p={t['pval']:.1e}) | 平趋 δ1={e['delta1']:+.4f} (p={e['p1']:.4f}) -> {'通过' if e['pt_held_p05'] else '拒绝'}")

# ---------- 5. V4.1 对账 ----------
anchor = {
  "DEP_full": {"twfe": -0.0102, "pt_delta1": -0.0048, "pt_se": 0.0032, "pt_p": 0.140},
  "ln_gift_amount": {"twfe": -0.600, "pt_delta1": -0.376},
  "deep_count": {"twfe": -2.476, "pt_delta1": -1.905},
}
print("\n===== V4.1 对账 =====")
for y, a in anchor.items():
    t = results[f"twfe_{y}"]; e = results[f"event_{y}"]
    print(f"{y}: TWFE 复跑={t['coef']:+.4f} vs V4.1={a['twfe']:+.4f} | 平趋 δ1 复跑={e['delta1']:+.4f}(p={e['p1']:.4f}) vs V4.1={a['pt_delta1']}(p={a.get('pt_p','n/a')})")

# ---------- 6. 落盘 + md5 申报 ----------
results["provenance"] = {
  "script": "did_dep_repro.py (Coze 2026-09-23, v1)",
  "data_path": PANEL_PATH,
  "data_md5": md5_of(PANEL_PATH),
  "spec": "PSI W1->W3 pct_change; treated<-30%, control ±10%; post=W3; PanelOLS entity+time FE, cluster(user)",
  "seed": None, "note": "frequentist, 无随机抽样",
  "DEP_construct": "comment/(click+like+comment), denom>0 else NaN",
}
out_json = os.path.join(OUT_DIR, "did_dep_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print(f"\nJSON: {out_json}")
print(f"JSON md5: {md5_of(out_json)}")
print(f"数据 md5: {results['provenance']['data_md5']}")
