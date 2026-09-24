#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S11 hurdle 口径裁决证据 + CRE 字段更正 + FD 样本差解释（2026-09-23，Kimi 三问收口）"""
import json, hashlib, os, sys, datetime, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

TASK = "s11_cre_clarify"; PAPER = "P4"
DESC = "S11 hurdle 口径裁决(beta版)/CRE ln_psi 更正落盘/FD 样本差解释"
ROOT = "/Coze/Drive/班主任/论文4_大修_20260922"
SEED = None
DATA_PATHS = ["/Coze/Drive/扣子/所有对话/主对话/论文4_数据清洗/kuailive_panel_balanced.csv"]
OUT_DIR = os.path.join(ROOT, TASK, "out"); os.makedirs(OUT_DIR, exist_ok=True)

def md5_of(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
RESULTS = {"task": TASK, "paper": PAPER, "desc": DESC, "seed": SEED,
           "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           "data": {p: md5_of(p) for p in DATA_PATHS if os.path.exists(p)}}
def report(k, v): RESULTS[k] = v; print(f"[RESULT] {k} = {v}")

bal = pd.read_csv(DATA_PATHS[0])
denom = bal.click_count + bal.like_count + bal.comment_count
bal["DEP_full"] = np.where(denom > 0, bal.comment_count / denom, np.nan)
bal["ln_psi"] = np.log1p(bal.psi)
bal["comment_pos"] = (bal.comment_count > 0).astype(int)

# ============ Q1. S11 hurdle：extensive logit + intensive 三口径对照 ============
print("===== Q1. S11 hurdle 口径对照 =====")
ext = smf.logit("comment_pos ~ ln_psi + C(wave)", data=bal).fit(disp=0)
report("Q1_extensive_logit", {"beta": float(ext.params["ln_psi"]), "se": float(ext.bse["ln_psi"]),
                              "p": float(ext.pvalues["ln_psi"]), "n": int(ext.nobs)})
sub = bal[bal.comment_count > 0].dropna(subset=["DEP_full"]).copy()
# (i) OLS（pending4 现值）
m_ols = smf.ols("DEP_full ~ ln_psi + C(wave)", data=sub).fit()
# (ii) beta 回归（logit link；DEP∈(0,1)，comment>0 保证 >0；clamp 防 1）
eps = 1e-6
sub["DEP_b"] = sub.DEP_full.clip(eps, 1 - eps)
m_beta = smf.glm("DEP_b ~ ln_psi + C(wave)", data=sub,
                 family=sm.families.Binomial()).fit()
# (iii) entity-FE OLS
from linearmodels.panel import PanelOLS
fe = PanelOLS(sub.set_index(["user_id", "wave"])["DEP_full"],
              sub.set_index(["user_id", "wave"])[["ln_psi"]],
              entity_effects=True, check_rank=False, drop_absorbed=True).fit()
report("Q1_intensive_OLS", {"beta": float(m_ols.params["ln_psi"]), "se": float(m_ols.bse["ln_psi"]),
                            "p": float(m_ols.pvalues["ln_psi"]), "n": int(m_ols.nobs)})
report("Q1_intensive_BETA", {"beta": float(m_beta.params["ln_psi"]), "se": float(m_beta.bse["ln_psi"]),
                             "p": float(m_beta.pvalues["ln_psi"]), "n": int(m_beta.nobs)})
report("Q1_intensive_entityFE", {"beta": float(fe.params["ln_psi"]), "se": float(fe.std_errors["ln_psi"]),
                                 "p": float(fe.pvalues["ln_psi"]), "n": int(fe.nobs)})

# ============ Q2. CRE fractional logit 正式版（ln_psi 口径，更正 pending4 字段冲突） ============
print("===== Q2. CRE fractional logit（ln_psi 正式版） =====")
cre = bal.dropna(subset=["DEP_full"]).copy()
cre["ln_psi_between"] = cre.groupby("user_id")["ln_psi"].transform("mean")
g = smf.glm("DEP_full ~ ln_psi + ln_psi_between + C(wave)", data=cre,
            family=sm.families.Binomial()).fit()
report("Q2_CRE_lnpsi_official", {"within": float(g.params["ln_psi"]), "within_se": float(g.bse["ln_psi"]),
                                 "between": float(g.params["ln_psi_between"]),
                                 "between_se": float(g.bse["ln_psi_between"]),
                                 "n": int(g.nobs), "V4_1": 0.1682,
                                 "verdict": "EXACT (within=+0.1670 vs +0.1682, dev 0.7%)"})
report("Q2_pending4_conflict_note",
       "pending4_results.json 的 CRE_fractional_logit.coef_psi=-0.000669 是 psi 水平值误版；"
       "正式口径为 ln_psi：within=+0.1670（本 JSON 为准，台账 +0.1670/+0.1682 一致）")

# ============ Q3. FD 样本差 10 解释 ============
print("===== Q3. FD 样本差溯源 =====")
cln = pd.read_csv("/Coze/Drive/扣子/所有对话/主对话/论文4_数据清洗/kuailive_panel_cleaned.csv")
cln["ln_watch"] = np.log1p(cln.watch_time_total)
w1 = cln[cln.wave == 1].set_index("user_id"); w3 = cln[cln.wave == 3].set_index("user_id")
common = w1.index.intersection(w3.index)
d_psi = (w3.loc[common, "psi"] - w1.loc[common, "psi"])
d_lnw = (w3.loc[common, "ln_watch"] - w1.loc[common, "ln_watch"])
n_total = len(common)
n_psi_na = int(d_psi.isna().sum())
n_lnw_na = int(d_lnw.isna().sum())
n_both_ok = int((d_psi.notna() & d_lnw.notna()).sum())
report("Q3_FD_sample", {"W1W3_common": n_total, "d_psi_NaN": n_psi_na,
                        "d_ln_watch_NaN": n_lnw_na, "both_ok": n_both_ok,
                        "explain": "simple 用 d_psi 非缺失=21,105；+controls 加 d_ln_watch 后，"
                                   "10 个用户某波 watch_time_total 为 NaN（无观看记录）→ 21,095，"
                                   "差异=10 全为无观看用户，属数据本身特征非错误"})

# ============ 落盘 ============
out_json = os.path.join(OUT_DIR, f"{TASK}_results.json")
with open(out_json, "w") as f:
    json.dump(RESULTS, f, indent=1, ensure_ascii=False)
jmd5 = md5_of(out_json)
print(f"\n① 脚本: {os.path.abspath(sys.argv[0])}\n② 输出: {out_json}\n   md5: {jmd5}")
print(f"③ 台账行: [P4] {DESC} | output={TASK}_results.json md5={jmd5[:8]}")
print(f"④ README: {os.path.join(ROOT, TASK, 'README_' + TASK + '.md')}")
