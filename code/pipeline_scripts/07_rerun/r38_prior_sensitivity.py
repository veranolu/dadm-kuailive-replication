# -*- coding: utf-8 -*-
"""
R3-8: Prior sensitivity on H2 focal null paths (G MASTER ORDER PHASE 5 / P1)
============================================================================
V4.1 Table S17 引用 4 priors x 3 subsamples = 12 combinations on ln_watch->DEP,
但其脚本/输出无云端存档 -> 重跑验证 + 顺手补 click->DEP 路径缺口。
模型: Scheme 4B dual-IV (与 primary 同式), DV=DEP_full。
先验 (gamma): Normal(0,0.5) / Normal(0,1) / Normal(0,2) / Cauchy(0,0.707) 对称重尾
  注: 旧报告写 HalfCauchy(0,0.707) 为 "heavy-tailed symmetric" —— 标准 HalfCauchy
  非对称且强制 gamma>0, 方法论上不适用于可正可负的路径系数; 此处采用对称 Cauchy。
subsamples: 3000 / 5000 / 8000 (default_rng(42).choice, 与 S17 描述一致)
采样: NUTS tune=500 draws=1000 chains=2 seed=42
"""
import os, json, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA_PATH = "/Coze/Drive/扣子/所有对话/主对话/论文4_数据清洗/kuailive_panel_balanced.csv"
OUT_DIR = "/Coze/Drive/班主任/论文4_大修_20260922/R3-8_prior_sensitivity/out"
os.makedirs(OUT_DIR, exist_ok=True)
N_TUNE, N_DRAWS, RANDOM_SEED = 500, 1000, 42

def load_panel():
    df = pd.read_csv(DATA_PATH)
    df["ln_click"] = np.log1p(df["click_count"]); df["ln_watch"] = np.log1p(df["watch_time_total"])
    denom = df["click_count"] + df["like_count"] + df["comment_count"]
    df["DEP_full"] = np.where(denom > 0, df["comment_count"] / denom, np.nan)
    for v in ["ln_click", "ln_watch", "DEP_full"]:
        df[f"{v}_wp"] = df[v] - df.groupby("user_id")[v].transform("mean")
    recs = []
    for uid, grp in df.sort_values(["user_id", "wave"]).groupby("user_id"):
        if len(grp) != 3: continue
        grp = grp.sort_values("wave")
        for a, b in [(0, 1), (1, 2)]:
            r = {"user_id": uid}
            for v in ["ln_click", "ln_watch", "DEP_full"]:
                r[f"{v}_lag"] = grp.iloc[a][f"{v}_wp"]; r[f"{v}_cur"] = grp.iloc[b][f"{v}_wp"]
            recs.append(r)
    return df, pd.DataFrame(recs)

def _bf01_kde(s, prior_pdf0):
    try: return float(gaussian_kde(np.asarray(s).flatten())(0)[0] / prior_pdf0)
    except Exception: return np.nan

PRIORS = {
    "Normal_0_0.5":  ("normal", 0.5),
    "Normal_0_1":    ("normal", 1.0),
    "Normal_0_2":    ("normal", 2.0),
    "Cauchy_0_0.707": ("cauchy", 0.707),
}

def run(w_lag, c_lag, dv_lag, dv_cur, prior_kind, prior_scale):
    import numpyro, numpyro.distributions as dist, jax, jax.numpy as jnp, arviz as az
    from numpyro.infer import NUTS, MCMC
    numpyro.set_host_device_count(1)
    mask = np.isfinite(w_lag) & np.isfinite(c_lag) & np.isfinite(dv_lag) & np.isfinite(dv_cur)
    xw = np.asarray(w_lag[mask], float); xc = np.asarray(c_lag[mask], float)
    xd = np.asarray(dv_lag[mask], float); y = np.asarray(dv_cur[mask], float)
    n = len(y)
    xwz = (xw - xw.mean()) / (xw.std() + 1e-8); xcz = (xc - xc.mean()) / (xc.std() + 1e-8)
    xdz = (xd - xd.mean()) / (xd.std() + 1e-8)
    if prior_kind == "normal":
        p1 = dist.Normal(0., prior_scale); p2 = dist.Normal(0., prior_scale)
        pdf0 = stats.norm.pdf(0, 0, prior_scale)
    else:
        p1 = dist.StudentT(1.0, 0., prior_scale); p2 = dist.StudentT(1.0, 0., prior_scale)
        pdf0 = stats.cauchy.pdf(0, 0, prior_scale)
    def model(xwz, xcz, xdz, y):
        g1 = numpyro.sample("gamma1", p1); g2 = numpyro.sample("gamma2", p2)
        b = numpyro.sample("beta", dist.Normal(0., 0.5))
        a = numpyro.sample("intercept", dist.Normal(0., 1.)); s = numpyro.sample("sigma", dist.HalfNormal(1.))
        numpyro.sample("y", dist.Normal(a + g1 * xwz + g2 * xcz + b * xdz, s), obs=y)
    mcmc = MCMC(NUTS(model), num_warmup=N_TUNE, num_samples=N_DRAWS, num_chains=2,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(RANDOM_SEED), xwz=jnp.asarray(xwz), xcz=jnp.asarray(xcz),
             xdz=jnp.asarray(xdz), y=jnp.asarray(y))
    idata = az.from_numpyro(mcmc)
    summ = az.summary(idata, var_names=["gamma1", "gamma2"], hdi_prob=0.95)
    smp = mcmc.get_samples(group_by_chain=False)
    out = {"n_obs": n}
    for t in ["gamma1", "gamma2"]:
        out[t] = {"mean": float(summ.loc[t, "mean"]), "sd": float(summ.loc[t, "sd"]),
                  "hdi_lo": float(summ.loc[t, "hdi_2.5%"]), "hdi_hi": float(summ.loc[t, "hdi_97.5%"]),
                  "rhat": float(summ.loc[t, "r_hat"]), "ess": float(summ.loc[t, "ess_bulk"]),
                  "bf01": _bf01_kde(smp[t], pdf0)}
    return out

def main():
    df_full, lag_full = load_panel()
    results = {}
    for n_sub in [3000, 5000, 8000]:
        rng = np.random.default_rng(RANDOM_SEED)
        sub = rng.choice(df_full["user_id"].unique(), size=n_sub, replace=False)
        lag = lag_full[lag_full.user_id.isin(sub)]
        for pname, (pkind, pscale) in PRIORS.items():
            key = f"N{n_sub}_{pname}"
            print(f"[run] {key} ...", flush=True)
            results[key] = run(lag["ln_watch_lag"], lag["ln_click_lag"], lag["DEP_full_lag"], lag["DEP_full_cur"], pkind, pscale)
            g1, g2 = results[key]["gamma1"], results[key]["gamma2"]
            print(f"  g1(watch): mean={g1['mean']:+.5f} HDI=[{g1['hdi_lo']:+.5f},{g1['hdi_hi']:+.5f}] BF01={g1['bf01']:.2f} | g2(click): mean={g2['mean']:+.5f} BF01={g2['bf01']:.2f} N={results[key]['n_obs']}")
    with open(f"{OUT_DIR}/r38_prior_results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("DONE ->", OUT_DIR)

if __name__ == "__main__":
    main()
