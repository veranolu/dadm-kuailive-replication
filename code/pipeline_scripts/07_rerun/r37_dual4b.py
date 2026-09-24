# -*- coding: utf-8 -*-
"""
R3-7b: Unconstrained-transition RI-CLPM — Scheme 4B dual-IV 对齐版
==================================================================
与 V4.1 primary (Scheme 4B) 同模型式:
  dv_cur = a + gamma1*ln_watch_lag_z + gamma2*ln_click_lag_z + beta*dv_lag_z + e
每条 focal path 报告: pooled(对照4B冻结值) / split W1->W2 / split W2->W3 /
interaction (gamma_diff 正式检验)。
数据/抽样/先验/采样与 bayesian_riclpm_V105B.py、回填脚本完全一致:
  default_rng(42).choice 5000; wp 去中心化; z 标准化 IV; Normal(0,0.5); NUTS 500+1000x2 seed=42
"""
import os, json, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DATA_PATH = "/Coze/Drive/扣子/所有对话/主对话/论文4_数据清洗/kuailive_panel_balanced.csv"
OUT_DIR = "/Coze/Drive/班主任/论文4_大修_20260922/R3-7_unconstrained/out_dual4b"
os.makedirs(OUT_DIR, exist_ok=True)
N_SUBSAMPLE, N_TUNE, N_DRAWS, RANDOM_SEED = 5000, 500, 1000, 42

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df["ln_click"] = np.log1p(df["click_count"])
    df["ln_comment"] = np.log1p(df["comment_count"])
    df["ln_watch"] = np.log1p(df["watch_time_total"])
    denom = df["click_count"] + df["like_count"] + df["comment_count"]
    df["DEP_full"] = np.where(denom > 0, df["comment_count"] / denom, np.nan)
    rng = np.random.default_rng(RANDOM_SEED)
    sub = rng.choice(df["user_id"].unique(), size=N_SUBSAMPLE, replace=False)
    df = df[df.user_id.isin(sub)].copy()
    for v in ["ln_click", "ln_comment", "ln_watch", "DEP_full"]:
        df[f"{v}_wp"] = df[v] - df.groupby("user_id")[v].transform("mean")
    recs = []
    for uid, grp in df.sort_values(["user_id", "wave"]).groupby("user_id"):
        if len(grp) != 3: continue
        grp = grp.sort_values("wave")
        for a, b in [(0, 1), (1, 2)]:
            r = {"user_id": uid, "lag_pair": f"W{a+1}_W{b+1}"}
            for v in ["ln_click", "ln_comment", "ln_watch", "DEP_full"]:
                r[f"{v}_lag"] = grp.iloc[a][f"{v}_wp"]; r[f"{v}_cur"] = grp.iloc[b][f"{v}_wp"]
            recs.append(r)
    return pd.DataFrame(recs)

def _bf01(s, psd=0.5):
    try: return float(gaussian_kde(np.asarray(s).flatten())(0)[0] / stats.norm.pdf(0, 0, psd))
    except Exception: return np.nan

def run_dual(w_lag, c_lag, dv_lag, dv_cur, trans_idx=None):
    import numpyro, numpyro.distributions as dist, jax, jax.numpy as jnp, arviz as az
    from numpyro.infer import NUTS, MCMC
    numpyro.set_host_device_count(1)
    mask = np.isfinite(w_lag) & np.isfinite(c_lag) & np.isfinite(dv_lag) & np.isfinite(dv_cur)
    xw = np.asarray(w_lag[mask], float); xc = np.asarray(c_lag[mask], float)
    xd = np.asarray(dv_lag[mask], float); y = np.asarray(dv_cur[mask], float)
    n = len(y)
    if n < 100: return {"error": f"n={n}"}
    xwz = (xw - xw.mean()) / (xw.std() + 1e-8); xcz = (xc - xc.mean()) / (xc.std() + 1e-8)
    xdz = (xd - xd.mean()) / (xd.std() + 1e-8)

    if isinstance(trans_idx, np.ndarray):
        t2 = np.asarray(trans_idx[mask], float)
        def model(xwz, xcz, xdz, t2, y):
            g1 = numpyro.sample("gamma1", dist.Normal(0., 0.5)); g1d = numpyro.sample("gamma1_diff", dist.Normal(0., 0.5))
            g2 = numpyro.sample("gamma2", dist.Normal(0., 0.5)); g2d = numpyro.sample("gamma2_diff", dist.Normal(0., 0.5))
            b = numpyro.sample("beta", dist.Normal(0., 0.5))
            a = numpyro.sample("intercept", dist.Normal(0., 1.)); s = numpyro.sample("sigma", dist.HalfNormal(1.))
            mu = a + (g1 + g1d * t2) * xwz + (g2 + g2d * t2) * xcz + b * xdz
            numpyro.sample("y", dist.Normal(mu, s), obs=y)
        mcmc = MCMC(NUTS(model), num_warmup=N_TUNE, num_samples=N_DRAWS, num_chains=2,
                    chain_method="sequential", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(RANDOM_SEED), xwz=jnp.asarray(xwz), xcz=jnp.asarray(xcz),
                 xdz=jnp.asarray(xdz), t2=jnp.asarray(t2), y=jnp.asarray(y))
        varnames = ["gamma1", "gamma1_diff", "gamma2", "gamma2_diff", "beta"]
        derived = {"gamma1_12": None, "gamma1_23": None, "gamma2_12": None, "gamma2_23": None}
    else:
        def model(xwz, xcz, xdz, y):
            g1 = numpyro.sample("gamma1", dist.Normal(0., 0.5))
            g2 = numpyro.sample("gamma2", dist.Normal(0., 0.5))
            b = numpyro.sample("beta", dist.Normal(0., 0.5))
            a = numpyro.sample("intercept", dist.Normal(0., 1.)); s = numpyro.sample("sigma", dist.HalfNormal(1.))
            numpyro.sample("y", dist.Normal(a + g1 * xwz + g2 * xcz + b * xdz, s), obs=y)
        mcmc = MCMC(NUTS(model), num_warmup=N_TUNE, num_samples=N_DRAWS, num_chains=2,
                    chain_method="sequential", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(RANDOM_SEED), xwz=jnp.asarray(xwz), xcz=jnp.asarray(xcz),
                 xdz=jnp.asarray(xdz), y=jnp.asarray(y))
        varnames = ["gamma1", "gamma2", "beta"]; derived = {}

    idata = az.from_numpyro(mcmc)
    summ = az.summary(idata, var_names=varnames, hdi_prob=0.95)
    smp = mcmc.get_samples(group_by_chain=False)
    out = {"n_obs": n}
    for t in varnames:
        out[t] = {"mean": float(summ.loc[t, "mean"]), "sd": float(summ.loc[t, "sd"]),
                  "hdi_lo": float(summ.loc[t, "hdi_2.5%"]), "hdi_hi": float(summ.loc[t, "hdi_97.5%"]),
                  "rhat": float(summ.loc[t, "r_hat"]), "ess": float(summ.loc[t, "ess_bulk"]),
                  "bf01": _bf01(smp[t])}
    if derived:
        g1, g1d = np.asarray(smp["gamma1"]), np.asarray(smp["gamma1_diff"])
        g2, g2d = np.asarray(smp["gamma2"]), np.asarray(smp["gamma2_diff"])
        for nm, arr in [("gamma1_12", g1), ("gamma1_23", g1 + g1d),
                        ("gamma2_12", g2), ("gamma2_23", g2 + g2d)]:
            hdi = az.hdi(arr, hdi_prob=0.95)
            out[nm] = {"mean": float(arr.mean()), "sd": float(arr.std()),
                       "hdi_lo": float(hdi[0]), "hdi_hi": float(hdi[1]), "bf01": _bf01(arr)}
    return out

def fmt(v, n=None):
    if not v or "error" in (v or {}): return str(v)
    return f"mean={v['mean']:+.4f} HDI=[{v['hdi_lo']:+.4f},{v['hdi_hi']:+.4f}] BF01={v['bf01']:.2f}" + (f" N={n}" if n else "")

def main():
    lag = load_and_prepare()
    t2 = (lag["lag_pair"] == "W2_W3").astype(float).values
    results = {}
    for dv, dv_tag in [("ln_comment", "H1_DV_comment"), ("DEP_full", "H2_DV_DEP")]:
        print(f"\n########## DV = {dv} ({dv_tag}) ##########")
        r = {}
        r["pooled"] = run_dual(lag["ln_watch_lag"], lag["ln_click_lag"], lag[f"{dv}_lag"], lag[f"{dv}_cur"])
        print("pooled  : g1(watch)", fmt(r["pooled"].get("gamma1", r["pooled"]), r["pooled"].get("n_obs")))
        print("pooled  : g2(click)", fmt(r["pooled"].get("gamma2", {})))
        for sp, lab in [(0, "W1_W2"), (1, "W2_W3")]:
            sub = lag[lag["lag_pair"] == lab]
            r[f"split_{lab}"] = run_dual(sub["ln_watch_lag"], sub["ln_click_lag"], sub[f"{dv}_lag"], sub[f"{dv}_cur"])
            print(f"{lab}: g1", fmt(r[f"split_{lab}"].get("gamma1", {})))
            print(f"{lab}: g2", fmt(r[f"split_{lab}"].get("gamma2", {})))
        r["interaction"] = run_dual(lag["ln_watch_lag"], lag["ln_click_lag"], lag[f"{dv}_lag"], lag[f"{dv}_cur"], trans_idx=t2)
        gi = r["interaction"]
        for k in ["gamma1_12", "gamma1_23", "gamma1_diff", "gamma2_12", "gamma2_23", "gamma2_diff"]:
            if k in gi: print(f"inter : {k}", fmt(gi[k], gi.get("n_obs")))
        results[dv_tag] = r
    with open(f"{OUT_DIR}/r37_dual4b_results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("\nDONE ->", OUT_DIR)

if __name__ == "__main__":
    main()
