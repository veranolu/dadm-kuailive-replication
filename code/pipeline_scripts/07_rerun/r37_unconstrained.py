# -*- coding: utf-8 -*-
"""
R3-7: Unconstrained-transition RI-CLPM sensitivity (G MASTER ORDER PHASE 5 / P0)
================================================================================
目的：V4.1 pooled RI-CLPM 假设跨转移路径相等 (gamma_W1W2 = gamma_W2W3)。
本脚本对三条 focal path 分别报告：
  - gamma_12 (W1->W2) [mean/sd/95% HDI/BF01/Rhat/ESS/N]
  - gamma_23 (W2->W3) [同上]
  - pooled gamma (与 V4.1 现值对账)
  - gamma_diff = gamma_23 - gamma_12 的后验（统一交互模型）
数据/设定严格镜像主分析 codeact/scripts/bayesian_riclpm_V105B.py 与
回填脚本 paper4_kimi_analyses_20260917.py：
  - 同一 CSV, 同一 default_rng(42).choice 5000 用户子样本
  - 用户内去中心化; 预测变量 z 标准化; DV(wp) 不标准化
  - Normal(0,0.5) 路径先验; NUTS tune=500 draws=1000 chains=2 seed=42
留账：开工 2026-09-22 / 输出 out/ 下 json+md
"""
import os, json, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_PATH = "/Coze/Drive/扣子/所有对话/主对话/论文4_数据清洗/kuailive_panel_balanced.csv"
OUT_DIR = "/Coze/Drive/班主任/论文4_大修_20260922/R3-7_unconstrained/out"
os.makedirs(OUT_DIR, exist_ok=True)
N_SUBSAMPLE, N_TUNE, N_DRAWS, RANDOM_SEED = 5000, 500, 1000, 42

# ---------- 数据准备（与主分析同代码路径）----------
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df["ln_click"] = np.log1p(df["click_count"])
    df["ln_comment"] = np.log1p(df["comment_count"])
    df["ln_watch"] = np.log1p(df["watch_time_total"])
    denom_full = df["click_count"] + df["like_count"] + df["comment_count"]
    df["DEP_full"] = np.where(denom_full > 0, df["comment_count"] / denom_full, np.nan)

    rng = np.random.default_rng(RANDOM_SEED)
    sample_users = rng.choice(df["user_id"].unique(), size=N_SUBSAMPLE, replace=False)
    df = df[df["user_id"].isin(sample_users)].copy()

    CENTER = ["ln_click", "ln_comment", "ln_watch", "DEP_full"]
    for v in CENTER:
        df[f"{v}_wp"] = df[v] - df.groupby("user_id")[v].transform("mean")

    recs = []
    for uid, grp in df.sort_values(["user_id", "wave"]).groupby("user_id"):
        if len(grp) != 3:
            continue
        grp = grp.sort_values("wave")
        for t_lag, t_cur in [(0, 1), (1, 2)]:
            rec = {"user_id": uid, "lag_pair": f"W{t_lag+1}_W{t_cur+1}"}
            for v in CENTER:
                rec[f"{v}_lag"] = grp.iloc[t_lag][f"{v}_wp"]
                rec[f"{v}_cur"] = grp.iloc[t_cur][f"{v}_wp"]
            recs.append(rec)
    return pd.DataFrame(recs)

# ---------- MCMC 工具 ----------
def _bf01(samples, prior_sd=0.5):
    prior_d0 = stats.norm.pdf(0, loc=0, scale=prior_sd)
    s = np.asarray(samples).flatten()
    if len(s) < 100: return np.nan
    try: return float(gaussian_kde(s)(0)[0] / prior_d0)
    except Exception: return np.nan

def _extract(mcmc, terms, n_obs, extra_samples=None):
    import arviz as az
    idata = az.from_numpyro(mcmc)
    summ = az.summary(idata, var_names=terms, hdi_prob=0.95)
    out = {"n_obs": n_obs}
    samples = mcmc.get_samples(group_by_chain=False)
    if extra_samples: samples.update(extra_samples)
    for t in list(terms) + list((extra_samples or {}).keys()):
        if t in summ.index:
            out[t] = {"mean": float(summ.loc[t, "mean"]), "sd": float(summ.loc[t, "sd"]),
                      "hdi_lo": float(summ.loc[t, "hdi_2.5%"]), "hdi_hi": float(summ.loc[t, "hdi_97.5%"]),
                      "rhat": float(summ.loc[t, "r_hat"]), "ess_bulk": float(summ.loc[t, "ess_bulk"]),
                      "bf01": _bf01(samples[t])}
        else:
            arr = np.asarray(samples[t]).flatten()
            hdi = az.hdi(arr, hdi_prob=0.95)
            out[t] = {"mean": float(arr.mean()), "sd": float(arr.std()),
                      "hdi_lo": float(hdi[0]), "hdi_hi": float(hdi[1]),
                      "rhat": np.nan, "ess_bulk": np.nan, "bf01": _bf01(arr)}
    return out

def run_model(iv_lag, dv_lag, dv_cur, trans_idx=None):
    """y_cur = a + gamma*x_lag_z + beta*y_lag_z (+ gamma_d*I(trans2)*x_lag_z) + e
    trans_idx: None=pooled; 0/1 数组=交互模型; 'split0'/'split1'=分转移独立"""
    import numpyro, numpyro.distributions as dist, jax, jax.numpy as jnp
    from numpyro.infer import NUTS, MCMC
    numpyro.set_host_device_count(1)

    mask = np.isfinite(iv_lag) & np.isfinite(dv_lag) & np.isfinite(dv_cur)
    x1 = np.asarray(iv_lag[mask], float); x2 = np.asarray(dv_lag[mask], float)
    y = np.asarray(dv_cur[mask], float)
    n = len(y)
    if n < 100: return {"error": f"n={n}<100"}
    x1z = (x1 - x1.mean()) / (x1.std() + 1e-8)
    x2z = (x2 - x2.mean()) / (x2.std() + 1e-8)

    if isinstance(trans_idx, np.ndarray):
        t2 = np.asarray(trans_idx[mask], float)  # 1 = W2->W3
        def model(x1z, x2z, t2, y):
            gamma = numpyro.sample("gamma", dist.Normal(0., 0.5))
            gamma_d = numpyro.sample("gamma_diff", dist.Normal(0., 0.5))
            beta = numpyro.sample("beta", dist.Normal(0., 0.5))
            a = numpyro.sample("intercept", dist.Normal(0., 1.))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.))
            mu = a + (gamma + gamma_d * t2) * x1z + beta * x2z
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=N_TUNE, num_samples=N_DRAWS, num_chains=2,
                    chain_method="sequential", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(RANDOM_SEED), x1z=jnp.asarray(x1z), x2z=jnp.asarray(x2z),
                 t2=jnp.asarray(t2), y=jnp.asarray(y))
        s = mcmc.get_samples(group_by_chain=False)
        extra = {"gamma_12": np.asarray(s["gamma"]), "gamma_23": np.asarray(s["gamma"]) + np.asarray(s["gamma_diff"])}
        return _extract(mcmc, ["gamma", "gamma_diff", "beta"], n, extra)
    else:
        def model(x1z, x2z, y):
            gamma = numpyro.sample("gamma", dist.Normal(0., 0.5))
            beta = numpyro.sample("beta", dist.Normal(0., 0.5))
            a = numpyro.sample("intercept", dist.Normal(0., 1.))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.))
            numpyro.sample("y", dist.Normal(a + gamma * x1z + beta * x2z, sigma), obs=y)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=N_TUNE, num_samples=N_DRAWS, num_chains=2,
                    chain_method="sequential", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(RANDOM_SEED), x1z=jnp.asarray(x1z), x2z=jnp.asarray(x2z), y=jnp.asarray(y))
        return _extract(mcmc, ["gamma", "beta"], n)

# ---------- 主流程 ----------
PATHS = [
    ("H1", "ln_click", "ln_comment"),
    ("H2_watch", "ln_watch", "DEP_full"),
    ("H2_click", "ln_click", "DEP_full"),
]

def fmt(v, n_obs=None):
    if v is None or "error" in v: return str(v)
    n_str = f" N={n_obs}" if n_obs is not None else ""
    return (f"mean={v['mean']:+.4f} sd={v['sd']:.4f} HDI=[{v['hdi_lo']:+.4f},{v['hdi_hi']:+.4f}] "
            f"BF01={v['bf01']:.2f} Rhat={v['rhat']:.3f} ESS={v['ess_bulk']:.0f}{n_str}")

def main():
    lag = load_and_prepare()
    t2 = (lag["lag_pair"] == "W2_W3").astype(float).values
    results, lines = {}, []
    for tag, iv, dv in PATHS:
        print(f"\n===== {tag}: {iv}(t) -> {dv}(t+1) =====")
        r = {}
        # pooled（对照 V4.1）
        r["pooled"] = run_model(lag[f"{iv}_lag"], lag[f"{dv}_lag"], lag[f"{dv}_cur"])
        print("pooled   :", fmt(r["pooled"].get("gamma", r["pooled"]), r["pooled"].get("n_obs")))
        # 分转移独立估计
        for sp, lab in [(0, "W1_W2"), (1, "W2_W3")]:
            sub = lag[lag["lag_pair"] == lab]
            key = f"split_{lab}"
            r[key] = run_model(sub[f"{iv}_lag"], sub[f"{dv}_lag"], sub[f"{dv}_cur"])
            print(f"{key}:", fmt(r[key].get("gamma", r[key]), r[key].get("n_obs")))
        # 统一交互模型（gamma_diff 正式检验）
        r["interaction"] = run_model(lag[f"{iv}_lag"], lag[f"{dv}_lag"], lag[f"{dv}_cur"], trans_idx=t2)
        gi = r["interaction"]
        if "error" not in gi:
            print("interact : gamma_12", fmt(gi["gamma_12"], gi.get("n_obs")))
            print("interact : gamma_23", fmt(gi["gamma_23"], gi.get("n_obs")))
            print("interact : diff   ", fmt(gi["gamma_diff"], gi.get("n_obs")))
        results[tag] = r
        lines.append(f"## {tag}: {iv}(t) -> {dv}(t+1)")
        for k, v in r.items():
            lines.append(f"- {k}: " + (fmt(v.get("gamma", v)) if isinstance(v, dict) else str(v)))
    with open(f"{OUT_DIR}/r37_results.json", "w") as f:
        json.dump(results, f, indent=1)
    with open(f"{OUT_DIR}/r37_report.md", "w", encoding="utf-8") as f:
        f.write("# R3-7 Unconstrained-transition RI-CLPM sensitivity\n\n" + "\n".join(lines))
    print("\nDONE ->", OUT_DIR)

if __name__ == "__main__":
    main()
