#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文4 (DADM / IEEE Access 大修) — 两项预注册新分析 (Kimi 工单执行脚本)
=====================================================================
分析A: DEP_noclick 敏感性 (Bayesian RI-CLPM 三路径) -> 主文 Table 10 / SM Table S27
分析B: baseline_orient 纯基线调节 (三模型)          -> 主文 Table 11 / SM Table S28

数据协议严格镜像主分析脚本 codeact/scripts/bayesian_riclpm_V105B_fullsample.py:
  - 同一数据文件 论文4_数据清洗/kuailive_panel_balanced.csv (20,576 users x 3 waves)
  - 同一子样本: np.random.default_rng(42).choice(unique_users, 5000, replace=False)
  - 用户内去中心化 (RI-CLPM 分解, Mulder & Hamaker 2021, Normal 似然)
  - lag 对: 恰好3波用户, W1->W2 与 W2->W3 (pooled stacked)
  - 预测变量 z 标准化 (mask 后计算 mean/sd); DV (wp 值) 不标准化
  - 先验: 路径系数 Normal(0,0.5), 截距 Normal(0,1), sigma HalfNormal(1)
  - 采样: NUTS, tune=500, draws=1000, chains=2, seed=42
  - BF01 = Savage-Dickey: gaussian_kde(后验)(0) / N(0,0.5).pdf(0)

与主分析的两点差异 (已在回报异常申报中注明):
  1. 工单口径: DEP 分母为 0 记缺失 (NaN); 主分析 new_IDI 分母为 0 记 0.0
  2. 采样器: 本脚本直接调用 numpyro NUTS (主分析经 PyMC 接口调 numpyro 后端;
     算法/调参/链数/种子一致)

baseline_orient: 工单规定"如主分析已有; 无则由执行方用 Wave 1 的 DEP 构造并注明"。
  经查, 主分析无合规的时不变基线取向变量 (旧 h13c_h13d 版为打赏行为全面板
  K-means 后验聚类标签, 工单明确禁止) -> 按工单兜底: baseline_orient = 用户
  Wave 1 的 DEP_full 原始值, 跨用户 z 标准化 (含居中), Wave 1 DEP_full 缺失的
  用户在分析 B 中按 mask 剔除。
"""

import os
import sys
import json
import platform
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

# ── 配置 (与主分析一致) ──
DATA_PATH = "论文4_数据清洗/kuailive_panel_balanced.csv"
N_SUBSAMPLE = 5000
N_TUNE = 500
N_DRAWS = 1000
RANDOM_SEED = 42
OUT_DIR = "paper4_kimi回填_20260917"
os.makedirs(OUT_DIR, exist_ok=True)

CENTER_VARS = ["ln_click", "ln_comment", "DEP_full", "DEP_noclick"]


# ═══════════════════════════════════════════════════════════
# 1. 数据加载与变量构造 (工单 §2 逐字公式)
# ═══════════════════════════════════════════════════════════

def load_and_prepare():
    print("[1] Loading data ...")
    df = pd.read_csv(DATA_PATH)
    n_users_all = df["user_id"].nunique()
    print(f"  Raw: {df.shape[0]} rows, {n_users_all} users, waves={sorted(df.wave.unique())}")

    # ln_* = log1p (工单 §2)
    df["ln_click"] = np.log1p(df["click_count"])
    df["ln_comment"] = np.log1p(df["comment_count"])
    df["ln_like"] = np.log1p(df["like_count"])

    # DEP_full = comment/(click+like+comment), 分母为 0 记缺失 (工单 §2)
    denom_full = df["click_count"] + df["like_count"] + df["comment_count"]
    df["DEP_full"] = np.where(denom_full > 0, df["comment_count"] / denom_full, np.nan)
    # DEP_noclick = comment/(like+comment), 分母为 0 记缺失 (工单 §2)
    denom_nc = df["like_count"] + df["comment_count"]
    df["DEP_noclick"] = np.where(denom_nc > 0, df["comment_count"] / denom_nc, np.nan)

    # 子样本: 与主分析完全同一代码路径 (同 CSV 同 unique 顺序同种子)
    rng = np.random.default_rng(RANDOM_SEED)
    sample_users = rng.choice(df["user_id"].unique(), size=N_SUBSAMPLE, replace=False)
    df = df[df["user_id"].isin(sample_users)].copy()
    print(f"  Subsample: {df['user_id'].nunique()} users, {df.shape[0]} rows (seed={RANDOM_SEED})")

    # baseline_orient = Wave 1 DEP_full (工单兜底构造), 跨用户 z 标准化
    w1 = df[df["wave"] == 1].set_index("user_id")["DEP_full"]
    mu, sd = np.nanmean(w1.values), np.nanstd(w1.values)
    orient_map = ((w1 - mu) / sd).to_dict()  # NaN 保持 NaN
    df["baseline_orient"] = df["user_id"].map(orient_map)
    n_orient_ok = int(np.isfinite(list(orient_map.values())).sum())
    print(f"  baseline_orient = Wave1 DEP_full, z-standardized; 有效用户 {n_orient_ok}/{len(orient_map)}")

    # ── 用户内去中心化 (RI-CLPM 分解; person mean 按非缺失波次计算) ──
    print("[2] Within-person centering ...")
    for v in CENTER_VARS:
        bp = df.groupby("user_id")[v].transform("mean")  # skipna=True (默认)
        df[f"{v}_wp"] = df[v] - bp

    # ── lag 对 (恰好 3 波用户; W1->W2, W2->W3) ──
    print("[3] Lagged dataset ...")
    recs = []
    for uid, grp in df.sort_values(["user_id", "wave"]).groupby("user_id"):
        if len(grp) != 3:
            continue
        grp = grp.sort_values("wave")
        for t_lag, t_cur in [(0, 1), (1, 2)]:
            rec = {"user_id": uid, "lag_pair": f"W{t_lag+1}_W{t_cur+1}",
                   "baseline_orient": grp.iloc[t_lag]["baseline_orient"]}
            for v in CENTER_VARS:
                rec[f"{v}_lag"] = grp.iloc[t_lag][f"{v}_wp"]
                rec[f"{v}_cur"] = grp.iloc[t_cur][f"{v}_wp"]
            recs.append(rec)
    lag_df = pd.DataFrame(recs)
    print(f"  Lagged rows: {lag_df.shape[0]}")
    return df, lag_df


# ═══════════════════════════════════════════════════════════
# 2. 贝叶斯模型 (numpyro NUTS 直调; 设定与主分析一致)
# ═══════════════════════════════════════════════════════════

def _mcmc_sample(model_fn, model_kwargs):
    import jax
    import numpyro
    from numpyro.infer import NUTS, MCMC
    numpyro.set_host_device_count(1)
    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=N_TUNE, num_samples=N_DRAWS,
                num_chains=2, chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(RANDOM_SEED), **model_kwargs)
    return mcmc


def _bf01(posterior_samples, prior_sd=0.5):
    prior_d0 = stats.norm.pdf(0, loc=0, scale=prior_sd)
    s = np.asarray(posterior_samples).flatten()
    if len(s) < 100:
        return np.nan
    try:
        return float(gaussian_kde(s)(0)[0] / prior_d0)
    except Exception:
        return np.nan


def _extract(mcmc, terms, n_obs):
    import arviz as az
    idata = az.from_numpyro(mcmc)
    summ = az.summary(idata, var_names=terms, hdi_prob=0.95)
    out = {"n_obs": n_obs}
    samples = mcmc.get_samples(group_by_chain=False)
    for t in terms:
        out[t] = {
            "mean": float(summ.loc[t, "mean"]),
            "sd": float(summ.loc[t, "sd"]),
            "hdi_lo": float(summ.loc[t, "hdi_2.5%"]),
            "hdi_hi": float(summ.loc[t, "hdi_97.5%"]),
            "rhat": float(summ.loc[t, "r_hat"]),
            "ess_bulk": float(summ.loc[t, "ess_bulk"]),
            "bf01": _bf01(samples[t]),
        }
    return out


def run_single_iv(iv_lag, dv_lag, dv_cur):
    """y_cur = a + gamma*x_lag_z + beta*y_lag_z + e  (镜像主分析 single-IV)"""
    import numpyro
    import numpyro.distributions as dist
    import jax.numpy as jnp

    mask = np.isfinite(iv_lag) & np.isfinite(dv_lag) & np.isfinite(dv_cur)
    x1 = np.asarray(iv_lag[mask], dtype=np.float64)
    x2 = np.asarray(dv_lag[mask], dtype=np.float64)
    y = np.asarray(dv_cur[mask], dtype=np.float64)
    n = len(y)
    if n < 100:
        return {"error": f"Insufficient data: n={n}"}
    x1z = (x1 - x1.mean()) / (x1.std() + 1e-8)
    x2z = (x2 - x2.mean()) / (x2.std() + 1e-8)

    def model(x1z, x2z, y):
        gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.5))
        beta = numpyro.sample("beta", dist.Normal(0.0, 0.5))
        a = numpyro.sample("intercept", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        numpyro.sample("y", dist.Normal(a + gamma * x1z + beta * x2z, sigma), obs=y)

    mcmc = _mcmc_sample(model, {"x1z": jnp.asarray(x1z), "x2z": jnp.asarray(x2z), "y": jnp.asarray(y)})
    return _extract(mcmc, ["gamma", "beta", "intercept", "sigma"], n)


def run_moderation(iv_lag, orient, dv_cur):
    """y_cur = a + g1*x_lag_z + g2*orient_z + g3*(x_lag_z*orient_z) + e  (工单 §4 逐字)"""
    import numpyro
    import numpyro.distributions as dist
    import jax.numpy as jnp

    mask = np.isfinite(iv_lag) & np.isfinite(orient) & np.isfinite(dv_cur)
    x1 = np.asarray(iv_lag[mask], dtype=np.float64)
    x2 = np.asarray(orient[mask], dtype=np.float64)   # 已跨用户 z 标准化
    y = np.asarray(dv_cur[mask], dtype=np.float64)
    n = len(y)
    if n < 100:
        return {"error": f"Insufficient data: n={n}"}
    x1z = (x1 - x1.mean()) / (x1.std() + 1e-8)
    x2z = (x2 - x2.mean()) / (x2.std() + 1e-8)  # mask 内重标准化, 保持先验尺度一致
    inter = x1z * x2z  # 交互 = z 分数乘积 (不再重标准化)

    def model(x1z, x2z, inter, y):
        g1 = numpyro.sample("gamma_main", dist.Normal(0.0, 0.5))
        g2 = numpyro.sample("gamma_orient", dist.Normal(0.0, 0.5))
        g3 = numpyro.sample("gamma_inter", dist.Normal(0.0, 0.5))
        a = numpyro.sample("intercept", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        numpyro.sample("y", dist.Normal(a + g1 * x1z + g2 * x2z + g3 * inter, sigma), obs=y)

    mcmc = _mcmc_sample(model, {"x1z": jnp.asarray(x1z), "x2z": jnp.asarray(x2z),
                                "inter": jnp.asarray(inter), "y": jnp.asarray(y)})
    return _extract(mcmc, ["gamma_main", "gamma_orient", "gamma_inter", "intercept", "sigma"], n)


# ═══════════════════════════════════════════════════════════
# 3. 主流程
# ═══════════════════════════════════════════════════════════

def main():
    import numpyro, arviz, jax
    env = {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "numpyro": numpyro.__version__, "jax": jax.__version__, "arviz": arviz.__version__,
        "sampler": "numpyro NUTS (direct)", "tune": N_TUNE, "draws": N_DRAWS,
        "chains": 2, "seed": RANDOM_SEED,
    }
    print("Environment:", json.dumps(env, ensure_ascii=False))

    df, lag_df = load_and_prepare()
    n_users = df["user_id"].nunique()

    # ── 描述统计: DEP 两版 pooled 相关 + 分波描述 ──
    print("\n[4] DEP descriptives ...")
    both = df[["DEP_full", "DEP_noclick"]].dropna()
    r_p, p_p = stats.pearsonr(both["DEP_full"], both["DEP_noclick"])
    r_s, p_s = stats.spearmanr(both["DEP_full"], both["DEP_noclick"])
    print(f"  pooled Pearson r={r_p:.6f} (N={len(both)}), Spearman rho={r_s:.6f}")

    desc_rows = []
    for w in [1, 2, 3]:
        sub = df[df["wave"] == w]
        for v in ["DEP_full", "DEP_noclick"]:
            x = sub[v].dropna()
            desc_rows.append({"wave": w, "var": v, "N": len(x), "N_missing": int(sub[v].isna().sum()),
                              "mean": x.mean(), "sd": x.std(), "median": x.median(),
                              "min": x.min(), "max": x.max()})
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(os.path.join(OUT_DIR, "paper4_kimi_desc_20260917.csv"), index=False)
    print(desc_df.round(4).to_string(index=False))

    # ── 分析 A: 三路径 ──
    print("\n[5] Analysis A ...")
    A = {}
    print("  A1: ln_comment(t+1) <- ln_comment(t) + DEP_noclick(t)")
    A["A1"] = run_single_iv(lag_df["DEP_noclick_lag"], lag_df["ln_comment_lag"], lag_df["ln_comment_cur"])
    print(f"      N={A['A1'].get('n_obs')}, gamma={A['A1'].get('gamma',{}).get('mean')}")
    print("  A2: DEP_noclick(t+1) <- DEP_noclick(t) + ln_comment(t)")
    A["A2"] = run_single_iv(lag_df["ln_comment_lag"], lag_df["DEP_noclick_lag"], lag_df["DEP_noclick_cur"])
    print(f"      N={A['A2'].get('n_obs')}, gamma={A['A2'].get('gamma',{}).get('mean')}")
    print("  A3: DEP_noclick(t+1) <- DEP_noclick(t) + ln_click(t)")
    A["A3"] = run_single_iv(lag_df["ln_click_lag"], lag_df["DEP_noclick_lag"], lag_df["DEP_noclick_cur"])
    print(f"      N={A['A3'].get('n_obs')}, gamma={A['A3'].get('gamma',{}).get('mean')}")

    # ── 分析 B: 三模型 ──
    print("\n[6] Analysis B ...")
    B = {}
    print("  B1: DEP_full(t+1) <- ln_comment(t) x baseline_orient")
    B["B1"] = run_moderation(lag_df["ln_comment_lag"], lag_df["baseline_orient"], lag_df["DEP_full_cur"])
    print(f"      N={B['B1'].get('n_obs')}")
    print("  B2: DEP_full(t+1) <- ln_click(t) x baseline_orient")
    B["B2"] = run_moderation(lag_df["ln_click_lag"], lag_df["baseline_orient"], lag_df["DEP_full_cur"])
    print(f"      N={B['B2'].get('n_obs')}")
    print("  B3: ln_comment(t+1) <- ln_comment(t) x baseline_orient")
    B["B3"] = run_moderation(lag_df["ln_comment_lag"], lag_df["baseline_orient"], lag_df["ln_comment_cur"])
    print(f"      N={B['B3'].get('n_obs')}")

    # ── 结果 CSV (长表) ──
    rows = []
    for mid, res in {**{k: ("A", v) for k, v in A.items()}, **{k: ("B", v) for k, v in B.items()}}.items():
        grp, r = res
        if "error" in r:
            rows.append({"model": mid, "term": "ERROR", "mean": r["error"]})
            continue
        for t in ["gamma", "beta", "gamma_main", "gamma_orient", "gamma_inter", "intercept", "sigma"]:
            if t in r:
                rows.append({"model": mid, "term": t, **r[t], "n_obs": r["n_obs"]})
    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(os.path.join(OUT_DIR, "paper4_kimi_results_20260917.csv"), index=False)

    # ── 汇总 JSON (供报告生成) ──
    summary = {"env": env, "n_users": int(n_users), "n_rows_panel": int(df.shape[0]),
               "n_lag_rows": int(lag_df.shape[0]),
               "corr": {"pearson_r": float(r_p), "pearson_p": float(p_p),
                        "spearman_rho": float(r_s), "spearman_p": float(p_s),
                        "N_both_finite": int(len(both))},
               "A": A, "B": B}
    with open(os.path.join(OUT_DIR, "paper4_kimi_summary_20260917.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
