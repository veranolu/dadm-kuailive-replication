#!/usr/bin/env python3
"""
Bayesian RI-CLPM Analysis for Paper 4 — Corrected IV/DV (Scheme B)
====================================================================
Runs Bayesian Random-Intercept Cross-Lagged Panel Models with two IV schemes:
  - Scheme 4A: IV = ln(watch_time_total+1) only
  - Scheme 4B: IV = ln(watch_time_total+1) and ln(click_count+1) dual IV
DVs: ln(comment_count+1), ln(like_count+1), new_IDI, ln(gift_count+1), ln(gift_amount_total+1)

Method: Within-person centering (RI-CLPM decomposition) + PyMC NUTS + Normal likelihood
Priors: Normal(0, 0.5) for cross-lagged paths
Output: Cross-lagged coefficients, 95% HDI, Savage-Dickey BF₀₁

Note: All count DVs are log-transformed before within-person centering, consistent
with the standard RI-CLPM approach (Mulder & Hamaker, 2021) which uses Normal
likelihood on the latent within-person components.
"""

import asyncio
import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ── Parameters ──
RESULT_MODE = sys.argv[1] if len(sys.argv) > 1 else "display_only"
N_SUBSAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
N_TUNE = int(sys.argv[3]) if len(sys.argv) > 3 else 500
N_DRAWS = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
RANDOM_SEED = 42

DATA_PATH = "论文4_数据清洗/kuailive_panel_balanced.csv"
OUTPUT_DIR = "codeact/output"
REPORT_DIR = "论文4_数据分析"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# All variables for within-person centering (after transformation)
CENTER_VARS = [
    'ln_watch', 'ln_click',
    'ln_comment', 'ln_like', 'new_IDI',
    'ln_gift_count', 'ln_gift_amount',
]


# ═══════════════════════════════════════════════════════════
# 1. Data Loading & Preprocessing
# ═══════════════════════════════════════════════════════════

def load_and_prepare_data():
    """Load panel data and compute within-person centered variables."""
    print("[1] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw data: {df.shape[0]} rows, {df.shape[1]} cols, {df['user_id'].nunique()} users")

    # ── Transform variables ──
    df['ln_watch'] = np.log1p(df['watch_time_total'])
    df['ln_click'] = np.log1p(df['click_count'])
    df['ln_comment'] = np.log1p(df['comment_count'])
    df['ln_like'] = np.log1p(df['like_count'])
    df['ln_gift_count'] = np.log1p(df['gift_count'])
    df['ln_gift_amount'] = np.log1p(df['gift_amount_total'])

    # new_IDI: comment/(click+like+comment), handle zeros
    denom = df['click_count'] + df['like_count'] + df['comment_count']
    df['new_IDI'] = np.where(denom > 0, df['comment_count'] / denom, 0.0)

    # Subsample for computational tractability
    if N_SUBSAMPLE < df['user_id'].nunique():
        rng = np.random.default_rng(RANDOM_SEED)
        sample_users = rng.choice(df['user_id'].unique(), size=N_SUBSAMPLE, replace=False)
        df = df[df['user_id'].isin(sample_users)].copy()
        print(f"  Subsampled to {N_SUBSAMPLE} users, {df.shape[0]} rows")

    # ── Within-person centering (RI-CLPM decomposition) ──
    print("[2] Computing within-person centered variables...")
    for var in CENTER_VARS:
        if var in df.columns:
            bp = df.groupby('user_id')[var].transform('mean')
            wp = df[var] - bp
            df[f'{var}_bp'] = bp
            df[f'{var}_wp'] = wp
    print(f"  Centering complete. {df.shape[0]} observations")

    # ── Create lagged dataset ──
    print("[3] Creating lagged dataset...")
    df_sorted = df.sort_values(['user_id', 'wave']).copy()

    lagged_records = []
    for uid, grp in df_sorted.groupby('user_id'):
        if len(grp) != 3:
            continue
        grp = grp.sort_values('wave')
        for pair_idx, (t_lag, t_cur) in enumerate([(0, 1), (1, 2)]):
            rec = {'user_id': uid, 'lag_pair': f'W{t_lag+1}_W{t_cur+1}'}
            for v in CENTER_VARS:
                wp_col = f'{v}_wp'
                if wp_col in grp.columns:
                    rec[f'{v}_lag'] = grp.iloc[t_lag][wp_col]
                    rec[f'{v}_cur'] = grp.iloc[t_cur][wp_col]
                else:
                    rec[f'{v}_lag'] = np.nan
                    rec[f'{v}_cur'] = np.nan
            lagged_records.append(rec)

    lag_df = pd.DataFrame(lagged_records)
    print(f"  Lagged dataset: {lag_df.shape[0]} observations, columns: {lag_df.columns.tolist()}")

    return df, lag_df


# ═══════════════════════════════════════════════════════════
# 2. Bayesian Cross-Lagged Models (Normal Likelihood)
# ═══════════════════════════════════════════════════════════

def run_bayesian_clpm_single_iv(iv_lag, dv_lag, dv_cur, iv_label, dv_label,
                                 prior_sd=0.5, n_tune=N_TUNE, n_draws=N_DRAWS):
    """
    Bayesian cross-lagged model with single IV and Normal likelihood.
    Model: dv_cur = intercept + gamma * iv_lag + beta * dv_lag + epsilon
    All inputs are within-person centered values.
    """
    import pymc as pm

    # Clean data
    mask = np.isfinite(iv_lag) & np.isfinite(dv_lag) & np.isfinite(dv_cur)
    x1 = iv_lag[mask].values.astype(np.float64)
    x2 = dv_lag[mask].values.astype(np.float64)
    y = dv_cur[mask].values.astype(np.float64)
    n = len(y)
    if n < 100:
        return {'error': f'Insufficient data: n={n}'}

    # Standardize for numerical stability
    x1m, x1s = x1.mean(), x1.std() + 1e-8
    x2m, x2s = x2.mean(), x2.std() + 1e-8
    x1z = (x1 - x1m) / x1s
    x2z = (x2 - x2m) / x2s

    print(f"  N={n}, IV range=[{x1.min():.3f}, {x1.max():.3f}], DV range=[{y.min():.3f}, {y.max():.3f}]")

    with pm.Model() as model:
        gamma = pm.Normal('gamma', mu=0, sigma=prior_sd)  # cross-lagged
        beta = pm.Normal('beta', mu=0, sigma=prior_sd)    # autoregressive
        intercept = pm.Normal('intercept', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=1)

        mu = intercept + gamma * x1z + beta * x2z
        pm.Normal('y', mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=n_draws, tune=n_tune, chains=2,
            random_seed=RANDOM_SEED, cores=1,
            progressbar=False, return_inferencedata=True,
            nuts_sampler='numpyro'
        )

    return {
        'trace': trace, 'n_obs': n,
        'iv_label': iv_label, 'dv_label': dv_label,
    }


def run_bayesian_clpm_dual_iv(iv1_lag, iv2_lag, dv_lag, dv_cur,
                               iv1_label, iv2_label, dv_label,
                               prior_sd=0.5, n_tune=N_TUNE, n_draws=N_DRAWS):
    """
    Bayesian cross-lagged model with dual IV and Normal likelihood.
    Model: dv_cur = intercept + gamma1 * iv1_lag + gamma2 * iv2_lag + beta * dv_lag + epsilon
    """
    import pymc as pm

    mask = np.isfinite(iv1_lag) & np.isfinite(iv2_lag) & np.isfinite(dv_lag) & np.isfinite(dv_cur)
    x1 = iv1_lag[mask].values.astype(np.float64)
    x2 = iv2_lag[mask].values.astype(np.float64)
    x3 = dv_lag[mask].values.astype(np.float64)
    y = dv_cur[mask].values.astype(np.float64)
    n = len(y)
    if n < 100:
        return {'error': f'Insufficient data: n={n}'}

    x1m, x1s = x1.mean(), x1.std() + 1e-8
    x2m, x2s = x2.mean(), x2.std() + 1e-8
    x3m, x3s = x3.mean(), x3.std() + 1e-8
    x1z = (x1 - x1m) / x1s
    x2z = (x2 - x2m) / x2s
    x3z = (x3 - x3m) / x3s

    print(f"  N={n}, IV1 range=[{x1.min():.3f}, {x1.max():.3f}], DV range=[{y.min():.3f}, {y.max():.3f}]")

    with pm.Model() as model:
        gamma1 = pm.Normal('gamma1', mu=0, sigma=prior_sd)
        gamma2 = pm.Normal('gamma2', mu=0, sigma=prior_sd)
        beta = pm.Normal('beta', mu=0, sigma=prior_sd)
        intercept = pm.Normal('intercept', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=1)

        mu = intercept + gamma1 * x1z + gamma2 * x2z + beta * x3z
        pm.Normal('y', mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=n_draws, tune=n_tune, chains=2,
            random_seed=RANDOM_SEED, cores=1,
            progressbar=False, return_inferencedata=True,
            nuts_sampler='numpyro'
        )

    return {
        'trace': trace, 'n_obs': n,
        'iv1_label': iv1_label, 'iv2_label': iv2_label, 'dv_label': dv_label,
    }


# ═══════════════════════════════════════════════════════════
# 3. Savage-Dickey Bayes Factor & Result Extraction
# ═══════════════════════════════════════════════════════════

def savage_dickey_bf(posterior_samples, prior_sd=0.5):
    """Compute Savage-Dickey BF₀₁ = p(θ=0|data) / p(θ=0|prior)."""
    prior_density_at_zero = stats.norm.pdf(0, loc=0, scale=prior_sd)
    samples = np.asarray(posterior_samples).flatten()
    if len(samples) < 100:
        return np.nan
    try:
        kde = gaussian_kde(samples)
        posterior_density_at_zero = kde(0)[0]
    except Exception:
        return np.nan
    return posterior_density_at_zero / prior_density_at_zero


def interpret_bf(bf_01):
    """Jeffreys (1961) interpretation."""
    if np.isnan(bf_01):
        return "N/A"
    if bf_01 < 1/100: return "Decisive for H₁"
    if bf_01 < 1/30:  return "Very strong for H₁"
    if bf_01 < 1/10:  return "Strong for H₁"
    if bf_01 < 1/3:   return "Substantial for H₁"
    if bf_01 < 3:     return "Anecdotal"
    if bf_01 < 10:    return "Substantial for H₀"
    if bf_01 < 30:    return "Strong for H₀"
    if bf_01 < 100:   return "Very strong for H₀"
    return "Decisive for H₀"


def extract_single_iv(result, prior_sd=0.5):
    """Extract results from single-IV model."""
    if 'error' in result:
        return result
    import arviz as az
    trace = result['trace']

    gamma_s = trace.posterior['gamma'].values.flatten()
    beta_s = trace.posterior['beta'].values.flatten()
    sigma_s = trace.posterior['sigma'].values.flatten()

    g_hdi = az.hdi(gamma_s, hdi_prob=0.95)
    b_hdi = az.hdi(beta_s, hdi_prob=0.95)

    return {
        'scheme': result.get('scheme', '?'),
        'iv_label': result['iv_label'],
        'dv_label': result['dv_label'],
        'n_obs': result['n_obs'],
        'gamma_mean': float(np.mean(gamma_s)),
        'gamma_median': float(np.median(gamma_s)),
        'gamma_hdi_lo': float(g_hdi[0]),
        'gamma_hdi_hi': float(g_hdi[1]),
        'gamma_sd': float(np.std(gamma_s)),
        'gamma_BF01': float(savage_dickey_bf(gamma_s, prior_sd)),
        'beta_mean': float(np.mean(beta_s)),
        'beta_hdi_lo': float(b_hdi[0]),
        'beta_hdi_hi': float(b_hdi[1]),
        'beta_BF01': float(savage_dickey_bf(beta_s, prior_sd)),
        'sigma_mean': float(np.mean(sigma_s)),
        'BF_interp': interpret_bf(savage_dickey_bf(gamma_s, prior_sd)),
    }


def extract_dual_iv(result, prior_sd=0.5):
    """Extract results from dual-IV model."""
    if 'error' in result:
        return result
    import arviz as az
    trace = result['trace']

    g1_s = trace.posterior['gamma1'].values.flatten()
    g2_s = trace.posterior['gamma2'].values.flatten()
    beta_s = trace.posterior['beta'].values.flatten()
    sigma_s = trace.posterior['sigma'].values.flatten()

    g1_hdi = az.hdi(g1_s, hdi_prob=0.95)
    g2_hdi = az.hdi(g2_s, hdi_prob=0.95)
    b_hdi = az.hdi(beta_s, hdi_prob=0.95)

    bf_g1 = savage_dickey_bf(g1_s, prior_sd)
    bf_g2 = savage_dickey_bf(g2_s, prior_sd)

    return {
        'scheme': result.get('scheme', '?'),
        'iv1_label': result['iv1_label'],
        'iv2_label': result['iv2_label'],
        'dv_label': result['dv_label'],
        'n_obs': result['n_obs'],
        'gamma1_mean': float(np.mean(g1_s)),
        'gamma1_hdi_lo': float(g1_hdi[0]),
        'gamma1_hdi_hi': float(g1_hdi[1]),
        'gamma1_sd': float(np.std(g1_s)),
        'gamma1_BF01': float(bf_g1),
        'gamma1_interp': interpret_bf(bf_g1),
        'gamma2_mean': float(np.mean(g2_s)),
        'gamma2_hdi_lo': float(g2_hdi[0]),
        'gamma2_hdi_hi': float(g2_hdi[1]),
        'gamma2_sd': float(np.std(g2_s)),
        'gamma2_BF01': float(bf_g2),
        'gamma2_interp': interpret_bf(bf_g2),
        'beta_mean': float(np.mean(beta_s)),
        'beta_hdi_lo': float(b_hdi[0]),
        'beta_hdi_hi': float(b_hdi[1]),
        'sigma_mean': float(np.mean(sigma_s)),
    }


# ═══════════════════════════════════════════════════════════
# 4. Run All Models
# ═══════════════════════════════════════════════════════════

def run_all_models(lag_df):
    """Run all Bayesian RI-CLPM models for Scheme 4A and 4B."""

    # DVs to test: (variable_base_name, display_label, hypothesis)
    dv_list = [
        ('ln_comment',   'ln(comment+1)',  'H1'),
        ('ln_like',      'ln(like+1)',     'Robust'),
        ('new_IDI',      'new_IDI',        'H2'),
        ('ln_gift_count', 'ln(gift_count+1)', 'H3a'),
        ('ln_gift_amount', 'ln(gift_amount+1)', 'H3b'),
    ]

    results_4A = []
    results_4B = []

    # ═══ Scheme 4A: IV = ln(watch+1) only ═══
    print("\n" + "="*70)
    print("SCHEME 4A: IV = ln(watch_time_total+1) [single IV]")
    print("="*70)

    for dv_var, dv_label, hyp in dv_list:
        iv_lag_col = 'ln_watch_lag'
        dv_lag_col = f'{dv_var}_lag'
        dv_cur_col = f'{dv_var}_cur'

        print(f"\n[4A] {hyp}: ln_watch → {dv_label}")

        missing = [c for c in [iv_lag_col, dv_lag_col, dv_cur_col] if c not in lag_df.columns]
        if missing:
            print(f"  SKIP: missing columns {missing}")
            results_4A.append({'scheme': '4A', 'dv_label': dv_label, 'hypothesis': hyp, 'error': f'missing cols: {missing}'})
            continue

        try:
            result = run_bayesian_clpm_single_iv(
                lag_df[iv_lag_col], lag_df[dv_lag_col], lag_df[dv_cur_col],
                iv_label='ln_watch', dv_label=dv_label,
                prior_sd=0.5, n_tune=N_TUNE, n_draws=N_DRAWS
            )
            result['scheme'] = '4A'
            summary = extract_single_iv(result, prior_sd=0.5)
            summary['hypothesis'] = hyp
            results_4A.append(summary)

            if 'error' not in summary:
                hdi_zero = summary['gamma_hdi_lo'] <= 0 <= summary['gamma_hdi_hi']
                print(f"  γ = {summary['gamma_mean']:+.4f}, "
                      f"95% HDI [{summary['gamma_hdi_lo']:+.4f}, {summary['gamma_hdi_hi']:+.4f}], "
                      f"BF₀₁ = {summary['gamma_BF01']:.2f} ({summary['BF_interp']})")
                print(f"  → HDI {'includes' if hdi_zero else 'excludes'} 0, "
                      f"{'null effect' if hdi_zero else 'significant effect'}")
            else:
                print(f"  ERROR: {summary['error']}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results_4A.append({'scheme': '4A', 'dv_label': dv_label, 'hypothesis': hyp, 'error': str(e)})

    # ═══ Scheme 4B: IV = ln(watch+1) + ln(click+1) ═══
    print("\n" + "="*70)
    print("SCHEME 4B: IV = ln(watch+1) + ln(click+1) [dual IV]")
    print("="*70)

    for dv_var, dv_label, hyp in dv_list:
        iv1_lag_col = 'ln_watch_lag'
        iv2_lag_col = 'ln_click_lag'
        dv_lag_col = f'{dv_var}_lag'
        dv_cur_col = f'{dv_var}_cur'

        print(f"\n[4B] {hyp}: ln_watch + ln_click → {dv_label}")

        missing = [c for c in [iv1_lag_col, iv2_lag_col, dv_lag_col, dv_cur_col] if c not in lag_df.columns]
        if missing:
            print(f"  SKIP: missing columns {missing}")
            results_4B.append({'scheme': '4B', 'dv_label': dv_label, 'hypothesis': hyp, 'error': f'missing cols: {missing}'})
            continue

        try:
            result = run_bayesian_clpm_dual_iv(
                lag_df[iv1_lag_col], lag_df[iv2_lag_col], lag_df[dv_lag_col], lag_df[dv_cur_col],
                iv1_label='ln_watch', iv2_label='ln_click', dv_label=dv_label,
                prior_sd=0.5, n_tune=N_TUNE, n_draws=N_DRAWS
            )
            result['scheme'] = '4B'
            summary = extract_dual_iv(result, prior_sd=0.5)
            summary['hypothesis'] = hyp
            results_4B.append(summary)

            if 'error' not in summary:
                hdi1_zero = summary['gamma1_hdi_lo'] <= 0 <= summary['gamma1_hdi_hi']
                hdi2_zero = summary['gamma2_hdi_lo'] <= 0 <= summary['gamma2_hdi_hi']
                print(f"  γ₁(ln_watch) = {summary['gamma1_mean']:+.4f}, "
                      f"95% HDI [{summary['gamma1_hdi_lo']:+.4f}, {summary['gamma1_hdi_hi']:+.4f}], "
                      f"BF₀₁ = {summary['gamma1_BF01']:.2f} ({summary['gamma1_interp']})")
                print(f"  γ₂(ln_click) = {summary['gamma2_mean']:+.4f}, "
                      f"95% HDI [{summary['gamma2_hdi_lo']:+.4f}, {summary['gamma2_hdi_hi']:+.4f}], "
                      f"BF₀₁ = {summary['gamma2_BF01']:.2f} ({summary['gamma2_interp']})")
            else:
                print(f"  ERROR: {summary['error']}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results_4B.append({'scheme': '4B', 'dv_label': dv_label, 'hypothesis': hyp, 'error': str(e)})

    return results_4A, results_4B


# ═══════════════════════════════════════════════════════════
# 5. Report Generation
# ═══════════════════════════════════════════════════════════

def generate_report(df, results_4A, results_4B):
    """Generate comprehensive markdown report."""

    # Descriptive statistics for original variables
    desc_vars = ['watch_time_total', 'click_count', 'comment_count', 'like_count',
                 'gift_count', 'gift_amount_total', 'new_IDI']
    desc_stats = df[desc_vars].describe().T
    desc_stats['zero_pct'] = (df[desc_vars] == 0).sum() / len(df) * 100

    # ICC computation
    icc_vals = {}
    for var in CENTER_VARS:
        if f'{var}_bp' in df.columns:
            bp_var = df[f'{var}_bp']
            total_var = df[var].var()
            between_var = bp_var.var()
            icc_vals[var] = between_var / total_var if total_var > 0 else 0

    import pymc as _pm

    report = []
    report.append("# 论文4 Bayesian RI-CLPM 分析报告 — 修正IV/DV (方案B)")
    report.append("")
    report.append(f"**生成日期**: 2026-07-13")
    report.append(f"**分析方法**: Bayesian Random-Intercept Cross-Lagged Panel Model (RI-CLPM)")
    report.append(f"**估计方法**: PyMC {_pm.__version__} + NumPyro NUTS sampler")
    report.append(f"**先验设定**: Normal(0, 0.5) for cross-lagged and autoregressive paths")
    report.append(f"**似然函数**: Normal (all DVs log-transformed for count variables)")
    report.append(f"**样本量**: N = {df['user_id'].nunique():,} users × 3 waves (subsample from 20,576)")
    report.append(f"**MCMC设定**: {N_TUNE} tune + {N_DRAWS} draws × 2 chains, seed = {RANDOM_SEED}")
    report.append(f"**Bayes Factor**: Savage-Dickey density ratio")
    report.append("")

    # ── Section 1: Variable Definitions ──
    report.append("## 1. 变量定义 (修正后)")
    report.append("")
    report.append("### 1.1 自变量 (IV)")
    report.append("")
    report.append("| 方案 | IV | 定义 | 说明 |")
    report.append("|------|-----|------|------|")
    report.append("| 4A | ln(watch_time_total+1) | 对数化观看总时长 | 文献共识推荐，避免criterion contamination |")
    report.append("| 4B-1 | ln(watch_time_total+1) | 对数化观看时长 | 参与深度 (engagement depth) |")
    report.append("| 4B-2 | ln(click_count+1) | 对数化点击次数 | 参与频率 (engagement frequency) |")
    report.append("")
    report.append("### 1.2 因变量 (DV)")
    report.append("")
    report.append("| 分类 | DV | 假设 | 定义 | 说明 |")
    report.append("|------|-----|------|------|------|")
    report.append("| 免费互动 | ln(comment_count+1) | H1 | 对数化弹幕次数 | 免费深度互动 |")
    report.append("| 免费互动 | ln(like_count+1) | Robust | 对数化点赞次数 | 匿名行为对照 |")
    report.append("| 信任深化 | new_IDI | H2 | comment/(click+like+comment) | 信任深化指数(去gift) |")
    report.append("| 付费商业 | ln(gift_count+1) | H3a | 对数化送礼次数 | 商业转化频率 |")
    report.append("| 付费商业 | ln(gift_amount_total+1) | H3b | 对数化送礼金额 | 商业转化金额 |")
    report.append("")
    report.append("**关键修正**: ")
    report.append("- V104原PSI复合 = watch + gift + comment → IV含gift导致criterion contamination")
    report.append("- 修正后IV仅含watch/click行为(免费浅层), DV含comment/like/gift(免费深层+付费)")
    report.append("- new_IDI = comment/(click+like+comment), 去除gift的影响")
    report.append("")

    # ── Section 2: Descriptive Statistics ──
    report.append("## 2. 描述性统计")
    report.append("")
    report.append("### 2.1 原始变量")
    report.append("")
    report.append("| 变量 | N | Mean | SD | Min | Median | Max | Zero% |")
    report.append("|------|-----|------|-----|-----|--------|-----|-------|")
    for var in desc_vars:
        if var in desc_stats.index:
            s = desc_stats.loc[var]
            report.append(f"| {var} | {int(s['count']):,} | {s['mean']:.2f} | {s['std']:.2f} | "
                          f"{s['min']:.2f} | {s['50%']:.2f} | {s['max']:.2f} | "
                          f"{desc_stats.loc[var, 'zero_pct']:.1f}% |")
    report.append("")

    report.append("### 2.2 转换后变量ICC")
    report.append("")
    report.append("| 变量 | ICC | 解释 |")
    report.append("|------|-----|------|")
    for var, icc in icc_vals.items():
        interp = "trait-like" if icc > 0.7 else ("moderate" if icc > 0.4 else "state-like")
        report.append(f"| {var} | {icc:.3f} | {interp} |")
    report.append("")
    report.append("ICC > 0.7 表明变量主要受个体间差异驱动(稳定特质)，ICC < 0.4 表明主要受个体内波动驱动(状态)。")
    report.append("PSI(原)的ICC = 0.788(trait-like) vs IDI(原)的ICC = 0.367(state-like) → 解耦的关键统计基础。")
    report.append("")

    # ── Section 3: Scheme 4A Results ──
    report.append("## 3. 方案4A结果: IV = ln(watch_time_total+1) [单一IV]")
    report.append("")
    report.append("Cross-lagged model: W_DV_t = γ × W_IV_(t-1) + β × W_DV_(t-1) + ε")
    report.append("All variables are within-person centered. γ captures the within-person cross-lagged effect.")
    report.append("")
    report.append("| 假设 | DV | γ (cross-lagged) | 95% HDI | SD | BF₀₁ | 证据强度 | AR β | HDI含0? |")
    report.append("|------|-----|-------------------|---------|-----|------|----------|------|---------|")
    for r in results_4A:
        if 'error' in r:
            report.append(f"| {r.get('hypothesis','')} | {r.get('dv_label','')} | ERROR | {r['error']} | - | - | - | - | - |")
        else:
            hdi_zero = r['gamma_hdi_lo'] <= 0 <= r['gamma_hdi_hi']
            report.append(f"| {r['hypothesis']} | {r['dv_label']} | {r['gamma_mean']:+.4f} | "
                          f"[{r['gamma_hdi_lo']:+.4f}, {r['gamma_hdi_hi']:+.4f}] | "
                          f"{r['gamma_sd']:.4f} | {r['gamma_BF01']:.2f} | "
                          f"{r['BF_interp']} | {r['beta_mean']:+.4f} | "
                          f"{'是' if hdi_zero else '否'} |")
    report.append("")

    # ── Section 4: Scheme 4B Results ──
    report.append("## 4. 方案4B结果: IV = ln(watch+1) + ln(click+1) [双IV]")
    report.append("")
    report.append("Cross-lagged model: W_DV_t = γ₁ × W_ln_watch_(t-1) + γ₂ × W_ln_click_(t-1) + β × W_DV_(t-1) + ε")
    report.append("")
    report.append("| 假设 | DV | γ₁(ln_watch) | γ₁ 95% HDI | γ₁ BF₀₁ | γ₂(ln_click) | γ₂ 95% HDI | γ₂ BF₀₁ | AR β |")
    report.append("|------|-----|---------------|------------|----------|---------------|------------|----------|------|")
    for r in results_4B:
        if 'error' in r:
            report.append(f"| {r.get('hypothesis','')} | {r.get('dv_label','')} | ERROR | {r.get('error','')} | - | - | - | - | - |")
        else:
            report.append(f"| {r['hypothesis']} | {r['dv_label']} | {r['gamma1_mean']:+.4f} | "
                          f"[{r['gamma1_hdi_lo']:+.4f}, {r['gamma1_hdi_hi']:+.4f}] | "
                          f"{r['gamma1_BF01']:.2f} | "
                          f"{r['gamma2_mean']:+.4f} | "
                          f"[{r['gamma2_hdi_lo']:+.4f}, {r['gamma2_hdi_hi']:+.4f}] | "
                          f"{r['gamma2_BF01']:.2f} | "
                          f"{r['beta_mean']:+.4f} |")
    report.append("")

    # ── Section 5: Comparison Table ──
    report.append("## 5. 方案4A vs 4B对比表")
    report.append("")
    report.append("| DV | 4A: γ(watch) | 4A BF₀₁ | 4B: γ₁(watch) | 4B γ₁ BF₀₁ | 4B: γ₂(click) | 4B γ₂ BF₀₁ | 方向一致? |")
    report.append("|-----|-------------|----------|----------------|-------------|----------------|-------------|-----------|")
    for r4a in results_4A:
        dv = r4a.get('dv_label', '')
        r4b = next((r for r in results_4B if r.get('dv_label') == dv and 'error' not in r), None)
        if 'error' in r4a or r4b is None:
            report.append(f"| {dv} | - | - | - | - | - | - | - |")
            continue
        g4a = r4a['gamma_mean']
        g4b1 = r4b['gamma1_mean']
        consistent = "✓" if (g4a * g4b1 >= 0) else "✗"
        report.append(f"| {dv} | {g4a:+.4f} | {r4a['gamma_BF01']:.2f} | "
                      f"{g4b1:+.4f} | {r4b['gamma1_BF01']:.2f} | "
                      f"{r4b['gamma2_mean']:+.4f} | {r4b['gamma2_BF01']:.2f} | {consistent} |")
    report.append("")

    # ── Section 6: Comparison with V104 ──
    report.append("## 6. 与V104结果对比")
    report.append("")
    report.append("| 路径 | V104结果 | 4A结果 | 4B结果 | 一致性 |")
    report.append("|------|----------|--------|--------|--------|")

    # V104 key findings (from the paper)
    v104_findings = {
        'PSI→IDI': ('β=+0.007, null, BF₀₁=85.10', 'decisive null'),
        'PSI→deep_count': ('NB: null, BF₀₁=9.95', 'strong null'),
        'PSI→gift_price (direct)': ('β=−0.471, p=.002', 'negative significant'),
        'deep_count→gift_price': ('β=+0.573, p=.113', 'positive ns'),
    }

    # Map V104 paths to current DVs
    # PSI→IDI → watch→new_IDI
    idi_4a = next((r for r in results_4A if r.get('dv_label') == 'new_IDI' and 'error' not in r), None)
    idi_4b = next((r for r in results_4B if r.get('dv_label') == 'new_IDI' and 'error' not in r), None)

    if idi_4a:
        v104_idi = v104_findings['PSI→IDI'][0]
        r4a_str = f"γ={idi_4a['gamma_mean']:+.4f}, BF₀₁={idi_4a['gamma_BF01']:.2f}"
        r4b_str = ""
        if idi_4b:
            r4b_str = f"γ₁={idi_4b['gamma1_mean']:+.4f}(BF₀₁={idi_4b['gamma1_BF01']:.2f}), γ₂={idi_4b['gamma2_mean']:+.4f}(BF₀₁={idi_4b['gamma2_BF01']:.2f})"
        # Consistency: V104 was null, 4A is null if BF01 > 3
        consist_4a = "一致(null)" if idi_4a['gamma_BF01'] > 3 else "不一致"
        report.append(f"| watch→new_IDI | {v104_idi} | {r4a_str} | {r4b_str} | {consist_4a} |")

    # PSI→deep_count → watch→ln_comment (partial, since deep_count in V104 included gift)
    comment_4a = next((r for r in results_4A if r.get('dv_label') == 'ln(comment+1)' and 'error' not in r), None)
    if comment_4a:
        v104_dc = v104_findings['PSI→deep_count'][0]
        r4a_str = f"γ={comment_4a['gamma_mean']:+.4f}, BF₀₁={comment_4a['gamma_BF01']:.2f}"
        comment_4b = next((r for r in results_4B if r.get('dv_label') == 'ln(comment+1)' and 'error' not in r), None)
        r4b_str = ""
        if comment_4b:
            r4b_str = f"γ₁={comment_4b['gamma1_mean']:+.4f}(BF₀₁={comment_4b['gamma1_BF01']:.2f}), γ₂={comment_4b['gamma2_mean']:+.4f}(BF₀₁={comment_4b['gamma2_BF01']:.2f})"
        report.append(f"| watch→ln_comment | {v104_dc} (V104含gift) | {r4a_str} | {r4b_str} | 不可直接比较 |")

    # PSI→gift_price → watch→ln_gift_amount
    gift_4a = next((r for r in results_4A if r.get('dv_label') == 'ln(gift_amount+1)' and 'error' not in r), None)
    if gift_4a:
        v104_gift = v104_findings['PSI→gift_price (direct)'][0]
        r4a_str = f"γ={gift_4a['gamma_mean']:+.4f}, BF₀₁={gift_4a['gamma_BF01']:.2f}"
        gift_4b = next((r for r in results_4B if r.get('dv_label') == 'ln(gift_amount+1)' and 'error' not in r), None)
        r4b_str = ""
        if gift_4b:
            r4b_str = f"γ₁={gift_4b['gamma1_mean']:+.4f}(BF₀₁={gift_4b['gamma1_BF01']:.2f}), γ₂={gift_4b['gamma2_mean']:+.4f}(BF₀₁={gift_4b['gamma2_BF01']:.2f})"
        consist = "方向一致(负)" if gift_4a['gamma_mean'] < 0 else "方向不一致"
        report.append(f"| watch→ln_gift_amount | {v104_gift} | {r4a_str} | {r4b_str} | {consist} |")

    report.append("")
    report.append("**关键对比说明**:")
    report.append("- V104的PSI含gift，当前IV(watch)不含gift → IV→gift DV不再有criterion contamination")
    report.append("- V104的deep_count含gift，当前用comment_count(不含gift) → 不可直接数值对比")
    report.append("- V104 PSI→IDI null (BF₀₁=85.10) → 当前应检验watch→new_IDI是否仍为null")
    report.append("")

    # ── Section 7: Hypothesis Testing Summary ──
    report.append("## 7. 假设检验总结")
    report.append("")

    for r in results_4A:
        if 'error' in r:
            continue
        hyp = r.get('hypothesis', '?')
        dv = r['dv_label']
        g = r['gamma_mean']
        hdi_lo, hdi_hi = r['gamma_hdi_lo'], r['gamma_hdi_hi']
        bf = r['gamma_BF01']
        hdi_zero = hdi_lo <= 0 <= hdi_hi
        direction = "正向" if g > 0 else "负向" if g < 0 else "零"

        if bf > 30:
            evidence = f"强null证据(BF₀₁={bf:.1f})"
            conclusion = f"{direction}效应不存在(decisive null)"
        elif bf > 10:
            evidence = f"较强null证据(BF₀₁={bf:.1f})"
            conclusion = f"{direction}效应不存在(strong null)"
        elif bf > 3:
            evidence = f"中等null证据(BF₀₁={bf:.1f})"
            conclusion = f"{direction}效应不显著(substantial null)"
        elif bf > 1/3:
            evidence = f"证据不明确(BF₀₁={bf:.2f})"
            conclusion = f"数据不足以判断"
        else:
            evidence = f"支持H₁(BF₀₁={bf:.2f})"
            conclusion = f"{direction}效应存在"

        report.append(f"### {hyp}: watch → {dv}")
        report.append(f"- 4A: γ = {g:+.4f}, 95% HDI [{hdi_lo:+.4f}, {hdi_hi:+.4f}], {evidence}")
        report.append(f"- **结论**: {conclusion}")
        report.append("")

    # ── Section 8: Method ──
    report.append("## 8. 方法论说明")
    report.append("")
    report.append("### 8.1 RI-CLPM分解方法")
    report.append("- 随机截距(RI): 个体均值 = groupby(user_id).mean()")
    report.append("- 个体内偏差(WP): 观测值 - 个体均值")
    report.append("- 交叉滞后回归: W_DV_t = γ × W_IV_(t-1) + β × W_DV_(t-1) + ε")
    report.append("- 汇合W1→W2和W2→W3两个lag pair (stacked lagged regression)")
    report.append("")
    report.append("### 8.2 Bayesian估计")
    report.append(f"- PyMC {_pm.__version__} + NumPyro NUTS backend")
    report.append("- Cross-lagged priors: Normal(0, 0.5)")
    report.append("- Autoregressive priors: Normal(0, 0.5)")
    report.append("- Residual SD: HalfNormal(1)")
    report.append(f"- MCMC: {N_TUNE} tune + {N_DRAWS} draws × 2 chains")
    report.append(f"- Subsample: N = {N_SUBSAMPLE:,} users (from 20,576)")
    report.append("- 所有计数DV经ln(x+1)变换后使用Normal似然 (标准RI-CLPM框架)")
    report.append("")
    report.append("### 8.3 Bayes Factor计算")
    report.append("- Savage-Dickey density ratio: BF₀₁ = p(θ=0|data) / p(θ=0|prior)")
    report.append("- Prior: Normal(0, 0.5), density at 0 ≈ 0.798")
    report.append("- Posterior: Gaussian KDE估计")
    report.append("- Jeffreys (1961): BF₀₁ > 3 = substantial H₀, > 10 = strong H₀, > 30 = very strong, > 100 = decisive")
    report.append("")
    report.append("### 8.4 Criterion Contamination修正")
    report.append("- V104原PSI复合 = watch + gift + comment → IV含gift")
    report.append("- 当IV含gift时, PSI→gift_count/gift_amount存在逻辑循环 (criterion contamination)")
    report.append("- 修正后IV仅含watch/click (免费浅层行为), 不含任何付费行为")
    report.append("- 这使得watch→gift路径的检验无criterion contamination, 结论更有说服力")
    report.append("")
    report.append("### 8.5 与V104的关键差异")
    report.append("| 维度 | V104 | 修正后(V105B) |")
    report.append("|------|------|---------------|")
    report.append("| IV | PSI = watch + gift + comment | 4A: watch only; 4B: ln(watch)+ln(click) |")
    report.append("| DV-deep | deep_count = comment + gift | comment_count only (去gift) |")
    report.append("| DV-IDI | deep/shallow | comment/(click+like+comment) |")
    report.append("| DV-gift | gift_price (direct+mediated) | gift_count, gift_amount (无criterion contamination) |")
    report.append("| NB模型 | Bayesian NB RI-CLPM | Normal on log-transformed (标准RI-CLPM) |")
    report.append("")

    # ── Save ──
    report_text = "\n".join(report)

    # Save to both locations
    for path in [
        os.path.join(REPORT_DIR, "bayesian_riclpm_V105B_results.md"),
        os.path.join(OUTPUT_DIR, "bayesian_riclpm_V105B_results.md"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[Report] Saved to {path}")

    # Save structured JSON
    json_path = os.path.join(OUTPUT_DIR, "bayesian_riclpm_V105B_results.json")
    all_results = {
        'scheme_4A': [r for r in results_4A if 'error' not in r],
        'scheme_4B': [r for r in results_4B if 'error' not in r],
        'meta': {
            'n_subsample': N_SUBSAMPLE,
            'n_tune': N_TUNE,
            'n_draws': N_DRAWS,
            'prior_sd': 0.5,
            'seed': RANDOM_SEED,
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[JSON] Saved to {json_path}")

    return report_text, os.path.join(REPORT_DIR, "bayesian_riclpm_V105B_results.md")


# ═══════════════════════════════════════════════════════════
# 6. Async Main
# ═══════════════════════════════════════════════════════════

async def main():
    from codeact_sdk import CodeActSDK

    print("="*70)
    print("Bayesian RI-CLPM Analysis — Paper 4 Corrected IV/DV (Scheme B)")
    print(f"Parameters: N_SUBSAMPLE={N_SUBSAMPLE}, N_TUNE={N_TUNE}, N_DRAWS={N_DRAWS}")
    print("="*70)

    sdk = CodeActSDK()

    try:
        # 1. Load and prepare data
        df, lag_df = load_and_prepare_data()

        # 2. Run all Bayesian models
        results_4A, results_4B = run_all_models(lag_df)

        # 3. Generate report
        report_text, report_path = generate_report(df, results_4A, results_4B)

        # 4. Build summary message
        key_findings = []
        for r in results_4A:
            if 'error' not in r and 'gamma_BF01' in r:
                key_findings.append(
                    f"4A {r.get('hypothesis','')}: watch→{r['dv_label']} "
                    f"γ={r['gamma_mean']:+.4f}, BF₀₁={r['gamma_BF01']:.2f} ({r['BF_interp']})"
                )
        for r in results_4B:
            if 'error' not in r and 'gamma1_BF01' in r:
                key_findings.append(
                    f"4B {r.get('hypothesis','')}: watch→{r['dv_label']} "
                    f"γ₁={r['gamma1_mean']:+.4f}(BF₀₁={r['gamma1_BF01']:.2f}), "
                    f"click→{r['dv_label']} γ₂={r['gamma2_mean']:+.4f}(BF₀₁={r['gamma2_BF01']:.2f})"
                )

        abs_report_path = os.path.abspath(report_path)
        summary = "\n".join([
            "Bayesian RI-CLPM分析完成 (修正IV/DV, 方案B)",
            "",
            "核心发现:",
            *[f"- {f}" for f in key_findings[:8]],
            "",
            f"完整报告: [bayesian_riclpm_V105B_results.md](computer://{abs_report_path})",
        ])

        actual_mode = RESULT_MODE if RESULT_MODE != "auto" else "display_only"
        await sdk.submit_result(
            result_mode=actual_mode,
            status="success",
            message=summary,
            data={
                "report_path": report_path,
                "n_subsample": N_SUBSAMPLE,
                "n_results_4A": len([r for r in results_4A if 'error' not in r]),
                "n_results_4B": len([r for r in results_4B if 'error' not in r]),
            }
        )

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        await sdk.submit_result(
            result_mode="notify",
            status="error",
            message=f"Bayesian RI-CLPM分析执行失败: {str(e)[:200]}",
            data={"error_type": type(e).__name__}
        )


if __name__ == "__main__":
    asyncio.run(main())
