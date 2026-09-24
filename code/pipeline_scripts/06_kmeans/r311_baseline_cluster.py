# -*- coding: utf-8 -*-
"""
R3-11: baseline-only orientation clustering (G MASTER ORDER PHASE 5 / P0)
=========================================================================
Reviewer R3.11 原话: "consider defining user orientation from baseline-only
behavior or non-outcome variables" -> 明确要求重新分类, 现 Wave-1 DEP
baseline-composition 不得冒充。
步骤:
  1) 全期 EA2 分类重建: K-means(K=2) on standardized (mean log gift amount,
     SD log gift amount, CV of gift amounts), user-level, 全 21 天 gift.csv;
     验证是否复现 V4.1 的 emotional n=5,249 / instrumental n=15,327
  2) W1 baseline-only: 同特征仅用 W1 (2025-05-05~05-11) gift 记录; 覆盖率报告
  3) 两版标签一致性 (ARI / Cohen's kappa / 交叉表)
  4) baseline-only orientation 的 moderation 敏感性 (镜像分析B 交互模型):
     DEP_full(t+1) ~ ln_comment(t) x orient_W1 ; ~ ln_click(t) x orient_W1 ;
     ln_comment(t+1) ~ ln_comment(t) x orient_W1
留账: 开工 2026-09-22 / 输出 out/ 下 csv+json+md
"""
import os, json, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

GIFT = "/Coze/Drive/扣子/所有对话/主对话/KuaiLive/KuaiLive/gift.csv"
PANEL = "/Coze/Drive/扣子/所有对话/主对话/论文4_数据清洗/kuailive_panel_balanced.csv"
OUT_DIR = "/Coze/Drive/班主任/论文4_大修_20260922/R3-11_baseline_cluster/out"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_SEED, N_TUNE, N_DRAWS, N_SUBSAMPLE = 42, 500, 1000, 5000
W1_START, W1_END = "2025-05-05", "2025-05-12"  # [start, end)

# ---------- 1. user 级 gift 特征 ----------
def gift_features(df_g):
    """df_g: gift 记录子集 -> user 级 (mean_log, sd_log, cv)"""
    df_g = df_g.copy()
    df_g["ln_price"] = np.log(df_g["gift_price"].astype(float))
    agg = df_g.groupby("user_id").agg(
        n_gifts=("gift_price", "size"),
        mean_log=("ln_price", "mean"),
        sd_log=("ln_price", "std"),
        mean_amt=("gift_price", "mean"),
        sd_amt=("gift_price", "std"),
    )
    agg["cv_amt"] = agg["sd_amt"] / agg["mean_amt"]
    # 单笔 gift 用户: sd_log/sd_amt/cv 为 NaN -> 记 0 (无变异) 并标注
    agg["single_gift"] = (agg["n_gifts"] == 1).astype(int)
    agg[["sd_log", "sd_amt", "cv_amt"]] = agg[["sd_log", "sd_amt", "cv_amt"]].fillna(0.0)
    return agg

def kmeans2(feat):
    X = StandardScaler().fit_transform(feat[["mean_log", "sd_log", "cv_amt"]])
    km = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10).fit(X)
    lab = pd.Series(km.labels_, index=feat.index)
    # emotional = 高 mean_log 的簇 (与 V4.1 语义对齐: emotional~高金额)
    mean_by = feat.groupby(lab)["mean_log"].mean()
    emo_cluster = mean_by.idxmax()
    return (lab == emo_cluster).map({True: "emotional", False: "instrumental"}), km

def main():
    g = pd.read_csv(GIFT)
    g["ts"] = pd.to_datetime(g["timestamp"], unit="ms")
    panel_users = pd.read_csv(PANEL, usecols=["user_id"]).user_id.unique()
    g = g[g.user_id.isin(panel_users)]
    out = {}

    # ===== 1) 全期 EA2 重建验证 =====
    feat_all = gift_features(g)
    lab_all, km_all = kmeans2(feat_all)
    n_emo = int((lab_all == "emotional").sum())
    out["full_period"] = {
        "n_users": int(len(lab_all)),
        "n_emotional": n_emo,
        "n_instrumental": int((lab_all == "instrumental").sum()),
        "target_emotional": 5249, "target_instrumental": 15327,
        "n_gift_users": int(feat_all.shape[0]),
        "single_gift_share": float(feat_all.single_gift.mean()),
    }
    print(f"[1] 全期重建: emotional={n_emo} (V4.1=5,249), instrumental={out['full_period']['n_instrumental']} (V4.1=15,327)")

    # ===== 2) W1 baseline-only =====
    w1 = g[(g.ts >= W1_START) & (g.ts < W1_END)]
    feat_w1 = gift_features(w1)
    lab_w1, km_w1 = kmeans2(feat_w1)
    cover = len(lab_w1) / len(panel_users)
    out["w1_baseline"] = {
        "n_classifiable": int(len(lab_w1)), "n_panel": int(len(panel_users)),
        "coverage": float(cover),
        "n_emotional": int((lab_w1 == "emotional").sum()),
        "single_gift_share": float(feat_w1.single_gift.mean()),
        "ge2_gift_share": float((feat_w1.n_gifts >= 2).mean()),
    }
    print(f"[2] W1: 可分类 {len(lab_w1)}/{len(panel_users)} = {cover:.2%}; single-gift share={feat_w1.single_gift.mean():.2%}; >=2 gift={out['w1_baseline']['ge2_gift_share']:.2%}")

    # ===== 3) 一致性 =====
    common = lab_all.index.intersection(lab_w1.index)
    ari = adjusted_rand_score(lab_all[common], lab_w1[common])
    kap = cohen_kappa_score(lab_all[common], lab_w1[common])
    ct = pd.crosstab(lab_all[common], lab_w1[common], normalize="index").round(4)
    agree = float((lab_all[common] == lab_w1[common]).mean())
    out["agreement"] = {"n_common": int(len(common)), "ARI": float(ari),
                        "kappa": float(kap), "raw_agree": agree,
                        "crosstab_row_normalized": ct.to_dict()}
    print(f"[3] 一致性 (n={len(common)}): ARI={ari:.3f}, kappa={kap:.3f}, agree={agree:.2%}")
    print(ct.to_string())

    # 保存标签
    tags = pd.DataFrame({"user_id": common,
                         "orient_full": lab_all[common].values,
                         "orient_w1": lab_w1[common].values})
    tags.to_csv(f"{OUT_DIR}/orientation_labels.csv", index=False)

    # ===== 4) baseline-only moderation 敏感性 (Bayesian, 镜像分析B) =====
    df = pd.read_csv(PANEL)
    df["ln_click"] = np.log1p(df["click_count"]); df["ln_comment"] = np.log1p(df["comment_count"])
    denom = df["click_count"] + df["like_count"] + df["comment_count"]
    df["DEP_full"] = np.where(denom > 0, df["comment_count"] / denom, np.nan)
    rng = np.random.default_rng(RANDOM_SEED)
    sub = rng.choice(df["user_id"].unique(), size=N_SUBSAMPLE, replace=False)
    df = df[df.user_id.isin(sub)].copy()
    # W1 orientation (emotional=1), 跨用户 z; 不可分类用户 -> NaN -> mask 剔除
    emap = (lab_w1 == "emotional").astype(float)
    mu, sd = emap.mean(), emap.std()
    df["orient_w1"] = df.user_id.map(((emap - mu) / sd).to_dict())
    CENTER = ["ln_click", "ln_comment", "DEP_full"]
    for v in CENTER:
        df[f"{v}_wp"] = df[v] - df.groupby("user_id")[v].transform("mean")
    recs = []
    for uid, grp in df.sort_values(["user_id", "wave"]).groupby("user_id"):
        if len(grp) != 3: continue
        grp = grp.sort_values("wave")
        for a, b in [(0, 1), (1, 2)]:
            r = {"user_id": uid, "orient_w1": grp.iloc[a]["orient_w1"]}
            for v in CENTER:
                r[f"{v}_lag"] = grp.iloc[a][f"{v}_wp"]; r[f"{v}_cur"] = grp.iloc[b][f"{v}_wp"]
            recs.append(r)
    lag = pd.DataFrame(recs)

    import numpyro, numpyro.distributions as dist, jax, jax.numpy as jnp, arviz as az
    from numpyro.infer import NUTS, MCMC
    numpyro.set_host_device_count(1)
    def bf01(s, psd=0.5):
        try: return float(gaussian_kde(np.asarray(s).flatten())(0)[0] / stats.norm.pdf(0, 0, psd))
        except Exception: return np.nan
    def run_mod(iv_lag, orient, dv_cur):
        mask = np.isfinite(iv_lag) & np.isfinite(orient) & np.isfinite(dv_cur)
        x1 = np.asarray(iv_lag[mask], float); x2 = np.asarray(orient[mask], float)
        y = np.asarray(dv_cur[mask], float); n = len(y)
        if n < 100: return {"error": f"n={n}"}
        x1z = (x1 - x1.mean()) / (x1.std() + 1e-8); x2z = (x2 - x2.mean()) / (x2.std() + 1e-8)
        def model(x1z, x2z, y):
            g1 = numpyro.sample("gamma_main", dist.Normal(0., 0.5))
            g2 = numpyro.sample("gamma_orient", dist.Normal(0., 0.5))
            g3 = numpyro.sample("gamma_inter", dist.Normal(0., 0.5))
            a = numpyro.sample("intercept", dist.Normal(0., 1.))
            s = numpyro.sample("sigma", dist.HalfNormal(1.))
            numpyro.sample("y", dist.Normal(a + g1 * x1z + g2 * x2z + g3 * x1z * x2z, s), obs=y)
        mcmc = MCMC(NUTS(model), num_warmup=N_TUNE, num_samples=N_DRAWS, num_chains=2,
                    chain_method="sequential", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(RANDOM_SEED), x1z=jnp.asarray(x1z), x2z=jnp.asarray(x2z), y=jnp.asarray(y))
        idata = az.from_numpyro(mcmc)
        summ = az.summary(idata, var_names=["gamma_main", "gamma_orient", "gamma_inter"], hdi_prob=0.95)
        smp = mcmc.get_samples(group_by_chain=False)
        o = {"n_obs": n}
        for t in ["gamma_main", "gamma_orient", "gamma_inter"]:
            o[t] = {"mean": float(summ.loc[t, "mean"]), "sd": float(summ.loc[t, "sd"]),
                    "hdi_lo": float(summ.loc[t, "hdi_2.5%"]), "hdi_hi": float(summ.loc[t, "hdi_97.5%"]),
                    "rhat": float(summ.loc[t, "r_hat"]), "ess": float(summ.loc[t, "ess_bulk"]),
                    "bf01": bf01(smp[t])}
        return o
    mod = {}
    for name, iv, dv in [("M1_DEP_comment", "ln_comment", "DEP_full"),
                          ("M2_DEP_click", "ln_click", "DEP_full"),
                          ("M3_comment_comment", "ln_comment", "ln_comment")]:
        print(f"[4] {name} ...")
        mod[name] = run_mod(lag[f"{iv}_lag"].values, lag["orient_w1"].values, lag[f"{dv}_cur"].values)
        gi = mod[name].get("gamma_inter", {})
        if gi: print(f"    交互: mean={gi['mean']:+.4f} HDI=[{gi['hdi_lo']:+.4f},{gi['hdi_hi']:+.4f}] BF01={gi['bf01']:.2f} N={mod[name]['n_obs']}")
    out["moderation_w1"] = mod

    with open(f"{OUT_DIR}/r311_results.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("\nDONE ->", OUT_DIR)

if __name__ == "__main__":
    main()
