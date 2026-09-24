# -*- coding: utf-8 -*-
"""
论文4 DADM — 两项预注册新分析（Table 10/11 + Table S27/S28 回填用）
依赖: pandas numpy statsmodels  （PowerShell 里: pip install pandas numpy statsmodels）
用法:  python 论文4_预注册分析_runner.py 你的面板数据.csv
结果:  同目录生成 论文4_新分析结果.txt
"""
import sys, numpy as np, pandas as pd
import statsmodels.formula.api as smf

CSV = sys.argv[1] if len(sys.argv) > 1 else "panel.csv"
OUT = "论文4_新分析结果.txt"

# ============ 列名映射：改成你数据里的真实列名 ============
COL_USER    = "user_id"        # 用户ID
COL_WAVE    = "wave"           # 波次 1/2/3
COL_CLICK   = "click_count"    # 该波点击数
COL_LIKE    = "like_count"     # 该波点赞数
COL_COMMENT = "comment_count"  # 该波评论数
COL_WATCH   = "watch_time"     # 该波观看时长(秒)；没有就设为 None
COL_ORIENT  = "orientation"    # 基线(Wave 1)取向得分；只取 wave==1 的值
# 若没有现成 orientation 列，脚本会用 Wave 1 的 DEP 自动构造 baseline orientation
# =========================================================

df = pd.read_csv(CSV)
lines = []
def log(s=""):
    lines.append(str(s)); print(s)

# ---- 构造变量 ----
df["DEP_full"]   = df[COL_COMMENT] / (df[COL_CLICK] + df[COL_LIKE] + df[COL_COMMENT]).replace(0, np.nan)
df["DEP_noclick"]= df[COL_COMMENT] / (df[COL_LIKE] + df[COL_COMMENT]).replace(0, np.nan)
df["ln_comment"] = np.log1p(df[COL_COMMENT])
df["ln_click"]   = np.log1p(df[COL_CLICK])
if COL_WATCH and COL_WATCH in df.columns:
    df["ln_watch"] = np.log1p(df[COL_WATCH])
else:
    df["ln_watch"] = np.nan

if COL_ORIENT not in df.columns:
    base = df[df[COL_WAVE] == 1][[COL_USER, "DEP_full"]].rename(columns={"DEP_full": "baseline_orient"})
    df = df.merge(base, on=COL_USER, how="left")
    COL_ORIENT = "baseline_orient"
else:
    base = df[df[COL_WAVE] == 1][[COL_USER, COL_ORIENT]].rename(columns={COL_ORIENT: "baseline_orient"})
    df = df.merge(base, on=COL_USER, how="left")
    COL_ORIENT = "baseline_orient"

log(f"N users = {df[COL_USER].nunique()}, rows = {len(df)}")
log("\n== DEP 描述（原版 vs 无点击版）==")
log(df[[COL_WAVE, "DEP_full", "DEP_noclick"]].groupby(COL_WAVE).describe().to_string())
log(f"\n两版 DEP 相关（ pooled ）: r = {df['DEP_full'].corr(df['DEP_noclick']):.4f}")

# ---- 工具：用户内去均值（within-person）----
def demean(d, cols):
    d = d.copy()
    for c in cols:
        d[c + "_w"] = d[c] - d.groupby(COL_USER)[c].transform("mean")
    return d

def lag_panel(d, y, xs):
    """构造 t -> t+1 滞后对，只保留同一用户相邻波"""
    d = d.sort_values([COL_USER, COL_WAVE]).copy()
    for c in [y] + xs:
        d[c + "_lag"] = d.groupby(COL_USER)[c].shift(1)
        d["wave_lag"] = d.groupby(COL_USER)[COL_WAVE].shift(1)
    d = d[d[COL_WAVE] == d["wave_lag"] + 1]
    return d

# ============ 分析1：无点击 DEP 敏感性（Table S27 / 主文 Table 10）============
log("\n" + "=" * 60)
log("分析1：DEP(-click) = comment/(like+comment) 敏感性")
log("=" * 60)
specs = [
    ("H1fwd : ln_comment(t+1) ~ ln_comment(t) + DEP_noclick(t)", "ln_comment", ["ln_comment", "DEP_noclick"]),
    ("H2fwd : DEP_noclick(t+1) ~ DEP_noclick(t) + ln_comment(t)", "DEP_noclick", ["DEP_noclick", "ln_comment"]),
    ("H2fwd2: DEP_noclick(t+1) ~ DEP_noclick(t) + ln_click(t)", "DEP_noclick", ["DEP_noclick", "ln_click"]),
]
res1 = []
d0 = df.dropna(subset=["DEP_noclick"])
for name, y, xs in specs:
    dl = lag_panel(d0, y, xs)
    dl = demean(dl, [y] + [x + "_lag" for x in xs])
    f = y + "_w ~ " + " + ".join(x + "_lag_w" for x in xs)
    m = smf.ols(f, data=dl).fit(cov_type="cluster", cov_kwds={"groups": dl[COL_USER]})
    log(f"\n[{name}]  N pairs = {int(m.nobs)}")
    log(m.params.round(5).to_string())
    log("SE: " + ", ".join(f"{v:.5f}" for v in m.bse.values))
    log("p : " + ", ".join(f"{v:.4f}" for v in m.pvalues.values))
    res1.append((name, m))

# ============ 分析2：纯基线取向调节（Table S28 / 主文 Table 11）============
log("\n" + "=" * 60)
log("分析2：baseline orientation (Wave 1 only) 调节滞后路径")
log("=" * 60)
res2 = []
d0 = df.dropna(subset=["DEP_full", COL_ORIENT]).copy()
d0["orient_c"] = d0[COL_ORIENT] - d0[COL_ORIENT].mean()
for y, vol in [("DEP_full", "ln_comment"), ("DEP_full", "ln_click"), ("ln_comment", "ln_comment")]:
    dl = lag_panel(d0, y, [vol, "orient_c"])
    dl["interact"] = dl[vol + "_lag"] * dl["orient_c"]  # 基线取向为时不变量，用原始交互
    dl = demean(dl, [y, vol + "_lag"])
    f = y + "_w ~ " + vol + "_lag_w + orient_c + interact"
    m = smf.ols(f, data=dl).fit(cov_type="cluster", cov_kwds={"groups": dl[COL_USER]})
    log(f"\n[{y}(t+1) ~ {vol}(t) x baseline_orient]  N pairs = {int(m.nobs)}")
    log(m.params.round(5).to_string())
    log("SE: " + ", ".join(f"{v:.5f}" for v in m.bse.values))
    log("p : " + ", ".join(f"{v:.4f}" for v in m.pvalues.values))
    res2.append((y, vol, m))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n已写出 {OUT} —— 把这个文件发给我即可回填表 10/11/S27/S28。")
