---
AIGC:
    Label: "1"
    ContentProducer: 001191110102MACQD9K64018705
    ProduceID: 2521604947715392_0-data_volume/7651177745430511891-files/所有对话/主对话/paper4_kimi回填_20260917/paper4_kimi_report_20260917.md
    ReservedCode1: ""
    ContentPropagator: 001191110102MACQD9K64028705
    PropagateID: 2521604947715392#1789621734973
    ReservedCode2: ""
---
论文4 回填回报（Kimi 工单 §5 模板）
生成日期: 2026-09-17 | 执行方: Coze | 脚本: paper4_kimi_analyses_20260917.py

== 环境 ==
Python 3.13.15; numpyro 0.19.0 (NUTS 直调) + jax 0.8.2 + arviz 0.22.0; numpy 2.4.6, pandas 3.0.5, scipy 1.18.0
采样: tune=500, draws=1000, chains=2, seed=42; 先验: 路径 Normal(0,0.5), 截距 Normal(0,1), sigma HalfNormal(1)
数据: 论文4_数据清洗/kuailive_panel_balanced.csv; N users=5000 (20,576 中按主分析同款 default_rng(42).choice 抽取), N rows=15000, lag pairs=10000 (W1->W2 + W2->W3)
模型设定: RI-CLPM 用户内去中心化; 预测变量 z 标准化, DV(wp) 不标准化; Normal 似然 —— 与主分析 V105B 完全一致

== 分析A ==（DEP_noclick 敏感性, Bayesian RI-CLPM）
A1 (ln_comment(t+1) <- DEP_noclick(t) + AR): mean=-0.0090, sd=0.0050, hdi_lo=-0.0190, hdi_hi=0.0010, BF01=22.836, Rhat=1.0000, ESS=1699, N=8231
   AR 项: mean=-0.2000, sd=0.0050, hdi_lo=-0.2100, hdi_hi=-0.1900, BF01=0.000, Rhat=1.0000, ESS=1734, N=8231
A2 (DEP_noclick(t+1) <- ln_comment(t) + AR): mean=-0.0000, sd=0.0020, hdi_lo=-0.0050, hdi_hi=0.0040, BF01=212.520, Rhat=1.0000, ESS=1569, N=6996
   AR 项: mean=-0.1250, sd=0.0020, hdi_lo=-0.1290, hdi_hi=-0.1200, BF01=0.000, Rhat=1.0000, ESS=1855, N=6996
A3 (DEP_noclick(t+1) <- ln_click(t) + AR): mean=-0.0020, sd=0.0020, hdi_lo=-0.0060, hdi_hi=0.0030, BF01=181.058, Rhat=1.0000, ESS=2632, N=6996
   AR 项: mean=-0.1250, sd=0.0020, hdi_lo=-0.1300, hdi_hi=-0.1210, BF01=0.000, Rhat=1.0000, ESS=2723, N=6996

DEP 两版相关: Pearson r=0.4378 (N=12398), Spearman rho=0.5586 (p 均 < 1e-300)
分波描述统计:
 wave         var    N  N_missing   mean     sd  median    min    max
    1    DEP_full 4998          2 0.0728 0.0985  0.0323 0.0000 1.0000
    1 DEP_noclick 4012        988 0.5147 0.3271  0.5000 0.0000 1.0000
    2    DEP_full 5000          0 0.0750 0.0983  0.0380 0.0000 1.0000
    2 DEP_noclick 4219        781 0.5233 0.3252  0.5000 0.0000 1.0000
    3    DEP_full 4999          1 0.0742 0.0977  0.0385 0.0000 1.0000
    3 DEP_noclick 4167        833 0.5176 0.3192  0.5000 0.0000 1.0000

== 分析B ==（baseline_orient 纯基线调节; orient = Wave1 DEP_full, z 标准化）
B1 主效应: mean=-0.0150, sd=0.0010, hdi=[-0.0160, -0.0140], BF01=0.000
B1 交互:   mean=-0.0030, sd=0.0010, hdi=[-0.0040, -0.0020], BF01=0.000
B1 取向主效应(备查): mean=-0.0190, sd=0.0010, hdi=[-0.0210, -0.0180], BF01=0.000
B1 诊断: Rhat(max)=1.0000, ESS(min)=2146, N=9995

B2 主效应: mean=0.0020, sd=0.0010, hdi=[0.0000, 0.0030], BF01=33.815
B2 交互:   mean=-0.0000, sd=0.0010, hdi=[-0.0010, 0.0010], BF01=865.717
B2 取向主效应(备查): mean=-0.0210, sd=0.0010, hdi=[-0.0230, -0.0200], BF01=0.000
B2 诊断: Rhat(max)=1.0000, ESS(min)=2241, N=9995

B3 主效应: mean=-0.2040, sd=0.0040, hdi=[-0.2130, -0.1970], BF01=0.000
B3 交互:   mean=0.0090, sd=0.0040, hdi=[0.0010, 0.0170], BF01=12.566
B3 取向主效应(备查): mean=-0.0660, sd=0.0040, hdi=[-0.0750, -0.0580], BF01=0.000
B3 诊断: Rhat(max)=1.0000, ESS(min)=2586, N=9996

== 异常申报 ==
1. 口径差异(工单内定): DEP 分母为 0 记缺失(NaN), 按工单 §2 逐字执行; 主分析 new_IDI 分母为 0 记 0.0。两口径仅在零互动行有差别, 各模型 N 已如实报告。
2. baseline_orient 构造: 主分析无合规时不变基线取向变量(旧 h13c_h13d 版为打赏行为全面板 K-means 后验聚类标签, 工单明确禁止)。按工单 §1 兜底: baseline_orient = 用户 Wave 1 的 DEP_full 原始值, 跨用户 z 标准化(含居中)。Wave 1 DEP_full 缺失用户(零互动)在分析 B 中剔除, 各模型 N 已报告。
3. 采样器接口: 主分析经 PyMC 调 numpyro 后端; 本脚本直接调用 numpyro NUTS。算法、tune/draws/chains/种子/先验全部一致, 非降级方案, 非 frequentist fallback。
4. 收敛警示: B1/sigma: ESS=192 < 400; B2/sigma: ESS=223 < 400

附件: paper4_kimi_results_20260917.csv (全部系数长表), paper4_kimi_desc_20260917.csv (描述统计), paper4_kimi_analyses_20260917.py (可复跑脚本)

---

> 本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
