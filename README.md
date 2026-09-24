# Replication Package — DADM / KuaiLive (IEEE Access)

Replication code and output files for:

**"Engagement Volume and Behavioral Composition in Live-Streaming Commerce: A Dual-Axis Decoupling Model"**
(Manuscript ID: Access-2026-41164, major revision)

## Data

The study uses the public **KuaiLive** live-streaming dataset (8 raw CSV files; see Supplementary Table S1 for the schema and Table S3 for the sample-screening pipeline). Raw data files are **not** redistributed in this repository; download them from the original public source cited in the manuscript. All scripts below start from the user-wave analysis panel constructed per Supplementary Tables S2–S3.

## Reproducibility policy

- All stochastic analyses use **seed = 42** (Bayesian subsample draw: `numpy.random.default_rng(42)`, 5,000 users without replacement; MCMC: PyMC/NumPyro NUTS, 500 tune + 1,000 draws × 2 chains).
- Every reported number comes from a **program-written output file** (JSON/CSV) — never from terminal transcription. Each authoritative JSON is accompanied by an MD5 checksum (see `outputs/authoritative_json/MANIFEST_md5.json`).
- Environment: Python 3.x, PyMC 5.28.4 / NumPyro (NUTS), pandas, numpy, statsmodels, scikit-learn, matplotlib. Savage–Dickey density ratios use Gaussian KDE with Scott's default bandwidth.

## Repository layout

```
code/
  paper4_kimi_analyses_20260917.py     Posterior summaries & key model outputs (RI-CLPM family)
  preregistered_analysis_runner.py     Pre-registered analysis runner (fixed specs, seed = 42)
  figures/
    make_figure3.py                    Figure 3 (Bayesian posterior density, primary evidence)
    make_figure4.py                    Figure 4 (margin decomposition schematic)
    make_figure7.py                    Figure 7 (DADM conceptual 2×2 schematic)
outputs/
  kimi_analyses/                       Posterior summary CSVs + analysis report (paper4_kimi_*)
  figure_data/                         Source data for figures + figure_key_numbers_master.csv
                                       (every key figure number → its source)
  authoritative_json/                  Final authoritative result files (with MD5 manifest):
    r38_prior_results.json             → Tables S17 & S17b; Figure S7 (24-combo prior sensitivity)
    r37_results.json                   → Table S16b (unconstrained-transition sensitivity)
    r37_dual4b_results.json            → Table S16b (dual-IV 4B variant)
    r311_results.json                  → Table S8b (baseline-only Wave-1 orientation clustering)
    pending4_results.json              → Table S11 hurdle cells (logit +0.6730; beta −0.6265),
                                         CRE (ln_psi within = +0.1670), FD N = 21,095
    did_archaeology_results.json       → Tables S23–S26 (DID / PSM-DID / pre-trend / FD; entity-FE
                                         TWFE + first-difference pre-trend calibration)
    MANIFEST_md5.json                  → MD5 checksums of all of the above
    P4_MASTER_NUMERICAL_PROVENANCE.xlsx→ Master ledger: every manuscript number → source file
```

## Pipeline scripts (computation side, delivered 2026-09-23)

`code/pipeline_scripts/` contains the primary pipeline scripts delivered by the computation-side archive (Coze), each verified against the delivery MANIFEST (21/21 OK). Script → manuscript mapping:

| Directory | Script | Manuscript product |
|---|---|---|
| `01_main_analysis/` | `bayesian_riclpm_V105B.py` | Primary analysis: Scheme 4A/4B, 5,000-user subsample (built-in `default_rng(42)` draw) → Tables 3/4; SM S18/S19/S22; H1/H2/H3 BF01s |
| | `bayesian_riclpm_V105B_fullsample.py` | Full-sample sensitivity (N=20,576) → SM S22 full-sample row (BF01=1.77/433.26) |
| `02_click_free/` | `paper4_kimi_analyses_20260917.py` | Click-free DEP measure, three paths + baseline-composition moderation → SM S27/S28 (identical to `code/paper4_kimi_analyses_20260917.py`) |
| `03_hurdle_cre/` | `s11_cre_clarify.py` | Hurdle four-way decomposition (→ SM S11 replacement cells +0.6730/−0.6265) + CRE fractional logit (ln_psi within=+0.1670 → SM S14) + FD sample tracing |
| `04_did/` | `did_analysis_ORIGINAL.py` / `did_repro.py` / `did_dep_repro.py` | DID suite (TWFE/PSM-DID/event study), SDK-stripped rerunnable version, DEP-measure rerun |
| | `did_archaeology_repro.py` | DID adjudicated spec: TWFE entity-FE-only + first-difference pre-trend → SM S23–S25 |
| `05_pending4/` | `pending4_repro.py` | FD regression (→ SM S26) + extensive/intensive (OLS) + entropy multivariate regression (reproducible version) + CRE |
| `06_kmeans/` | `r311_baseline_cluster.py` | K-means orientation clustering: full-period rebuild (5,344 ≈ EA2's 5,249) + baseline-only W1 (→ SM S8b; coverage 53.21%) |
| `07_rerun/` | `r37_unconstrained.py` / `r37_dual4b.py` / `r38_prior_sensitivity.py` | R3-7 three-wave unconstrained (single/dual-IV) → r37 JSONs; R3-8 prior sensitivity, 24 combos → SM S17/S17b + Figure S7 |
| `pip_freeze.txt` | — | Environment: Python 3.13, numpyro 0.19.0, jax 0.8.2, numpy 2.4.6, pandas 3.0.5, statsmodels 0.14.6, linearmodels 7.0, arviz 0.22.0, scikit-learn 1.9.0 |

Re-run notes: unified seed=42; 500 tune + 1000 draws × 2 chains (sequential); **jax must be 0.8.2** (later jax breaks numpyro 0.19 with `xla_pmap_p`); panel data `kuailive_panel_balanced.csv` (md5 registered in the ledger) is not redistributed — see Data statement.

`code/pipeline_scripts/MISSING.md` discloses six scripts that could not be recovered from the archive (panel construction, NB RI-CLPM, lag-2 variant, original entropy regression, original EA2 K-means, original extensive/intensive), each with its adjudicated substitute evidence. No missing item was fabricated.

`outputs/coze_extra_json/` holds the five supplementary output JSONs delivered with the pipeline scripts (`did_dep`, `s17_archaeology` ×2 rounds, `s11_cre_clarify`, `make_s17_table`); `s11_cre_clarify_results.json` is byte-identical to the authoritative copy (md5 `2219cd909c6e570b429740ddb7688100`).

## License

MIT (code). The KuaiLive dataset remains under its original terms.
