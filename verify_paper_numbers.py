"""Recompute the reported metrics from the CSVs in ``results/``.

Run after regenerating the pipeline. A failure means the committed numbers
and the current outputs have drifted apart.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import paths

R = paths.RESULTS_DIR
FAILS: list[str] = []


def check(name: str, paper: float, computed: float, tol: float = 5e-3) -> None:
    ok = abs(paper - computed) <= tol
    flag = "ok  " if ok else "FAIL"
    print(f"  [{flag}] {name}: paper={paper}  computed={computed:.4f}")
    if not ok:
        FAILS.append(name)


def metric(m: pd.DataFrame, mode: str, model: str, subset: str, col: str) -> float:
    row = m[(m["mode"] == mode) & (m["model"] == model) & (m["subset"] == subset)]
    return float(row[col].iloc[0])


print("Table 1: headline results")
m = pd.read_csv(R / "pv_v4_metrics.csv")
for mode, model, subset, r2, nrmse in [
    ("TEMPORAL", "NNLSStack", "ALL", 0.719, 11.37),
    ("TEMPORAL", "NNLSStack", "daylight", 0.532, 17.59),
    ("TEMPORAL", "RidgeStack", "daylight", 0.572, 16.83),
    ("TEMPORAL", "CNN", "daylight", 0.505, 18.09),
    ("TEMPORAL", "SmartPersistence", "daylight", 0.478, 18.59),
    ("KFOLD", "NNLSStack", "ALL", 0.859, 8.41),
    ("KFOLD", "NNLSStack", "daylight", 0.765, 12.42),
    ("KFOLD", "RidgeStack", "daylight", 0.761, 12.52),
    ("KFOLD", "GBM", "daylight", 0.734, 13.23),
    ("KFOLD", "SmartPersistence", "daylight", 0.505, 18.04),
]:
    check(f"{mode}/{model}/{subset} R2", r2, metric(m, mode, model, subset, "R2"))
    check(
        f"{mode}/{model}/{subset} nRMSE",
        nrmse,
        metric(m, mode, model, subset, "nRMSE_pct"),
        0.006,
    )

print("\nSkill scores and ensemble gains")
# The ensemble side of these comparisons is always NNLSStack, the combination
# declared in advance; see significance_table in analyze_pv_v4.
sig = pd.read_csv(R / "pv_v4_significance.csv")
k = sig[(sig["mode"] == "KFOLD") & (sig["subset"] == "daylight")].iloc[0]
t = sig[(sig["mode"] == "TEMPORAL") & (sig["subset"] == "daylight")].iloc[0]
check("random-fold ensemble gain (%)", 6.1, k["rel_rmse_gain_pct"], 0.05)
check("rolling-origin ensemble gain (%)", 2.8, t["rel_rmse_gain_pct"], 0.05)
for name, row, limit in [("random-fold", k, 1e-7), ("rolling-origin", t, 0.01)]:
    ok = row["p_value"] < limit
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name} DM p-value {row['p_value']:.2g} < {limit}")
    if not ok:
        FAILS.append(f"{name} DM p-value")
SKILL = "skill_vs_SmartPersistence"
check("random-fold skill", 0.311, metric(m, "KFOLD", "NNLSStack", "daylight", SKILL))
check(
    "rolling-origin skill", 0.054, metric(m, "TEMPORAL", "NNLSStack", "daylight", SKILL)
)

print("\nStacking weights: what forcing a sum of one would cost")
sc = pd.read_csv(R / "pv_v4_stack_constraint.csv")


def stack(mode: str, variant: str) -> float:
    row = sc[(sc["mode"] == mode) & (sc["variant"] == variant)]
    return float(row["daylight_nRMSE_pct"].iloc[0])


for mode, fitted, renormalised, convex in [
    ("KFOLD", 12.42, 12.30, 12.54),
    ("TEMPORAL", 17.59, 18.03, 17.91),
]:
    check(f"{mode} stack as fitted", fitted, stack(mode, "NNLSStack"), 0.006)
    check(f"{mode} stack renormalised", renormalised, stack(mode, "NNLSRenormalised"), 0.006)
    check(f"{mode} stack convex", convex, stack(mode, "NNLSConvex"), 0.006)
for mode, total in [("KFOLD", 1.03), ("TEMPORAL", 1.06)]:
    row = sc[sc["mode"] == mode].iloc[0]
    check(f"{mode} mean weight sum", total, row["mean_raw_weight_sum"])

print("\nResidual correlations")
for mode, lo, hi, mean in [("kfold", 0.71, 0.74, 0.80), ("temporal", 0.80, 0.86, 0.83)]:
    c = pd.read_csv(R / f"pv_v4_residual_corr_{mode}.csv", index_col=0)
    cnn = c["CNN"].drop("CNN")
    check(f"{mode} CNN min rho", lo, cnn.min(), 0.005)
    check(f"{mode} CNN max rho", hi, cnn.max(), 0.005)
    a = c.to_numpy()
    check(f"{mode} mean off-diagonal", mean, a[~np.eye(len(a), dtype=bool)].mean(), 0.005)
c = pd.read_csv(R / "pv_v4_residual_corr_kfold.csv", index_col=0)
check("GBM/GBM_kt rho (random fold)", 0.93, c.loc["GBM", "GBM_kt"], 0.005)

print("\nMulti-seed repetition")
ms = pd.read_csv(R / "pv_v4_multiseed_summary.csv")


def seed(model: str, col: str) -> float:
    sel = ms[(ms["mode"] == "KFOLD") & (ms["model"] == model) & (ms["subset"] == "daylight")]
    return float(sel[col].iloc[0])


check("NNLS mean R2", 0.763, seed("NNLSStack", "R2_mean"))
check("NNLS sd R2", 0.005, seed("NNLSStack", "R2_sd"), 0.0005)
check("GBM mean R2", 0.731, seed("GBM", "R2_mean"))
check("GBM sd R2", 0.003, seed("GBM", "R2_sd"), 0.0005)

print("\nAblation table")
ab = pd.read_csv(R / "pv_v4_ablation.csv")
ab = ab[ab["subset"] == "daylight"]


def abl(label: str, mode: str, col: str) -> float:
    return float(ab[(ab["ablation"] == label) & (ab["mode"] == mode)][col].iloc[0])


for label, kr2, kn, tr2, tn in [
    ("full", 0.765, 12.42, 0.532, 17.59),
    ("no_alignment", 0.788, 12.02, 0.580, 17.01),
    ("calendar_only", 0.763, 12.46, 0.668, 14.83),
    ("no_temporal", 0.763, 12.49, 0.516, 17.90),
]:
    for mode, r2, nrmse in [("KFOLD", kr2, kn), ("TEMPORAL", tr2, tn)]:
        check(f"ablation {label} {mode} R2", r2, abl(label, mode, "ensemble_R2"))
        check(f"ablation {label} {mode} nRMSE", nrmse, abl(label, mode, "ensemble_nRMSE_pct"), 0.006)

print("\nTimestamp alignment")
sc_shift = pd.read_csv(R / "pv_time_shift_scan.csv")
check("R2 at shift 0", 0.42, float(sc_shift[sc_shift["shift_hours"] == 0]["R2"].iloc[0]))
check("R2 at shift -1", 0.60, float(sc_shift[sc_shift["shift_hours"] == -1]["R2"].iloc[0]))
check(
    "xcorr peak lag at shift -1",
    0,
    float(sc_shift[sc_shift["shift_hours"] == -1]["xcorr_peak_lag"].iloc[0]),
)

# The shift must be recoverable from training rows alone, and -1 h and -2 h must
# be the two candidates the training-only sweep cannot separate.
byfold = pd.read_csv(R / "pv_shift_by_fold.csv")
folds = byfold[byfold["block"] != "initial training pool"]
chosen = folds["selected_shift"].value_counts()
check("folds selecting -1 h", 7, float(chosen.get(-1, 0)), 0)
check("folds selecting -2 h", 2, float(chosen.get(-2, 0)), 0)

shift = pd.read_csv(R / "pv_v4_shift_sensitivity.csv")


def by_shift(mode: str, hours: int) -> float:
    row = shift[(shift["mode"] == mode) & (shift["shift_hours"] == hours)]
    return float(row["NNLS_nRMSE_pct"].iloc[0])


check("alignment cost KFOLD (pp)", 0.40, by_shift("KFOLD", -1) - by_shift("KFOLD", 0), 0.006)
check(
    "alignment cost TEMPORAL (pp)",
    0.58,
    by_shift("TEMPORAL", -1) - by_shift("TEMPORAL", 0),
    0.006,
)
check("shift -2 h KFOLD nRMSE", 12.55, by_shift("KFOLD", -2), 0.006)
check("shift -2 h TEMPORAL nRMSE", 17.25, by_shift("TEMPORAL", -2), 0.006)

print("\nReference-forecast fallbacks")
gaps = pd.read_csv(R / "pv_v4_baseline_gaps.csv")


def gap(mode: str, col: str) -> float:
    return float(gaps[gaps["protocol"] == mode][col].iloc[0])


check("lag-24 missing KFOLD (%)", 1.8, gap("KFOLD", "lag24_missing_pct"), 0.05)
check("lag-24 missing TEMPORAL (%)", 1.2, gap("TEMPORAL", "lag24_missing_pct"), 0.05)
check("unseen month-hour KFOLD (%)", 0.0, gap("KFOLD", "unseen_month_hour_pct"), 0.05)
check("unseen month-hour TEMPORAL (%)", 53.8, gap("TEMPORAL", "unseen_month_hour_pct"), 0.05)

print("\nWeather experiment")
ws = pd.read_csv(R / "pv_v4_weather_summary.csv")
wc = pd.read_csv(R / "pv_v4_weather_comparisons.csv")


def wsum(cfg: str, mode: str, col: str) -> float:
    return float(ws[(ws["configuration"] == cfg) & (ws["mode"] == mode)][col].iloc[0])


def wcmp(cmp_: str, mode: str, col: str) -> float:
    return float(wc[(wc["comparison"] == cmp_) & (wc["mode"] == mode)][col].iloc[0])


REL = "relative_rmse_change_pct"
for name, cmp_, mode, paper, tol in [
    ("matched-restriction cost KFOLD (%)", "dropped_cloud_levels", "KFOLD", 0.05, 0.005),
    ("matched-restriction gain TEMPORAL (%)", "dropped_cloud_levels", "TEMPORAL", -0.81, 0.005),
    ("forecast penalty KFOLD (%)", "operational_penalty", "KFOLD", 13.1, 0.05),
    ("forecast penalty TEMPORAL (%)", "operational_penalty", "TEMPORAL", 5.9, 0.05),
    ("ERA5 KFOLD (%)", "weather_product_sensitivity", "KFOLD", 4.9, 0.05),
    ("ERA5 TEMPORAL (%)", "weather_product_sensitivity", "TEMPORAL", -6.7, 0.05),
]:
    check(name, paper, wcmp(cmp_, mode, REL), tol)

WSKILL = "skill_vs_smart_persistence"
check("forecast R2 KFOLD", 0.699, wsum("forecast_day1", "KFOLD", "ensemble_R2"))
check("forecast R2 TEMPORAL", 0.484, wsum("forecast_day1", "TEMPORAL", "ensemble_R2"))
check("forecast skill KFOLD", 0.331, wsum("forecast_day1", "KFOLD", WSKILL))
check("analysis-matched skill KFOLD", 0.311, wsum("analysis_matched", "KFOLD", WSKILL))
check(
    "forecast smart-pers nRMSE KFOLD",
    21.01,
    wsum("forecast_day1", "KFOLD", "smart_persistence_nRMSE_pct"),
    0.006,
)

print("\nForecast-versus-analysis weather difference")
wd = pd.read_csv(R / "pv_v4_weather_difference.csv")
wd = wd[wd["subset"] == "daylight"]


def diff(var: str, col: str) -> float:
    return float(wd[wd["variable"] == var][col].iloc[0])


check("shortwave analysis mean", 305, diff("shortwave_radiation", "analysis_mean"), 0.5)
check("shortwave RMSE", 119, diff("shortwave_radiation", "rmse"), 0.5)
check("cloud-cover correlation", 0.62, diff("cloud_cover", "correlation"))

print("\nCost and marginal value")
cb = pd.read_csv(R / "pv_v4_cost_benefit.csv")


def cost(label: str, mode: str, col: str) -> float:
    return float(cb[(cb["label"] == label) & (cb["mode"] == mode)][col].iloc[0])


base = cb[cb["label"] != "Full ensemble"].drop_duplicates("label")
check("total fit seconds", 21.37, base["fit_seconds"].sum(), 0.05)
check("total predict ms per day", 2.61, base["predict_ms_per_day"].sum(), 0.05)
check("GBM alone fit seconds", 1.97, cost("Gradient boosting", "KFOLD", "fit_seconds"), 0.05)
check("CNN value, random fold (%)", 5.77, cost("CNN", "KFOLD", "rmse_increase_pct"), 0.05)
check(
    "POA-normalised GBM value, rolling (%)",
    2.90,
    cost("POA-normalised GBM", "TEMPORAL", "rmse_increase_pct"),
    0.05,
)

print("\nCommitment cost")
ov = pd.read_csv(R / "pv_v4_operational_value.csv")


def ops(label: str, protocol: str, col: str) -> float:
    return float(ov[(ov["label"] == label) & (ov["protocol"] == protocol)][col].iloc[0])


# The reordering claim: the POA-normalised GBM commits better at r=3 than the
# reported stack under both protocols, despite a higher RMSE in one of them.
for protocol in ("Random day-fold", "Rolling-origin"):
    cheaper = ops("POA-normalised GBM", protocol, "cost_r3") < ops(
        "NNLS stacking", protocol, "cost_r3"
    )
    print(f"  [{'ok  ' if cheaper else 'FAIL'}] POA-normalised GBM cheaper at r=3 ({protocol})")
    if not cheaper:
        FAILS.append(f"POA-normalised GBM not cheaper at r=3 ({protocol})")
for label, paper in [("Ridge stacking", 0.09), ("POA-normalised GBM", 0.26)]:
    check(f"{label} skill_r3 (rolling)", paper, ops(label, "Rolling-origin", "skill_r3"))

print("\nMask ablation")
ma = pd.read_csv(R / "pv_v4_mask_ablation.csv")
kens = ma[ma["mode"] == "KFOLD"]["ensemble_nRMSE_pct"]
check("ensemble spread across arms (pp)", 0.15, kens.max() - kens.min(), 0.006)
kcnn = ma[(ma["mode"] == "KFOLD")].set_index("arm")["cnn_nRMSE_pct"]
check("full CNN nRMSE (KFOLD)", 13.23, kcnn["full"], 0.006)
check("naive-fill CNN nRMSE (KFOLD)", 13.44, kcnn["neither"], 0.006)

print("\nClipping sensitivity")
cs = pd.read_csv(R / "pv_v4_clip_sensitivity.csv")
row = cs[(cs["kappa_clip"] == 1.5) & (cs["mode"] == "KFOLD")].iloc[0]
check("clip 1.5 matches headline (KFOLD)", 12.42, row["ensemble_nRMSE_pct"], 0.006)
for mode, spread in [("KFOLD", 0.03), ("TEMPORAL", 0.22)]:
    view = cs[cs["mode"] == mode]["ensemble_nRMSE_pct"]
    check(f"clip spread {mode} (pp)", spread, float(np.ptp(view.to_numpy())), 0.006)

print("\nStandard of reference")
sor = pd.read_csv(R / "pv_v4_standard_of_reference.csv")


def ref(name: str, mode: str, col: str = "nRMSE_pct") -> float:
    return float(sor[(sor["reference"] == name) & (sor["mode"] == mode)].iloc[0][col])


check("smart persistence nRMSE (KFOLD)", 18.04, ref("SmartPersistence", "KFOLD"), 0.006)
check("combination nRMSE (KFOLD)", 18.18, ref("ClimPersCombination", "KFOLD"), 0.006)
check("smart persistence nRMSE (TEMPORAL)", 18.59, ref("SmartPersistence", "TEMPORAL"), 0.006)
check("combination nRMSE (TEMPORAL)", 23.37, ref("ClimPersCombination", "TEMPORAL"), 0.006)
check("climatology nRMSE (TEMPORAL)", 28.28, ref("Climatology", "TEMPORAL"), 0.006)
# Skill scores use smart persistence, which should be the most accurate reference.
for mode in ("KFOLD", "TEMPORAL"):
    block = sor[sor["mode"] == mode]
    best = block.loc[block["nRMSE_pct"].idxmin(), "reference"]
    ok = best == "SmartPersistence"
    print(f"  [{'ok  ' if ok else 'FAIL'}] most accurate reference ({mode}): {best}")
    if not ok:
        FAILS.append(f"most accurate reference ({mode}) is {best}, not SmartPersistence")

print()
if FAILS:
    print(f"{len(FAILS)} MISMATCHES:")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("All checked numbers agree with the generated results.")
