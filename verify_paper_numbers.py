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
    ("TEMPORAL", "RidgeStack", "ALL", 0.738, 10.99),
    ("TEMPORAL", "RidgeStack", "daylight", 0.572, 16.83),
    ("TEMPORAL", "NNLSStack", "daylight", 0.509, 18.03),
    ("TEMPORAL", "CNN", "daylight", 0.505, 18.09),
    ("TEMPORAL", "SmartPersistence", "daylight", 0.478, 18.59),
    ("KFOLD", "NNLSStack", "ALL", 0.861, 8.33),
    ("KFOLD", "NNLSStack", "daylight", 0.769, 12.30),
    ("KFOLD", "RidgeStack", "daylight", 0.761, 12.52),
    ("KFOLD", "GBM", "daylight", 0.734, 13.23),
    ("KFOLD", "SmartPersistence", "daylight", 0.505, 18.04),
]:
    check(f"{mode}/{model}/{subset} R2", r2, metric(m, mode, model, subset, "R2"))
    check(
        f"{mode}/{model}/{subset} nRMSE", nrmse, metric(m, mode, model, subset, "nRMSE_pct"), 0.005
    )

print("\nSkill scores and ensemble gains")
sig = pd.read_csv(R / "pv_v4_significance.csv")
k = sig[(sig["mode"] == "KFOLD") & (sig["subset"] == "daylight")].iloc[0]
t = sig[(sig["mode"] == "TEMPORAL") & (sig["subset"] == "daylight")].iloc[0]
check("random-fold ensemble gain (%)", 7.0, k["rel_rmse_gain_pct"], 0.05)
check("rolling-origin ensemble gain (%)", 7.0, t["rel_rmse_gain_pct"], 0.05)
check("random-fold DM statistic", -6.5, k["dm_statistic"], 0.05)
check("rolling-origin DM statistic", -6.5, t["dm_statistic"], 0.05)
SKILL = "skill_vs_SmartPersistence"
check("random-fold skill", 0.32, metric(m, "KFOLD", "NNLSStack", "daylight", SKILL))
check("rolling-origin skill", 0.095, metric(m, "TEMPORAL", "RidgeStack", "daylight", SKILL))

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


check("NNLS mean R2", 0.767, seed("NNLSStack", "R2_mean"))
check("NNLS sd R2", 0.005, seed("NNLSStack", "R2_sd"), 0.0005)
check("GBM mean R2", 0.731, seed("GBM", "R2_mean"))
check("GBM sd R2", 0.003, seed("GBM", "R2_sd"), 0.0005)

print("\nStacking weights (Figure 5)")
pred = pd.read_csv(R / "pv_v4_predictions.csv")
print("  (weights checked against make_weights_figure output, see log)")

print("\nAblation table")
ab = pd.read_csv(R / "pv_v4_ablation.csv")
ab = ab[ab["subset"] == "daylight"]


def abl(label: str, mode: str, col: str) -> float:
    return float(ab[(ab["ablation"] == label) & (ab["mode"] == mode)][col].iloc[0])


for label, kr2, kn, tr2, tn in [
    ("full", 0.769, 12.30, 0.572, 16.83),
    ("no_alignment", 0.790, 11.97, 0.568, 17.25),
    ("calendar_only", 0.768, 12.34, 0.705, 13.98),
    ("no_temporal", 0.768, 12.34, 0.527, 17.70),
]:
    for mode, r2, nrmse in [("KFOLD", kr2, kn), ("TEMPORAL", tr2, tn)]:
        check(f"ablation {label} {mode} R2", r2, abl(label, mode, "ensemble_R2"))
        nr = abl(label, mode, "ensemble_nRMSE_pct")
        check(f"ablation {label} {mode} nRMSE", nrmse, nr, 0.005)

print("\nAlignment scan")
sc = pd.read_csv(R / "pv_time_shift_scan.csv")
check("R2 at shift 0", 0.42, float(sc[sc["shift_hours"] == 0]["R2"].iloc[0]))
check("R2 at shift -1", 0.60, float(sc[sc["shift_hours"] == -1]["R2"].iloc[0]))
check("xcorr peak lag at shift -1", 0, float(sc[sc["shift_hours"] == -1]["xcorr_peak_lag"].iloc[0]))

print("\nWeather experiment")
ws = pd.read_csv(R / "pv_v4_weather_summary.csv")
wc = pd.read_csv(R / "pv_v4_weather_comparisons.csv")


def wsum(cfg: str, mode: str, col: str) -> float:
    return float(ws[(ws["configuration"] == cfg) & (ws["mode"] == mode)][col].iloc[0])


def wcmp(cmp_: str, mode: str, col: str) -> float:
    return float(wc[(wc["comparison"] == cmp_) & (wc["mode"] == mode)][col].iloc[0])


REL = "relative_rmse_change_pct"
for name, cmp_, mode, paper, tol in [
    ("matched-restriction cost KFOLD (%)", "dropped_cloud_levels", "KFOLD", 0.23, 0.005),
    ("matched-restriction gain TEMPORAL (%)", "dropped_cloud_levels", "TEMPORAL", -1.05, 0.005),
    ("forecast penalty KFOLD (%)", "operational_penalty", "KFOLD", 12.1, 0.05),
    ("forecast penalty TEMPORAL (%)", "operational_penalty", "TEMPORAL", 8.2, 0.05),
    ("ERA5 KFOLD (%)", "weather_product_sensitivity", "KFOLD", 4.3, 0.05),
    ("ERA5 TEMPORAL (%)", "weather_product_sensitivity", "TEMPORAL", -4.9, 0.05),
]:
    check(name, paper, wcmp(cmp_, mode, REL), tol)

WSKILL = "skill_vs_smart_persistence"
check("forecast R2 KFOLD", 0.709, wsum("forecast_day1", "KFOLD", "ensemble_R2"))
check("forecast skill KFOLD", 0.342, wsum("forecast_day1", "KFOLD", WSKILL))
check("analysis-matched skill KFOLD", 0.316, wsum("analysis_matched", "KFOLD", WSKILL))
sp = wsum("forecast_day1", "KFOLD", "smart_persistence_nRMSE_pct")
check("forecast smart-pers nRMSE KFOLD", 21.01, sp, 0.005)

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
check("total fit seconds", 10.41, base["fit_seconds"].sum(), 0.05)
check("total predict ms per day", 1.32, base["predict_ms_per_day"].sum(), 0.05)
check("CNN value, random fold (%)", 5.77, cost("CNN", "KFOLD", "rmse_increase_pct"), 0.05)
check(
    "clearness GBM value, rolling (%)",
    2.90,
    cost("Clearness-index GBM", "TEMPORAL", "rmse_increase_pct"),
    0.05,
)

print("\nCommitment cost")
ov = pd.read_csv(R / "pv_v4_operational_value.csv")


def ops(label: str, protocol: str, col: str) -> float:
    return float(ov[(ov["label"] == label) & (ov["protocol"] == protocol)][col].iloc[0])


nnls_c = ops("NNLS stacking", "Random day-fold", "cost_r3")
kt_c = ops("Clearness-index GBM", "Random day-fold", "cost_r3")
check("clearness GBM cheaper at r=3 (%)", 7.4, 100 * (nnls_c - kt_c) / nnls_c, 0.1)
nnls_r = ops("NNLS stacking", "Random day-fold", "rmse")
kt_r = ops("Clearness-index GBM", "Random day-fold", "rmse")
check("clearness GBM worse RMSE (%)", 8.5, 100 * (kt_r - nnls_r) / nnls_r, 0.1)
for label, paper in [("Ridge stacking", 0.09), ("Clearness-index GBM", 0.26)]:
    check(f"{label} skill_r3 (rolling)", paper, ops(label, "Rolling-origin", "skill_r3"))

print("\nMask ablation")
ma = pd.read_csv(R / "pv_v4_mask_ablation.csv")
kens = ma[ma["mode"] == "KFOLD"]["ensemble_nRMSE_pct"]
check("ensemble spread across arms (pp)", 0.044, kens.max() - kens.min(), 0.005)

print("\nClipping sensitivity")
cs = pd.read_csv(R / "pv_v4_clip_sensitivity.csv")
row = cs[(cs["kappa_clip"] == 1.5) & (cs["mode"] == "KFOLD")].iloc[0]
check("clip 1.5 matches headline (KFOLD)", 12.30, row["ensemble_nRMSE_pct"], 0.005)

print("\nStandard of reference")
sor = pd.read_csv(R / "pv_v4_standard_of_reference.csv")


def ref(name: str, mode: str, col: str = "nRMSE_pct") -> float:
    return float(sor[(sor["reference"] == name) & (sor["mode"] == mode)].iloc[0][col])


check("smart persistence nRMSE (KFOLD)", 18.04, ref("SmartPersistence", "KFOLD"), 0.005)
check("combination nRMSE (KFOLD)", 18.18, ref("ClimPersCombination", "KFOLD"), 0.005)
check("smart persistence nRMSE (TEMPORAL)", 18.59, ref("SmartPersistence", "TEMPORAL"), 0.005)
check("combination nRMSE (TEMPORAL)", 23.37, ref("ClimPersCombination", "TEMPORAL"), 0.005)
check("climatology nRMSE (TEMPORAL)", 28.28, ref("Climatology", "TEMPORAL"), 0.005)
check(
    "combination weight on climatology (KFOLD)",
    0.77,
    ref("ClimPersCombination", "KFOLD", "combination_weight_on_climatology"),
    0.005,
)
check(
    "combination weight on climatology (TEMPORAL)",
    0.41,
    ref("ClimPersCombination", "TEMPORAL", "combination_weight_on_climatology"),
    0.005,
)
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
