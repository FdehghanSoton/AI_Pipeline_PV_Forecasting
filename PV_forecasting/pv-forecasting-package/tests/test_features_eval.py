from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_pv_v2 import flag_daylight_geometric, metric_set
from analyze_pv_v4 import apply_feature_ablation
from config import RunConfig


def _pv_frame(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2025-06-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"y": np.linspace(0, 100, n)}, index=idx)


def test_full_config_keeps_all_features() -> None:
    feats = ["shortwave_radiation", "sun_elev", "poa_global", "cloud_cover_lag1"]
    _, out = apply_feature_ablation(_pv_frame(), feats, RunConfig())
    assert out == feats


def test_calendar_only_drops_physics_and_adds_hour_encoding() -> None:
    feats = ["shortwave_radiation", "sun_elev", "poa_global", "clearness_kt"]
    cfg = RunConfig(use_physics_features=False)
    pv, out = apply_feature_ablation(_pv_frame(), feats, cfg)
    assert "sun_elev" not in out
    assert "poa_global" not in out
    assert "clearness_kt" not in out
    assert "shortwave_radiation" in out  # weather kept
    assert "hour_sin" in out and "hour_cos" in out
    assert "hour_sin" in pv.columns


def test_no_temporal_drops_lag_lead_roll() -> None:
    feats = ["shortwave_radiation", "cloud_cover_lag1", "poa_global_roll3", "x_lead1"]
    cfg = RunConfig(use_temporal_context=False)
    _, out = apply_feature_ablation(_pv_frame(), feats, cfg)
    assert "shortwave_radiation" in out
    assert not any(f.endswith(("_lag1", "_lead1", "_roll3")) for f in out)


def test_geometric_daylight_is_independent_of_pv_values() -> None:
    idx = pd.date_range("2025-06-01", periods=48, freq="h", tz="UTC")
    elev = np.where((idx.hour >= 6) & (idx.hour <= 18), 30.0, -10.0)
    base = pd.DataFrame({"y": np.linspace(0, 100, 48), "sun_elev": elev}, index=idx)
    scaled = base.copy()
    scaled["y"] = base["y"] * 7.3  # arbitrary scaling of the labels

    mask_a = flag_daylight_geometric(base.copy(), elevation_deg=5.0)["is_daylight"]
    mask_b = flag_daylight_geometric(scaled.copy(), elevation_deg=5.0)["is_daylight"]
    assert mask_a.equals(mask_b)  # mask does not depend on PV labels
    assert mask_a.to_numpy().tolist() == (elev > 5.0).astype(int).tolist()


def test_metric_set_perfect_forecast() -> None:
    y = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    m = metric_set(y, y, capacity=1000.0)
    assert np.isclose(m["R2"], 1.0)
    assert np.isclose(m["RMSE"], 0.0)
    assert m["n"] == 5
