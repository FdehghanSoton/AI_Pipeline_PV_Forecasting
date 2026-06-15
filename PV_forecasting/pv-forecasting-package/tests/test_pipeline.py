from pathlib import Path

import numpy as np
import pandas as pd

from analyze_pv_cnn2d import build_supervised_tensors
from analyze_pv_v2 import load_pv_v2


def test_timestamp_shift_changes_index_by_one_hour(tmp_path: Path) -> None:
    path = tmp_path / "pv.csv"
    pd.DataFrame(
        {
            "_time": ["2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z"],
            "_field": ["PPV", "PPV"],
            "_value": [1.0, 2.0],
        }
    ).to_csv(path, index=False)
    unshifted = load_pv_v2(path, time_shift_hours=0)
    shifted = load_pv_v2(path, time_shift_hours=-1)
    assert shifted.index[0] == unshifted.index[0] - pd.Timedelta(hours=1)


def test_cnn_missing_targets_receive_zero_loss_weight() -> None:
    days, hours = 9, 24
    pv = np.ones((days, hours), dtype=np.float32)
    wx = np.zeros((6, days, hours), dtype=np.float32)
    daylight = np.ones((days, hours), dtype=int)
    missing = np.zeros((days, hours), dtype=int)
    missing[-1, 12] = 1
    day_index = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")

    _, _, weights, eval_mask, target_days = build_supervised_tensors(
        pv, wx, day_index, daylight, missing, capacity=1.0
    )
    assert target_days[-1] == day_index[-1]
    assert weights[-1, 12].item() == 0.0
    assert eval_mask[-1, 12].item() == 0.0
    assert weights[-1, 11].item() == 4.0


def test_held_out_pv_is_hidden_from_cnn_history() -> None:
    days, hours = 10, 24
    pv = np.zeros((days, hours), dtype=np.float32)
    pv[7] = 99.0
    wx = np.zeros((6, days, hours), dtype=np.float32)
    daylight = np.ones((days, hours), dtype=int)
    missing = np.zeros((days, hours), dtype=int)
    day_index = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")

    X, *_ = build_supervised_tensors(
        pv,
        wx,
        day_index,
        daylight,
        missing,
        capacity=100.0,
        hidden_pv_days=pd.DatetimeIndex([day_index[7]]),
    )
    # Sample targeting day 9 includes day 7 in its historical window.
    sample = X[2]
    assert np.allclose(sample[0, 5].numpy(), 0.0)
    assert np.allclose(sample[2, 5].numpy(), 0.0)
