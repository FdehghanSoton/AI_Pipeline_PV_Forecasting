from pathlib import Path

import numpy as np
import pandas as pd

from analyze_pv_cnn2d import CNN_INPUT_CHANNELS, build_supervised_tensors
from analyze_pv_v2 import load_pv_v2
from analyze_pv_v4 import FoldRes, _split_temporal_validation_by_day, aggregate


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


def test_cnn_uses_nine_channels_and_distinguishes_missing_from_observed_zero() -> None:
    days, hours = 10, 24
    pv = np.zeros((days, hours), dtype=np.float32)
    pv[7, 12] = 99.0  # discarded because the availability flag says missing
    wx = np.zeros((6, days, hours), dtype=np.float32)
    daylight = np.ones((days, hours), dtype=int)
    missing = np.zeros((days, hours), dtype=int)
    missing[7, 12] = 1
    day_index = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")

    X, *_ = build_supervised_tensors(
        pv, wx, day_index, daylight, missing, capacity=100.0
    )

    assert len(CNN_INPUT_CHANNELS) == 9
    assert X.shape[1] == len(CNN_INPUT_CHANNELS)
    # Sample targeting day 9 contains day 7 at history position 5.
    sample = X[2]
    assert sample[0, 5, 12].item() == 0.0  # finite placeholder
    assert sample[2, 5, 12].item() == 0.0  # missing measurement
    assert sample[0, 5, 11].item() == 0.0  # genuine observed zero
    assert sample[2, 5, 11].item() == 1.0  # observed measurement


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


def test_missing_targets_are_excluded_from_all_aggregate_metrics() -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    fold = FoldRes(
        name="synthetic",
        test_idx=index,
        y_test=np.array([1.0, 2.0, 999.0, 4.0, 5.0, 6.0]),
        is_day_test=np.ones(6, dtype=bool),
        is_miss_test=np.array([False, False, True, False, False, False]),
        preds={"CNN": np.array([1.0, 2.0, -999.0, 4.0, 5.0, 6.0])},
    )

    metrics = aggregate([fold], capacity=10.0, mode="synthetic")
    all_hours = metrics[metrics["subset"] == "ALL"].iloc[0]
    daylight = metrics[metrics["subset"] == "daylight"].iloc[0]
    assert all_hours["n"] == 5
    assert all_hours["RMSE"] == 0.0
    assert daylight["n"] == 5
    assert daylight["RMSE"] == 0.0


def test_temporal_validation_split_is_calendar_day_disjoint() -> None:
    index = pd.date_range("2025-01-01", periods=20 * 24, freq="h", tz="UTC")
    frame = pd.DataFrame({"y": np.arange(len(index), dtype=float)}, index=index)
    frame = frame.drop(index[5 * 24 + 3])  # row-count splitting would divide a day

    train, validation = _split_temporal_validation_by_day(frame)
    train_days = set(train.index.normalize())
    validation_days = set(validation.index.normalize())

    assert train_days.isdisjoint(validation_days)
    assert max(train.index) < min(validation.index)
    assert len(validation_days) >= 7
