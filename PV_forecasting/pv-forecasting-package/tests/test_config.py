from __future__ import annotations

import config


def test_defaults_are_leakage_safe() -> None:
    cfg = config.RunConfig()
    assert cfg.capacity_policy == "fold_train"
    assert cfg.metric_capacity_policy == "global"
    assert cfg.daylight_policy == "geometric"
    assert cfg.time_shift_hours == -1


def test_env_overrides_are_read_at_call_time(monkeypatch) -> None:
    monkeypatch.setenv("PV_TIME_SHIFT", "0")
    monkeypatch.setenv("PV_DAYLIGHT_POLICY", "pv_median")
    monkeypatch.setenv("PV_USE_PHYSICS", "0")
    monkeypatch.setenv("PV_CNN_SEEDS", "0,1,2")
    cfg = config.load_config()
    assert cfg.time_shift_hours == 0
    assert cfg.daylight_policy == "pv_median"
    assert cfg.use_physics_features is False
    assert cfg.cnn_seeds == (0, 1, 2)


def test_run_tag_changes_output_path() -> None:
    untagged = config.RunConfig(run_tag="")
    tagged = config.RunConfig(run_tag="ablation_noshift")
    assert untagged.tagged("pv_v4_metrics.csv").name == "pv_v4_metrics.csv"
    assert tagged.tagged("pv_v4_metrics.csv").name == "pv_v4_metrics__ablation_noshift.csv"


def test_describe_is_serialisable() -> None:
    import json

    payload = json.dumps(config.RunConfig().describe())
    assert "capacity_policy" in payload
    assert "site" in payload
