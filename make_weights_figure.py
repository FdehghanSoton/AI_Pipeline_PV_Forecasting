"""NNLS stacking weights per base learner, mean +/- sd across folds.

Reruns both backtests and wraps ``build_ensembles``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

import analyze_pv_v4 as m
import model_labels
import paths
from config import load_config

OUT_DIR = paths.results_dir()

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

LABELS = model_labels.WRAPPED_LABEL
MODE_LABEL = model_labels.MODE_LABEL

_orig_build_ensembles = m.build_ensembles
_captured: dict[str, list[dict[str, float]]] = {"KFOLD": [], "TEMPORAL": []}
_current_mode = {"mode": "KFOLD"}


def _capturing_build_ensembles(val_preds, y_val, val_mask, test_preds):
    names = list(val_preds.keys())
    V = np.stack([val_preds[n] for n in names], axis=1)
    y_m, V_m = y_val[val_mask], V[val_mask]
    try:
        w, _ = nnls(V_m, y_m)
        w = w / w.sum() if w.sum() > 1e-6 else np.ones(len(names)) / len(names)
    except Exception:
        w = np.ones(len(names)) / len(names)
    _captured[_current_mode["mode"]].append(dict(zip(names, w, strict=False)))
    return _orig_build_ensembles(val_preds, y_val, val_mask, test_preds)


def main() -> None:
    m.build_ensembles = _capturing_build_ensembles
    cfg = load_config()
    pv, feats, capacity = m.build_dataset(cfg)

    _current_mode["mode"] = "KFOLD"
    m.kfold_backtest(
        pv, pv, capacity, feats, n_folds=cfg.kfold_n_folds, seed=cfg.seed, cfg=cfg
    )
    _current_mode["mode"] = "TEMPORAL"
    m.temporal_backtest(
        pv, pv, capacity, feats,
        n_folds=cfg.temporal_n_folds, first_test_days=cfg.first_test_days, cfg=cfg,
    )

    base = m.BASE_LEARNERS
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    for ax, mode in zip(axes, ("KFOLD", "TEMPORAL"), strict=False):
        folds = _captured[mode]
        mat = np.array([[fold.get(b, 0.0) for b in base] for fold in folds])
        mean = mat.mean(axis=0)
        sd = mat.std(axis=0, ddof=1) if len(folds) > 1 else np.zeros_like(mean)
        xpos = np.arange(len(base))
        ax.bar(xpos, mean, yerr=sd, capsize=3, color="steelblue", edgecolor="0.3")
        ax.set_xticks(xpos)
        ax.set_xticklabels([LABELS[b] for b in base], fontsize=8)
        ax.set_title(f"{MODE_LABEL[mode]}  ($n={len(folds)}$ folds)")
        ax.grid(axis="y", alpha=0.25)
        for x, v in zip(xpos, mean, strict=False):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylabel("Mean NNLS stacking weight")
    axes[0].set_ylim(0, max(0.6, axes[0].get_ylim()[1]))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"pv_v4_fig_app_weights.{ext}")
    plt.close(fig)
    print("Wrote pv_v4_fig_app_weights.pdf and .png")
    for mode in ("KFOLD", "TEMPORAL"):
        folds = _captured[mode]
        mat = np.array([[fold.get(b, 0.0) for b in base] for fold in folds])
        print(f"\n{mode} mean NNLS weights:")
        for b, w in zip(base, mat.mean(axis=0), strict=False):
            print(f"  {b:14s} {w:.3f}")


if __name__ == "__main__":
    main()
