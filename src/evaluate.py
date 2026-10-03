"""Metrics, splits and cross-validation helpers.

The split helpers are the second half of the leakage firewall: they guarantee
that the model never sees a row it will later be scored on, and they make the
"random split vs. time split" difference measurable instead of rhetorical.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

if TYPE_CHECKING:  # pragma: no cover
    from src.features import DesignMatrix

logger = logging.getLogger(__name__)

#: The quadruple of metrics reported everywhere in this project.
METRIC_NAMES: tuple[str, ...] = ("mae", "rmse", "r2", "mape", "mdape")


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """MAE / RMSE / R2 / MAPE / MdAPE in the original JPY-per-square-metre units.

    The target spans five orders of magnitude (a forest plot at a few hundred
    JPY/m2 next to Ginza at ~150,000 JPY/m2), so MAPE is dominated by the cheapest
    plots and is *not* a selection criterion. ``mdape`` (the median of the absolute
    percentage errors) is reported next to it as the robust counterpart and is what
    the reports use when the relative error matters.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr, y_pred_arr = y_true_arr[mask], y_pred_arr[mask]
    if y_true_arr.size == 0:
        raise ValueError("no finite values to score")
    nonzero = y_true_arr != 0
    if nonzero.any():
        relative = np.abs((y_true_arr[nonzero] - y_pred_arr[nonzero]) / y_true_arr[nonzero]) * 100
        mape = float(np.mean(relative))
        mdape = float(np.median(relative))
    else:
        mape = mdape = float("nan")
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if y_true_arr.size > 1 else float("nan"),
        "mape": mape,
        "mdape": mdape,
        "n": int(y_true_arr.size),
    }


def temporal_split(
    df: pd.DataFrame, cfg: Config, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the prediction year into validation and test halves.

    The two halves are disjoint sets of *sites*, sampled with a fixed seed. This
    keeps both halves equally representative of the prediction year (a
    deterministic "first half / second half" cut would confound the comparison
    with the file's internal geographic ordering, which runs from Chiyoda to the
    Izu islands).
    """
    seed = cfg.seed if seed is None else seed
    val_fraction = float(cfg.split["val_fraction"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    cut = int(round(len(df) * val_fraction))
    val_idx, test_idx = order[:cut], order[cut:]
    val = df.iloc[val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    return val, test


def random_split(
    df: pd.DataFrame, cfg: Config, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Control experiment: shuffle the *prediction year* into train/test.

    Rows of the same year, city and often the same street end up on both sides,
    so the test set stops being a forecast and becomes an interpolation task.
    The pipeline measures how much this inflates the scores.
    """
    seed = cfg.seed if seed is None else seed
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    cut = int(round(len(df) * float(cfg.split["val_fraction"])))
    train = df.iloc[order[:cut]].reset_index(drop=True)
    test = df.iloc[order[cut:]].reset_index(drop=True)
    return train, test


def cross_validate_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
) -> dict[str, float]:
    """K-fold CV on the training year, reported as mean +/- std of MAE and R2.

    A chronological CV is impossible here because a single year is a single
    timestamp, so plain K-fold on the training rows is the honest choice: it
    measures how sensitive the model is to *which* sites it learns from.
    """
    cv = KFold(
        n_splits=int(cfg.cv["n_splits"]),
        shuffle=True,
        random_state=cfg.seed,
    )
    out: dict[str, float] = {}
    for metric, short in (("neg_mean_absolute_error", "mae"), ("r2", "r2")):
        scores = cross_val_score(
            estimator, X, y, cv=cv, scoring=metric, n_jobs=1, error_score="raise"
        )
        # scikit-learn negates error-like metrics, so flip the sign back.
        values = -scores if metric.startswith("neg_") else scores
        out[f"cv_{short}_mean"] = float(np.mean(values))
        out[f"cv_{short}_std"] = float(np.std(values))
    return out


def unwrap_estimator(estimator: BaseEstimator) -> Any:
    """Peel the project's wrappers down to the underlying estimator.

    Layers, outermost first: ``TargetLogRegressor`` (log target) ->
    ``ModelPipeline`` (preprocessing + estimator) -> ``LGBMCategoricalRegressor``
    (categorical declaration) -> the concrete model exposing coefficients or
    ``feature_importances_``.
    """
    from src.models import (
        LGBMCategoricalRegressor,
        ModelPipeline,
        NonNegativeRegressor,
        TargetLogRegressor,
    )

    core: Any = estimator
    if isinstance(core, TargetLogRegressor):
        core = core.estimator_
    if isinstance(core, NonNegativeRegressor):
        core = core.estimator_
    if isinstance(core, ModelPipeline):
        core = core.estimator_
    if isinstance(core, LGBMCategoricalRegressor):
        core = core.estimator_
    if hasattr(core, "named_steps"):
        core = core.named_steps.get("model", core)
    return core


def feature_importance_table(
    estimator: BaseEstimator,
    feature_names: list[str],
    expander: Callable[[str], list[str]] | None = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract a ranked importance table from a fitted pipeline.

    Tree ensembles expose split-based ``feature_importances_`` and linear models
    expose standardised coefficients. When the preprocessor one-hot encoded the
    categories, the raw vector is longer than the design matrix; ``expander``
    maps each emitted column back to its source feature and the importances are
    summed per source feature, so every model is ranked on the same feature names
    and the comparison between them stays meaningful.
    """
    core = unwrap_estimator(estimator)

    if hasattr(core, "feature_importances_"):
        importances = np.asarray(core.feature_importances_, dtype=float)
    elif getattr(core, "coef_", None) is not None:
        importances = np.abs(np.ravel(core.coef_))
    else:
        logger.warning("model exposes neither feature_importances_ nor coef_; skipping")
        return pd.DataFrame(columns=["feature", "importance", "importance_share"])

    if len(importances) != len(feature_names):
        logger.warning(
            "importance length %d != emitted feature count %d; skipping",
            len(importances),
            len(feature_names),
        )
        return pd.DataFrame(columns=["feature", "importance", "importance_share"])

    totals: dict[str, float] = {}
    for token, value in zip(feature_names, importances):
        for source in (expander(token) if expander else [token]):
            totals[source] = totals.get(source, 0.0) + float(value)

    table = (
        pd.DataFrame({"feature": list(totals), "importance": list(totals.values())})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    total = float(table["importance"].sum())
    table["importance_share"] = table["importance"] / total if total else np.nan
    return table.head(top_n)


def metrics_frame(results: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Flatten ``{model: {split: metrics}}`` into a tidy DataFrame."""
    rows: list[dict[str, Any]] = []
    for model, splits in results.items():
        for split, metrics in splits.items():
            row = {"model": model, "split": split}
            row.update({k: v for k, v in metrics.items() if k in (*METRIC_NAMES, "n")})
            rows.append(row)
    return pd.DataFrame(rows)
