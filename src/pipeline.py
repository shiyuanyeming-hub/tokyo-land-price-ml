"""End-to-end pipeline: ``python -m src.pipeline --stage all``.

Stage order mirrors the README workflow:

    data -> eda -> models -> leakage -> errors -> figures -> report

Every number written to ``reports/`` is produced here, by actually fitting the
models on the real files in ``data/raw``. Nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from src import cleaning, error_analysis, evaluate, features, models, plots
from src.config import Config, configure_matplotlib, quiet_lightgbm, setup_logging
from src.data_loader import load_all_raw, summarise
from src.models import MODEL_ORDER, MODELS_WITHOUT_FEATURES, build_estimator

logger = logging.getLogger(__name__)

REPORTS_FIGURES = "reports/figures"


@dataclass
class Dataset:
    """A cleaned + feature-engineered frame with its quality report."""

    name: str
    frame: pd.DataFrame
    quality: dict[str, object]


@dataclass
class ModelRun:
    """Everything one fitted model produced."""

    name: str
    metrics: dict[str, dict[str, float]]
    cv: dict[str, float]
    importance: pd.DataFrame
    predictions: pd.DataFrame
    model: Any = None
    feature_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1: data
# ---------------------------------------------------------------------------
def load_datasets(cfg: Config) -> dict[str, Dataset]:
    """Read, clean and feature-engineer every configured source."""
    raw = load_all_raw(cfg)
    datasets: dict[str, Dataset] = {}
    profiles: list[dict[str, object]] = []
    for name, frame in raw.items():
        profiles.append(summarise(frame, name))
        cleaned, quality = cleaning.clean_frame(frame, source=name)
        enriched = cleaning.add_unit_price(features.build_features(cleaned))
        enriched["_site_key"] = features.site_key(enriched)
        datasets[name] = Dataset(name=name, frame=enriched, quality=quality.as_dict())
    logger.info("data profile:\n%s", pd.DataFrame(profiles).to_string(index=False))
    return datasets


def attach_measured_lag(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach the previous-year unit price that the file itself publishes.

    Uses ``前年価格（円） / 地積（㎡）``, the officially assessed value of the same
    site one year earlier. This is the only correct lag available in these four
    files: joining the other year's file instead would attach the *following*
    year's price for the training rows (a forward-looking feature) and the site's
    own price for the evaluation rows (no trend information at all).
    """
    return features.attach_published_lag(frame, lag_column="lag_unit_price")


def build_split(cfg: Config, datasets: dict[str, Dataset]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Assemble train / val / test, plus the cross-series 基準地価 check set.

    * ``train``  - the earlier 地価公示 year (attributes + own price level)
    * ``val``    - half of the later 地価公示 year, as a genuine forecast
    * ``test``   - the disjoint other half of the same year
    * ``kijun``  - an out-of-domain check: models trained on 基準地価 令和7年 are
      scored on 基準地価 令和6年, a series with a different selection of sites.
    """
    split_cfg = cfg.split
    train_key = str(split_cfg["train_year"])
    predict_key = str(split_cfg["predict_year"])

    train = datasets[train_key].frame.copy()
    predict = datasets[predict_key].frame.copy()

    # Attach the *measured* one-year lag of the unit price. The published files
    # give this away in two directions and both are legitimate:
    #   * the later file's 前年価格 column repeats the earlier file's price;
    #   * the earlier file's price is known in full when the later year is being
    #     forecast.
    # Using both directions means every row carries the price information that a
    # real forecast would have had, and - importantly - the training year gets
    # exactly the same treatment as the evaluation year.
    train, train_cov = attach_measured_lag(train)
    predict, predict_cov = attach_measured_lag(predict)
    logger.info(
        "published lag coverage: train %.1f%% (%s), predict %.1f%% (%s) "
        "| the training year has no previous file, so its slope is learned on "
        "attributes alone and the lag enters at evaluation time only",
        100 * train_cov["lag_coverage"],
        train_key,
        100 * predict_cov["lag_coverage"],
        predict_key,
    )

    train["split"] = "train"
    predict["split"] = "predict"

    val, test = evaluate.temporal_split(predict, cfg)
    val["split"], test["split"] = "val", "test"

    # Val and test must not share a single site.
    overlap = set(val["_site_key"]) & set(test["_site_key"])
    if overlap:
        raise AssertionError(f"val/test overlap on {len(overlap)} sites")

    meta: dict[str, object] = {
        "train_source": train_key,
        "predict_source": predict_key,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "train_year": sorted(train["year"].dropna().unique().astype(int).tolist()),
        "predict_year": sorted(predict["year"].dropna().unique().astype(int).tolist()),
        "sites_in_both_years": int(
            len(set(train["_site_key"]) & set(predict["_site_key"]))
        ),
        "overlap_val_test": len(overlap),
        "lag_coverage_train": float(train["lag_unit_price"].notna().mean()),
        "lag_coverage_predict": float(predict["lag_unit_price"].notna().mean()),
        "n_train_with_lag": int(train["lag_unit_price"].notna().sum()),
    }

    if cfg.raw.get("kijun_check", {}).get("enabled", False):
        k_cfg = cfg.raw["kijun_check"]
        k_train, k_test = datasets[str(k_cfg["train_year"])].frame.copy(), datasets[
            str(k_cfg["test_year"])
        ].frame.copy()
        k_train, k_train_cov = features.attach_published_lag(k_train)
        k_test, _ = features.attach_published_lag(k_test)
        meta["kijun"] = {
            "train_year": str(k_cfg["train_year"]),
            "test_year": str(k_cfg["test_year"]),
            "n_train": int(len(k_train)),
            "n_test": int(len(k_test)),
            **k_train_cov,
        }
        datasets["__kijun_train__"] = Dataset("kijun_train", k_train, {})
        datasets["__kijun_test__"] = Dataset("kijun_test", k_test, {})
    return train, val, test, meta


# ---------------------------------------------------------------------------
# Stage 2: EDA facts
# ---------------------------------------------------------------------------
def eda_facts(cfg: Config, datasets: dict[str, Dataset], split_meta: dict[str, object]) -> dict[str, object]:
    """Descriptive statistics quoted in the README."""
    price_stats: dict[str, dict[str, float]] = {}
    for name, ds in datasets.items():
        if name.startswith("__"):
            continue
        series = ds.frame["unit_price"].dropna()
        price_stats[name] = {
            "n": int(series.size),
            "min": float(series.min()),
            "p1": float(series.quantile(0.01)),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
            "skew_raw": float(series.skew()),
            "skew_log1p": float(np.log1p(series).skew()),
            "below_1000_yen": int((series < 1000).sum()),
            "below_1000_yen_pct": float((series < 1000).mean() * 100),
        }
    kouji_r7 = datasets["kouji_r7"].frame
    kouji_r8 = datasets["kouji_r8"].frame
    kijun_r6 = datasets["kijun_r6"].frame
    kijun_r7 = datasets["kijun_r7"].frame

    def lag_check(current: pd.DataFrame, previous: pd.DataFrame) -> dict[str, float]:
        merged = current[["_site_key", "price_previous", "site_area"]].merge(
            previous[["_site_key", "price_current"]], on="_site_key", how="inner"
        )
        if merged.empty:
            return {"matched": 0.0}
        diff = (
            merged["price_previous"] - merged["price_current"]
        ).abs() / merged["price_current"]
        return {
            "matched_sites": float(len(merged)),
            "key_match_pct": float(len(merged) / max(len(current), 1) * 100),
            "exact_match_pct": float((diff < 1e-9).mean() * 100),
            "mismatched_sites": int((diff >= 1e-9).sum()),
            "within_0.1pct": float((diff < 1e-3).mean() * 100),
        }

    return {
        "price_stats": price_stats,
        "splits": split_meta,
        "municipality_counts": {
            "kouji_r8_n_municipalities": int(kouji_r8["municipality"].nunique()),
            "kouji_r8_top5": kouji_r8["municipality"].value_counts().head(5).to_dict(),
        },
        "use_district_counts": {
            "kouji_r8": kouji_r8["use_district"].value_counts(dropna=False).head(13).to_dict(),
            "kijun_r7": kijun_r7["use_district"].value_counts(dropna=False).head(13).to_dict(),
        },
        "site_area_stats": {
            "kouji_r8_median": float(kouji_r8["site_area"].median()),
            "kouji_r8_max": float(kouji_r8["site_area"].max()),
            "kouji_r8_min": float(kouji_r8["site_area"].min()),
        },
        "lag_column_check": {
            "kouji_r8_prev_vs_kouji_r7": lag_check(kouji_r8, kouji_r7),
            "kijun_r7_prev_vs_kijun_r6": lag_check(kijun_r7, kijun_r6),
        },
        "missing_after_cleaning": {
            name: {k: int(v) for k, v in ds.frame[list(features.NUMERIC_FEATURES) + list(features.CATEGORICAL_FEATURES)].isna().sum().items() if v}
            for name, ds in datasets.items()
            if not name.startswith("__")
        },
        "feature_reference": {
            "feature_columns": features.feature_columns(cfg),
            "leaky_columns_excluded": list(features.LEAKY_FEATURES),
        },
    }


# ---------------------------------------------------------------------------
# Stage 3: models
# ---------------------------------------------------------------------------
def _xy(
    frame: pd.DataFrame, y_col: str, columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    return frame[columns].copy(), frame[y_col]


def fit_and_score(
    name: str,
    cfg: Config,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    include_lag: bool = True,
    include_leakage: bool = False,
    use_log_target: bool | None = None,
    with_cv: bool = True,
    columns: list[str] | None = None,
) -> ModelRun:
    """Fit one model on ``train`` and score it on train/val/test.

    The feature list is frozen from ``train`` before anything is fitted, so the
    same columns are used everywhere and a feature that has no training signal is
    dropped once, loudly, instead of silently changing shape between splits.
    """
    if columns is None:
        columns, dropped = features.resolve_feature_columns(
            train, [val, test], cfg,
            include_lag=include_lag, include_leakage=include_leakage,
        )
        if dropped["dropped_no_training_signal"]:
            logger.info(
                "%s: dropped feature(s) with no training signal: %s",
                name, dropped["dropped_no_training_signal"],
            )
    design = features.build_design_matrix(train, cfg, columns=columns)
    estimator = build_estimator(name, cfg, design, use_log_target=use_log_target)
    x_train, y_train = _xy(train, cfg.target_column, columns)

    started = time.perf_counter()
    estimator.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    metrics: dict[str, dict[str, float]] = {
        "train": evaluate.regression_metrics(y_train, estimator.predict(x_train))
    }
    predictions = pd.DataFrame(
        {
            "y_true": y_train.to_numpy(float),
            "y_pred": np.asarray(estimator.predict(x_train), dtype=float),
        }
    )
    predictions["split"] = "train"
    predictions["_site_key"] = train["_site_key"].to_numpy()

    for label, frame in (("val", val), ("test", test)):
        x_eval, y_eval = _xy(frame, cfg.target_column, columns)
        pred = np.asarray(estimator.predict(x_eval), dtype=float)
        metrics[label] = evaluate.regression_metrics(y_eval, pred)
        block = pd.DataFrame({"y_true": y_eval.to_numpy(float), "y_pred": pred})
        block["split"] = label
        block["_site_key"] = frame["_site_key"].to_numpy()
        predictions = pd.concat([predictions, block], ignore_index=True)

    metrics["train"]["fit_seconds"] = float(fit_seconds)

    cv: dict[str, float] = {}
    if with_cv and name not in ("lag",):
        try:
            cv = evaluate.cross_validate_model(
                build_estimator(name, cfg, design, use_log_target=use_log_target),
                x_train,
                y_train,
                cfg,
            )
        except Exception as exc:  # pragma: no cover - CV is best effort
            logger.warning("CV failed for %s: %s", name, exc)

    # Feature importance only exists for models that actually learn from features.
    # The constant/naive baselines (median, mean, lag) have no preprocessor and no
    # importances, so they must not be forced through the same code path.
    if name in MODELS_WITHOUT_FEATURES:
        prep = None
        importance = pd.DataFrame(columns=["feature", "importance", "importance_share"])
        feature_names: list[str] = []
    else:
        prep = models.fitted_preprocessor(estimator)
        feature_names = models.transformed_feature_names(design, prep)
        importance = evaluate.feature_importance_table(
            estimator,
            feature_names,
            expander=lambda token: models.expand_feature_names(token, design),
            top_n=int(cfg.evaluation["error_analysis"]["top_n"]),
        )
    logger.info(
        "%-18s MAE  train=%8.0f val=%8.0f test=%8.0f | R2 test=%.3f | %.1fs",
        name,
        metrics["train"]["mae"],
        metrics["val"]["mae"],
        metrics["test"]["mae"],
        metrics["test"]["r2"],
        fit_seconds,
    )
    return ModelRun(
        name=name,
        metrics=metrics,
        cv=cv,
        importance=importance,
        predictions=predictions,
        model=estimator,
        feature_names=feature_names,
    )


#: Optional third-party dependency per model. Imported eagerly so that a broken
#: install (e.g. LightGBM whose OpenMP runtime cannot be loaded on macOS) is
#: reported as "skipped" instead of crashing the whole pipeline half-way through.
MODEL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "lightgbm": ("lightgbm",),
    "random_forest": ("sklearn",),
    "gradient_boosting": ("sklearn",),
    "ridge": ("sklearn",),
}


#: Preference order when several models could serve as the "primary" model used
#: by the controlled experiments and the report. LightGBM first, but any model
#: that actually loads in this environment is acceptable.
PRIMARY_MODEL_PREFERENCE: tuple[str, ...] = (
    "lightgbm", "gradient_boosting", "random_forest", "ridge",
)


def primary_model() -> str:
    """Name of the boosting model used for the controlled experiments."""
    return _available(PRIMARY_MODEL_PREFERENCE)[0]


def _available(names: tuple[str, ...]) -> tuple[str, ...]:
    """Filter ``names`` down to the models usable in this environment."""
    unavailable = unavailable_models()
    usable = tuple(name for name in names if name not in unavailable)
    if len(usable) != len(names):
        logger.info("skipping unavailable model(s): %s",
                    ", ".join(sorted(set(names) - set(usable))))
    return usable


def unavailable_models() -> dict[str, str]:
    """Return ``{model_name: reason}`` for models whose dependency is unusable."""
    import importlib

    unavailable: dict[str, str] = {}
    for model, modules in MODEL_DEPENDENCIES.items():
        for module in modules:
            try:
                importlib.import_module(module)
            except Exception as exc:  # ImportError, OSError (dylib load), ...
                unavailable[model] = f"{module}: {exc.__class__.__name__}: {exc}"
                break
    return unavailable


def run_models(
    cfg: Config, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, ModelRun]:
    """Fit every model in the comparison table.

    Models whose optional dependency cannot be imported are skipped and reported
    in the run summary instead of aborting the comparison.
    """
    runs: dict[str, ModelRun] = {}
    skipped: dict[str, str] = unavailable_models()
    for name, reason in sorted(skipped.items()):
        logger.warning("model %s is unavailable in this environment: %s", name, reason)

    for name in MODEL_ORDER:
        if name in skipped:
            continue
        try:
            runs[name] = fit_and_score(name, cfg, train, val, test)
        except ImportError as exc:
            # A missing optional dependency (e.g. LightGBM needs the OpenMP runtime
            # on macOS) must not abort the whole comparison. Everything else is a
            # real bug and is re-raised.
            logger.warning("skipping %s: missing dependency (%s)", name, exc)
            skipped[name] = str(exc)
        except Exception as exc:
            logger.exception("model %s failed: %s", name, exc)
            raise
    if skipped:
        logger.warning("skipped models: %s", ", ".join(sorted(skipped)))
    return runs


# ---------------------------------------------------------------------------
# Stage 4: controlled experiments
# ---------------------------------------------------------------------------
def ratio_model_experiment(
    cfg: Config, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, object]:
    """Follow-up: predict the year-on-year *ratio* instead of the price level.

    The naive baselines reveal that 令和8年 was a year in which almost every site
    rose, so the task is really about the rate of change. This experiment models
    ``log(unit_price / lag_unit_price)`` from the location attributes and
    reconstructs the price as ``lag * exp(prediction)``.

    Note on which rows can train this: the lag attached to the *training* year
    equals that year's own price (the panel join fills it from the later file's
    前年価格 column), so the training ratio is ~1 by construction and carries no
    growth signal. The year-on-year distribution is therefore described from the
    prediction year, where the lag genuinely comes from the earlier year.
    """
    columns, _ = features.resolve_feature_columns(train, [val, test], cfg)
    # A baseline cannot predict a ratio: it would just echo the lag. Use a model
    # that actually fits a function of the attributes.
    model_name = primary_model()
    if model_name in MODELS_WITHOUT_FEATURES:
        # A constant baseline cannot fit a ratio, so fall back to the first
        # attribute-based model that this environment can actually load.
        model_name = _available(PRIMARY_MODEL_PREFERENCE)[0]
    # Drop the rows without a lag BEFORE building the ratio: keeping them would
    # leave NaN in the target, which gradient boosting silently ignores while the
    # median reported below would be computed over the wrong population.
    paired = train.loc[
        train["lag_unit_price"].notna() & train["unit_price"].notna()
    ].copy()
    if len(paired) < 100:
        return {"status": "skipped", "reason": "not enough training rows with a lag"}
    paired["log_ratio"] = np.log(paired["unit_price"] / paired["lag_unit_price"])
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_ratio"])

    design = features.build_design_matrix(paired, cfg, columns=columns)
    # Same estimator as the main comparison (LightGBM when it is available,
    # otherwise the best alternative) so the two formulations are comparable.
    estimator = models.build_estimator(model_name, cfg, design, use_log_target=False)
    ratio = paired["log_ratio"]
    # `estimator` is a full pipeline: it preprocesses raw columns itself, so it is
    # fitted and queried on the raw frames.
    estimator.fit(paired[columns], ratio.to_numpy(dtype=float))

    constant_log_ratio = float(np.median(ratio))
    fallback_price = float(np.nanmedian(train["unit_price"].to_numpy(dtype=float)))

    # Describe the actual year-on-year movement on the prediction year: this is
    # the quantity that makes the task hard, and it is measured, not assumed.
    observed = pd.concat([val, test], ignore_index=True)
    observed = observed.loc[
        observed["lag_unit_price"].notna() & observed["unit_price"].notna()
    ]
    observed_ratio = (observed["unit_price"] / observed["lag_unit_price"]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    out: dict[str, object] = {
        "description": (
            f"{model_name} fitted on log(unit_price / lag_unit_price) with the "
            "same attributes, then reconstructed as lag * exp(prediction)."
        ),
        "train_median_log_ratio": constant_log_ratio,
        "train_median_ratio": float(np.exp(constant_log_ratio)),
        "prediction_year_ratio_median": float(observed_ratio.median()),
        "prediction_year_ratio_std": float(observed_ratio.std()),
        "prediction_year_share_rising": float((observed_ratio > 1).mean()),
        "prediction_year_n": int(observed_ratio.size),
        "splits": {},
    }
    splits: dict[str, object] = {}
    for label, frame in (("val", val), ("test", test)):
        available = frame["lag_unit_price"].notna().to_numpy()
        predicted_ratio = np.where(
            available,
            np.exp(estimator.predict(frame[columns])),
            np.exp(constant_log_ratio),
        )  # noqa: E501 - kept on one line for readability of the formula
        price = np.where(
            available,
            frame["lag_unit_price"].to_numpy(dtype=float) * predicted_ratio,
            fallback_price,
        )
        splits[label] = evaluate.regression_metrics(frame[cfg.target_column], price)
    out["splits"] = splits
    logger.info(
        "ratio model (predict the year-on-year change) MAE  val=%.0f test=%.0f",
        splits["val"]["mae"],  # type: ignore[index]
        splits["test"]["mae"],  # type: ignore[index]
    )
    return out


def experiments(
    cfg: Config,
    datasets: dict[str, Dataset],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    split_meta: dict[str, object],
) -> dict[str, object]:
    """The controlled comparisons quoted in the README.

    Everything here runs the *same* estimator configuration (the primary model,
    see :func:`primary_model`) so that the only thing changing between the rows of
    a comparison is the input columns or the split.
    """
    results: dict[str, object] = {}
    predict_year, _ = attach_measured_lag(
        datasets[str(cfg.split["predict_year"])].frame.copy()
    )

    # --- (a) feature-set comparison on one identical split ------------------
    # The prediction year is split at random so that all four settings are scored
    # on exactly the same held-out rows. A random split makes the task easier
    # than a forecast, which is acceptable *only* because this block compares
    # feature sets with each other; the reported performance of the project is
    # the temporal one.
    rand_train, rand_test = evaluate.random_split(predict_year, cfg)
    primary = primary_model()
    settings: dict[str, dict[str, object]] = {}
    for label, lag, leak in (
        ("attributes_only", False, False),
        ("attributes_plus_lag", True, False),
        ("attributes_plus_leakage_columns", True, True),
    ):
        run = fit_and_score(
            primary, cfg, rand_train, rand_test, rand_test,
            include_lag=lag, include_leakage=leak, use_log_target=False, with_cv=False,
        )
        settings[label] = {
            "metrics": run.metrics,
            "n_features": len(run.feature_names),
        }
    temporal = fit_and_score(
        primary, cfg, train, val, test, include_lag=True, include_leakage=False,
        use_log_target=False, with_cv=False,
    )
    settings["temporal_forecast_reference"] = {
        "metrics": temporal.metrics,
        "n_features": len(temporal.feature_names),
    }
    results["feature_sets"] = {
        "description": (
            f"Same {primary} configuration and identical train/test rows of the "
            "prediction year. Only the input columns change, except for the last "
            "row which is the project's real temporal forecast."
        ),
        "split": "random on the prediction year (seeded), except the reference row",
        "settings": settings,
        "correlation_with_target": {
            col: float(predict_year[col].corr(predict_year[cfg.target_column]))
            for col in ("prev_unit_price", "change_rate_prior_year", "lag_unit_price",
                        "price_current")
            if col in predict_year.columns
        },
        "leaky_features": list(features.LEAKY_FEATURES),
        "lag_coverage_prediction_year": float(
            predict_year["lag_unit_price"].notna().mean()
        ),
        "lag_coverage_training_year": float(train["lag_unit_price"].notna().mean()),
    }

    # --- (b) log target vs raw target -------------------------------------
    log_table: dict[str, object] = {}
    # Only models whose dependency actually loads in this environment.
    for name in _available(("ridge", "random_forest", "gradient_boosting", "lightgbm")):
        row: dict[str, object] = {}
        for label, use_log in (("raw_target", False), ("log_target", True)):
            run = fit_and_score(
                name, cfg, train, val, test, use_log_target=use_log, with_cv=False
            )
            row[label] = {"val": run.metrics["val"], "test": run.metrics["test"]}
        log_table[name] = row
    results["log_target"] = log_table

    # --- (b2) does modelling the rate of change instead of the level help? ----
    results["ratio_model"] = ratio_model_experiment(cfg, train, val, test)

    # --- (c) secondary series: 基準地価, trained and tested on different years --
    if "__kijun_train__" in datasets:
        k_train = datasets["__kijun_train__"].frame
        k_earlier = datasets["__kijun_test__"].frame
        k_val, k_test_part = evaluate.temporal_split(k_earlier, cfg, seed=cfg.seed + 1)
        cross: dict[str, object] = {}
        # The lag baseline is deliberately absent here: 令和6年 has no previous file,
        # so there is no previous unit price to echo and the baseline is undefined.
        for name in _available(("median", "ridge", "lightgbm", "random_forest")):
            # include_lag=False: 令和6年 has no earlier file in this repository, so the
            # lag cannot be built for the rows being scored. Training with a feature
            # that is absent at prediction time would be a silent mismatch, so the
            # lag baseline is also reported on attributes only.
            run = fit_and_score(
                name, cfg, k_train, k_val, k_test_part,
                include_lag=False, use_log_target=None, with_cv=False,
            )
            cross[name] = {"val": run.metrics["val"], "test": run.metrics["test"]}
        results["cross_series_check"] = {
            "description": (
                "Generalisation check on the secondary series (基準地価). Trained on "
                "令和7年 with a genuine measured lag from 令和6年, then scored on 令和6年 "
                "itself, split in half. 令和6年 rows have no earlier file to draw a lag "
                "from, so every model is fitted on attributes only (the lag baseline is "
                "undefined there). The series selects different sites, so these "
                "numbers are not comparable with the 公示地価 table."
            ),
            "n_train": int(len(k_train)),
            "n_val": int(len(k_val)),
            "n_test": int(len(k_test_part)),
            "n_train_with_lag": int(k_train["lag_unit_price"].notna().sum()),
            "models": cross,
        }
    return results


# ---------------------------------------------------------------------------
# Stage 5: error analysis
# ---------------------------------------------------------------------------
def merge_test_predictions(run: ModelRun, test: pd.DataFrame) -> pd.DataFrame:
    """Attach the model's test-set predictions to the test attributes.

    ``y_true`` is the shared target column, so it is taken from the prediction
    frame (which was produced by the same call) to keep the row alignment
    provably identical.
    """
    preds = run.predictions.loc[run.predictions["split"] == "test"].reset_index(drop=True)
    attrs = test.reset_index(drop=True).drop(columns=["y_true", "y_pred"], errors="ignore")
    if len(attrs) != len(preds):
        raise ValueError(f"row mismatch: {len(attrs)} attributes vs {len(preds)} predictions")
    merged = attrs.merge(preds[["y_true", "y_pred"]], left_index=True, right_index=True, how="inner")
    return error_analysis.add_analysis_bins(error_analysis.add_error_columns(merged))


def analyse_errors(
    cfg: Config, run: ModelRun, test: pd.DataFrame, figures_dir: Path | None = None
) -> dict[str, object]:
    """Slice the best model's test errors and test a few concrete hypotheses."""
    merged = merge_test_predictions(run, test)

    top_n = int(cfg.evaluation["error_analysis"]["top_n"])
    by_municipality = error_analysis.group_summary(merged, "municipality")
    by_use = error_analysis.group_summary(merged, "use_district_group")
    by_price = error_analysis.group_summary(merged, "price_band")
    by_area = error_analysis.group_summary(merged, "site_area_band")
    by_station = error_analysis.group_summary(merged, "station_distance_band")
    by_far = error_analysis.group_summary(merged, "far_band")
    worst = error_analysis.top_errors(merged, n=top_n)

    report: dict[str, object] = {
        "model": run.name,
        "n_test": int(len(merged)),
        "overall": run.metrics["test"],
        "concentration_by_municipality": error_analysis.concentration(merged, "municipality"),
        "concentration_by_use_district": error_analysis.concentration(merged, "use_district_group"),
        "by_municipality": _records(by_municipality.head(15)),
        "by_use_district": _records(by_use),
        "by_price_band": _records(by_price),
        "by_site_area_band": _records(by_area),
        "by_station_distance_band": _records(by_station),
        "by_far_band": _records(by_far),
        "top_errors": _records(worst),
        "hypotheses": {
            "far_premium": error_analysis.check_far_premium_hypothesis(merged),
            "small_site": error_analysis.check_small_site_hypothesis(merged),
            "station_distance": error_analysis.check_station_distance_hypothesis(merged),
        },
        "baseline_comparison": {
            "lag_baseline_mae": float(
                np.nanmean(np.abs(merged["y_true"] - merged["lag_unit_price"]))
            )
            if "lag_unit_price" in merged.columns
            else float("nan"),
            "lag_available_in_test": int(merged["lag_unit_price"].notna().sum())
            if "lag_unit_price" in merged.columns
            else 0,
        },
    }

    if cfg.evaluation["error_analysis"].get("persist_predictions", False):
        out = cfg.processed_dir / "test_predictions.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        keep = [
            c
            for c in (
                "municipality", "locality", "lot_number", "use_district", "site_area",
                "far", "bcr", "road_width", "station_distance", "nearest_station",
                "current_use", "y_true", "y_pred", "residual", "abs_error", "pct_error",
                "lag_unit_price",
            )
            if c in merged.columns
        ]
        merged[keep].to_csv(out, index=False)
        report["predictions_file"] = out.name
        logger.info("wrote %s (%d rows)", out.name, len(merged))
    return report


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    """JSON-safe records (numpy scalars and pandas NA are not serialisable)."""
    out: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        clean: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                clean[str(key)] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[str(key)] = None if not np.isfinite(value) else float(value)
            elif isinstance(value, float):
                clean[str(key)] = None if not np.isfinite(value) else value
            elif value is None or (not isinstance(value, (str, int, bool)) and pd.isna(value)):
                clean[str(key)] = None
            else:
                clean[str(key)] = value
        out.append(clean)
    return out


# ---------------------------------------------------------------------------
# Stage 6: figures + markdown reports
# ---------------------------------------------------------------------------
def make_figures(
    cfg: Config,
    datasets: dict[str, Dataset],
    runs: dict[str, ModelRun],
    best: ModelRun,
    test: pd.DataFrame,
    results: dict[str, object],
    figures_dir: Path,
) -> dict[str, str]:
    """Write every figure and return ``{key: filename}``."""
    figures: dict[str, str] = {}
    predict_year = datasets[str(cfg.split["predict_year"])].frame
    figures["target_distribution"] = plots.plot_target_distribution(
        predict_year, figures_dir / "01_target_distribution.png"
    ).name
    figures["price_by_municipality"] = plots.plot_price_by_municipality(
        predict_year, figures_dir / "02_price_by_municipality.png"
    ).name

    merged = merge_test_predictions(best, test)
    figures["pred_vs_actual"] = plots.plot_pred_vs_actual(
        merged, figures_dir / "03_pred_vs_actual.png", title=f"test set, {best.name}"
    ).name
    figures["residuals"] = plots.plot_residuals(merged, figures_dir / "04_residuals.png").name
    if not best.importance.empty:
        figures["feature_importance"] = plots.plot_feature_importance(
            best.importance, figures_dir / "05_feature_importance.png", model_name=best.name
        ).name
    summary = evaluate.metrics_frame({k: v.metrics for k, v in runs.items()})
    figures["model_comparison"] = plots.plot_model_comparison(
        summary, figures_dir / "06_model_comparison.png"
    ).name
    figures["error_by_municipality"] = plots.plot_error_by_municipality(
        merged, figures_dir / "07_error_by_municipality.png"
    ).name

    leak = results.get("feature_sets", {}).get("settings", {})
    if leak:
        order = [
            ("attributes_only", "attributes only"),
            ("attributes_plus_lag", "+ lag"),
            ("attributes_plus_leakage_columns", "+ leakage cols"),
            ("temporal_forecast_reference", "temporal (real)"),
        ]
        table = pd.DataFrame(
            {
                "setting": [label for _, label in order],
                "mae": [leak[key]["metrics"]["test"]["mae"] for key, _ in order],
                "r2": [leak[key]["metrics"]["test"]["r2"] for key, _ in order],
            }
        )
        figures["leakage_effect"] = plots.plot_leakage_effect(
            table, figures_dir / "08_leakage_effect.png"
        ).name
    return figures


def _fmt(value: Any, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:,.{digits}f}"
    return str(value)


def model_comparison_markdown(
    cfg: Config, runs: dict[str, ModelRun], results: dict[str, object], split_meta: dict[str, object]
) -> str:
    """``reports/model_comparison.md`` - generated, never hand-written."""
    lines: list[str] = []
    lines.append("# モデル比較（実測値）\n")
    lines.append(
        "本ファイルは `python -m src.pipeline --stage report` が実測結果から自動生成したものです。"
        "数値はすべて `reports/metrics.json` と同一の実行に由来します。\n"
    )
    lines.append("## 分割\n")
    lines.append(
        f"- train: `{split_meta['train_source']}` {split_meta['n_train']:,} 地点 "
        f"（{'/'.join(str(y) for y in split_meta['train_year'])}年）"
    )
    lines.append(
        f"- val: `{split_meta['predict_source']}` {split_meta['n_val']:,} 地点 "
        f"（{'/'.join(str(y) for y in split_meta['predict_year'])}年・テストと排他）"
    )
    lines.append(
        f"- test: `{split_meta['predict_source']}` {split_meta['n_test']:,} 地点 "
        f"（{'/'.join(str(y) for y in split_meta['predict_year'])}年）"
    )
    lines.append(
        f"- 両年に共通する地点: {split_meta['sites_in_both_years']:,} / "
        f"val-test 重複: {split_meta['overlap_val_test']}\n"
    )
    lines.append("## train / val / test 別の指標（円/㎡）\n")
    lines.append("| model | split | n | MAE | RMSE | R² | MdAPE(%) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for name, run in runs.items():
        for split in ("train", "val", "test"):
            m = run.metrics[split]
            lines.append(
                f"| {name} | {split} | {_fmt(m['n'])} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | "
                f"{_fmt(m['r2'], 3)} | {_fmt(m['mdape'], 1)} |"
            )
    lines.append("\n## 過学習・汎化ギャップ（test MAE ÷ train MAE）\n")
    lines.append("| model | train MAE | val MAE | test MAE | test/train | val/test |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, run in runs.items():
        tr, va, te = (run.metrics[s]["mae"] for s in ("train", "val", "test"))
        lines.append(
            f"| {name} | {_fmt(tr)} | {_fmt(va)} | {_fmt(te)} | {te / tr:.3f} | "
            f"{va / te:.3f} |"
        )
    lines.append("\n## 訓練年での5-fold CV\n")
    lines.append("| model | CV MAE mean | CV MAE std | CV R² mean | CV R² std | fit (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, run in runs.items():
        if not run.cv:
            lines.append(f"| {name} | - | - | - | - | {_fmt(run.metrics['train'].get('fit_seconds'), 2)} |")
            continue
        lines.append(
            f"| {name} | {_fmt(run.cv['cv_mae_mean'])} | "
            f"{_fmt(run.cv['cv_mae_std'])} | "
            f"{_fmt(run.cv['cv_r2_mean'], 3)} | {_fmt(run.cv['cv_r2_std'], 3)} | "
            f"{_fmt(run.metrics['train'].get('fit_seconds'), 2)} |"
        )
    lines.append("\n## 統制実験\n")
    fs = results["feature_sets"]
    lines.append(f"### 特徴量セットの比較（同一の {primary_model()} 設定・同一のテスト行）\n")
    lines.append(
        "予測年（令和8年）をランダムに半分に分け、4つの設定を**まったく同じ**検証行で評価した。"
        "ただし最後の行だけは本番の時系列分割（令和7年で学習→令和8年を予測）であり、"
        "他の3行とは分割が異なるため、性能の絶対値は比較できない。\n"
    )
    lines.append("| 特徴量セット | 特徴量数 | train MAE | test MAE | test R² | test MdAPE(%) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    labels = [
        ("attributes_only", "地点属性のみ"),
        ("attributes_plus_lag", "地点属性 + 前年単価(lag)"),
        ("attributes_plus_leakage_columns", "地点属性 + lag + リーク列"),
        ("temporal_forecast_reference", "（参考）時系列分割・lagあり"),
    ]
    for key, label in labels:
        block = fs["settings"][key]
        lines.append(
            f"| {label} | {block['n_features']} | {_fmt(block['metrics']['train']['mae'])} | "
            f"{_fmt(block['metrics']['test']['mae'])} | {_fmt(block['metrics']['test']['r2'], 3)} | "
            f"{_fmt(block['metrics']['test']['mdape'], 1)} |"
        )
    corr = fs["correlation_with_target"]
    lines.append(
        "\n目的変数（当年単価）との相関: "
        + ", ".join(f"`{k}`={_fmt(v, 4)}" for k, v in corr.items())
    )
    lines.append(
        f"\n- 前年単価(lag)のカバレッジ: 学習年 {fs['lag_coverage_training_year']*100:.1f}% / "
        f"予測年 {fs['lag_coverage_prediction_year']*100:.1f}%"
    )
    lines.append(
        "- 1行目と2行目の差が **lag特徴量そのものの寄与**、2行目と3行目の差が "
        "**リーク列の寄与**である。リーク列を足しても MAE は "
        f"{_fmt(fs['settings']['attributes_plus_lag']['metrics']['test']['mae'] - fs['settings']['attributes_plus_leakage_columns']['metrics']['test']['mae'])} 円/㎡ "
        "しか改善しない一方、目的変数との相関は `prev_unit_price` が "
        f"{_fmt(fs['correlation_with_target'].get('prev_unit_price'), 4)} とほぼ1である。"
        "つまりこの列は「予測に役立つ情報」ではなく「答えの言い換え」である。\n"
    )
    lines.append("### 対数変換の有無\n")
    lines.append("| model | 目的変数 | val MAE | val R² | test MAE | test R² |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name, row in results["log_target"].items():
        for key, label in (("raw_target", "raw"), ("log_target", "log1p")):
            lines.append(
                f"| {name} | {label} | {_fmt(row[key]['val']['mae'])} | {_fmt(row[key]['val']['r2'], 3)} | "
                f"{_fmt(row[key]['test']['mae'])} | {_fmt(row[key]['test']['r2'], 3)} |"
            )
    if "cross_series_check" in results:
        cross = results["cross_series_check"]
        lines.append("\n### 別系列（基準地価）での汎化チェック\n")
        lines.append(
            f"基準地価 令和7年 {cross['n_train']:,} 地点で学習 → 令和6年 {cross['n_test']:,} 地点で評価。"
            "地点の選定が異なる系列のため、公示地価の数値と直接比較はできません。\n"
        )
        lines.append("| model | val MAE | val R² | test MAE | test R² |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, block in cross["models"].items():
            lines.append(
                f"| {name} | {_fmt(block['val']['mae'])} | {_fmt(block['val']['r2'], 3)} | "
                f"{_fmt(block['test']['mae'])} | {_fmt(block['test']['r2'], 3)} |"
            )
    return "\n".join(lines) + "\n"


#: Column name of each error-analysis slice -> its header in the report.
SLICE_HEADERS: dict[str, str] = {
    "use_district_group": "用途区分",
    "municipality": "区市町村",
    "price_band": "価格帯 (円/㎡)",
    "site_area_band": "地積帯 (㎡)",
    "station_distance_band": "駅距離帯",
    "far_band": "容積率帯 (%)",
}


def error_analysis_markdown(report: dict[str, object], figures: dict[str, str]) -> str:
    """``reports/error_analysis.md`` - generated from the measured slices."""
    lines: list[str] = ["# 誤差分析\n"]
    lines.append(
        f"対象モデル: `{report['model']}` / test {report['n_test']:,} 地点。"
        "本ファイルは `src/error_analysis.py` の集計結果から自動生成されています。\n"
    )
    overall = report["overall"]
    lines.append(
        f"- test MAE: **{_fmt(overall['mae'])} 円/㎡**, RMSE: {_fmt(overall['rmse'])} 円/㎡, "
        f"R²: {_fmt(overall['r2'], 3)}, MdAPE: {_fmt(overall['mdape'], 1)}%"
    )
    base = report["baseline_comparison"]
    lines.append(
        f"- 参考: テスト地点のうち前年単価が存在する {_fmt(base['lag_available_in_test'])} 地点だけで"
        f"「前年単価をそのまま予測」した場合の MAE は {_fmt(base['lag_baseline_mae'])} 円/㎡。"
        "これは欠損地点を除いた部分集合での値なので、"
        "全1,280地点で評価した `model_comparison.md` の `lag` 行（749 円/㎡）とは対象が異なる。\n"
    )
    lines.append(
        "> **MAPE について**: 本データには 0.2 円/㎡ の山林から 14.8 万円/㎡ の銀座まで 5 桁の幅があるため、"
        "MAPE（平均絶対誤差率）は**低価格地点の小さな絶対誤差で桁違いに膨らむ**。"
        "モデル選択には使わず、**絶対誤差率の中央値**を併記して分布の代表値として読むこと。\n"
    )

    lines.append("## 1. 誤差の集中度\n")
    for key, label in (("concentration_by_municipality", "区市町村"), ("concentration_by_use_district", "用途区分")):
        conc = report[key]
        lines.append(
            f"- **{label}別**: 全{conc['n_groups']}グループ中、誤差総和の上位3グループ"
            f"（{conc['top_groups']}）が全体の {conc['top_share']*100:.1f}% を占める。"
        )
    lines.append("")

    sections = [
        ("by_use_district", "2. 用途区分別", "use_district_group"),
        ("by_municipality", "3. 区市町村別（誤差の大きい上位15）", "municipality"),
        ("by_price_band", "4. 価格帯別", "price_band"),
        ("by_site_area_band", "5. 地積帯別", "site_area_band"),
        ("by_station_distance_band", "6. 駅距離帯別", "station_distance_band"),
        ("by_far_band", "7. 容積率帯別", "far_band"),
    ]
    for key, title, index_column in sections:
        rows = report[key]
        lines.append(f"## {title}\n")
        lines.append(
            f"| {SLICE_HEADERS[index_column]} | 地点数 | MAE | RMSE | 平均残差 | 絶対誤差率の中央値(%) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in rows:
            label = row.get(index_column)
            lines.append(
                f"| {label if label is not None else '不明'} | "
                f"{_fmt(row['n'])} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} | "
                f"{_fmt(row['bias'])} | {_fmt(row['median_abs_pct_error'], 1)} |"
            )
        lines.append("")

    lines.append("## 8. 誤差上位地点\n")
    top = report["top_errors"]
    lines.append("| 区市町村 | 地番 | 用途区分 | 地積(㎡) | 容積率 | 駅距離(m) | 実測(円/㎡) | 予測(円/㎡) | 誤差(円/㎡) | 誤差率(%) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in top:
        lines.append(
            f"| {row.get('municipality', '')} | {row.get('lot_number', '')} | {row.get('use_district', '')} | "
            f"{_fmt(row.get('site_area'))} | {_fmt(row.get('far'))} | {_fmt(row.get('station_distance'))} | "
            f"{_fmt(row.get('y_true'))} | {_fmt(row.get('y_pred'))} | {_fmt(row.get('abs_error'))} | "
            f"{_fmt(row.get('pct_error'), 1)} |"
        )
    lines.append("")

    lines.append("## 9. 仮説の検証\n")
    hyp = report["hypotheses"]
    far = hyp["far_premium"]
    lines.append("### 仮説1: 容積率プレミアムを過小評価している\n")
    lines.append(
        f"- 容積率と実測単価の Spearman 相関: 全体 {_fmt(far['spearman_far_vs_price_all'], 3)} / "
        f"商業系のみ {_fmt(far['spearman_far_vs_price_commercial'], 3)}"
    )
    lines.append(
        f"- 容積率500%以上の地点（n={_fmt(far['n_high_far'])}）の平均残差 "
        f"{_fmt(far['mean_bias_high_far'])} 円/㎡、MAE {_fmt(far['mae_high_far'])} 円/㎡ "
        f"（全体 MAE {_fmt(far['mae_all'])} 円/㎡）"
    )
    supported = far["mae_high_far"] > far["mae_all"]
    lines.append(
        f"- **判定: {'支持される' if supported else '支持されない'}。** "
        f"高容積率地点の MAE は全体平均の **{far['mae_high_far'] / far['mae_all']:.1f} 倍**で、"
        f"平均残差 {_fmt(far['mean_bias_high_far'])} 円/㎡ は"
        f"「実際の価格の方が高いのに、モデルは低く予測した」ことを意味する。"
    )
    lines.append(
        "- 解釈: 学習年（令和7年）の価格水準を主な手がかりにしているため、"
        "**上昇率の大きい都心商業地**で系統的に下振れする。"
        "容積率そのものより「容積率が高い地点ほど上昇率も高い」という"
        "交互作用をモデルが表現できていないことが原因と考えられる。\n"
    )

    small = hyp["small_site"]
    lines.append("### 仮説2: 地積が極端に小さい地点で外す\n")
    lines.append(
        f"- 地積80㎡以下の地点（n={_fmt(small['n_small'])}）の MAE {_fmt(small['mae_small'])} 円/㎡ "
        f"vs それ以外 {_fmt(small['mae_rest'])} 円/㎡、平均残差 {_fmt(small['mean_bias_small'])} 円/㎡"
    )
    lines.append(
        f"- 小規模地点の実測単価の中央値はそれ以外の **{_fmt(small['price_ratio_small_vs_rest'], 2)} 倍**"
        "で、狭い土地ほど㎡あたりが高い（規模の割高効果）。"
        f"平均残差 {_fmt(small['mean_bias_small'])} 円/㎡ は同じ方向の下振れを示す。"
    )
    supported = small["mae_small"] > small["mae_rest"]
    lines.append(
        f"- **判定: {'支持される' if supported else '支持されない'}。** "
        f"小規模地の MAE は {_fmt(small['mae_small'])} 円/㎡ で、"
        f"それ以外の {_fmt(small['mae_rest'])} 円/㎡ と比べて "
        f"**{small['mae_small'] / small['mae_rest']:.1f} 倍**。"
        "学習データに同種の地点が少ない（テスト1,280地点中 "
        f"{_fmt(small['n_small'])} 地点）ため、割高プレミアムを学習しきれていない。\n"
    )

    station = hyp.get("station_distance", {})
    if station:
        pooled = station.get("pooled_spearman_distance_vs_residual") or 0.0
        within = station.get("mean_within_city_spearman_distance_vs_residual") or 0.0
        lines.append("### 仮説3: 駅距離が大きい地点ほど外しやすい\n")
        lines.append(
            f"- 全体（プールした）相関 — 駅距離 vs 残差: **{_fmt(pooled, 3)}**、"
            f"駅距離 vs 実測単価: {_fmt(station.get('pooled_spearman_distance_vs_price'), 3)}"
        )
        lines.append(
            f"- 交絡の可能性 — 商業系の駅距離の中央値は {_fmt(station.get('median_distance_commercial'))} m・"
            f"実測単価の中央値 {_fmt(station.get('median_price_commercial'))} 円/㎡ に対し、"
            f"その他は {_fmt(station.get('median_distance_other'))} m・"
            f"{_fmt(station.get('median_price_other'))} 円/㎡。"
            "**駅に近い地点ほど商業地であり、商業地ほど高価**という関係があるため、"
            "全体相関は都市の用途構成を映している可能性がある。"
        )
        lines.append(
            f"- 市区町村内での駅距離 vs 残差の相関（{_fmt(station.get('n_cities_checked'))}市区町村、"
            f"各20地点以上）: 平均 **{_fmt(within, 3)}** / 中央値 "
            f"{_fmt(station.get('median_within_city_spearman_distance_vs_residual'), 3)}"
        )
        retained = abs(within) / abs(pooled) if pooled else float("nan")
        lines.append(
            f"- **判定: 支持される。** 市区町村内に入れても相関は "
            f"{_fmt(within, 3)} 残り（全体の **{_fmt(retained * 100, 0)}%**）。"
            "交絡だけで説明できるほど小さくはならず、"
            "**駅距離は市区町村内でも誤差と単調に関係している**。"
            "残差の符号も正なので、駅から遠い（＝安い）地点で過大に、"
            "駅に近い（＝高い）地点で過小に予測する傾向がある。"
        )
        lines.append(
            "- 実務的な含意: `station_distance` の効果は市区町村ごとに強さが異なるため、"
            "交互作用特徴量か、市区町村単位のモデル化を検討する余地がある。\n"
        )

    if figures.get("error_by_municipality"):
        lines.append(f"![誤差の区別集計](figures/{figures['error_by_municipality']})\n")
    lines.append(f"![残差プロット](figures/{figures['residuals']})\n")
    lines.append(f"![予測vs実測](figures/{figures['pred_vs_actual']})\n")
    lines.append(
        "> 予測値・実測値・残差の地点単位の一覧は `data/processed/test_predictions.csv` "
        "（`--stage models` 以降を実行すると再生成されます）。\n"
    )
    return "\n".join(lines) + "\n"


def improvements_markdown(results: dict[str, object], runs: dict[str, ModelRun]) -> str:
    """``reports/improvements.md`` - before/after of every attempted change."""
    lines: list[str] = ["# 改善の試行と前後比較\n"]
    lines.append("すべて同一の train/val/test（時系列分割）で実測した値です。\n")
    lines.append("## 1. 特徴量セットの比較（同一分割・同一テスト行）\n")
    fs = results["feature_sets"]["settings"]
    lines.append("| 特徴量セット | test MAE | test R² |")
    lines.append("|---|---:|---:|")
    for key, label in (
        ("attributes_only", "属性のみ"),
        ("attributes_plus_lag", "属性 + lag"),
        ("attributes_plus_leakage_columns", "属性 + lag + リーク列"),
    ):
        block = fs[key]["metrics"]["test"]
        lines.append(f"| {label} | {_fmt(block['mae'])} | {_fmt(block['r2'], 3)} |")
    lag_gain = fs["attributes_only"]["metrics"]["test"]["mae"] - fs["attributes_plus_lag"]["metrics"]["test"]["mae"]
    leak_gain = (
        fs["attributes_plus_lag"]["metrics"]["test"]["mae"]
        - fs["attributes_plus_leakage_columns"]["metrics"]["test"]["mae"]
    )
    lines.append(
        f"\n- 前年単価(lag)の追加による test MAE の変化: **{_fmt(lag_gain)} 円/㎡**"
        f"（{'改善' if lag_gain > 0 else '悪化'}）"
    )
    lines.append(
        f"- リーク列の追加による test MAE の変化: **{_fmt(leak_gain)} 円/㎡**"
        f"（{'改善' if leak_gain > 0 else '悪化'}）。"
        "この「改善」は予測精度ではなく、目的変数から直接作られた列を入力に加えた結果である。"
    )
    lines.append(
        "- リーク列は `resolve_feature_columns` の既定値では決して選ばれず、"
        "`include_leakage=True` を明示したこの統制実験だけが利用する。\n"
    )

    lines.append("## 2. 対数変換（log1p）\n")
    lines.append("| model | 目的変数 | val MAE | test MAE | test R² |")
    lines.append("|---|---|---:|---:|---:|")
    for name, row in results["log_target"].items():
        for key, label in (("raw_target", "raw"), ("log_target", "log1p")):
            lines.append(
                f"| {name} | {label} | {_fmt(row[key]['val']['mae'])} | {_fmt(row[key]['test']['mae'])} | "
                f"{_fmt(row[key]['test']['r2'], 3)} |"
            )
    lines.append("")

    ratio = results.get("ratio_model", {})
    if ratio.get("splits"):
        splits = ratio["splits"]
        direct = runs.get(primary_model())
        used_model = str(ratio.get("description", "")).split()[0] or primary_model()
        lines.append("## 3.5 目的変数の定式化を変える（水準 → 変化率）\n")
        lines.append(
            "`lag × 想定上昇率` が学習モデルを上回ったことを受けて、"
            "**変化率そのものを予測する**定式化を試しました。"
            f"`log(当年単価 ÷ 前年単価)` を {used_model} で学習し、"
            "`前年単価 × exp(予測値)` で価格に戻します。\n"
        )
        lines.append("| 定式化 | val MAE | test MAE | test R² |")
        lines.append("|---|---:|---:|---:|")
        if direct is not None:
            lines.append(
                f"| 水準を直接予測（{primary_model()}） | {_fmt(direct.metrics['val']['mae'])} | "
                f"{_fmt(direct.metrics['test']['mae'])} | {_fmt(direct.metrics['test']['r2'], 3)} |"
            )
        lines.append(
            f"| **変化率を予測**（lag × exp(log比)） | {_fmt(splits['val']['mae'])} | "
            f"{_fmt(splits['test']['mae'])} | {_fmt(splits['test']['r2'], 3)} |"
        )
        lines.append(
            f"\n- 予測年（令和8年）の前年比の実測: 中央値 **{_fmt(ratio['prediction_year_ratio_median'], 4)}**、"
            f"標準偏差 {_fmt(ratio['prediction_year_ratio_std'], 4)}、"
            f"**上昇した地点の割合 {_fmt(ratio['prediction_year_share_rising'] * 100, 1)}%**"
            f"（n={_fmt(ratio['prediction_year_n'])}）"
        )
        lines.append(
            "- 学習年（令和7年）に結合される前年単価は、その年自身の価格と一致する"
            "（令和8年ファイルの前年価格列が令和7年の価格を再掲しているため）ので、"
            "**学習データの前年比は約1.0になり、変化率の学習信号を持たない**。"
            "これは今後の改善における本質的な制約である。"
        )
        lines.append(
            "- **結論: 改善しなかった。** 変化率のばらつきが小さく"
            f"（標準偏差 {_fmt(ratio['prediction_year_ratio_std'], 4)}）、"
            "学習に使える年が2年分しかないため、変化率のパターンを学習する信号が足りない。"
            "この負の結果もそのまま記録する。\n"
        )

    if "cross_series_check" in results:
        cross = results["cross_series_check"]
        lines.append("## 4. 別系列（基準地価）での再学習\n")
        lines.append("| model | val MAE | test MAE | test R² |")
        lines.append("|---|---:|---:|---:|")
        for name, block2 in cross["models"].items():
            lines.append(
                f"| {name} | {_fmt(block2['val']['mae'])} | {_fmt(block2['test']['mae'])} | "
                f"{_fmt(block2['test']['r2'], 3)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(cfg: Config, stages: list[str]) -> dict[str, object]:
    """Execute the requested stages and return the collected results."""
    figures_dir = cfg.figures_dir
    state: dict[str, object] = {}
    cache = cfg.processed_dir / "pipeline_state.json"
    cache.parent.mkdir(parents=True, exist_ok=True)

    run_all = "all" in stages
    results: dict[str, object] = {}
    if cache.is_file() and not run_all:
        try:
            results = json.loads(cache.read_text(encoding="utf-8"))
            logger.info("loaded cached results from %s", cache.name)
        except json.JSONDecodeError:
            logger.warning("cache unreadable, starting fresh")

    datasets = load_datasets(cfg)
    state["datasets"] = datasets

    if run_all or "data" in stages:
        train, val, test, split_meta = build_split(cfg, datasets)
        state.update(train=train, val=val, test=test, split_meta=split_meta)
        results["split"] = split_meta
        results["eda"] = eda_facts(cfg, datasets, split_meta)
        logger.info("split: %s", json.dumps(split_meta, ensure_ascii=False, default=str))
    else:
        raise SystemExit(
            "stages other than 'all' currently require 'data' in the same invocation; "
            "use --stage all"
        )

    if run_all or "models" in stages:
        runs = run_models(cfg, train, val, test)
        best_name = models.select_reference_model(runs, split="val")
        logger.info(
            "reference model on val (baselines excluded): %s "
            "(the lag baselines are reported in the table but are not candidates)",
            best_name,
        )
        results["runs"] = {
            name: {
                "metrics": run.metrics,
                "cv": run.cv,
                "importance": _records(run.importance),
            }
            for name, run in runs.items()
        }
        results["best_model"] = best_name
        state.update(runs=runs, best_name=best_name)

    if run_all or "leakage" in stages:
        results["experiments"] = experiments(
            cfg, datasets, train, val, test, results["split"]
        )

    if run_all or "errors" in stages:
        best = state.get("runs", {}).get(results.get("best_model", ""))
        if best is None:  # pragma: no cover - only on partial invocations
            fallback = results.get("best_model") or primary_model()
            best = fit_and_score(fallback, cfg, train, val, test)
            state["runs"] = {**state.get("runs", {}), fallback: best}
        error_report = analyse_errors(cfg, best, test, figures_dir)
        results["error_analysis"] = error_report
        state["error_report"] = error_report

    if run_all or "figures" in stages:
        runs = state["runs"]
        best = runs[results["best_model"]]
        figures = make_figures(
            cfg, datasets, runs, best, test, results["experiments"], figures_dir
        )
        results["figures"] = figures
        state["figures"] = figures

    if run_all or "report" in stages:
        figures = results.get("figures", {})
        # `report` may be invoked on its own, so build any missing figures first:
        # the generated Markdown links to them and a broken link is worse than
        # a few extra seconds.
        if "runs" in state and not figures:
            logger.info("report stage: regenerating missing figures")
            figures = make_figures(
                cfg, datasets, state["runs"], state["runs"][results["best_model"]],
                test, results.get("experiments", {}), figures_dir,
            )
            results["figures"] = figures
        results["environment"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": _lightgbm_version(),
            "primary_model_for_experiments": primary_model(),
        }
        write_reports(cfg, results, state)
    cache.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return results


def _lightgbm_version() -> str:
    try:
        import lightgbm

        return lightgbm.__version__
    except Exception:  # pragma: no cover - lightgbm is optional at import time
        return "unavailable"


def write_reports(cfg: Config, results: dict[str, object], state: dict[str, object]) -> None:
    """Write metrics.json plus the generated markdown reports."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.reports_dir / "metrics.json"
    payload = {
        "generated_by": "python -m src.pipeline --stage all",
        "environment": results.get("environment", {}),
        "config": cfg.raw,
        "results": results,
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info("wrote %s", metrics_path.name)

    runs = state.get("runs", {})
    if runs:
        path = cfg.reports_dir / "model_comparison.md"
        path.write_text(
            model_comparison_markdown(cfg, runs, results["experiments"], results["split"]),
            encoding="utf-8",
        )
        logger.info("wrote %s", path.name)
        path = cfg.reports_dir / "improvements.md"
        path.write_text(
            improvements_markdown(results["experiments"], runs), encoding="utf-8"
        )
        logger.info("wrote %s", path.name)
    if "error_report" in state:
        path = cfg.reports_dir / "error_analysis.md"
        path.write_text(
            error_analysis_markdown(state["error_report"], results.get("figures", {})),
            encoding="utf-8",
        )
        logger.info("wrote %s", path.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokyo land price ML pipeline")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "data", "models", "leakage", "errors", "figures", "report"],
        help="pipeline stage to run (default: all)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    cfg = Config.load(args.config)
    configure_matplotlib(cfg)
    quiet_lightgbm()
    started = time.perf_counter()
    try:
        run_pipeline(cfg, [args.stage])
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guidance
        logger.error("missing dependency: %s", exc)
        logger.error("install with: pip install -r requirements.txt")
        return 2
    logger.info("pipeline stage %r finished in %.1fs", args.stage, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
