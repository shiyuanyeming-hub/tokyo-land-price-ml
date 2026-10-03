"""Feature engineering and the leakage firewall.

Two ideas carry this module:

1. **Panel construction.** Sites can be followed across years through the key
   ``admin_code + use_code + seq_no``. Joining consecutive years gives a *measured*
   one-year lag of the price per square metre for the very same site. That lag is
   legitimate information for a forecast, unlike the ``前年価格``/``対前年変動率``
   columns of the file whose target year is being predicted.

2. **An explicit feature contract.** :func:`feature_columns` is the single
   source of truth for what a model may see. Leakage-prone columns can only enter
   through the ``include_leakage`` flag, which exists solely for the controlled
   experiment, and :mod:`tests.test_no_leakage` asserts they never enter silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.data_loader import PANEL_KEY

logger = logging.getLogger(__name__)

#: Numeric features that exist in the cleaned frame.
NUMERIC_FEATURES: tuple[str, ...] = (
    "site_area",
    "frontage_ratio",
    "depth_ratio",
    "road_width",
    "station_distance",
    "bcr",
    "far",
    "floors_total",
    "far_bcr_ratio",
)

#: Categorical features.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "municipality",
    "use_district",
    "current_use",
    "structure",
    "road_type",
    "road_direction",
    "fire_zone",
    "nearest_station",
    "locality",
    "pref_zone_1",
    "pref_zone_2",
)

#: Binned (ordinal) features derived from continuous columns.
BINNED_FEATURES: tuple[str, ...] = (
    "site_area_band",
    "station_distance_band",
    "road_width_band",
)

#: The one legitimate lag feature.
LAG_FEATURE: str = "lag_unit_price"

#: Columns that are leakage for a same-year target. They may only be requested
#: through ``include_leakage=True``, which the pipeline uses for one controlled
#: experiment and nothing else.
LEAKY_FEATURES: tuple[str, ...] = ("prev_unit_price", "change_rate_prior_year")

UNKNOWN_CATEGORY: str = "unknown"


def add_station_band(df: pd.DataFrame) -> pd.DataFrame:
    """Bin the walking distance to the nearest station.

    ``交通施設までの道路距離（m）`` is heavily right-skewed with a long tail up to
    ~7 km for the islands and mountain areas, so equal-width bins would put 95%
    of the sites in one bucket. The edges below follow the official 地価公示
    "駅からの距離" categories, extended with a tail bin.
    """
    edges = [-np.inf, 100, 250, 500, 1000, 2000, 5000, np.inf]
    labels = ["<=100m", "100-250m", "250-500m", "500m-1km", "1-2km", "2-5km", ">5km"]
    out = df.copy()
    out["station_distance_band"] = pd.cut(
        out["station_distance"], bins=edges, labels=labels, right=True
    ).astype("string")
    return out


def add_area_band(df: pd.DataFrame) -> pd.DataFrame:
    """Bin the site area, because price per square metre is not area-neutral."""
    edges = [-np.inf, 100, 150, 200, 300, 500, 1000, 5000, np.inf]
    labels = ["<=100", "100-150", "150-200", "200-300", "300-500", "500-1k", "1k-5k", ">5k"]
    out = df.copy()
    out["site_area_band"] = pd.cut(
        out["site_area"], bins=edges, labels=labels, right=True
    ).astype("string")
    return out


def add_road_width_band(df: pd.DataFrame) -> pd.DataFrame:
    """Bin the frontage road width.

    The first edge is ``-inf`` on purpose: 29 sites in 令和8年 publish a road width
    of exactly 0 m, which is the official way of recording "this lot does not
    front a road". A bin starting at 0 with ``right=True`` would turn those into
    ``NaN`` and silently discard a meaningful category.
    """
    edges = [-np.inf, 4, 6, 8, 12, 20, np.inf]
    labels = ["<=4m", "4-6m", "6-8m", "8-12m", "12-20m", ">20m"]
    out = df.copy()
    out["road_width_band"] = pd.cut(
        out["road_width"], bins=edges, labels=labels, right=True
    ).astype("string")
    return out


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the floor-area / coverage ratio, a proxy for development intensity."""
    out = df.copy()
    bcr = out["bcr"].replace(0, np.nan)
    out["far_bcr_ratio"] = out["far"] / bcr
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every deterministic feature transformation to a cleaned frame."""
    out = df.copy()
    out = add_ratio_features(out)
    out = add_station_band(out)
    out = add_area_band(out)
    out = add_road_width_band(out)
    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = out[col].astype("string").fillna(UNKNOWN_CATEGORY)
    out["locality"] = out["locality"].astype("string").fillna(UNKNOWN_CATEGORY)
    return out


def site_key(df: pd.DataFrame) -> pd.Series:
    """Build the cross-year site identifier from the panel key columns."""
    return df[PANEL_KEY].astype("string").agg("|".join, axis=1)


def attach_published_lag(
    frame: pd.DataFrame, lag_column: str = LAG_FEATURE
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach the previous-year unit price that the file itself publishes.

    Each file carries ``前年価格（円）``, the officially assessed value of *this
    site* one year earlier. Dividing it by the current area gives a genuine
    one-year lag of the unit price, available at forecast time and for exactly
    the year being predicted.

    Why not join the other year's file instead: doing so produces a lag from the
    *wrong* year. Joining 令和7年 against 令和8年's ``当年価格`` would attach the
    following year's price - a forward-looking feature - and, worse, that value
    is the site's own current price, so the "lag" would carry no trend at all.
    The published column is the only correct source among these four files, and
    where it is missing (newly selected sites) the lag is genuinely unknown and
    stays ``NaN``.
    """
    out = frame.copy()
    area = out["site_area"].replace(0, np.nan)
    out[lag_column] = out["price_previous"] / area
    coverage = {
        "rows": float(len(out)),
        "lag_available": float(out[lag_column].notna().sum()),
        "lag_coverage": float(out[lag_column].notna().mean()),
    }
    logger.info(
        "published lag: %.1f%% of %d sites have a previous-year value",
        100 * coverage["lag_coverage"],
        len(out),
    )
    return out, coverage


def attach_lag_unit_price(
    df: pd.DataFrame,
    previous: pd.DataFrame,
    lag_column: str = LAG_FEATURE,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach a previous-year unit price by joining the site key across files.

    Kept for the cross-year panel experiments: ``previous`` must contain
    ``unit_price``, and the lag on a row becomes the unit price that the
    *previous* frame observed for the same site. Only use this when ``previous``
    really is the earlier year; otherwise use :func:`attach_published_lag`.
    """
    left = df.copy()
    left["_site_key"] = site_key(left)
    if left["_site_key"].duplicated().any():
        raise ValueError("duplicate panel keys in the left frame; the lag join would fan out")
    right = previous.copy()
    right["_site_key"] = site_key(right)
    right = right[["_site_key", "unit_price"]].rename(columns={"unit_price": lag_column})
    left = left.merge(right, on="_site_key", how="left")
    coverage = {
        "rows": float(len(left)),
        "lag_available": float(left[lag_column].notna().sum()),
        "lag_coverage": float(left[lag_column].notna().mean()),
    }
    logger.info(
        "key join: %.1f%% of %d sites found in the companion frame",
        100 * coverage["lag_coverage"],
        len(left),
    )
    return left, coverage


def feature_columns(
    cfg: Config, include_lag: bool = True, include_leakage: bool = False
) -> list[str]:
    """Return the ordered feature list a model is allowed to consume.

    This function is the leakage firewall. The two price-derived columns that
    describe the *target year itself* (``prev_unit_price``, i.e. 前年価格 ÷ 地積,
    and ``change_rate_prior_year``) are appended **only** when
    ``include_leakage=True``, a flag reserved for the controlled experiment in
    ``src.pipeline.experiments``. ``tests/test_no_leakage.py`` asserts that the
    default list never contains them.

    The legitimately lagged ``lag_unit_price`` is different: it is the previous
    year's assessed value, which a forecast genuinely has, so it is included by
    default and can be switched off with ``include_lag=False``.
    """
    cols = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *BINNED_FEATURES]
    if include_lag:
        cols.append(LAG_FEATURE)
    if include_leakage:
        cols.extend(LEAKY_FEATURES)
    return cols


def resolve_feature_columns(
    train: pd.DataFrame,
    eval_frames: list[pd.DataFrame],
    cfg: Config,
    include_lag: bool = True,
    include_leakage: bool = False,
) -> tuple[list[str], dict[str, object]]:
    """Freeze one feature list for the whole experiment.

    The list is resolved **once** from the training frame and then reused for
    validation and test, so the model never has to deal with a different column
    set at prediction time. A feature that is entirely absent in training (not a
    single observed value) is dropped and reported: it could still be filled in
    by an imputer, but a column with no training signal is not a feature.
    """
    wanted = feature_columns(cfg, include_lag=include_lag, include_leakage=include_leakage)
    dropped = [c for c in wanted if c not in train.columns or train[c].notna().sum() == 0]
    kept = [c for c in wanted if c not in dropped]
    for frame in eval_frames:
        missing = [c for c in kept if c not in frame.columns]
        if missing:
            raise KeyError(f"evaluation frame is missing frozen feature(s): {missing}")
    return kept, {"dropped_no_training_signal": dropped}


def build_design_matrix(
    df: pd.DataFrame,
    cfg: Config,
    include_lag: bool = True,
    include_leakage: bool = False,
    columns: list[str] | None = None,
) -> DesignMatrix:
    """Select the model input columns present in ``df``.

    Passing explicit ``columns`` reuses a feature list frozen by
    :func:`resolve_feature_columns` and is strict: a missing column raises, so a
    silent shape change between splits is impossible. Calling without ``columns``
    skips features that were never built, which is convenient for one-off use.
    """
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"feature column(s) missing from the frame: {missing}")
        wanted = columns
    else:
        # Ad-hoc use: skip optional features the caller has not built (the lag is
        # the usual case). The pipeline always passes a frozen column list, which
        # takes the strict branch above.
        wanted = [
            c
            for c in feature_columns(cfg, include_lag=include_lag, include_leakage=include_leakage)
            if c in df.columns
        ]
    numeric = [c for c in wanted if c in NUMERIC_FEATURES or c in (*LEAKY_FEATURES, LAG_FEATURE)]
    categorical = [c for c in wanted if c in CATEGORICAL_FEATURES or c in BINNED_FEATURES]
    ordered = [*numeric, *categorical]
    return DesignMatrix(
        frame=df[ordered].copy(),
        numeric=numeric,
        categorical=categorical,
        features=ordered,
    )


@dataclass
class DesignMatrix:
    """A modelling matrix plus the bookkeeping needed to interpret it.

    ``numeric`` and ``categorical`` drive the preprocessor, and ``features`` is
    the ordered column list the estimator receives. Keeping the three together
    is what lets the important task of naming columns survive the one-hot
    expansion (see :func:`src.models.transformed_feature_names`).
    """

    frame: pd.DataFrame
    numeric: list[str]
    categorical: list[str]
    features: list[str]
