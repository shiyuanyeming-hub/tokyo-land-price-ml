"""Error analysis: who does the model get wrong, and why?

Aggregate metrics hide the structure of the mistakes. This module slices the
prediction errors by the attributes that matter for land appraisal - city,
official use district, price level, site area, station distance - and then tests
a few concrete hypotheses about *why* the model fails where it does.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Human-readable labels for the machine column names used in the tables.
LABELS: dict[str, str] = {
    "municipality": "区市町村",
    "use_district": "用途区分",
    "use_district_group": "用途区分",
    "price_band": "価格帯 (円/㎡)",
    "site_area_band": "地積帯 (㎡)",
    "station_distance_band": "駅距離帯",
    "far_band": "容積率帯 (%)",
    "bcr_band": "建蔽率帯 (%)",
    "road_width_band": "前面道路幅員帯",
    "shape": "形状区分",
    "n": "地点数",
    "mae": "MAE (円/㎡)",
    "rmse": "RMSE (円/㎡)",
    "bias": "平均残差 (予測-実測)",
    "median_abs_pct_error": "絶対誤差率の中央値 (%)",
    "mean_abs_pct_error": "絶対誤差率の平均 (%)",
}

#: Official 用途区分 codes mapped to a small, readable grouping.
USE_DISTRICT_GROUPS: dict[str, str] = {
    "１低専": "住居系(低層)",
    "２低専": "住居系(低層)",
    "１中専": "住居系(中高層)",
    "２中専": "住居系(中高層)",
    "１住居": "住居系(中高層)",
    "２住居": "住居系(中高層)",
    "準住居": "住居系(中高層)",
    "近商": "商業系",
    "商業": "商業系",
    "準工": "工業系",
    "工業": "工業系",
    "工専": "工業系",
}


def add_error_columns(
    df: pd.DataFrame, prediction_col: str = "y_pred", target_col: str = "y_true"
) -> pd.DataFrame:
    """Attach signed, absolute and relative error columns."""
    out = df.copy()
    out["residual"] = out[prediction_col] - out[target_col]
    out["abs_error"] = out["residual"].abs()
    finite = out[target_col].abs() > 0
    out["pct_error"] = np.nan
    out.loc[finite, "pct_error"] = (
        out.loc[finite, "residual"] / out.loc[finite, target_col] * 100
    )
    out["abs_pct_error"] = out["pct_error"].abs()
    return out


def add_analysis_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Bin the continuous attributes used as error-analysis slices."""
    out = df.copy()
    out["use_district_group"] = (
        out["use_district"].map(USE_DISTRICT_GROUPS).fillna("その他/不明")
    )
    out["price_band"] = pd.cut(
        out["y_true"],
        bins=[0, 1_000, 5_000, 10_000, 30_000, 100_000, np.inf],
        labels=["<1k", "1k-5k", "5k-10k", "10k-30k", "30k-100k", ">100k"],
    ).astype("string")
    out["far_band"] = pd.cut(
        out["far"],
        bins=[-np.inf, 100, 200, 300, 400, 600, np.inf],
        labels=["<=100", "100-200", "200-300", "300-400", "400-600", ">600"],
    ).astype("string")
    out["bcr_band"] = pd.cut(
        out["bcr"],
        bins=[-np.inf, 40, 60, 70, 80, np.inf],
        labels=["<=40", "40-60", "60-70", "70-80", ">80"],
    ).astype("string")
    return out


def group_summary(df: pd.DataFrame, by: str, min_count: int = 1) -> pd.DataFrame:
    """Error summary per group, sorted by MAE (worst first)."""
    grouped = df.groupby(by, dropna=False, observed=True)
    table = grouped.agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        rmse=("residual", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        bias=("residual", "mean"),
        median_abs_pct_error=("abs_pct_error", "median"),
        mean_abs_pct_error=("abs_pct_error", "mean"),
    ).reset_index()
    table = table.loc[table["n"] >= min_count]
    return table.sort_values("mae", ascending=False).reset_index(drop=True)


def top_errors(df: pd.DataFrame, n: int = 20, columns: list[str] | None = None) -> pd.DataFrame:
    """The n worst absolute errors, with the land attributes that explain them."""
    columns = columns or [
        "municipality",
        "locality",
        "lot_number",
        "use_district",
        "site_area",
        "far",
        "bcr",
        "station_distance",
        "nearest_station",
        "road_width",
        "current_use",
        "y_true",
        "y_pred",
        "residual",
        "abs_error",
        "pct_error",
    ]
    present = [c for c in columns if c in df.columns]
    return df.nlargest(n, "abs_error")[present].reset_index(drop=True)


def concentration(df: pd.DataFrame, column: str, top_k: int = 3) -> dict[str, float]:
    """How much of the total absolute error the worst groups carry."""
    by_group = df.groupby(column, dropna=False, observed=True)["abs_error"].sum()
    total = float(by_group.sum())
    share = (by_group / total).sort_values(ascending=False)
    return {
        "total_abs_error": total,
        "top_groups": ", ".join(str(i) for i in share.head(top_k).index),
        "top_share": float(share.head(top_k).sum()),
        "n_groups": int(share.size),
    }


def check_far_premium_hypothesis(df: pd.DataFrame) -> dict[str, object]:
    """Hypothesis: unit price rises with the permitted floor-area ratio, and the
    model under-predicts the high-FAR commercial lots because 容積率 interacts
    with the use district instead of acting additively.

    Returns the observed correlation by use-district group plus a direct test of
    whether the largest FAR values are systematically under-predicted.
    """
    commercial = df.loc[df["use_district_group"] == "商業系"]
    corr_overall = float(
        df[["far", "y_true"]].dropna().corr(method="spearman").iloc[0, 1]
    )
    corr_commercial = (
        float(commercial[["far", "y_true"]].dropna().corr(method="spearman").iloc[0, 1])
        if len(commercial) > 2
        else float("nan")
    )
    high_far = df.loc[df["far"] >= 500]
    return {
        "spearman_far_vs_price_all": corr_overall,
        "spearman_far_vs_price_commercial": corr_commercial,
        "n_high_far": int(len(high_far)),
        "mean_bias_high_far": float(high_far["residual"].mean()) if len(high_far) else float("nan"),
        "mae_high_far": float(high_far["abs_error"].mean()) if len(high_far) else float("nan"),
        "mae_all": float(df["abs_error"].mean()),
    }


def check_small_site_hypothesis(df: pd.DataFrame, area_threshold: float = 80.0) -> dict[str, object]:
    """Hypothesis: very small sites carry a scarcity premium that the model,
    having few such examples, cannot reproduce."""
    small = df.loc[df["site_area"] <= area_threshold]
    rest = df.loc[df["site_area"] > area_threshold]
    return {
        "n_small": int(len(small)),
        "mae_small": float(small["abs_error"].mean()) if len(small) else float("nan"),
        "mae_rest": float(rest["abs_error"].mean()) if len(rest) else float("nan"),
        "mean_bias_small": float(small["residual"].mean()) if len(small) else float("nan"),
        "price_ratio_small_vs_rest": (
            float(small["y_true"].median() / rest["y_true"].median())
            if len(small) and len(rest)
            else float("nan")
        ),
    }


def check_station_distance_hypothesis(df: pd.DataFrame) -> dict[str, object]:
    """Hypothesis: within the same municipality, station distance explains the residual.

    This check deliberately reports the relationship *three* ways, because the
    pooled correlation is confounded and on its own it would support a wrong
    conclusion. In this data, commercial districts sit much closer to stations
    (median ~230 m) and are much more expensive (median ~8,400 JPY/m2) than other
    districts (median ~820 m, ~2,500 JPY/m2). A pooled correlation between
    distance and residual therefore mostly measures "this site is commercial",
    not "this site is far from a station".
    """
    if "station_distance" not in df.columns or df["station_distance"].isna().all():
        return {}

    results: dict[str, object] = {}
    pooled = df[["station_distance", "residual"]].dropna()
    results["pooled_spearman_distance_vs_residual"] = (
        float(pooled.corr(method="spearman").iloc[0, 1]) if len(pooled) > 2 else float("nan")
    )
    price = df[["station_distance", "y_true"]].dropna()
    results["pooled_spearman_distance_vs_price"] = (
        float(price.corr(method="spearman").iloc[0, 1]) if len(price) > 2 else float("nan")
    )

    # Confounder evidence: commercial vs. other districts.
    commercial = df["use_district_group"] == "商業系"
    results["median_distance_commercial"] = float(
        df.loc[commercial, "station_distance"].median()
    )
    results["median_distance_other"] = float(
        df.loc[~commercial, "station_distance"].median()
    )
    results["median_price_commercial"] = float(df.loc[commercial, "y_true"].median())
    results["median_price_other"] = float(df.loc[~commercial, "y_true"].median())

    # The honest test: within a municipality, does distance relate to the residual?
    per_city = df.dropna(subset=["station_distance"]).groupby("municipality", observed=True)
    rows: list[float] = []
    for _, group in per_city:
        if len(group) < 20 or group["station_distance"].nunique() < 3:
            continue
        value = group[["station_distance", "residual"]].corr(method="spearman").iloc[0, 1]
        if np.isfinite(value):
            rows.append(float(value))
    results["n_cities_checked"] = len(rows)
    results["mean_within_city_spearman_distance_vs_residual"] = (
        float(np.mean(rows)) if rows else float("nan")
    )
    results["median_within_city_spearman_distance_vs_residual"] = (
        float(np.median(rows)) if rows else float("nan")
    )
    return results
