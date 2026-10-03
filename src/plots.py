"""Figure generation.

All figure text is English on purpose: a bare matplotlib install has no Japanese
font, so Japanese labels would render as empty boxes (tofu) on the reviewer's
machine. Numbers, titles and axis labels are generated from the actual run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGSIZE_WIDE = (11.0, 5.0)
FIGSIZE_SQUARE = (6.4, 5.6)

#: Axis labels are bilingual so the figures stay readable whether or not a
#: Japanese font is installed (see ``config.configure_japanese_font``).
JA = {
    "unit_price": "平米単価 unit price (円/㎡)",
    "unit_price_short": "平米単価 (円/㎡)",
    "sites": "地点数 number of sites",
    "municipality": "区市町村 municipality",
    "actual": "実測 actual (円/㎡)",
    "predicted": "予測 predicted (円/㎡)",
    "residual": "残差 residual (予測 - 実測, 円/㎡)",
    "site_area": "地積 site area (㎡, log)",
    "importance": "重要度 importance (split gain / |標準化係数|)",
    "model": "モデル model",
    "count": "地点数 sites",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote figure %s", path.name)
    return path


def plot_target_distribution(df: pd.DataFrame, out: Path) -> Path:
    """(1) Target distribution in raw and log space."""
    values = df["unit_price"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    axes[0].hist(values, bins=80, color="#3b6ea5", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel(JA["unit_price"])
    axes[0].set_ylabel(JA["sites"])
    axes[0].set_title(f"生の目的変数 raw target (skew={values.skew():.1f})")
    axes[0].axvline(values.median(), color="#c0392b", linestyle="--", linewidth=1.2,
                    label=f"median {values.median():,.0f}")
    axes[0].axvline(values.mean(), color="#27ae60", linestyle=":", linewidth=1.2,
                    label=f"mean {values.mean():,.0f}")
    axes[0].legend(fontsize=8)

    axes[1].hist(np.log10(values.where(values > 0)), bins=80, color="#8e6ea5",
                 edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("log10(平米単価 円/㎡)")
    axes[1].set_ylabel(JA["sites"])
    axes[1].set_title(f"対数変換後 log10 target (skew={np.log1p(values).skew():.1f})")
    fig.suptitle(f"目的変数の分布 target distribution - {len(values):,} sites", fontsize=12)
    return _save(fig, out)


def plot_price_by_municipality(df: pd.DataFrame, out: Path, top_n: int = 20) -> Path:
    """(2) Median unit price by municipality."""
    stats = (
        df.groupby("municipality", observed=True)["unit_price"]
        .agg(["median", "count"])
        .sort_values("median", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.barh(stats.index[::-1], stats["median"][::-1], color="#3b6ea5")
    for i, (value, count) in enumerate(zip(stats["median"][::-1], stats["count"][::-1])):
        ax.text(value, i, f" {value:,.0f} (n={count})", va="center", fontsize=8)
    ax.set_xlabel(JA["unit_price_short"] + " / median")
    ax.set_title(f"区市町村別の中央単価 Top {top_n} municipalities by median unit price")
    ax.margins(x=0.18)
    return _save(fig, out)


def plot_pred_vs_actual(df: pd.DataFrame, out: Path, title: str = "Test set") -> Path:
    """(3) Predicted vs. actual with the identity line."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    for ax, logscale in zip(axes, (False, True)):
        x, y = df["y_true"].to_numpy(float), df["y_pred"].to_numpy(float)
        ax.scatter(x, y, s=7, alpha=0.35, color="#3b6ea5", edgecolors="none")
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], color="#c0392b", linewidth=1.2, label="y = x")
        if logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("log-log scale")
        else:
            ax.set_title("linear scale")
        ax.set_xlabel(JA["actual"])
        ax.set_ylabel(JA["predicted"])
        ax.legend(fontsize=8)
    fig.suptitle(f"予測 vs 実測 predicted vs actual - {title} (n={len(df):,})", fontsize=12)
    return _save(fig, out)


def plot_residuals(df: pd.DataFrame, out: Path) -> Path:
    """(4) Residual diagnostics: vs. fitted value, vs. site area, distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    axes[0].scatter(df["y_pred"], df["residual"], s=7, alpha=0.35, color="#3b6ea5")
    axes[0].axhline(0, color="#c0392b", linewidth=1.2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(JA["predicted"] + " (log)")
    axes[0].set_ylabel(JA["residual"])
    axes[0].set_title("残差 vs 予測値 residual vs prediction")

    axes[1].scatter(df["site_area"], df["residual"], s=7, alpha=0.35, color="#8e6ea5")
    axes[1].axhline(0, color="#c0392b", linewidth=1.2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(JA["site_area"])
    axes[1].set_ylabel(JA["residual"])
    axes[1].set_title("残差 vs 地積 residual vs site area")

    axes[2].hist(df["residual"], bins=70, color="#27ae60", edgecolor="white", linewidth=0.3)
    axes[2].axvline(0, color="#c0392b", linewidth=1.2)
    axes[2].set_xlabel(JA["residual"])
    axes[2].set_ylabel(JA["sites"])
    axes[2].set_title(f"残差の分布 (mean={df['residual'].mean():,.0f})")
    fig.suptitle("残差診断 residual diagnostics", fontsize=12)
    return _save(fig, out)


def plot_feature_importance(table: pd.DataFrame, out: Path, model_name: str = "") -> Path:
    """(5) Top-N feature importance, plotted as a share of the total.

    Raw split gains span three orders of magnitude (the top feature reaches
    ~1,100 while the tail sits below 10), so plotting them on a linear axis makes
    every bar except the first look like zero. Normalising to a share of the
    total keeps all fifteen bars readable and is the quantity quoted in the text.
    """
    data = table.iloc[::-1].copy()
    total = float(data["importance"].sum())
    data["share"] = (data["importance"] / total * 100) if total else 0.0
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.barh(data["feature"], data["share"], color="#3b6ea5")
    for y, share in enumerate(data["share"]):
        ax.text(share, y, f" {share:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("重要度シェア importance share of total (%)")
    title = f"特徴量重要度 Top {len(table)} features"
    ax.set_title(f"{title} ({model_name})" if model_name else title)
    ax.margins(x=0.14)
    return _save(fig, out)


def plot_model_comparison(summary: pd.DataFrame, out: Path) -> Path:
    """(6) MAE and RMSE per model, on validation and test."""
    models = summary["model"].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8))
    for ax, metric, label in zip(axes, ("mae", "rmse"), ("MAE", "RMSE")):
        for offset, split, colour in ((-width / 2, "val", "#3b6ea5"), (width / 2, "test", "#e67e22")):
            values = [
                float(summary.loc[(summary["model"] == m) & (summary["split"] == split), metric].mean())
                for m in models
            ]
            ax.bar(x + offset, values, width, label=split, color=colour)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel(f"{label} (円/㎡)")
        ax.set_title(f"モデル別 {label} by model")
        ax.legend(fontsize=8)
    fig.suptitle("モデル比較 model comparison (低いほど良い lower is better)", fontsize=12)
    return _save(fig, out)


def plot_error_by_municipality(df: pd.DataFrame, out: Path, top_n: int = 15) -> Path:
    """(7) Map substitute: total absolute error and MAE per municipality."""
    stats = (
        df.groupby("municipality", observed=True)
        .agg(total_abs_error=("abs_error", "sum"), mae=("abs_error", "mean"), n=("abs_error", "size"))
        .sort_values("total_abs_error", ascending=False)
        .head(top_n)
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))
    axes[0].barh(stats.index[::-1], stats["total_abs_error"][::-1] / 1e6, color="#c0392b")
    axes[0].set_xlabel("絶対誤差の合計 (百万円/㎡)")
    axes[0].set_title(f"誤差の質量 top {top_n} municipalities by total absolute error")
    axes[1].barh(stats.index[::-1], stats["mae"][::-1], color="#e67e22")
    for i, (mae, n) in enumerate(zip(stats["mae"][::-1], stats["n"][::-1])):
        axes[1].text(mae, i, f" n={n}", va="center", fontsize=8)
    axes[1].set_xlabel("MAE (円/㎡)")
    axes[1].set_title("区市町村別の平均絶対誤差 MAE per municipality")
    fig.suptitle("誤差の区別集計 error by municipality (test set)", fontsize=12)
    return _save(fig, out)


def plot_learning_curve(
    train_sizes: np.ndarray, train_scores: np.ndarray, val_scores: np.ndarray, out: Path
) -> Path:
    """Auxiliary figure: generalisation gap as the training year grows."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    ax.plot(train_sizes, train_scores, "o-", color="#3b6ea5", label="train MAE")
    ax.plot(train_sizes, val_scores, "s-", color="#e67e22", label="val MAE")
    ax.set_xlabel("学習地点数 training sites")
    ax.set_ylabel("MAE (円/㎡)")
    ax.set_title("学習曲線 learning curve (LightGBM)")
    ax.legend(fontsize=8)
    return _save(fig, out)


def plot_leakage_effect(table: pd.DataFrame, out: Path) -> Path:
    """Auxiliary figure: the measured cost of the leakage columns."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    labels = table["setting"].tolist()
    x = np.arange(len(labels))
    ax.bar(x - 0.2, table["mae"], 0.4, label="MAE (円/㎡)", color="#3b6ea5")
    ax.set_ylabel("MAE (円/㎡)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, table["r2"], 0.4, label="R2", color="#27ae60")
    ax2.set_ylabel("R2")
    ax2.set_ylim(0, 1.05)
    for i, r2 in enumerate(table["r2"]):
        ax2.text(i + 0.2, min(float(r2) + 0.02, 1.02), f"{r2:.3f}", ha="center", fontsize=8)
    for i, mae in enumerate(table["mae"]):
        ax.text(i - 0.2, float(mae), f"{mae:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("リーク列が同じテストセットに与える影響")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper left")
    return _save(fig, out)
