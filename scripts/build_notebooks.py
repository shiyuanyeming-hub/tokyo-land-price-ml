"""Generate the two notebooks as source, so they can be reviewed in git.

The notebooks import the same `src` modules the CLI uses: they are a narrative
layer over the pipeline, not a second implementation. Their outputs are produced
by actually executing them (see scripts/run_notebooks.py).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

BOOTSTRAP = """\
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Notebooks live one level below the repository root.
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import cleaning, error_analysis, evaluate, features, plots
from src.config import Config, configure_matplotlib, quiet_lightgbm
from src.data_loader import load_all_raw, summarise

logging.basicConfig(level=logging.WARNING)
cfg = Config.load()
configure_matplotlib(cfg)
quiet_lightgbm()
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 60)

print(f"repository : {REPO_ROOT.name}")
print(f"config     : {cfg.path.name}")
print(f"seed       : {cfg.seed}")
from IPython.display import Image, display, display as display_image

print(f"train year : {cfg.split['train_year']}  ->  predict year: {cfg.split['predict_year']}")
"""


def _source_lines(text: str) -> list[str]:
    """Split cell text into nbformat source lines, keeping the newlines.

    ``str.split("\n")`` drops the separators, which made nbclient concatenate
    the whole cell into one line and fail with a SyntaxError. ``splitlines(True)``
    preserves them, which is the format nbformat expects.
    """
    body = text.strip("\n")
    return [line if line.endswith("\n") else line + "\n" for line in body.splitlines()]


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(source),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(source),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


EDA_CELLS = [
    markdown(
        """
# 01 — 探索的データ分析（EDA）

**目的**: 東京都の公式地価データ（地価公示・基準地価）を読み込み、
モデリング前に「何を予測するのか」「どこに罠があるのか」を実データで確認する。

この notebook は `src/` のモジュールをそのまま呼び出します。
CLI（`python -m src.pipeline`）と同じコードを通るので、
ここで得た知見とパイプラインの挙動が食い違うことはありません。
"""
    ),
    markdown("## 0. セットアップ"),
    code(BOOTSTRAP),
    markdown(
        """
## 1. 生データをそのまま読む

4ファイルは CP932・1行目タイトル行・数値は文字列（カンマ＋末尾スペース入り）。
まずは**加工せずに**素の姿を確認します。
"""
    ),
    code(
        """
raw = load_all_raw(cfg)
profile = pd.DataFrame([summarise(frame, name) for name, frame in raw.items()])
profile
"""
    ),
    code(
        """
# 素の値はこんな形で入っている（カンマと末尾スペースに注意）
frame = raw["kouji_r8"]
print("raw dtypes :", frame["price_current"].dtype)
print("raw sample :", repr(frame.loc[0, "price_current"]))
print("raw year   :", repr(frame.loc[0, "year_wareki"]), "<- 令和8年ファイルだけ和暦の漢字表記")
print("column names containing brackets:")
print([c for c in frame.columns if "距離" in c or "幅員" in c])
"""
    ),
    markdown(
        """
### 確認できたこと

- 生の行数は 2,561 / 1,281 / 1,278 行で、**最終行が完全な空行**。
- `対象年（和暦）` は3ファイルが `'7 '` のような数字、`kouji_r8` だけ `'令和8年'`。
- `交通施設までの道路距離（m)` は**開き括弧が全角・閉じ括弧が半角**。
  基準地価は `（m）` と両方全角。→ `data_loader._normalise_name` で吸収。
"""
    ),
    markdown(
        """
## 2. クリーニングと目的変数の定義

`当年価格（円）` は**地点の総額**です。これをそのまま目的変数にすると
「面積が大きいほど高額」という自明な関係を当てるだけになります。
`地積（㎡）` で割って **平米単価（円/㎡）** を目的変数にします。
"""
    ),
    code(
        """
frames, reports = {}, {}
for name, frame in raw.items():
    cleaned_frame, report = cleaning.clean_frame(frame, source=name)
    frames[name] = cleaning.add_unit_price(features.build_features(cleaned_frame))
    reports[name] = report

quality = pd.DataFrame([r.as_dict() for r in reports.values()])
quality[["source", "rows_in", "rows_out", "dropped_invalid", "notes"]]
"""
    ),
    code(
        """
kouji_r7, kouji_r8 = frames["kouji_r7"], frames["kouji_r8"]
kijun_r6, kijun_r7 = frames["kijun_r6"], frames["kijun_r7"]

summary = []
for name, frame in frames.items():
    price = frame["unit_price"]
    summary.append(
        {
            "source": name,
            "n": len(frame),
            "median": round(price.median()),
            "mean": round(price.mean()),
            "p1": round(price.quantile(0.01)),
            "p99": round(price.quantile(0.99)),
            "max": round(price.max()),
            "skew_raw": round(price.skew(), 1),
            "skew_log1p": round(np.log1p(price).skew(), 2),
            "under_1000yen_pct": round((price < 1000).mean() * 100, 1),
        }
    )
pd.DataFrame(summary)
"""
    ),
    markdown(
        """
### 確認できたこと

- 平米単価の中央値は約 3,000〜3,400 円/㎡、平均は 5,700〜6,600 円/㎡。
  **平均が中央値の約2倍**で、右に強く裾を引いています（歪度 > 10）。
- 最小値は 0.2 円/㎡、最大値は約 14.8 万円/㎡ で **5桁の開き**。
- 1,000 円/㎡ 未満の地点が 15〜19% あります。これは誤りではなく
  **山林・原野・島しょ部**が実際に含まれているためです（次で確認）。
- 対数変換すると歪度が 10 前後から 0 付近まで下がります。
"""
    ),
    markdown("## 3. 外れ値の正体を確認する（削除する前に）"),
    code(
        """
cheap = kouji_r8.nsmallest(6, "unit_price")[
    ["municipality", "lot_number", "current_use", "site_area",
     "price_current", "unit_price", "use_district", "forest_law"]
]
expensive = kouji_r8.nlargest(6, "unit_price")[
    ["municipality", "lot_number", "current_use", "site_area",
     "price_current", "unit_price", "use_district", "far"]
]
print("=== 平米単価が最も低い地点 ===")
display(cheap)
print("=== 平米単価が最も高い地点 ===")
display(expensive)
"""
    ),
    markdown(
        """
### 確認できたこと

最安値の地点は `利用の現況 = 山林（雑木林）`、`森林法 = 地森計` の**森林**でした。
最高値は千代田区・中央区の商業地です。
つまり外れ値は**データ誤りではなく土地用途の多様性そのもの**なので、
削除せず、対数変換と地点属性（用途区分・容積率）で扱うのが正しい判断です。
"""
    ),
    markdown("## 4. 目的変数の分布"),
    code(
        """
path = plots.plot_target_distribution(kouji_r8, cfg.figures_dir / "01_target_distribution.png")
plt.close("all")
print(f"saved: {path.relative_to(REPO_ROOT)}")
"""
    ),
    code('display_image(filename=str(cfg.figures_dir / "01_target_distribution.png"))'),
    markdown(
        """
**この図から読み取れること**: 生の平米単価は右に強く裾を引いた分布（歪度 17.2）で、
平均が中央値の約2倍。対数（log10）を取るとほぼ左右対称の釣鐘型（歪度 -0.7）になり、
線形モデルの誤差仮定と整合します。
"""
    ),
    markdown("## 5. 地域による価格差"),
    code(
        """
path = plots.plot_price_by_municipality(kouji_r8, cfg.figures_dir / "02_price_by_municipality.png")
plt.close("all")
display_image(filename=str(path))
"""
    ),
    code(
        """
by_use = (
    kouji_r8.groupby("use_district", observed=True)["unit_price"]
    .agg(["count", "median", "mean"])
    .sort_values("median", ascending=False)
)
by_use.round(0)
"""
    ),
    markdown(
        """
**この図から読み取れること**: 中央単価の上位は千代田区・中央区・港区・渋谷区で、
下位とは 1 桁以上差があります。「どの区か」が最重要の説明変数になることが分かります。
用途区分でも商業 > 住居系 > 工業系 > 山林・原野の順に並びます。
"""
    ),
    markdown(
        """
## 6. データリーク源の確認（このプロジェクトの最重要ポイント）

`前年価格（円）` と `対前年変動率（％）` は、**同じ年の価格から作られた列**です。
翌年の価格を予測するとき、この列は使えません（存在しないため）。
ところが「同じ年で学習して同じ年を評価する」と、この列は
**目的変数の言い換え**として機能してしまいます。

まず、どれくらい目的変数と相関しているかを実測します。
"""
    ),
    code(
        """
leak_table = pd.DataFrame(
    {
        "column": ["prev_unit_price (前年価格/地積)", "change_rate_prior_year (対前年変動率)",
                   "lag_unit_price (前年ファイルから結合)", "far (指定容積率)",
                   "station_distance (駅距離)", "site_area (地積)"],
        "pearson_with_target": [
            kouji_r8["prev_unit_price"].corr(kouji_r8["unit_price"]),
            kouji_r8["change_rate_prior_year"].corr(kouji_r8["unit_price"]),
            float("nan"),
            kouji_r8["far"].corr(kouji_r8["unit_price"]),
            kouji_r8["station_distance"].corr(kouji_r8["unit_price"]),
            kouji_r8["site_area"].corr(kouji_r8["unit_price"]),
        ],
    }
)
# lag は「前年ファイル」との結合で作る（ファイル内の前年価格列は使わない）
lagged, coverage = features.attach_lag_unit_price(kouji_r8, kouji_r7)
leak_table.loc[2, "pearson_with_target"] = lagged["lag_unit_price"].corr(lagged["unit_price"])
print(f"lag coverage: {coverage['lag_coverage']*100:.1f}% ({int(coverage['lag_available'])}/{int(coverage['rows'])} sites)")
leak_table.round(4)
"""
    ),
    markdown(
        """
### 確認できたこと

- `前年価格` 系の列は目的変数と **0.95 前後**の相関を持ちます。これは
  「予測」ではなく「答えの言い換え」に近い状態です。
- 一方 `lag_unit_price`（前年ファイルから地点キーで結合した**実測の前年単価**）も
  相関は高いものの、**別の年に観測された値**なので、翌年を予測する時点で
  実際に手に入る情報です。これがリークとの本質的な違いです。
- 区別のため、本プロジェクトは
  **ファイル内の前年価格列は使わず、前年ファイルから結合した lag を使う**方針にしています。
"""
    ),
    markdown("## 7. クロス年度リンクの検証"),
    code(
        """
key_cols = ["admin_code", "use_code", "seq_no"]
merged = kouji_r8[key_cols + ["price_previous"]].merge(
    kouji_r7[key_cols + ["price_current"]], on=key_cols, how="inner"
)
relative = (merged["price_previous"] - merged["price_current"]).abs() / merged["price_current"]

# 行番号で比較した場合との違いを実際に示す
rowwise = (kouji_r8["price_previous"].to_numpy() == kouji_r7["price_current"].to_numpy())
print(f"キーで整列して比較 : {(relative < 1e-9).mean()*100:5.1f}% が完全一致 (n={len(merged)})")
print(f"行番号で比較       : {rowwise.mean()*100:5.1f}% が完全一致  <- 並び順が違うため誤った結論になる")
print(f"両年に共通する地点 : {len(set(features.site_key(kouji_r7)) & set(features.site_key(kouji_r8)))} / {len(kouji_r8)}")
"""
    ),
    markdown(
        """
### 確認できたこと

キー（市区町村コード＋用途＋連番）で整列すれば **97.8%** が完全一致しますが、
**行番号で比較すると 42%** しか一致しません。ファイルの並び順は年度間で
同一ではないため、行番号で対応を取ると誤った結論を導きます。
これは開発中に実際に踏んだ罠です（→ `docs/INTERVIEW_GUIDE.md`）。
"""
    ),
    markdown("## 8. 欠損とカテゴリの扱い"),
    code(
        """
missing = (
    kouji_r8[list(features.NUMERIC_FEATURES) + list(features.CATEGORICAL_FEATURES)]
    .isna()
    .sum()
    .loc[lambda s: s > 0]
    .sort_values(ascending=False)
)
print("=== 特徴量として使う列の欠損（クリーニング後） ===")
print(missing if len(missing) else "欠損なし（カテゴリは 'unknown' として明示的に保持）")
print()
print("=== 用途区分が unknown の地点（実在する） ===")
unknown = kouji_r8.loc[kouji_r8["use_district"] == features.UNKNOWN_CATEGORY]
print(f"n = {len(unknown)}")
print(unknown[["municipality", "lot_number", "current_use", "unit_price", "forest_law"]].head(5))
"""
    ),
    markdown(
        """
### 確認できたこと

`用途区分` が空欄の地点が実在します（公示 令和8年で 34 地点）。
山林・農地など市街化区域外の地点で、数値の欠損ではなく
**「区分が存在しない」という意味のある情報**です。
そのため `NaN` のまま埋めるのではなく、文字列 `"unknown"` という
明示的なカテゴリとして扱います（`OneHotEncoder` / `OrdinalEncoder` が扱えるようにするため）。
"""
    ),
    markdown(
        """
## 9. EDA のまとめ（モデリングへの申し送り）

| 論点 | 実データから分かったこと | モデリングでの対処 |
|---|---|---|
| 目的変数 | 総額は面積に比例する自明な量 | **平米単価**（当年価格 ÷ 地積）を使う |
| 分布 | 歪度 17.2、平均が中央値の2倍 | **対数変換**を試し、raw と比較する |
| 外れ値 | 山林〜都心商業地で5桁の開き。誤りではない | 削除しない。用途区分・容積率で説明する |
| リーク | 前年価格列は目的変数と 0.95 相関 | **既定の特徴量から除外**し、統制実験でのみ使う |
| 正しいラグ | 前年ファイルとの結合で 98.4% の地点に実測ラグがある | `lag_unit_price` として特徴量化する |
| 年度対応 | キー整列で 97.8% 一致（行番号だと 42%） | `site_key` によるパネル結合を使う |
| カテゴリ欠損 | 用途区分に実在する空欄（34地点） | `"unknown"` を明示カテゴリにする |
| 分割 | 同一地点が複数年・複数区に現れる | **地点単位**で train/val/test を分離する |

次の `02_modeling.ipynb` では、この申し送りに沿って
時系列分割 → ベースライン → モデル比較 → 誤差分析 の順に進めます。
"""
    ),
]

MODELING_CELLS = [
    markdown(
        """
# 02 — モデリングと評価

**問い**: 東京都の公示地価について、**翌年の平米単価**をどこまで当てられるか。
そして「当たっている」という結論は、どこまで信用できるのか。

この notebook は CLI パイプライン（`python -m src.pipeline --stage all`）と
同じ `src/` の関数を呼びます。数値は `reports/metrics.json` と一致します。
"""
    ),
    markdown("## 0. セットアップ"),
    code(BOOTSTRAP + """
from src import models as model_lib
from src.models import MODEL_ORDER

# 両 notebook は独立して実行できるようにしてある。01_eda.ipynb と同じ
# クリーニングを通したフレームをここでも用意する（コードは src/ を共用）。
raw = load_all_raw(cfg)
frames = {}
for _name, _frame in raw.items():
    _clean, _ = cleaning.clean_frame(_frame, source=_name)
    _built = cleaning.add_unit_price(features.build_features(_clean))
    _built["_site_key"] = features.site_key(_built)
    frames[_name] = _built

kouji_r7, kouji_r8 = frames["kouji_r7"], frames["kouji_r8"]
kijun_r6, kijun_r7 = frames["kijun_r6"], frames["kijun_r7"]
print({name: len(frame) for name, frame in frames.items()})
"""),
    markdown(
        """
## 1. 分割の設計（ここが評価の信頼性を決める）

```
train : 地価公示 令和7年 (2,560 地点)   ← 学習。価格水準も含む
val   : 地価公示 令和8年 (1,280 地点)   ← 翌年を予測（モデル選択用）
test  : 地価公示 令和8年 (1,280 地点)   ← 翌年を予測（最終評価、val と排他）
```

**なぜランダム分割にしないのか**を、この notebook の最後で実測して示します。
"""
    ),
    code(
        """
from src import pipeline

datasets = pipeline.load_datasets(cfg)
train, val, test, split_meta = pipeline.build_split(cfg, datasets)
print(json.dumps(split_meta, ensure_ascii=False, indent=2, default=str) if (json := __import__("json")) else "")
"""
    ),
    code(
        """
# 地点キーが分割をまたいで重複していないことを確認（同一地点の記憶を防ぐ）
train_keys, val_keys, test_keys = (set(f[["admin_code", "use_code", "seq_no"]].astype(str).agg("|".join, axis=1))
                                   for f in (train, val, test))
print(f"train ∩ val  = {len(train_keys & val_keys)}")
print(f"train ∩ test = {len(train_keys & test_keys)}")
print(f"val   ∩ test = {len(val_keys & test_keys)}")
print()
print("lag_unit_price のカバレッジ")
for label, frame in (("train(令和7年)", train), ("test(令和8年)", test)):
    print(f"  {label}: {frame['lag_unit_price'].notna().mean()*100:.1f}%")
"""
    ),
    markdown(
        """
## 2. 学習に使う特徴量（リーク防止の確認）

`features.feature_columns()` がモデルに入れてよい列の唯一の定義です。
"""
    ),
    code(
        """
allowlist = features.feature_columns(cfg)
print(f"既定の特徴量: {len(allowlist)} 列")
for column in allowlist:
    print("  -", column)
print()
print("リーク列（既定では絶対に入らない）:", list(features.LEAKY_FEATURES))
print()
raw_price_columns = {"price_current", "price_previous", "unit_price", "prev_unit_price"}
print("生の価格列が特徴量に含まれていないか:",
      "OK" if raw_price_columns.isdisjoint(allowlist) else "NG")
"""
    ),
    markdown("## 3. ベースラインとモデルの比較"),
    code(
        """
runs = pipeline.run_models(cfg, train, val, test)
table = evaluate.metrics_frame({name: run.metrics for name, run in runs.items()})
table = table.pivot(index="model", columns="split", values=["mae", "rmse", "r2"])
table = table.reindex(MODEL_ORDER)
table.round(3)
"""
    ),
    code(
        """
comparison = pd.DataFrame(
    {
        "train MAE": [runs[n].metrics["train"]["mae"] for n in MODEL_ORDER],
        "val MAE": [runs[n].metrics["val"]["mae"] for n in MODEL_ORDER],
        "test MAE": [runs[n].metrics["test"]["mae"] for n in MODEL_ORDER],
        "test R2": [runs[n].metrics["test"]["r2"] for n in MODEL_ORDER],
        "test/train gap": [runs[n].metrics["test"]["mae"] / runs[n].metrics["train"]["mae"] for n in MODEL_ORDER],
    },
    index=MODEL_ORDER,
)
comparison.round(2)
"""
    ),
    markdown(
        """
### 結果の読み方

- **median / mean（定数予測）**: MAE 約 4,900〜5,500 円/㎡。R² はほぼ 0 か負。
  これが「何も学習しない」水準です。
- **lag（前年単価をそのまま予測）**: MAE 約 750 円/㎡。**最も強いベースライン**です。
  地価は1年ではほとんど動かないため、これは当然の結果であり、
  「機械学習がこのベースラインを超えられるか」が本当の勝負になります。
- **ridge**: 線形モデルでも 1,600 円/㎡ 程度まで下がります。
- **random_forest / gradient_boosting / lightgbm**: 1,280〜1,350 円/㎡。
  train MAE は 88〜385 円/㎡ と極端に小さく、**過学習が明確**です。
"""
    ),
    code(
        """
path = plots.plot_model_comparison(
    evaluate.metrics_frame({n: runs[n].metrics for n in MODEL_ORDER}),
    cfg.figures_dir / "06_model_comparison.png",
)
plt.close("all")
display_image(filename=str(path))
"""
    ),
    markdown(
        """
**この図から読み取れること**: val と test の MAE がほぼ同じで、
分割が安定していることが分かります（val と test は同じ年の別地点）。
勾配ブースティング系は線形モデル（ridge）より 20% ほど良いものの、
前年単価ベースライン（lag）には遠く及びません。
"""
    ),
    markdown("## 4. 過学習と汎化の確認（train / val / test の乖離）"),
    code(
        """
gap = pd.DataFrame(
    {
        "train MAE": [runs[n].metrics["train"]["mae"] for n in MODEL_ORDER],
        "test MAE": [runs[n].metrics["test"]["mae"] for n in MODEL_ORDER],
    },
    index=MODEL_ORDER,
)
gap["test/train"] = (gap["test MAE"] / gap["train MAE"]).round(2)
gap["CV MAE (train年5-fold)"] = [runs[n].cv.get("cv_mae_mean", float("nan")) for n in MODEL_ORDER]
gap["CV std"] = [runs[n].cv.get("cv_mae_std", float("nan")) for n in MODEL_ORDER]
gap.round(1)
"""
    ),
    markdown(
        """
### 確認できたこと

- 決定木系は **train MAE が test MAE の 1/10 以下**（gap 10〜15倍）。
  深さを制限していない `random_forest` が最も極端です。
- `ridge` の gap は 1.13 倍で、過学習はほとんどありません。
- 訓練年での 5-fold CV（MAE の標準偏差）はモデル間の相対的な安定性を示します。

地価データは「同じ場所は似た価格」という性質が極めて強いため、
決定木は学習地点を丸暗記しやすく、**地点を分けた瞬間に精度が落ちます**。
これがこの課題の本質的な難しさです。
"""
    ),
    markdown("## 5. 特徴量重要度"),
    code(
        """
importance = runs["lightgbm"].importance
display(importance.round(4))
path = plots.plot_feature_importance(importance, cfg.figures_dir / "05_feature_importance.png")
plt.close("all")
display_image(filename=str(path))
"""
    ),
    markdown(
        """
**この図から読み取れること**: 前年単価（`num_lag_unit_price`）が圧倒的首位（シェア約 37%）で、
次に地積・容積率・前面道路幅員・駅距離が続きます。
つまり**「去年の価格」＋「土地の物理的属性」**でほぼ説明でき、
残差に効くのは都市計画（容積率）と立地（駅距離）です。
"""
    ),
    markdown("## 6. 予測 vs 実測 / 残差"),
    code(
        """
best_name = min((n for n in MODEL_ORDER if n != "lag"),
                key=lambda n: runs[n].metrics["val"]["mae"])
print("val MAE 最小のモデル:", best_name)

merged = pipeline.merge_test_predictions(runs[best_name], test)
path = plots.plot_pred_vs_actual(merged, cfg.figures_dir / "03_pred_vs_actual.png", title=f"test, {best_name}")
plt.close("all")
display_image(filename=str(path))
path = plots.plot_residuals(merged, cfg.figures_dir / "04_residuals.png")
plt.close("all")
display_image(filename=str(path))
"""
    ),
    markdown(
        """
**この図から読み取れること**: 線形スケールでは低価格帯に点が密集し、
高価格帯（10万円/㎡超）で過小予測（点が y=x より下）が目立ちます。
対数スケールでは全体にばらつきが均一になり、**相対誤差で見れば
価格帯によらず同程度の精度**であることが分かります。
残差は 0 を中心に分布しますが、右裾が長く（大きな過小予測がある）、
これは都心の高額地点を取り切れていないことを示します。
"""
    ),
    markdown("## 7. 誤差分析"),
    code(
        """
from src import error_analysis

by_use = error_analysis.group_summary(merged, "use_district_group")
display(by_use.round(0))
"""
    ),
    code(
        """
by_price = error_analysis.group_summary(merged, "price_band")
by_area = error_analysis.group_summary(merged, "site_area_band")
print("=== 価格帯別 ===")
display(by_price.round(0))
print("=== 地積帯別 ===")
display(by_area.round(0))
"""
    ),
    code(
        """
worst = error_analysis.top_errors(merged, n=10)
display(worst[["municipality", "lot_number", "use_district", "site_area", "far",
               "station_distance", "y_true", "y_pred", "abs_error", "pct_error"]].round(0))
"""
    ),
    markdown("## 8. 仮説の検証"),
    code(
        """
hypotheses = {
    "容積率プレミアム": error_analysis.check_far_premium_hypothesis(merged),
    "小規模地": error_analysis.check_small_site_hypothesis(merged),
    "駅距離": error_analysis.check_station_distance_hypothesis(merged),
}
for name, values in hypotheses.items():
    print(f"=== {name} ===")
    for key, value in values.items():
        print(f"  {key:52s} {value}")
    print()
"""
    ),
    markdown(
        """
### 検証の結論

上記の数値は `reports/error_analysis.md` に自動で書き出されます。
実際に確かめられたこと・確かめられなかったことはそちらにまとめています
（**支持された仮説と、支持されなかった仮説の両方**を記載しています）。
"""
    ),
    markdown(
        """
## 9. 統制実験: リーク列とランダム分割は何を隠すか

同じモデル（LightGBM）・同じテスト行で、**入力列だけを変えて**比較します。
"""
    ),
    markdown(
        """
## 9. 統制実験: リーク列とランダム分割は何を隠すか

同じモデル（LightGBM）・同じテスト行で、**入力列だけを変えて**比較します。
"""
    ),
    code(
        """
from src.models import build_estimator

# 予測年（令和8年）を、下の3設定すべてで同じ train/test 行になるように分割する。
# `test` は時系列分割のテスト側で、実測ラグが結合済み。ここからランダムに
# 半分を取ることで、時系列分割と同じデータを使いつつ分割方法だけを変えられる。
rand_train, rand_test = evaluate.random_split(test, cfg)
print(f"ランダム分割: train {len(rand_train)} / test {len(rand_test)} (同じ年・別地点)")
print(f"時系列分割  : train {len(train)} (令和7年) / test {len(test)} (令和8年)")
print()

settings = {}
for label, lag, leak in (
    ("属性のみ", False, False),
    ("属性 + 前年単価(lag)", True, False),
    ("属性 + lag + リーク列", True, True),
):
    columns, _ = features.resolve_feature_columns(
        rand_train, [rand_test], cfg, include_lag=lag, include_leakage=leak
    )
    design = features.build_design_matrix(rand_train, cfg, columns=columns)
    estimator = build_estimator("lightgbm", cfg, design, use_log_target=False)
    estimator.fit(rand_train[columns], rand_train[cfg.target_column])
    settings[label] = {
        "n_features": len(columns),
        "train MAE": evaluate.regression_metrics(rand_train[cfg.target_column], estimator.predict(rand_train[columns]))["mae"],
        "test MAE": evaluate.regression_metrics(rand_test[cfg.target_column], estimator.predict(rand_test[columns]))["mae"],
        "test R2": evaluate.regression_metrics(rand_test[cfg.target_column], estimator.predict(rand_test[columns]))["r2"],
    }

# 参考: 本番の時系列分割
settings["（参考）時系列分割・lagあり"] = {
    "n_features": len(runs["lightgbm"].feature_names),
    "train MAE": runs["lightgbm"].metrics["train"]["mae"],
    "test MAE": runs["lightgbm"].metrics["test"]["mae"],
    "test R2": runs["lightgbm"].metrics["test"]["r2"],
}
pd.DataFrame(settings).T.round(3)
"""
    ),
    markdown(
        """
### この表がこのプロジェクトの結論です

1. **リーク列を入れると test MAE が劇的に下がる。** しかしこれは予測精度ではなく、
   目的変数から直接作られた列を入力に加えた結果です。実運用では
   「翌年の価格」を知るために「翌年の前年価格」が必要、という循環になります。
2. **ランダム分割はリーク列なしでもスコアを甘く見せる。** 同じ年の別地点を当てるのは
   補間（interpolation）であり、翌年を当てる外挿（extrapolation）とは別のタスクです。
3. 本プロジェクトが報告する性能は、**必ず時系列分割の数値**です。

### 改善の試行

`reports/improvements.md` に、ラグ特徴量の追加・対数変換・別系列での再学習を
**前後比較**でまとめています（改善しなかった試行も正直に記載）。
"""
    ),
    code(
        """
# 生成物の確認（CLI パイプラインの出力と一致するはず）
metrics_path = cfg.reports_dir / "metrics.json"
if metrics_path.exists():
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    stored = payload["results"]["runs"]["lightgbm"]["metrics"]["test"]
    live = runs["lightgbm"].metrics["test"]
    print("reports/metrics.json の lightgbm test 指標 vs この notebook の計算:")
    for key in ("mae", "rmse", "r2"):
        print(f"  {key:5s} json={stored[key]:12,.4f}  notebook={live[key]:12,.4f}  "
              f"一致={'OK' if abs(stored[key]-live[key]) < 1e-6 else 'NG'}")
else:
    print("reports/metrics.json が未生成です。先に `python -m src.pipeline --stage all` を実行してください。")
"""
    ),
    markdown(
        """
---
### まとめ（面接で説明する 3 点）

1. **目的変数の設計**: 総額ではなく平米単価にした。総額では「広いほど高い」という
   自明な関係を当てるだけになる。
2. **リークの実測**: 前年価格列は目的変数と 0.95 の相関を持ち、
   入れるとスコアが跳ね上がる。時系列分割と特徴量ホワイトリストで構造的に排除し、
   その効果を統制実験として定量化した。
3. **正直な結論**: この課題の最強ベースラインは「前年単価をそのまま使う」（MAE 約 750 円/㎡）で、
   勾配ブースティング（MAE 約 1,300 円/㎡）はそれを超えられない。
   一方で属性のみの線形モデル（MAE 約 1,600 円/㎡）は明確に改善しており、
   **「機械学習が効く範囲」と「効かない範囲」を切り分けた**ことが成果です。
"""
    ),
]


def main() -> None:
    EDA_CELLS_FILE = NOTEBOOK_DIR / "01_eda.ipynb"
    MODELING_CELLS_FILE = NOTEBOOK_DIR / "02_modeling.ipynb"
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    EDA_CELLS_FILE.write_text(
        json.dumps(notebook(EDA_CELLS), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    MODELING_CELLS_FILE.write_text(
        json.dumps(notebook(MODELING_CELLS), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote {EDA_CELLS_FILE.name} ({len(EDA_CELLS)} cells)")
    print(f"wrote {MODELING_CELLS_FILE.name} ({len(MODELING_CELLS)} cells)")


if __name__ == "__main__":
    main()
