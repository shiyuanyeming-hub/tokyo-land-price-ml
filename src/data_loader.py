"""Loading of the raw Tokyo land price CSVs.

Two official series are used:

* **地価公示** (``kouji``) - the national official land price publication, where
  the site identifier columns are ``標準地名`` / ``標準地番号``.
* **基準地価** (``kijun``) - the Tokyo prefectural land price survey, which uses
  ``基準地名`` / ``基準地番号`` and carries two extra prefecture-level zoning
  columns.

Both files are CP932-encoded, start with a decorative title row and store every
number as text with thousands separators. :func:`load_raw` normalises the column
names so that the two series can be handled by the same downstream code, but it
never changes a single data value.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

# Japanese column name -> ASCII identifier. Names are normalised (whitespace
# removed) before this lookup, so '交通施設までの道路距離（m）' and
# '交通施設までの道路距離（m)' both resolve to the mid-ASCII bracket variant.
COLUMN_MAP: dict[str, str] = {
    "継続地点": "continuity",
    "都道府県市区町村コード": "admin_code",
    "対象年（和暦）": "year_wareki",
    "標準地名": "locality",
    "基準地名": "locality",
    "標準地番号（用途）": "use_code",
    "基準地番号（用途）": "use_code",
    "標準地番号（連番）": "seq_no",
    "基準地番号（連番）": "seq_no",
    "都分類地区１": "pref_zone_1",
    "都分類地区２": "pref_zone_2",
    "区市町村名": "municipality",
    "地番": "lot_number",
    "住居表示": "address",
    "当年価格（円）": "price_current",
    "前年価格（円）": "price_previous",
    "対前年変動率（％）": "change_rate_prior_year",
    "地積（㎡）": "site_area",
    "形状区分": "shape",
    "間口（比率）": "frontage_ratio",
    "奥行（比率）": "depth_ratio",
    "利用の現況": "current_use",
    "構造": "structure",
    "地上階": "floors_above_raw",
    "地下階": "floors_below_raw",
    "周辺の土地の利用の現況": "surrounding_use",
    "前面道路区分": "road_type",
    "前面道路の方位": "road_direction",
    "前面道路の幅員（m）": "road_width",
    "前面道路の駅前区分": "road_station_front",
    "側道区分": "side_road_type",
    "側道方位": "side_road_direction",
    "ガス": "gas",
    "水道": "water",
    "下水道": "sewer",
    "主要交通施設": "nearest_station",
    "交通施設との近接区分": "station_proximity",
    "交通施設までの道路距離（m）": "station_distance",
    "用途区分": "use_district",
    "防火地域": "fire_zone",
    "指定建蔽率（％）": "bcr",  # 建蔽率 = building coverage ratio
    "指定容積率（％）": "far",  # 容積率  = floor area ratio
    "割増容積率考慮区分": "far_bonus",
    "高度地区": "height_district",
    "区域区分": "area_division",
    "森林法": "forest_law",
    "公園法": "park_law",
    "公園法普通特別": "park_law_special",
    "共通地点区分": "common_point",
    "前回地価公示都道府県市区町村コード": "prev_admin_code",
    "前回地価公示番号（用途）": "prev_use_code",
    "前回地価公示番号（連番）": "prev_seq_no",
    "前回価公示価格（円）": "prev_public_price",
    "前回地価調査都道府県市区町村コード": "prev_admin_code",
    "前回地価調査番号（用途）": "prev_use_code",
    "前回地価調査番号（連番）": "prev_seq_no",
    "前回地価調査価格（円）": "prev_survey_price",
    "対前回地価調査変動率（％）": "change_rate_prior_survey",
}

#: Columns that form the cross-year site identifier (panel key).
PANEL_KEY: list[str] = ["admin_code", "use_code", "seq_no"]

#: The base (non-price) columns shared with the metadata of every series.
KEY_COLUMNS: list[str] = [*PANEL_KEY, "locality", "municipality", "lot_number"]


def _normalise_name(name: str) -> str:
    """Canonicalise a raw column name before the lookup table is applied.

    The published files are not internally consistent: 地価公示 writes
    ``交通施設までの道路距離（m)`` with a full-width opening and a half-width
    closing bracket, while 基準地価 writes ``（m）`` with both full width. Removing
    all whitespace and folding the bracket variants makes the two series agree
    without touching a single data value.
    """
    text = str(name).replace(" ", "").replace("\u3000", "").strip()
    return text.replace("(", "（").replace(")", "）")


def read_raw_csv(path: str | Path, cfg: Config) -> pd.DataFrame:
    """Read one raw CSV exactly as published (string dtype, CP932)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"raw data file missing: {path}. See data/DATA_CARD.md for the download source."
        )
    df = pd.read_csv(
        path,
        encoding=cfg.encoding,
        skiprows=cfg.skiprows,
        dtype=str,
        low_memory=False,
    )
    df.columns = [_normalise_name(c) for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)
    logger.info("read %s -> %d rows x %d cols", path.name, *df.shape)
    return df


def drop_trailing_blank_rows(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Drop the fully blank trailing row(s) present in every published file.

    The guard is deliberately narrow: a row is dropped only when the whole panel
    key is missing, so a real but partially filled record can never be lost.
    """
    key = [c for c in PANEL_KEY if c in df.columns]
    blank = df[key].isna().all(axis=1)
    if blank.any():
        logger.info("%s: dropped %d fully blank row(s)", source, int(blank.sum()))
    return df.loc[~blank].reset_index(drop=True)


def load_raw(key: str, cfg: Config) -> pd.DataFrame:
    """Load one raw source by its config key (e.g. ``"kouji_r8"``)."""
    df = read_raw_csv(cfg.source_path(key), cfg)
    return drop_trailing_blank_rows(df, key)


def load_all_raw(cfg: Config) -> dict[str, pd.DataFrame]:
    """Load every configured raw source."""
    return {key: load_raw(key, cfg) for key in cfg.sources}


def summarise(df: pd.DataFrame, name: str) -> dict[str, object]:
    """Small profiling record used by the data-quality report."""
    key = [c for c in PANEL_KEY if c in df.columns]
    return {
        "source": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "unique_sites": int(df[key].drop_duplicates().shape[0]) if key else -1,
        "duplicate_site_keys": int(df.duplicated(subset=key).sum()) if key else -1,
        "all_null_columns": sorted(df.columns[df.isna().all()].tolist()),
    }
