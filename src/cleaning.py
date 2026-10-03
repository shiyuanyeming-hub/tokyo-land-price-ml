"""Cleaning of the raw Tokyo land price records.

Everything in this module is deliberately conservative: the published files are
small and official, so cleaning only *parses* what is there and never imputes,
filters or otherwise fabricates values. Records that cannot be parsed are kept
and surfaced through the quality report instead of being silently dropped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Raw string columns that must become numeric. Keys are cleaned column names.
NUMERIC_COLUMNS: dict[str, str] = {
    "price_current": "price_current",
    "price_previous": "price_previous",
    "change_rate_prior_year": "change_rate_prior_year",
    "site_area": "site_area",
    "frontage_ratio": "frontage_ratio",
    "depth_ratio": "depth_ratio",
    "road_width": "road_width",
    "station_distance": "station_distance",
    "bcr": "bcr",
    "far": "far",
    "prev_public_price": "prev_public_price",
    "prev_survey_price": "prev_survey_price",
    "change_rate_prior_survey": "change_rate_prior_survey",
}

#: Categorical text columns: full-width spaces are normalised away, then any
#: blank string becomes NaN so that "missing" has exactly one representation.
TEXT_COLUMNS: tuple[str, ...] = (
    "continuity",
    "locality",
    "municipality",
    "lot_number",
    "address",
    "shape",
    "current_use",
    "structure",
    "surrounding_use",
    "road_type",
    "road_direction",
    "road_station_front",
    "side_road_type",
    "side_road_direction",
    "gas",
    "water",
    "sewer",
    "nearest_station",
    "station_proximity",
    "use_district",
    "fire_zone",
    "far_bonus",
    "height_district",
    "area_division",
    "forest_law",
    "park_law",
    "park_law_special",
    "common_point",
    "pref_zone_1",
    "pref_zone_2",
)

#: Panel key columns stay strings but must be whitespace-free.
KEY_TEXT_COLUMNS: tuple[str, ...] = ("admin_code", "use_code", "seq_no")

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９．－", "0123456789.-")
_ERA_BASE = {"令和": 2018, "平成": 1988, "昭和": 1925}
_FLOOR_RE = re.compile(r"(-?\d+)")


def to_number(series: pd.Series) -> pd.Series:
    """Parse a published numeric text column into floats.

    The published files store numbers such as ``'3,960,000 '`` (thousands
    separators and a trailing space) and occasionally use full-width digits.
    Anything unparseable becomes ``NaN`` rather than raising.
    """
    text = (
        series.astype("string")
        .str.translate(_FULLWIDTH_DIGITS)
        .str.replace(",", "", regex=False)
        .str.replace("\u3000", "", regex=False)
        .str.strip()
    )
    text = text.replace({"": pd.NA, "-": pd.NA, "－": pd.NA})
    return pd.to_numeric(text, errors="coerce")


def parse_wareki_year(series: pd.Series) -> pd.Series:
    """Convert a Japanese era year into a Western calendar year.

    The files are inconsistent: 令和7年 is published as ``'7 '`` in three of the
    four files and as ``'令和8年'`` in ``tokyo_kouji_r8.csv``. Both spellings are
    handled, and an unrecognised format yields ``NaN`` rather than a wrong year.

    Implementation note: the result is built from numpy arrays and re-wrapped in
    a Series at the end. Mixing pandas Series with the ``numpy.where`` result
    silently misaligns on the index, which produced ``NaN`` for every row until
    it was caught by the tests.
    """
    text = series.astype("string").str.strip()
    values = np.full(len(text), np.nan, dtype="float64")

    plain = pd.to_numeric(text, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    has_plain = np.isfinite(plain)
    values[has_plain] = plain[has_plain]

    # A bare number is an era year, not a calendar year: 地価公示 is published in
    # 令和, so '7' means 令和7年 = 2025. Anything that already looks like a
    # Gregorian year (1868+) is kept as-is so the function is safe either way.
    is_calendar_year = has_plain & (plain >= 1868)
    values[is_calendar_year] = plain[is_calendar_year]
    needs_era_base = has_plain & (plain < 1868)
    values[needs_era_base] = plain[needs_era_base] + _ERA_BASE["令和"]

    raw = text.to_numpy(dtype=object)
    for era, base in _ERA_BASE.items():
        digits = np.array(
            [_FLOOR_RE.search(str(item)).group(1) if _FLOOR_RE.search(str(item)) else None
             for item in raw],
            dtype=object,
        )
        prefixed = np.array([str(item).startswith(era) for item in raw], dtype=bool)
        numeric = pd.to_numeric(pd.Series(digits), errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
        take = prefixed & np.isfinite(numeric)
        values[take] = numeric[take] + base
        needs_era_base &= ~prefixed

    return pd.Series(values, index=series.index, name="year")


def parse_floor_count(series: pd.Series) -> pd.Series:
    """Extract the floor count from strings such as ``'10F'`` or ``'1B'``.

    Basements are returned as negative numbers so that a single numeric column
    carries the sign convention "above ground positive, below ground negative".
    """
    text = series.astype("string").str.translate(_FULLWIDTH_DIGITS).str.strip().str.upper()
    digits = text.str.extract(_FLOOR_RE, expand=False)
    value = pd.to_numeric(digits, errors="coerce")
    basement = text.str.endswith(("B", "Ｂ")).fillna(False)
    return value.where(~basement, -value)


def strip_text(series: pd.Series) -> pd.Series:
    """Normalise a text column: trim, collapse inner spaces, blank -> NaN."""
    text = (
        series.astype("string")
        .str.replace("\u3000", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return text.replace({"": pd.NA, "nan": pd.NA})


@dataclass
class QualityReport:
    """Per-source record of what cleaning observed (never of what it invented)."""

    source: str
    rows_in: int
    rows_out: int
    numeric_missing: dict[str, int] = field(default_factory=dict)
    numeric_unparseable: dict[str, int] = field(default_factory=dict)
    dropped_invalid: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "numeric_missing": self.numeric_missing,
            "numeric_unparseable": self.numeric_unparseable,
            "dropped_invalid": self.dropped_invalid,
            "notes": self.notes,
        }


def clean_frame(df: pd.DataFrame, source: str = "unknown") -> tuple[pd.DataFrame, QualityReport]:
    """Clean one raw frame and return it together with a quality report.

    Steps, in order:

    1. normalise key/text columns (trim, blank -> ``NaN``);
    2. parse every numeric column (thousands separators, full-width digits);
    3. derive ``year`` from the Japanese era year and floor counts from ``xF``;
    4. apply *structural* validity rules and report - not hide - what they drop.

    The structural rules are minimal and unavoidable: a record without a
    positive site area cannot yield a price per square metre, and the target is
    undefined for it. Both rules are counted in the report.
    """
    report = QualityReport(source=source, rows_in=int(len(df)), rows_out=int(len(df)))
    out = df.copy()

    for col in (*TEXT_COLUMNS, *KEY_TEXT_COLUMNS):
        if col in out.columns:
            out[col] = strip_text(out[col])

    for raw_col, clean_col in NUMERIC_COLUMNS.items():
        if raw_col not in out.columns:
            continue
        original = out[raw_col].astype("string")
        parsed = to_number(out[raw_col])
        report.numeric_missing[clean_col] = int(original.isna().sum())
        report.numeric_unparseable[clean_col] = int(
            (parsed.isna() & original.notna()).sum()
        )
        out[clean_col] = parsed

    if "year_wareki" in out.columns:
        out["year"] = parse_wareki_year(out["year_wareki"])
        if out["year"].isna().any():
            report.notes.append(f"{int(out['year'].isna().sum())} row(s) with unparsed era year")

    if "floors_above_raw" in out.columns:
        out["floors_above"] = parse_floor_count(out["floors_above_raw"])
    if "floors_below_raw" in out.columns:
        out["floors_below"] = parse_floor_count(out["floors_below_raw"])
    out["floors_total"] = out.get("floors_above", 0).fillna(0) + out.get(
        "floors_below", 0
    ).fillna(0)
    out["floors_total"] = out["floors_total"].replace(0, np.nan)

    # --- structural validity (counted, never silent) ----------------------
    before = len(out)
    out = out.loc[out["site_area"].notna() & (out["site_area"] > 0)].copy()
    report.dropped_invalid["site_area_not_positive"] = before - len(out)

    before = len(out)
    out = out.loc[out["price_current"].notna() & (out["price_current"] > 0)].copy()
    report.dropped_invalid["price_current_not_positive"] = before - len(out)

    out = out.reset_index(drop=True)
    report.rows_out = int(len(out))
    logger.info(
        "cleaned %-10s rows %d -> %d (%s)",
        source,
        report.rows_in,
        report.rows_out,
        ", ".join(f"{k}={v}" for k, v in report.dropped_invalid.items() if v) or "no drops",
    )
    return out, report


def add_unit_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add the modelling target: price per square metre (JPY/m2).

    ``当年価格（円）`` is the value of the *whole site*, so predicting it directly
    would mostly learn "bigger site, bigger total". Dividing by ``地積（㎡）``
    removes that trivial scale effect.

    ``prev_unit_price`` is the file's own ``前年価格（円）`` column divided by the
    *current* area. The published series revises the previous price when the
    site boundary was re-drawn, so this column is **not** a clean lag feature;
    the pipeline uses the measured cross-year lag built in :mod:`src.features`
    instead and keeps this column only for the leakage experiment.
    """
    out = df.copy()
    area = out["site_area"].replace(0, np.nan)
    out["unit_price"] = out["price_current"] / area
    out["prev_unit_price"] = out["price_previous"] / area
    return out
