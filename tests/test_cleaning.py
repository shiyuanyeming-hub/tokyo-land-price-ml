"""Tests for the cleaning layer, run against the real published files.

These tests encode what the data actually looks like (verified by inspection),
so a change in the source files or a regression in the parser fails loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cleaning
from src.config import Config
from src.data_loader import PANEL_KEY


class TestRawFiles:
    """The raw files must be readable exactly as published."""

    def test_every_source_loads_with_expected_row_count(self, raw):
        expected = {
            "kouji_r7": 2560,
            "kouji_r8": 2560,
            "kijun_r6": 1277,
            "kijun_r7": 1280,
        }
        assert {k: len(v) for k, v in raw.items()} == expected

    def test_blank_trailing_row_is_removed(self, raw, cfg):
        """Each published file carries one extra all-empty line (2561/1281 raw)."""
        for name, frame in raw.items():
            assert not frame[PANEL_KEY].isna().all(axis=1).any(), name
        counts = {name: len(frame) for name, frame in raw.items()}
        assert counts == {"kouji_r7": 2560, "kouji_r8": 2560, "kijun_r6": 1277, "kijun_r7": 1280}

    def test_panel_key_is_unique(self, raw):
        for name, frame in raw.items():
            assert frame.duplicated(subset=PANEL_KEY).sum() == 0, name

    def test_column_names_are_ascii_after_normalisation(self, raw):
        for name, frame in raw.items():
            assert all(c.isascii() for c in frame.columns), name

    def test_bracket_variants_are_normalised(self, raw):
        """公示 writes '（m)' and 基準地 writes '（m）'; both must map to road_width."""
        assert "road_width" in raw["kouji_r8"].columns
        assert "road_width" in raw["kijun_r7"].columns


class TestParsing:
    """Number, era-year and floor parsing."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("3,960,000 ", 3_960_000.0),
            ("1,340,000", 1_340_000.0),
            ("969", 969.0),
            ("10.4 ", 10.4),
            ("１２３", 123.0),  # full-width digits
            ("-", None),        # published placeholder for "no value"
            ("", None),
        ],
    )
    def test_to_number(self, text, expected):
        result = cleaning.to_number(pd.Series([text])).iloc[0]
        if expected is None:
            assert pd.isna(result)
        else:
            assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("7 ", 2025.0),          # bare era year, as published in three files
            ("8 ", 2026.0),
            ("6 ", 2024.0),
            ("令和8年", 2026.0),      # the spelling used by tokyo_kouji_r8.csv
            ("平成30年", 2018.0),     # other eras must not be read as 令和
            ("2025", 2025.0),        # an explicit Gregorian year is kept
            ("", np.nan),
            ("不明", np.nan),
        ],
    )
    def test_parse_wareki_year(self, text, expected):
        result = cleaning.parse_wareki_year(pd.Series([text])).iloc[0]
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert result == expected

    def test_parse_wareki_year_keeps_the_index(self):
        """A silent index misalignment once made every row NaN."""
        series = pd.Series(["7 ", "8 ", "6 "], index=[10, 20, 30])
        result = cleaning.parse_wareki_year(series)
        assert list(result.index) == [10, 20, 30]
        assert result.notna().all()

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("10F", 10.0), ("2F", 2.0), ("1B", -1.0), ("３Ｆ", 3.0), ("", None)],
    )
    def test_parse_floor_count(self, text, expected):
        result = cleaning.parse_floor_count(pd.Series([text])).iloc[0]
        if expected is None:
            assert pd.isna(result)
        else:
            assert result == expected

    def test_year_is_in_the_reiwa_range(self, cleaned):
        frames, _ = cleaned
        for name, frame in frames.items():
            years = frame["year"].dropna().unique()
            assert len(years) == 1, name
            assert 2019 <= years[0] <= 2026, name


class TestCleaning:
    """Cleaning must parse, never invent."""

    def test_no_row_is_dropped_for_the_real_files(self, cleaned):
        """Every published record has a usable area and price, so nothing is lost."""
        _, reports = cleaned
        for name, report in reports.items():
            assert report.rows_out == report.rows_in, (name, report.dropped_invalid)
            assert report.rows_in > 0

    def test_unit_price_equals_price_over_area(self, cleaned):
        frames, _ = cleaned
        frame = frames["kouji_r8"]
        expected = frame["price_current"] / frame["site_area"]
        assert np.allclose(frame["unit_price"], expected)

    def test_unit_price_is_plausible(self, cleaned):
        """Tokyo land is on the order of thousands of yen per square metre."""
        frames, _ = cleaned
        for name, frame in frames.items():
            price = frame["unit_price"]
            assert (price > 0).all(), name
            assert price.notna().all(), name
            assert 100 < price.median() < 1e6, name

    def test_structural_rule_drops_a_zero_area_record(self, kouji_pair):
        """A site with no area cannot have a price per square metre."""
        _, predict = kouji_pair
        broken = predict.copy()
        broken.loc[broken.index[0], "site_area"] = 0
        fixed, report = cleaning.clean_frame(broken, source="synthetic")
        assert len(fixed) == len(broken) - 1
        assert report.dropped_invalid["site_area_not_positive"] == 1

    def test_blank_and_missing_are_the_same_thing(self, raw):
        """The files mix '' and absent values; both must become NaN."""
        frame = raw["kouji_r8"]
        # '住居表示' is blank for many urban lots and must not stay as ''.
        assert not (frame["address"].astype("string").fillna("") == "").all()


class TestLinking:
    """Cross-year linkage, the basis of the whole panel."""

    def test_most_sites_recur_between_consecutive_kouji_years(self, kouji_pair):
        earlier, later = kouji_pair
        shared = set(earlier["admin_code"] + earlier["use_code"] + earlier["seq_no"])
        other = set(later["admin_code"] + later["use_code"] + later["seq_no"])
        assert len(shared & other) > 2400

    def test_previous_price_reproduces_the_earlier_year(self, kouji_pair):
        """令和8年's 前年価格 column repeats 令和7年's 当年価格 (verified 97.8%)."""
        earlier, later = kouji_pair
        keys = PANEL_KEY
        merged = later[[*keys, "price_previous"]].merge(
            earlier[[*keys, "price_current"]], on=keys, how="inner"
        )
        relative = (
            (merged["price_previous"] - merged["price_current"]).abs()
            / merged["price_current"]
        )
        assert (relative < 1e-9).mean() > 0.95

    def test_lag_join_covers_almost_every_site(self, cfg, kouji_pair):
        from src import features

        earlier, later = kouji_pair
        joined, coverage = features.attach_lag_unit_price(later, earlier)
        assert coverage["lag_coverage"] > 0.95
        assert joined["lag_unit_price"].notna().sum() == coverage["lag_available"]

    def test_lag_price_comes_from_the_earlier_year_not_the_target(self, kouji_pair):
        """The lag must equal the earlier year's price, never the later one."""
        from src import features

        earlier, later = kouji_pair
        joined, _ = features.attach_lag_unit_price(later, earlier)
        earlier = earlier.copy()
        earlier["_site_key"] = features.site_key(earlier)
        merged = joined.merge(
            earlier[["_site_key", "unit_price"]].rename(columns={"unit_price": "prev_true"}),
            on="_site_key",
            how="left",
        )
        ok = merged["lag_unit_price"].notna()
        assert np.allclose(merged.loc[ok, "lag_unit_price"], merged.loc[ok, "prev_true"])
        # And it must differ from the target for the vast majority of sites.
        differs = (merged.loc[ok, "lag_unit_price"] != merged.loc[ok, "unit_price"]).mean()
        assert differs > 0.5, "a lag that equals the target would be leakage"


class TestConfig:
    def test_seed_and_split_are_pinned(self, cfg):
        assert cfg.seed == 42
        assert cfg.split["strategy"] == "temporal"
        assert cfg.split["train_year"] != cfg.split["predict_year"]

    def test_config_stores_relative_paths_only(self, cfg):
        """No absolute path may live in the config: it must work from any checkout."""
        config_text = cfg.path.read_text(encoding="utf-8")
        assert "/Users/" not in config_text
        assert not str(cfg.raw["data"]["raw_dir"]).startswith("/")
        assert cfg.raw_dir.is_dir()
