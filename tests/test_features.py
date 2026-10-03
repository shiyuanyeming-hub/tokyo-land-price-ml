"""Tests for feature engineering and the panel join."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features
from src.config import Config
from src.data_loader import PANEL_KEY


class TestFeatureContract:
    """``feature_columns`` is the single source of truth for model inputs."""

    def test_leakage_columns_are_never_in_the_default_set(self, cfg):
        cols = features.feature_columns(cfg)
        for leaky in features.LEAKY_FEATURES:
            assert leaky not in cols

    def test_leakage_columns_only_appear_when_explicitly_requested(self, cfg):
        cols = features.feature_columns(cfg, include_leakage=True)
        assert set(features.LEAKY_FEATURES).issubset(cols)

    def test_lag_can_be_switched_off(self, cfg):
        assert features.LAG_FEATURE in features.feature_columns(cfg, include_lag=True)
        assert features.LAG_FEATURE not in features.feature_columns(cfg, include_lag=False)

    def test_config_leakage_declaration_matches_the_code(self, cfg):
        """If a new leaky source column is declared in YAML, the code must list it."""
        declared = set(cfg.leakage_columns)
        assert declared, "config must declare the leakage columns"
        assert declared == {"前年価格（円）", "対前年変動率（％）"}

    def test_raw_price_columns_are_not_treated_as_features(self, cfg):
        """The raw price/rate columns must never reach the model as features."""
        cols = set(features.feature_columns(cfg, include_leakage=True))
        for raw_price_column in ("price_current", "price_previous", "unit_price"):
            assert raw_price_column not in cols


class TestDerivedFeatures:
    @pytest.mark.parametrize(
        ("band", "source"),
        [
            ("site_area_band", "site_area"),
            ("station_distance_band", "station_distance"),
            ("road_width_band", "road_width"),
        ],
    )
    def test_band_is_present_and_only_missing_where_the_source_is(self, cleaned, band, source):
        """A band must never invent a value, and must never lose a real one."""
        frames, _ = cleaned
        for name, frame in frames.items():
            assert band in frame.columns, name
            if frame[source].notna().all():
                assert frame[band].notna().all(), (name, band)
            else:
                # Missing exactly where the source column is missing (29 sites in
                # 令和8年 have no published road width).
                assert frame[band].isna().equals(frame[source].isna()), (name, band)

    def test_station_distance_band_edges_match_the_official_classes(self, cleaned):
        frames, _ = cleaned
        frame = frames["kouji_r8"]
        # 100 m or less must land in the first bucket, 7 km in the tail bucket.
        row = frame.index[frame["station_distance"] < 100][0]
        assert frame.loc[row, "station_distance_band"] == "<=100m"
        far = frame.index[frame["station_distance"] > 5000]
        if len(far):
            assert frame.loc[far[0], "station_distance_band"] == ">5km"

    def test_far_bcr_ratio(self, cleaned):
        frames, _ = cleaned
        frame = frames["kouji_r8"].dropna(subset=["bcr", "far"])
        frame = frame.loc[frame["bcr"] > 0]
        expected = frame["far"] / frame["bcr"]
        assert np.allclose(frame["far_bcr_ratio"], expected)

    def test_categoricals_have_no_missing_values(self, cleaned):
        """'unknown' is an explicit category, so the encoders never see NA."""
        frames, _ = cleaned
        frame = frames["kouji_r8"]
        for column in features.CATEGORICAL_FEATURES:
            assert frame[column].notna().all(), column
            assert (frame[column].astype(str).str.len() > 0).all(), column

    def test_missing_use_district_becomes_unknown(self, cleaned):
        """用途区分 is blank for forest and farmland lots in the real files."""
        frames, _ = cleaned
        frame = frames["kouji_r8"]
        assert (frame["use_district"] == features.UNKNOWN_CATEGORY).sum() > 0


class TestDesignMatrix:
    def test_numeric_and_categorical_partition_the_features(self, cfg, kouji_pair):
        train, _ = kouji_pair
        design = features.build_design_matrix(train, cfg)
        assert set(design.numeric).isdisjoint(design.categorical)
        assert len(design.numeric) + len(design.categorical) == len(design.features)
        assert list(design.frame.columns) == design.features

    def test_missing_feature_raises_when_columns_are_frozen(self, cfg, kouji_pair):
        """The strict path: the pipeline always passes a frozen column list."""
        train, predict = kouji_pair
        columns, _ = features.resolve_feature_columns(train, [predict], cfg)
        broken = train.drop(columns=["site_area"])
        with pytest.raises(KeyError, match="site_area"):
            features.build_design_matrix(broken, cfg, columns=columns)

    def test_resolve_feature_columns_drops_columns_without_training_signal(
        self, cfg, kouji_pair
    ):
        """令和7年 has no previous-year price, so the lag must be dropped for it."""
        train, predict = kouji_pair
        without_lag = train.drop(columns=["lag_unit_price"], errors="ignore")
        columns, info = features.resolve_feature_columns(
            without_lag, [predict], cfg, include_lag=True
        )
        assert features.LAG_FEATURE in info["dropped_no_training_signal"]
        assert features.LAG_FEATURE not in columns

    def test_resolve_feature_columns_keeps_the_lag_when_it_has_signal(
        self, cfg, kouji_pair
    ):
        from src import features as feat

        earlier, _ = kouji_pair
        joined, _ = feat.attach_published_lag(earlier)
        columns, info = features.resolve_feature_columns(joined, [joined], cfg)
        assert features.LAG_FEATURE in columns
        assert features.LAG_FEATURE not in info["dropped_no_training_signal"]

    def test_frozen_columns_are_enforced_on_evaluation_frames(self, cfg, kouji_pair):
        train, predict = kouji_pair
        columns, _ = features.resolve_feature_columns(train, [predict], cfg)
        broken = predict.drop(columns=[features.NUMERIC_FEATURES[0]])
        with pytest.raises(KeyError):
            features.build_design_matrix(broken, cfg, columns=columns)


class TestLagFeature:
    """The lag must come from the previous year - never from the target year."""

    def test_published_lag_is_previous_price_over_area(self, kouji_pair):
        from src import features as feat

        _, later = kouji_pair
        joined, coverage = feat.attach_published_lag(later)
        expected = later["price_previous"] / later["site_area"]
        observed = joined[feat.LAG_FEATURE].notna()
        assert np.allclose(joined.loc[observed, feat.LAG_FEATURE], expected[observed])
        assert coverage["lag_available"] == int(observed.sum())

    def test_training_year_lag_is_its_own_price_not_the_next_year(self, kouji_pair):
        """The regression this test exists for.

        An earlier version joined the *later* file to build the training lag, so
        the training rows received the following year's price - a forward-looking
        feature - while the evaluation rows received the training year's price.
        The model could therefore never observe a real year-on-year trend. The
        published ``前年価格`` column is the only correct source.
        """
        from src import features as feat

        earlier, later = kouji_pair
        joined, _ = feat.attach_published_lag(earlier)
        observed = joined[feat.LAG_FEATURE].notna()
        # The lag equals the site's own assessed value from one year before, so
        # for the earlier file it must NOT equal that file's own current price.
        same_as_current = np.isclose(
            joined.loc[observed, feat.LAG_FEATURE], joined.loc[observed, "unit_price"]
        )
        assert same_as_current.mean() < 0.1, (
            "the training lag looks like the current year's own price, which "
            "means it was joined from the following year's file"
        )

    def test_lag_never_uses_the_following_year(self, kouji_pair):
        from src import features as feat

        earlier, later = kouji_pair
        joined, _ = feat.attach_published_lag(earlier)
        merged = joined[["_site_key", feat.LAG_FEATURE]].merge(
            later[["_site_key", "unit_price"]].rename(columns={"unit_price": "next_year"}),
            on="_site_key",
            how="inner",
        )
        both = merged[[feat.LAG_FEATURE, "next_year"]].dropna()
        assert not np.allclose(both[feat.LAG_FEATURE], both["next_year"]), (
            "the lag equals the following year's price: it is forward-looking"
        )

    def test_lag_carries_a_real_year_on_year_trend(self, kouji_pair):
        """Both years must show the same, plausible growth distribution."""
        from src import features as feat

        for label, frame in zip(("train", "predict"), kouji_pair):
            joined, _ = feat.attach_published_lag(frame)
            pair = joined[[feat.LAG_FEATURE, "unit_price"]].dropna()
            ratio = (pair["unit_price"] / pair[feat.LAG_FEATURE]).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            median = float(ratio.median())
            assert 0.9 < median < 1.2, (label, median)
            # A year in which almost everything moved the same way: this is the
            # property that makes a "lag x drift" baseline so strong.
            assert (ratio > 1).mean() > 0.5, (label, float((ratio > 1).mean()))

    def test_new_sites_get_a_missing_lag_not_a_fabricated_one(self, kouji_pair):
        from src import features as feat

        _, later = kouji_pair
        joined, _ = feat.attach_published_lag(later)
        missing = joined.loc[joined[feat.LAG_FEATURE].isna()]
        assert len(missing) > 0, "the files do contain newly selected sites"
        assert missing["price_previous"].isna().all(), (
            "a lag may only be missing where the published previous price is missing"
        )


class TestCrossYearJoin:
    """The generic key join, used by the cross-series checks."""

    def test_site_key_is_stable_across_years(self, kouji_pair):
        earlier, later = kouji_pair
        shared = set(features.site_key(earlier)) & set(features.site_key(later))
        assert len(shared) > 2400
        sample = later.iloc[0]
        assert features.site_key(later).iloc[0] == "|".join(str(sample[c]) for c in PANEL_KEY)

    def test_join_does_not_duplicate_rows(self, kouji_pair):
        earlier, later = kouji_pair
        joined, coverage = features.attach_lag_unit_price(later, earlier)
        assert len(joined) == len(later)
        assert coverage["lag_available"] <= coverage["rows"]

    def test_duplicate_keys_in_the_left_frame_are_rejected(self, kouji_pair):
        earlier, later = kouji_pair
        doubled = pd.concat([later, later.head(5)], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            features.attach_lag_unit_price(doubled, earlier)


class TestTarget:
    def test_unit_price_is_price_per_square_metre(self, cleaned):
        frames, _ = cleaned
        frame = frames["kouji_r8"]
        assert np.allclose(frame["unit_price"], frame["price_current"] / frame["site_area"])

    def test_target_is_not_explained_by_area_alone(self, cleaned):
        """Predicting the total price would just learn 'bigger site, bigger total'."""
        frames, _ = cleaned
        frame = frames["kouji_r8"]
        corr_total = frame[["site_area", "price_current"]].corr().iloc[0, 1]
        corr_unit = frame[["site_area", "unit_price"]].corr().iloc[0, 1]
        assert abs(corr_total) > abs(corr_unit)
        assert corr_unit < 0  # larger sites are cheaper per square metre
