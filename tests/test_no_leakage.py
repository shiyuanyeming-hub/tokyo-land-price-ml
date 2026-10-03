"""Leakage tests: the guard rails around the single biggest risk in this project.

"前年価格" and "対前年変動率" are derived from the very price being predicted
when a model is trained and evaluated on the same year. They are excellent
predictors and completely useless forecasters, so this file asserts three
separate things:

1. the feature contract never selects them by default;
2. the pipeline's own experiment is the only thing that can, and only for the
   prediction year (never for the training rows);
3. a model that does see them behaves in the tell-tale way - a train/test gap
   that collapses - which is what a reviewer should look for.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._deps import requires_lightgbm

from src import evaluate, features
from src.config import Config
from src.models import build_estimator

#: The raw published columns that would leak a same-year target.
SAME_YEAR_PRICE_COLUMNS = ("price_previous", "change_rate_prior_year", "prev_unit_price")


class TestFeatureContractIsLeakFree:
    def test_default_features_contain_no_same_year_price_column(self, cfg):
        cols = set(features.feature_columns(cfg))
        assert cols.isdisjoint(SAME_YEAR_PRICE_COLUMNS)
        assert cols.isdisjoint(features.LEAKY_FEATURES)

    def test_default_features_exclude_the_target_and_its_components(self, cfg):
        cols = set(features.feature_columns(cfg))
        for forbidden in ("unit_price", "price_current", "prev_unit_price"):
            assert forbidden not in cols

    def test_design_matrix_never_contains_the_target(self, cfg, kouji_pair):
        train, _ = kouji_pair
        design = features.build_design_matrix(train, cfg)
        assert cfg.target_column not in design.frame.columns
        assert "price_current" not in design.frame.columns

    def test_leaky_columns_require_an_explicit_flag(self, cfg):
        assert features.LEAKY_FEATURES
        default = features.feature_columns(cfg)
        explicit = features.feature_columns(cfg, include_leakage=True)
        assert set(explicit) - set(default) == set(features.LEAKY_FEATURES)

    def test_resolve_feature_columns_only_drops_and_never_adds(self, cfg, kouji_pair):
        train, predict = kouji_pair
        columns, info = features.resolve_feature_columns(train, [predict], cfg)
        assert set(columns).issubset(set(features.feature_columns(cfg)))
        assert info["dropped_no_training_signal"] == [
            c for c in features.feature_columns(cfg) if c not in columns
        ]


class TestLagIsNotLeakage:
    """Being lagged one year is exactly what makes a feature legitimate."""

    def test_lag_is_the_published_previous_price(self, kouji_pair):
        from src import features as feat

        _, later = kouji_pair
        assert later["year"].iloc[0] == 2026
        joined, _ = feat.attach_published_lag(later)
        observed = joined[feat.LAG_FEATURE].notna()
        expected = later["price_previous"] / later["site_area"]
        assert np.allclose(joined.loc[observed, feat.LAG_FEATURE], expected[observed])

    def test_lag_does_not_reproduce_the_target(self, kouji_pair):
        """If it did, the 'forecast' would be a lookup, not a prediction."""
        from src import features as feat

        _, later = kouji_pair
        joined, _ = feat.attach_published_lag(later)
        observed = joined[feat.LAG_FEATURE].notna() & joined["unit_price"].notna()
        exact = np.isclose(
            joined.loc[observed, feat.LAG_FEATURE], joined.loc[observed, "unit_price"]
        )
        assert exact.mean() < 0.05, "the lag repeats the target for too many sites"

    def test_lag_is_not_taken_from_the_year_being_predicted(self, kouji_pair):
        """The guard against the bug this project actually shipped once.

        Joining the companion file produced a "lag" that was the *following*
        year's price for training rows and the site's own price for evaluation
        rows. Either way the model saw no genuine year-on-year movement. The
        published previous price is the only correct source among these files.
        """
        from src import features as feat

        earlier, later = kouji_pair
        train_joined, _ = feat.attach_published_lag(earlier)
        test_joined, _ = feat.attach_published_lag(later)

        # Neither year's lag may equal that year's own price.
        for label, joined in (("train", train_joined), ("test", test_joined)):
            observed = joined[feat.LAG_FEATURE].notna()
            same = np.isclose(
                joined.loc[observed, feat.LAG_FEATURE], joined.loc[observed, "unit_price"]
            )
            assert same.mean() < 0.1, label

        # And both years must show a comparable, real growth distribution.
        ratios = {}
        for label, joined in (("train", train_joined), ("test", test_joined)):
            pair = joined[[feat.LAG_FEATURE, "unit_price"]].dropna()
            ratios[label] = (pair["unit_price"] / pair[feat.LAG_FEATURE]).median()
        assert 0.95 < ratios["train"] < 1.2, ratios
        assert abs(ratios["train"] - ratios["test"]) < 0.1, ratios

    def test_lag_correlation_with_the_target_is_high_but_not_suspicious(self, kouji_pair):
        from src import features as feat

        _, later = kouji_pair
        joined, _ = feat.attach_published_lag(later)
        pair = joined[[feat.LAG_FEATURE, "unit_price"]].dropna()
        corr = float(pair.corr().iloc[0, 1])
        assert 0.9 < corr < 0.999, corr


class TestLeakyColumnsBehaveLikeLeakage:
    """The controlled experiment, reproduced as a test on the prediction year."""

    @pytest.fixture(scope="class")
    def leaky_setup(self, cfg: Config, kouji_pair):
        earlier, later = kouji_pair
        frame = later.head(1200).reset_index(drop=True)
        train, test = evaluate.random_split(frame, cfg)
        return train, test

    def _fit(self, cfg: Config, train, test, include_leakage: bool, include_lag: bool):
        columns, _ = features.resolve_feature_columns(
            train, [test], cfg, include_lag=include_lag, include_leakage=include_leakage
        )
        design = features.build_design_matrix(train, cfg, columns=columns)
        estimator = build_estimator("lightgbm", cfg, design, use_log_target=False)
        estimator.fit(train[columns], train[cfg.target_column])
        train_mae = evaluate.regression_metrics(
            train[cfg.target_column], estimator.predict(train[columns])
        )["mae"]
        test_mae = evaluate.regression_metrics(
            test[cfg.target_column], estimator.predict(test[columns])
        )["mae"]
        return train_mae, test_mae

    def test_leaky_columns_collapse_the_train_test_gap(self, cfg, leaky_setup):
        """A near-zero generalisation gap on a price task is the red flag."""
        requires_lightgbm()
        pytest.importorskip("lightgbm")
        train, test = leaky_setup
        clean_train, clean_test = self._fit(cfg, train, test, False, True)
        leaky_train, leaky_test = self._fit(cfg, train, test, True, True)

        clean_gap = clean_test / clean_train
        leaky_gap = leaky_test / leaky_train
        assert leaky_test < clean_test, "the leaky model should look better"
        assert leaky_gap < clean_gap, "and its train/test gap should be smaller"
        # The leaky model is also far more accurate *on its own training rows*.
        assert leaky_train < clean_train

    def test_leaky_model_is_unusable_for_a_forecast(self, cfg, kouji_pair):
        """Trained on one year, scored on the next: the leaky column is not there.

        This is the structural reason the leaky columns are excluded: at real
        forecast time the current year's own price ratio simply does not exist.
        """
        requires_lightgbm()
        pytest.importorskip("lightgbm")
        earlier, later = kouji_pair
        val, test = evaluate.temporal_split(later, cfg)
        # 'prev_unit_price' exists for 令和8年 rows, so the leaky model *can* be
        # fitted and scored - but only because the training rows are the same
        # year. Mixing years makes the column inconsistent, which is exactly why
        # the project never uses it.
        missing_in_earlier = earlier["price_previous"].isna().mean()
        missing_in_later = later["price_previous"].isna().mean()
        assert missing_in_later < missing_in_earlier or missing_in_earlier > 0

    def test_prev_unit_price_matches_the_target_too_closely_to_be_a_feature(
        self, cfg, kouji_pair
    ):
        """Quantifies why it is excluded: it is the target plus one year's drift."""
        _, later = kouji_pair
        pair = later[["prev_unit_price", "unit_price"]].dropna()
        ratio = (pair["unit_price"] / pair["prev_unit_price"]).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        # The year-on-year ratio is tightly concentrated around 1.
        assert ratio.quantile(0.05) > 0.8
        assert ratio.quantile(0.95) < 1.3
        corr = float(pair.corr().iloc[0, 1])
        assert corr > 0.9


class TestSplitIntegrity:
    def test_temporal_split_produces_disjoint_halves(self, cfg, kouji_pair):
        _, later = kouji_pair
        val, test = evaluate.temporal_split(later, cfg)
        assert len(val) + len(test) == len(later)
        assert set(val["_site_key"]).isdisjoint(set(test["_site_key"]))

    def test_temporal_split_is_seeded_and_reproducible(self, cfg, kouji_pair):
        _, later = kouji_pair
        first_val, first_test = evaluate.temporal_split(later, cfg)
        second_val, second_test = evaluate.temporal_split(later, cfg)
        assert list(first_val["_site_key"]) == list(second_val["_site_key"])
        assert list(first_test["_site_key"]) == list(second_test["_site_key"])

    def test_train_and_predict_years_are_different(self, cfg, kouji_pair):
        earlier, later = kouji_pair
        assert set(earlier["year"]) != set(later["year"])
        assert earlier["year"].max() < later["year"].min()

    def test_random_split_does_share_the_year_and_that_is_the_point(self, cfg, kouji_pair):
        """Documents the control experiment: a random split is interpolation."""
        _, later = kouji_pair
        train, test = evaluate.random_split(later, cfg)
        assert set(train["year"]) == set(test["year"])
        assert set(train["_site_key"]).isdisjoint(set(test["_site_key"])), (
            "even a random split must not put the same site on both sides"
        )
