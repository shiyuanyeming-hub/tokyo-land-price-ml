"""Tests for the model layer: pipelines fit, predict and stay comparable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import evaluate, features, models
from src.config import Config
from src.models import MODEL_ORDER, build_estimator
from tests._deps import AVAILABLE_FITTING_MODELS, AVAILABLE_MODELS, requires_lightgbm


def _design(train: pd.DataFrame, cfg: Config, **kwargs):
    columns, _ = features.resolve_feature_columns(train, [], cfg, **kwargs)
    return features.build_design_matrix(train, cfg, columns=columns)


class TestEstimatorConstruction:
    @pytest.mark.parametrize("name", [m for m in MODEL_ORDER if m in AVAILABLE_MODELS])
    def test_every_model_builds(self, cfg, kouji_pair, name):
        train, _ = kouji_pair
        design = _design(train, cfg)
        assert build_estimator(name, cfg, design) is not None

    def test_unknown_model_is_rejected(self, cfg, kouji_pair):
        train, _ = kouji_pair
        with pytest.raises(ValueError, match="unknown model"):
            build_estimator("no_such_model", cfg, _design(train, cfg))

    def test_linear_models_use_one_hot_encoding(self, cfg, kouji_pair):
        """An integer code would impose a false order on the categories."""
        train, _ = kouji_pair
        design = _design(train, cfg)
        prep = models.build_preprocessor(design, models.LINEAR_ENCODER)
        prep.fit(design.frame)
        assert prep.transform(design.frame).shape[1] > len(design.features)

    def test_tree_models_use_a_dense_ordinal_matrix(self, cfg, kouji_pair):
        train, _ = kouji_pair
        design = _design(train, cfg)
        prep = models.build_preprocessor(design, models.TREE_ENCODER)
        prep.fit(design.frame)
        assert prep.transform(design.frame).shape[1] == len(design.features)


class TestFitting:
    @pytest.mark.parametrize("name", [m for m in ["median", "lag", "ridge", "random_forest", "lightgbm"] if m in AVAILABLE_MODELS])
    def test_model_fits_and_predicts_finite_values(self, cfg, small_panel, name):
        train, val, _ = small_panel
        columns, _ = features.resolve_feature_columns(train, [val], cfg)
        design = features.build_design_matrix(train, cfg, columns=columns)
        estimator = build_estimator(name, cfg, design)
        estimator.fit(train[columns], train[cfg.target_column])
        pred = np.asarray(estimator.predict(val[columns]), dtype=float)
        assert pred.shape == (len(val),)
        assert np.isfinite(pred).all()
        assert (pred >= 0).all(), "land prices cannot be negative"

    def test_predictions_improve_on_the_train_median(self, cfg, small_panel):
        """A model that cannot beat 'always predict the median' is not useful."""
        train, val, _ = small_panel
        columns, _ = features.resolve_feature_columns(train, [val], cfg)
        design = features.build_design_matrix(train, cfg, columns=columns)
        y_val = val[cfg.target_column]
        median_pred = np.full(len(val), float(train[cfg.target_column].median()))
        median_mae = evaluate.regression_metrics(y_val, median_pred)["mae"]
        for name in AVAILABLE_FITTING_MODELS:
            estimator = build_estimator(name, cfg, design)
            estimator.fit(train[columns], train[cfg.target_column])
            mae = evaluate.regression_metrics(y_val, estimator.predict(val[columns]))["mae"]
            assert mae < median_mae, f"{name} failed to beat the median baseline"

    def test_median_baseline_predicts_the_median(self, cfg, small_panel):
        train, val, _ = small_panel
        estimator = build_estimator("median", cfg, _design(train, cfg))
        estimator.fit(train[[features.NUMERIC_FEATURES[0]]], train[cfg.target_column])
        pred = estimator.predict(val[[features.NUMERIC_FEATURES[0]]])
        assert np.allclose(pred, train[cfg.target_column].median())

    def test_lag_baseline_reproduces_the_previous_price(self, cfg, small_panel):
        train, val, _ = small_panel
        estimator = build_estimator("lag", cfg, _design(train, cfg))
        estimator.fit(train[[features.LAG_FEATURE]], train[cfg.target_column])
        pred = estimator.predict(val[[features.LAG_FEATURE]])
        available = val[features.LAG_FEATURE].notna().to_numpy()
        assert available.any(), "the fixture must contain sites that recur"
        assert np.allclose(pred[available], val.loc[available, features.LAG_FEATURE])

    def test_lag_baseline_falls_back_when_the_lag_is_missing(self, cfg, small_panel):
        train, val, _ = small_panel
        estimator = build_estimator("lag", cfg, _design(train, cfg))
        estimator.fit(train[[features.LAG_FEATURE]], train[cfg.target_column])
        frame = val[[features.LAG_FEATURE]].copy()
        frame.iloc[0, 0] = np.nan
        pred = estimator.predict(frame)
        assert np.isfinite(pred).all()
        assert pred[0] == pytest.approx(train[cfg.target_column].median())

    def test_log_target_is_inverted_correctly(self, cfg, small_panel):
        """TargetLogRegressor must return raw JPY/m2, not log values."""
        train, val, _ = small_panel
        columns, _ = features.resolve_feature_columns(train, [val], cfg)
        design = features.build_design_matrix(train, cfg, columns=columns)
        plain = build_estimator("ridge", cfg, design, use_log_target=False)
        logged = build_estimator("ridge", cfg, design, use_log_target=True)
        for estimator in (plain, logged):
            estimator.fit(train[columns], train[cfg.target_column])
            pred = estimator.predict(val[columns])
            assert pred.max() > 100, "predictions must be on the JPY/m2 scale"
        assert isinstance(logged, models.TargetLogRegressor)


class TestReproducibility:
    def test_same_seed_gives_identical_predictions(self, cfg, small_panel):
        train, val, _ = small_panel
        columns, _ = features.resolve_feature_columns(train, [val], cfg)
        design = features.build_design_matrix(train, cfg, columns=columns)
        preds = []
        for _ in range(2):
            estimator = build_estimator("random_forest", cfg, design)
            estimator.fit(train[columns], train[cfg.target_column])
            preds.append(np.asarray(estimator.predict(val[columns])))
        assert np.allclose(preds[0], preds[1])

    def test_lightgbm_predictions_are_reproducible(self, cfg, small_panel):
        requires_lightgbm()
        train, val, _ = small_panel
        columns, _ = features.resolve_feature_columns(train, [val], cfg)
        design = features.build_design_matrix(train, cfg, columns=columns)
        preds = []
        for _ in range(2):
            estimator = build_estimator("lightgbm", cfg, design)
            estimator.fit(train[columns], train[cfg.target_column])
            preds.append(np.asarray(estimator.predict(val[columns])))
        assert np.allclose(preds[0], preds[1])


class TestFeatureImportance:
    @pytest.mark.parametrize("name", [m for m in ["ridge", "random_forest", "gradient_boosting", "lightgbm"] if m in AVAILABLE_MODELS])
    def test_importance_table_is_readable_and_ranked(self, cfg, small_panel, name):
        train, val, _ = small_panel
        columns, _ = features.resolve_feature_columns(train, [val], cfg)
        design = features.build_design_matrix(train, cfg, columns=columns)
        estimator = build_estimator(name, cfg, design)
        estimator.fit(train[columns], train[cfg.target_column])
        fitted_prep = models.fitted_preprocessor(estimator)
        table = evaluate.feature_importance_table(
            estimator,
            models.transformed_feature_names(design, fitted_prep),
            expander=lambda token: models.expand_feature_names(token, design),
            top_n=15,
        )
        assert not table.empty, f"{name} produced no importances"
        assert table["importance"].is_monotonic_decreasing
        known = set(design.features)
        assert set(table["feature"]).issubset(known), "importance names must be real features"

    def test_lag_dominates_when_it_is_available(self, cfg, small_panel):
        """Sanity check on the ranking, not a performance claim."""
        requires_lightgbm()
        train, val, _ = small_panel
        columns, _ = features.resolve_feature_columns(train, [val], cfg)
        design = features.build_design_matrix(train, cfg, columns=columns)
        estimator = build_estimator("lightgbm", cfg, design)
        estimator.fit(train[columns], train[cfg.target_column])
        fitted_prep = models.fitted_preprocessor(estimator)
        table = evaluate.feature_importance_table(
            estimator,
            models.transformed_feature_names(design, fitted_prep),
            expander=lambda token: models.expand_feature_names(token, design),
            top_n=15,
        )
        assert features.LAG_FEATURE in set(table["feature"])
