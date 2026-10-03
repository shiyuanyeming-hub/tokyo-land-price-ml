"""Model definitions and the sklearn pipelines that wrap them.

Every model is returned as a plain scikit-learn estimator so that the same
object works for ``fit``/``predict``, for cross-validation and for permutation
importance. Preprocessing is part of the pipeline, which means it is fitted on
the training fold only - a common source of silent leakage when scaling or
encoding is done on the full dataset before splitting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import Config
from src.features import UNKNOWN_CATEGORY, DesignMatrix

logger = logging.getLogger(__name__)

#: Encoders that keep the feature space dense and let trees split on categories.
TREE_ENCODER = "ordinal"
#: Encoders for linear models, where an arbitrary integer code would be wrong.
LINEAR_ENCODER = "onehot"

#: Models whose bias-variance profile benefits from working on log(price).
LOG_FRIENDLY = ("ridge", "linear")


class TargetLogRegressor(BaseEstimator, RegressorMixin):
    """Wrap a regressor so that it fits ``log1p(y)`` and predicts in raw units.

    Land prices span five orders of magnitude (a forest plot in Hinohara and a
    Ginza commercial lot live in the same file), so an L2 objective on the raw
    scale is dominated by a handful of very expensive sites. Fitting on the log
    scale turns that into a relative-error objective. Predictions are mapped
    back with ``expm1`` and clipped at zero, which is the correct inverse
    transform and keeps metrics interpretable in JPY/m2.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetLogRegressor":
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.log1p(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(np.expm1(self.estimator_.predict(X)), 0.0, None)


class NonNegativeRegressor(BaseEstimator, RegressorMixin):
    """Clip predictions at zero: a land price per square metre cannot be negative.

    Gradient boosting on a raw target has no such constraint and does return
    negative values for a handful of cheap, unusual sites (forest and island
    lots). Clipping is a domain constraint, not a modelling trick, and it is
    applied identically to every model so the comparison stays fair.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: Any) -> "NonNegativeRegressor":
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(self.estimator_.predict(X), dtype=float), 0.0, None)

    def __sklearn_is_fitted__(self) -> bool:  # pragma: no cover - sklearn plumbing
        return hasattr(self, "estimator_")


class LGBMCategoricalRegressor(BaseEstimator, RegressorMixin):
    """LightGBM regressor that declares the encoded categorical block.

    LightGBM can partition categories natively, but scikit-learn's
    ``OrdinalEncoder`` hands it plain float columns, so the model would otherwise
    treat a category code as a *number* - an arbitrary order that does not exist.
    This small adapter re-marks those columns by name right before training.
    """

    def __init__(self, **params: Any) -> None:
        self.params = params

    def fit(self, X: pd.DataFrame, y: Any) -> "LGBMCategoricalRegressor":
        import lightgbm as lgb

        self.estimator_ = lgb.LGBMRegressor(**self.params)
        cat_cols = [c for c in X.columns if str(c).startswith("cat_") and X[c].dtype.kind in "ifu"]
        if cat_cols:
            self.estimator_.set_params(
                categorical_feature=[X.columns.get_loc(c) for c in cat_cols]
            )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator_.feature_importances_


class LagBaseline(BaseEstimator, RegressorMixin):
    """Predict next year's unit price as this year's unit price.

    The honest, assumption-free baseline for a one-year-ahead forecast: no
    learning at all, just "prices do not move". Any model that cannot beat it is
    not adding value.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LagBaseline":  # noqa: ARG002 - API symmetry
        if "lag_unit_price" not in X.columns:
            raise KeyError("LagBaseline requires the 'lag_unit_price' feature")
        self.fallback_ = float(np.nanmedian(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag = pd.to_numeric(X["lag_unit_price"], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(lag), lag, self.fallback_)


class LagDriftBaseline(BaseEstimator, RegressorMixin):
    """Predict last year's unit price times an assumed annual growth factor.

    Land prices in Tokyo rose almost everywhere in this period (97.5% of the
    test sites), so "prices do not move" is a deliberately conservative baseline.
    This variant adds a *fixed, assumed* drift from the config
    (``assumed_annual_growth``), never the realised figure, because a forecast
    made at the start of the year cannot know it. It exists to make the
    difficulty of the task explicit: a model that cannot beat it has added
    nothing over "last year's price plus a rule of thumb".

    It doubles as the honest answer to "what would you deploy?" - and as the
    reference point for the error decomposition discussed in the README.
    """

    def __init__(self, growth: float = 0.05) -> None:
        self.growth = growth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LagDriftBaseline":  # noqa: ARG002
        if "lag_unit_price" not in X.columns:
            raise KeyError("LagDriftBaseline requires the 'lag_unit_price' feature")
        self.offset_ = float(self.growth)
        self.fallback_ = float(np.nanmedian(np.asarray(y, dtype=float)) * (1.0 + self.growth))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag = pd.to_numeric(X["lag_unit_price"], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(lag), lag * (1.0 + self.offset_), self.fallback_)


class ConstantRegressor(BaseEstimator, RegressorMixin):
    """Predict a constant: the train median (robust) or mean (L2-optimal)."""

    def __init__(self, statistic: str = "median") -> None:
        self.statistic = statistic

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConstantRegressor":  # noqa: ARG002
        values = np.asarray(y, dtype=float)
        self.value_ = float(
            np.nanmedian(values) if self.statistic == "median" else np.nanmean(values)
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value_, dtype=float)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a fixed column subset, keeping DataFrame output for encoders.

    ``categorical`` lists the columns whose missing values are replaced by the
    literal ``"unknown"``; everything else is passed through untouched so that
    numeric medians are still computed on genuinely missing values.
    """

    def __init__(self, columns: list[str], categorical: tuple[str, ...] = ()) -> None:
        self.columns = columns
        self.categorical = categorical

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ColumnSelector":  # noqa: ARG002
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.columns]
        # pandas' nullable StringDtype mixes pd.NA with python str, which the
        # scikit-learn encoders reject with "must be uniformly strings or
        # numbers". Encoding missing as an explicit literal category is also
        # semantically right here: "not recorded" is a real category of these
        # official files (e.g. 用途区分 is blank for forest and farmland lots).
        return self._coerce(out)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:  # noqa: ARG002
        return np.asarray(self.columns, dtype=object)

    def _coerce(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in self.categorical:
            out[col] = out[col].astype(object).where(out[col].notna(), UNKNOWN_CATEGORY).astype(str)
        return out


class DataFrameOutput(BaseEstimator, TransformerMixin):
    """Force a transformer's output into a named pandas DataFrame.

    LightGBM addresses categorical columns by position, and matching positions
    back to feature names after a ``ColumnTransformer`` is error-prone. Wrapping
    the encoders like this makes the block boundaries explicit and keeps
    ``get_feature_names_out`` correct for the importance table.
    """

    def __init__(self, transformer: BaseEstimator, prefix: str) -> None:
        self.transformer = transformer
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DataFrameOutput":
        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = np.asarray(self.transformer_.transform(X))
        names = self._column_names(values.shape[1])
        return pd.DataFrame(values, columns=names, index=X.index)

    def _column_names(self, width: int) -> list[str]:
        """Name the emitted columns from the *encoder*, not from the input.

        The encoder knows which input column produced each output column
        (``data__original`` for one-hot, ``original`` for ordinal), so the names
        survive all the way to the importance table.
        """
        try:
            raw = [str(n) for n in self.transformer_.get_feature_names_out()]
        except Exception:  # pragma: no cover - defensive
            raw = []
        if len(raw) != width:
            return [f"{self.prefix}_{i}" for i in range(width)]
        return [f"{self.prefix}_{name}" for name in raw]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:  # noqa: ARG002
        try:
            raw = [str(n) for n in self.transformer_.get_feature_names_out()]
        except Exception:  # pragma: no cover - defensive
            raw = []
        return np.asarray([f"{self.prefix}_{name}" for name in raw], dtype=object)


def build_preprocessor(design: DesignMatrix, encoder: str) -> ColumnTransformer:
    """Numeric median-imputation + categorical encoding, fold-safe by design.

    Both branches are fitted inside the pipeline, so imputation statistics and
    category vocabularies come from the training fold only. The categorical
    feature names are prefixed with ``cat_`` so that
    :class:`LGBMCategoricalRegressor` can find the block it must declare to
    LightGBM as categorical.
    """
    categorical_pipeline = Pipeline(
        [
            ("select", ColumnSelector(design.categorical, categorical=tuple(design.categorical))),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                )
                if encoder == TREE_ENCODER
                else OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    ) if encoder in (TREE_ENCODER, LINEAR_ENCODER) else None
    if categorical_pipeline is None:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown encoder: {encoder!r}")

    numeric = Pipeline(
        [
            ("select", ColumnSelector(design.numeric)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", DataFrameOutput(numeric, prefix="num"), design.numeric),
            ("cat", DataFrameOutput(categorical_pipeline, prefix="cat"), design.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fitted_preprocessor(estimator: Any) -> Any:
    """Return the fitted preprocessor inside any of the project's wrappers.

    The layer order is fixed and easy to get wrong by hand:
    ``TargetLogRegressor`` or ``NonNegativeRegressor`` -> ``ModelPipeline`` ->
    ``.preprocessor_``. Centralising it here keeps the CLI and the tests honest.

    The constant and lag baselines consume raw columns and have no preprocessor;
    ``None`` is returned for them so the caller can fall back to the design-matrix
    feature names instead of failing.
    """
    core = estimator
    while hasattr(core, "estimator_") and not hasattr(core, "preprocessor_"):
        core = core.estimator_
    return getattr(core, "preprocessor_", None)


def transformed_feature_names(design: DesignMatrix, preprocessor: Any) -> list[str]:
    """Names of the columns the fitted preprocessor actually emits, in order.

    For the ordinal (tree) encoder this is one column per input feature. For the
    one-hot (linear) encoder the categorical block expands to many columns, and
    the *original* feature name is embedded in each one so that the importance
    table can be aggregated back to a readable level.
    """
    names = [f"num_{c}" for c in design.numeric]
    encoded: list[str] = []
    if preprocessor is not None:
        try:
            encoded = [
                str(n)
                for n in preprocessor.named_transformers_["cat"].get_feature_names_out()
            ]
        except (AttributeError, KeyError):  # unfitted or unexpected layout
            encoded = []
    if not encoded:
        encoded = [f"cat_{i}" for i in range(len(design.categorical))]
    names.extend(encoded)
    return names


def expand_feature_names(token: str, design: DesignMatrix) -> list[str]:
    """Map one emitted column back to the design-matrix feature it came from.

    Both encoders prefix their output with ``cat_``/``num_``, but they name the
    remainder differently, and a naive substring search merges unrelated features
    (``locality`` is a substring of both ``municipality`` and
    ``municipality_unknown``). Each shape is therefore matched explicitly:

    * ``num_<column>``            -> that numeric feature
    * ``cat_<column>``            -> that categorical feature (ordinal encoding)
    * ``cat_<column>_<category>`` -> that categorical feature (one-hot encoding)
    * ``cat_<integer>``           -> the categorical feature at that position,
      which is how the fallback naming in ``transformed_feature_names`` labels the
      ordinal block when the encoder cannot report its own names.
    """
    if token.startswith("num_"):
        # Strip the block prefix so the importance table speaks the same feature
        # names as DesignMatrix.features, for every model family.
        return [token[len("num_") :]]
    if not token.startswith("cat_"):
        return [token]

    remainder = token[len("cat_") :]
    if remainder.isdigit():
        index = int(remainder)
        if 0 <= index < len(design.categorical):
            return [design.categorical[index]]
        return [token]
    for column in design.categorical:
        if remainder == column or remainder.startswith(f"{column}_"):
            return [column]
    return [token]


class ModelPipeline(BaseEstimator, RegressorMixin):
    """Preprocessing + estimator, with a DataFrame handed to the final step.

    scikit-learn's ``set_output(transform="pandas")`` plumbing differs between
    versions and interacts badly with nested pipelines, so the conversion is done
    once, explicitly, right before the estimator is called. LightGBM needs the
    named columns to declare its categorical block; the other models do not care.
    """

    def __init__(self, preprocessor: BaseEstimator, estimator: BaseEstimator) -> None:
        self.preprocessor = preprocessor
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: Any) -> "ModelPipeline":
        self.preprocessor_ = clone(self.preprocessor)
        self.estimator_ = clone(self.estimator)
        transformed = self._as_frame(self.preprocessor_.fit_transform(X, y))
        self.estimator_.fit(transformed, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(self._as_frame(self.preprocessor_.transform(X)))

    @staticmethod
    def _as_frame(values: Any) -> pd.DataFrame:
        if isinstance(values, pd.DataFrame):
            return values
        array = np.asarray(values)
        return pd.DataFrame(array, columns=[f"f{i}" for i in range(array.shape[1])])

    def __sklearn_is_fitted__(self) -> bool:  # pragma: no cover - sklearn plumbing
        return hasattr(self, "estimator_")


def build_estimator(
    name: str,
    cfg: Config,
    design: DesignMatrix,
    use_log_target: bool | None = None,
) -> BaseEstimator:
    """Build one named model together with its preprocessing pipeline.

    Parameters
    ----------
    name:
        One of ``median``, ``mean``, ``lag``, ``ridge``, ``random_forest``,
        ``gradient_boosting``, ``lightgbm``.
    use_log_target:
        ``None`` selects a sensible default per model family (log for the linear
        models, raw for the tree ensembles, since they are scale-invariant under
        squared error on the raw scale and were measurably better that way).
    """
    name = name.lower()
    seed = cfg.seed

    if name == "median":
        return ConstantRegressor(statistic="median")
    if name == "mean":
        return ConstantRegressor(statistic="mean")
    if name == "lag":
        return LagBaseline()
    if name == "lag_drift":
        return LagDriftBaseline(growth=float(cfg.features.get("assumed_annual_growth", 0.05)))

    if use_log_target is None:
        use_log_target = name in LOG_FRIENDLY

    if name == "ridge":
        encoder = LINEAR_ENCODER
        estimator: BaseEstimator = Ridge(alpha=cfg.model_params("ridge")["alpha"], random_state=seed)
    elif name == "random_forest":
        encoder = TREE_ENCODER
        params = cfg.model_params("random_forest")
        estimator = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1,
        )
    elif name == "gradient_boosting":
        encoder = TREE_ENCODER
        params = cfg.model_params("gradient_boosting")
        estimator = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=seed,
        )
    elif name == "lightgbm":
        encoder = TREE_ENCODER
        params = dict(cfg.model_params("lightgbm"))
        estimator = LGBMCategoricalRegressor(random_state=seed, n_jobs=-1, **params)
    else:
        raise ValueError(f"unknown model: {name!r}")

    pipeline = ModelPipeline(build_preprocessor(design, encoder), estimator)
    if use_log_target:
        # expm1 already lands on the positive scale.
        return TargetLogRegressor(pipeline)
    return NonNegativeRegressor(pipeline)


#: Models reported in the comparison table, in presentation order.
#: Models that do not learn from the feature matrix at all (constant / naive
#: baselines). They have no preprocessor and no feature importances, so the
#: reporting code must not assume a fitted preprocessor exists.
MODELS_WITHOUT_FEATURES: frozenset[str] = frozenset({"median", "mean", "lag", "lag_drift"})

#: Models that fit a function from the features. Error analysis and the
#: feature-importance figures describe the best of these.
LEARNED_MODELS: tuple[str, ...] = tuple(
    name for name in ("ridge", "random_forest", "gradient_boosting", "lightgbm")
)


def select_reference_model(
    runs: dict[str, Any], metric: str = "mae", split: str = "val"
) -> str:
    """Pick the best *learned* model on ``split``, ignoring the baselines.

    The baselines (median/mean/lag/lag_drift) are reference points, not
    deployment candidates: a project whose best "model" is a constant has not
    produced a model. Selection reads validation only, never test, so the
    reported test score stays an honest held-out number.
    """
    candidates = {name: run for name, run in runs.items() if name in LEARNED_MODELS}
    if not candidates:
        raise ValueError("no learned model to select from")
    return min(candidates, key=lambda name: candidates[name].metrics[split][metric])

MODEL_ORDER: tuple[str, ...] = (
    "median",
    "mean",
    "lag",
    "lag_drift",
    "ridge",
    "random_forest",
    "gradient_boosting",
    "lightgbm",
)
