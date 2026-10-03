"""Test helpers for optional dependencies.

Some models depend on native libraries that are not always installable (LightGBM
and XGBoost need the OpenMP runtime ``libomp`` on macOS). Failing the whole suite
because of a missing shared library would hide real regressions, so tests that
need those models are skipped/reduced instead, with the reason reported.
"""

from __future__ import annotations

import pytest

from src.pipeline import unavailable_models

#: Models that can be constructed and fitted in this environment.
UNAVAILABLE: dict[str, str] = unavailable_models()
AVAILABLE_MODELS: tuple[str, ...] = tuple(
    name for name in ("median", "mean", "lag", "lag_drift", "ridge",
                      "random_forest", "gradient_boosting", "lightgbm")
    if name not in UNAVAILABLE
)
#: Models that learn from the feature matrix (excludes constant/naive baselines).
AVAILABLE_FITTING_MODELS: tuple[str, ...] = tuple(
    name for name in AVAILABLE_MODELS
    if name not in ("median", "mean", "lag", "lag_drift")
)


def available(names: list[str]) -> list[str]:
    """Filter ``names`` down to what this environment can actually run."""
    return [name for name in names if name not in UNAVAILABLE]


def requires_lightgbm() -> None:
    """Skip the current test if LightGBM cannot be imported.

    LightGBM raises ``OSError`` (not ``ImportError``) when the native library is
    present but its OpenMP dependency is missing, so ``importorskip`` alone is
    not enough.
    """
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"lightgbm unavailable: {exc.__class__.__name__}: {exc}")
