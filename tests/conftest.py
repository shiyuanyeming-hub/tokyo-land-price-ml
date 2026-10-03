"""Shared pytest fixtures: real data, cleaned once per session."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import cleaning, features  # noqa: E402
from src.config import Config  # noqa: E402
from src.data_loader import load_all_raw  # noqa: E402

logging.getLogger("src").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The project configuration, as shipped."""
    return Config.load()


@pytest.fixture(scope="session")
def raw(cfg: Config):
    """Every raw source, exactly as published (all columns are strings)."""
    return load_all_raw(cfg)


@pytest.fixture(scope="session")
def cleaned(cfg: Config, raw):
    """Cleaned frames keyed by source name, plus their quality reports."""
    frames, reports = {}, {}
    for name, frame in raw.items():
        frame_clean, report = cleaning.clean_frame(frame, source=name)
        built = cleaning.add_unit_price(features.build_features(frame_clean))
        # The panel key is part of every usable frame: splits, joins and the
        # leakage tests all rely on it being present.
        built["_site_key"] = features.site_key(built)
        frames[name] = built
        reports[name] = report
    return frames, reports


@pytest.fixture(scope="session")
def kouji_pair(cleaned):
    """The 地価公示 令和7年 / 令和8年 pair: the project's main panel."""
    frames, _ = cleaned
    return frames["kouji_r7"], frames["kouji_r8"]


@pytest.fixture(scope="session")
def kijun_pair(cleaned):
    """The 基準地価 令和6年 / 令和7年 pair: the secondary series."""
    frames, _ = cleaned
    return frames["kijun_r6"], frames["kijun_r7"]


@pytest.fixture(scope="session")
def small_panel(cfg: Config, kouji_pair):
    """A small matched train/val/test triple, to keep the model tests fast."""
    from src import evaluate, features as feat

    train, predict = kouji_pair
    train_small, _ = feat.attach_lag_unit_price(train.head(500).reset_index(drop=True), predict)
    predict_small, _ = feat.attach_lag_unit_price(
        predict.head(500).reset_index(drop=True), train
    )
    val, test = evaluate.temporal_split(predict_small, cfg)
    return train_small, val, test
