"""Configuration loading for the Tokyo land price ML pipeline.

A single YAML file (``configs/config.yaml``) drives the whole pipeline so that
experiments are reproducible from the command line. Paths inside the config are
relative to the repository root and are resolved to absolute paths here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Repository root = parent of the ``src`` package directory.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "configs" / "config.yaml"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, for CLI runs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass(frozen=True)
class Config:
    """Typed accessor around the YAML config."""

    raw: dict[str, Any]
    path: Path

    # ---- convenience accessors ------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw["project"]["random_seed"])

    @property
    def encoding(self) -> str:
        return str(self.raw["data"]["encoding"])

    @property
    def skiprows(self) -> int:
        return int(self.raw["data"]["skiprows"])

    @property
    def raw_dir(self) -> Path:
        return REPO_ROOT / self.raw["data"]["raw_dir"]

    @property
    def processed_dir(self) -> Path:
        return REPO_ROOT / self.raw["data"]["processed_dir"]

    @property
    def sources(self) -> dict[str, str]:
        return dict(self.raw["data"]["sources"])

    @property
    def figures_dir(self) -> Path:
        return REPO_ROOT / "reports" / "figures"

    @property
    def reports_dir(self) -> Path:
        return REPO_ROOT / "reports"

    @property
    def split(self) -> dict[str, Any]:
        return dict(self.raw["split"])

    @property
    def features(self) -> dict[str, Any]:
        return dict(self.raw["features"])

    @property
    def leakage_columns(self) -> list[str]:
        return list(self.raw["leakage_columns"])

    @property
    def target(self) -> dict[str, Any]:
        return dict(self.raw["target"])

    @property
    def target_column(self) -> str:
        """Name of the engineered target column (JPY per square metre)."""
        return "unit_price"

    @property
    def log_target(self) -> bool:
        return bool(self.target.get("log_transform", False))

    def model_params(self, name: str) -> dict[str, Any]:
        return dict(self.raw["models"][name])

    @property
    def cv(self) -> dict[str, Any]:
        return dict(self.raw["cv"])

    @property
    def evaluation(self) -> dict[str, Any]:
        return dict(self.raw["evaluation"])

    @property
    def plots(self) -> dict[str, Any]:
        return dict(self.raw["plots"])

    def source_path(self, key: str) -> Path:
        """Absolute path of one raw CSV, resolved by its config key."""
        return self.raw_dir / self.sources[key]

    # ---- constructors ---------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        logger.info("loaded config from %s", cfg_path.relative_to(REPO_ROOT))
        return cls(raw=data, path=cfg_path)


def quiet_lightgbm() -> None:
    """Stop LightGBM's C++ layer from writing to stderr.

    LightGBM prints training banners ("Total Bins", "force_col_wise", ...)
    directly from C++ and routes them through Python's ``logging`` at INFO, which
    drowns the pipeline log. ``verbose=-1`` in the params silences the booster
    itself; this call silences the library wrapper.
    """
    try:
        import lightgbm
    except Exception:  # pragma: no cover - lightgbm is optional at import time
        logger.debug("lightgbm not available for logger registration")
        return
    quiet = logging.getLogger("lightgbm.quiet")
    quiet.handlers = [logging.NullHandler()]
    quiet.propagate = False  # keep the C++ banners out of the pipeline log
    lightgbm.register_logger(quiet)


#: Fonts that carry Japanese glyphs, in preference order. A bare matplotlib
#: install ships DejaVu Sans only, which renders every Japanese character as an
#: empty box ("tofu"), so the figures would silently lose their labels.
CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
)


def configure_japanese_font() -> str | None:
    """Register a Japanese-capable font with matplotlib, if one is installed.

    Returns the family name that was selected, or ``None`` when nothing suitable
    exists (figures then fall back to English-only labels, which the plot code
    already provides for every axis).
    """
    from matplotlib import font_manager

    for path in CJK_FONT_CANDIDATES:
        if not Path(path).is_file():
            continue
        try:
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
        except Exception as exc:  # pragma: no cover - depends on the local font
            logger.debug("could not register %s: %s", path, exc)
    logger.info(
        "no Japanese font found; figures fall back to English labels "
        "(install a CJK font, or set plots.language: en)"
    )
    return None


def configure_matplotlib(cfg: Config) -> None:
    """Apply the plotting style globally (call once per process)."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed for CLI runs
    import matplotlib.pyplot as plt

    style = cfg.plots.get("style")
    if style:
        try:
            plt.style.use(style)
        except OSError:  # pragma: no cover - depends on the mpl version
            logger.warning("matplotlib style %r unavailable, using default", style)

    family = configure_japanese_font()
    if family:
        # Prepend so Japanese glyphs resolve, while maths and Latin text keep
        # whatever the style already selected.
        plt.rcParams["font.family"] = [family, *plt.rcParams.get("font.family", ["sans-serif"])]
        plt.rcParams["axes.unicode_minus"] = False
        logger.info("matplotlib font: %s (Japanese labels enabled)", family)

