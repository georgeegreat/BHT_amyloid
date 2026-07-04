"""Load package configuration from TOML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from amyloid_wrappers.paths import DEFAULT_CONFIG_PATH


@dataclass
class CacheConfig:
    root: str = "cache"
    enabled: bool = True


@dataclass
class MetascoreConfig:
    method: str = "weighted_sum"
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class AppConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    metascore: MetascoreConfig = field(default_factory=MetascoreConfig)
    predictors: dict[str, dict[str, Any]] = field(default_factory=dict)
    runners: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)


def resolve_config_path(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env_path = os.environ.get("AMYLOID_WRAPPERS_CONFIG")
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> AppConfig:
    """Load and validate configuration (cached by path string)."""
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    cache_raw = raw.get("cache", {})
    cache = CacheConfig(
        root=str(cache_raw.get("root", "cache")),
        enabled=bool(cache_raw.get("enabled", True)),
    )

    meta_raw = raw.get("metascore", {})
    weights = {str(k): float(v) for k, v in meta_raw.get("weights", {}).items()}
    if weights:
        total = sum(weights.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"metascore.weights must sum to 1.0, got {total:.6f}")

    metascore = MetascoreConfig(
        method=str(meta_raw.get("method", "weighted_sum")),
        weights=weights,
    )

    predictors: dict[str, dict[str, Any]] = {}
    if "predictors" in raw and isinstance(raw["predictors"], dict):
        predictors = {str(k): dict(v) for k, v in raw["predictors"].items()}

    runners: dict[str, dict[str, Any]] = {}
    if "runners" in raw and isinstance(raw["runners"], dict):
        runners = {str(k): dict(v) for k, v in raw["runners"].items()}

    return AppConfig(
        cache=cache,
        metascore=metascore,
        predictors=predictors,
        runners=runners,
        source_path=config_path,
    )


def predictor_options(name: str, config: AppConfig | None = None) -> dict[str, Any]:
    """Return parser kwargs from config for a registry key."""
    cfg = config or load_config()
    return dict(cfg.predictors.get(name, {}))


def runner_options(name: str, config: AppConfig | None = None) -> dict[str, Any]:
    """Return runner kwargs from config for a registry key."""
    cfg = config or load_config()
    return dict(cfg.runners.get(name, {}))
