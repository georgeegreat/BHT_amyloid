"""Load package configuration from INI (.cfg) files."""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from amyloid_wrappers.paths import DEFAULT_CONFIG_PATH

METASCORE_PRESET_ENV = "AMYLOID_METASCORE_PRESET"
DEFAULT_METASCORE_PRESET = "predictor_specificity"
PRESET_SECTION_PREFIX = "metascore.presets."
PREDICTOR_SECTION_PREFIX = "predictors."
RUNNER_SECTION_PREFIX = "runners."


@dataclass
class CacheConfig:
    root: str = "cache"
    enabled: bool = False


@dataclass
class MetascoreConfig:
    method: str = "weighted_sum"
    preset: str = DEFAULT_METASCORE_PRESET
    presets: dict[str, dict[str, float]] = field(default_factory=dict)
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


def _validate_weights(name: str, weights: dict[str, float]) -> None:
    if not weights:
        raise ValueError(f"metascore preset {name!r} has no weights")
    total = sum(weights.values())
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"metascore preset {name!r} must sum to 1.0, got {total:.6f}")


def _coerce_option_value(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _section_options(parser: configparser.ConfigParser, section: str) -> dict[str, Any]:
    if section not in parser:
        return {}
    return {key: _coerce_option_value(value) for key, value in parser.items(section)}


def _parse_presets(parser: configparser.ConfigParser) -> dict[str, dict[str, float]]:
    presets: dict[str, dict[str, float]] = {}
    for section in parser.sections():
        if not section.startswith(PRESET_SECTION_PREFIX):
            continue
        preset_name = section[len(PRESET_SECTION_PREFIX) :]
        weights = {key: float(value) for key, value in _section_options(parser, section).items()}
        _validate_weights(preset_name, weights)
        presets[preset_name] = weights
    return presets


def _resolve_active_preset(
    meta_options: dict[str, Any],
    presets: dict[str, dict[str, float]],
) -> str:
    env_preset = os.environ.get(METASCORE_PRESET_ENV)
    preset = str(env_preset or meta_options.get("preset", DEFAULT_METASCORE_PRESET))

    if preset in presets:
        return preset

    if presets:
        known = ", ".join(sorted(presets))
        raise ValueError(f"Unknown metascore preset {preset!r}. Known presets: {known}")

    legacy = meta_options.get("weights")
    if isinstance(legacy, dict) and legacy:
        return preset

    raise ValueError("No metascore presets configured in config.cfg")


def _active_weights(
    meta_options: dict[str, Any],
    presets: dict[str, dict[str, float]],
    preset: str,
) -> dict[str, float]:
    if preset in presets:
        return dict(presets[preset])

    legacy = meta_options.get("weights")
    if isinstance(legacy, dict) and legacy:
        weights = {str(k): float(v) for k, v in legacy.items()}
        _validate_weights("weights", weights)
        return weights

    raise ValueError(f"Metascore preset {preset!r} not found and no legacy [metascore] weights")


def _read_config_parser(config_path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # preserve key casing
    read_ok = parser.read(config_path, encoding="utf-8")
    if not read_ok:
        raise FileNotFoundError(f"Config not found: {config_path}")
    return parser


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> AppConfig:
    """Load and validate configuration (cached by path string)."""
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    parser = _read_config_parser(config_path)

    cache_options = _section_options(parser, "cache")
    cache = CacheConfig(
        root=str(cache_options.get("root", "cache")),
        enabled=bool(cache_options.get("enabled", True)),
    )

    meta_options = _section_options(parser, "metascore")
    presets = _parse_presets(parser)
    preset = _resolve_active_preset(meta_options, presets)
    weights = _active_weights(meta_options, presets, preset)

    metascore = MetascoreConfig(
        method=str(meta_options.get("method", "weighted_sum")),
        preset=preset,
        presets=presets,
        weights=weights,
    )

    predictors: dict[str, dict[str, Any]] = {}
    for section in parser.sections():
        if section.startswith(PREDICTOR_SECTION_PREFIX):
            key = section[len(PREDICTOR_SECTION_PREFIX) :]
            predictors[key] = _section_options(parser, section)

    runners: dict[str, dict[str, Any]] = {}
    for section in parser.sections():
        if section.startswith(RUNNER_SECTION_PREFIX):
            key = section[len(RUNNER_SECTION_PREFIX) :]
            runners[key] = _section_options(parser, section)

    return AppConfig(
        cache=cache,
        metascore=metascore,
        predictors=predictors,
        runners=runners,
        source_path=config_path,
    )


def reload_config(path: str | None = None) -> AppConfig:
    """Clear cached config and load again (useful after editing config.cfg)."""
    load_config.cache_clear()
    return load_config(path)


def list_metascore_presets(config: AppConfig | None = None) -> list[str]:
    cfg = config or load_config()
    return sorted(cfg.metascore.presets)


def predictor_options(name: str, config: AppConfig | None = None) -> dict[str, Any]:
    """Return parser kwargs from config for a registry key."""
    cfg = config or load_config()
    return dict(cfg.predictors.get(name, {}))


def runner_options(name: str, config: AppConfig | None = None) -> dict[str, Any]:
    """Return runner kwargs from config for a registry key."""
    cfg = config or load_config()
    return dict(cfg.runners.get(name, {}))
