"""Predictor runner registry."""

from __future__ import annotations

from typing import Any

from amyloid_wrappers.core.config import load_config, runner_options
from amyloid_wrappers.core.schema import resolve_predictor_key
from amyloid_wrappers.runners.appnn import APPNNRunner
from amyloid_wrappers.runners.base import BasePredictorRunner
from amyloid_wrappers.runners.path import PATHRunner

RUNNER_REGISTRY: dict[str, type[BasePredictorRunner]] = {
    "path": PATHRunner,
    "appnn": APPNNRunner,
}


def get_runner(
    name: str,
    *,
    config_path: str | None = None,
    **overrides: Any,
) -> BasePredictorRunner:
    key = resolve_predictor_key(name)
    if key not in RUNNER_REGISTRY:
        raise KeyError(f"No runner registered for {name!r}. Known: {sorted(RUNNER_REGISTRY)}")

    cfg = load_config(config_path)
    options = runner_options(key, cfg)
    options.update(overrides)
    return RUNNER_REGISTRY[key](**options)


def list_runners() -> list[str]:
    return sorted(RUNNER_REGISTRY)
