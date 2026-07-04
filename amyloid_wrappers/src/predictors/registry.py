"""Predictor parser registry."""

from __future__ import annotations

from typing import Any

from amyloid_wrappers.core.config import load_config, predictor_options
from amyloid_wrappers.core.schema import resolve_predictor_key
from amyloid_wrappers.predictors.aggreprot import AggreProtParser
from amyloid_wrappers.predictors.appnn import APPNNParser
from amyloid_wrappers.predictors.archcandy import ArchCandyParser
from amyloid_wrappers.predictors.base import BasePredictorParser
from amyloid_wrappers.predictors.crossbeta import CrossBetaParser
from amyloid_wrappers.predictors.pasta import PASTAParser
from amyloid_wrappers.predictors.path import PATHParser
from amyloid_wrappers.predictors.waltz import WALTZParser

PARSER_REGISTRY: dict[str, type[BasePredictorParser]] = {
    "path": PATHParser,
    "appnn": APPNNParser,
    "waltz": WALTZParser,
    "pasta": PASTAParser,
    "aggreprot": AggreProtParser,
    "archcandy": ArchCandyParser,
    "crossbeta": CrossBetaParser,
}


def get_parser(
    name: str,
    *,
    config_path: str | None = None,
    **overrides: Any,
) -> BasePredictorParser:
    key = resolve_predictor_key(name)
    if key not in PARSER_REGISTRY:
        raise KeyError(f"No parser registered for {name!r}")

    cfg = load_config(config_path)
    options = predictor_options(key, cfg)
    options.update(overrides)
    return PARSER_REGISTRY[key](**options)


def list_parsers() -> list[str]:
    return sorted(PARSER_REGISTRY)
