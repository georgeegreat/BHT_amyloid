"""Tests for metascore weight presets and widemerge reference validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from amyloid_wrappers.core.config import load_config
from amyloid_wrappers.core.validate import compare_score_columns, load_reference_table
from amyloid_wrappers.paths import bht_reference_root

FIXTURES = Path(__file__).parent / "fixtures"


def test_functional_amyloid_preset_weights() -> None:
    cfg = load_config()
    weights = cfg.metascore.presets["functional_amyloids"]
    assert weights["waltz"] == pytest.approx(0.24)
    assert weights["path"] == pytest.approx(0.22)
    assert weights["appnn"] == pytest.approx(0.20)
    assert weights["crossbeta"] == pytest.approx(0.10)
    assert weights["pasta"] == pytest.approx(0.16)
    assert weights["archcandy"] == pytest.approx(0.05)
    assert weights["aggrescan"] == pytest.approx(0.03)
    assert "aggreprot" not in weights


def test_pathogenic_amyloid_preset_weights() -> None:
    weights = load_config().metascore.presets["pathogenic_amyloids"]
    assert weights["crossbeta"] == pytest.approx(0.20)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_widemerge_reference_self_compare(tmp_path: Path) -> None:
    ref_path = bht_reference_root() / "all" / "RPS2_human_all.csv"
    if not ref_path.is_file():
        pytest.skip("BHT reference CSV not available")

    ref = load_reference_table(ref_path)
    out = tmp_path / "merged.csv"
    ref.to_csv(out, index=False)

    mismatches = compare_score_columns(ref, ref)
    assert mismatches == {}


def test_runner_timeout_zero_from_config() -> None:
    cfg = load_config()
    assert cfg.runners["path"]["timeout_seconds"] == 0
    assert cfg.runners["appnn"]["timeout_seconds"] == 0
    assert cfg.cache.enabled is False


def test_guess_predictor_import() -> None:
    from amyloid_wrappers.core.inputs import guess_predictor_from_filename

    assert guess_predictor_from_filename(Path("RPS2_PATH.csv")) == "path"
    assert guess_predictor_from_filename(Path("RPL27_crossbeta.csv")) == "crossbeta"
