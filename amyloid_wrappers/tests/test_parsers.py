"""Tests for amyloid_wrappers parsers and merge."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from amyloid_wrappers.core.merge import merge_predictor_tables
from amyloid_wrappers.core.schema import get_predictor_spec
from amyloid_wrappers.predictors.aggreprot import AggreProtParser
from amyloid_wrappers.predictors.archcandy import ArchCandyParser
from amyloid_wrappers.predictors.crossbeta import CrossBetaParser
from amyloid_wrappers.predictors.pasta import PASTAParser
from amyloid_wrappers.predictors.waltz import WALTZParser

FIXTURES = Path(__file__).parent / "fixtures"
SEQUENCE = "MADKG"  # 5 residues


@pytest.fixture
def waltz_dat(tmp_path: Path) -> Path:
    path = tmp_path / "test.dat"
    path.write_text("1\t0.0\n2\t1.5\n3\t0.0\n4\t0.0\n5\t2.0\n")
    return path


@pytest.fixture
def pasta_energy(tmp_path: Path) -> Path:
    path = tmp_path / "pasta.txt"
    path.write_text("-3\n-6\n-4\n-2\n-7\n")
    return path


@pytest.fixture
def aggreprot_csv(tmp_path: Path) -> Path:
    path = tmp_path / "aggreprot.csv"
    path.write_text(
        "header\n"
        "position,amino_acid,aggregation,sasa,transmembrane,struct_position\n"
        "1,M,0.1,1,0,1\n"
        "2,A,0.3,1,0,2\n"
        "3,D,0.1,1,0,3\n"
        "4,K,0.5,1,0,4\n"
        "5,G,0.1,1,0,5\n"
    )
    return path


@pytest.fixture
def archcandy_csv(tmp_path: Path) -> Path:
    path = tmp_path / "archcandy.csv"
    pd.DataFrame(
        {"Start": [2], "Stop": [3], "Score": [0.85]},
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def crossbeta_json(tmp_path: Path) -> Path:
    path = tmp_path / "crossbeta.json"
    payload = {
        "query": [
            {
                "AA_list": [
                    {"index": i, "amino_acid": aa, "mean_confidence": score}
                    for i, (aa, score) in enumerate(zip(SEQUENCE, [0.1, 0.6, 0.2, 0.8, 0.3]))
                ]
            }
        ]
    }
    path.write_text(json.dumps(payload))
    return path


def test_waltz_parser(waltz_dat: Path) -> None:
    result = WALTZParser().parse(waltz_dat, protein_id="t", sequence=SEQUENCE)
    assert result.scores[1] == 1.5
    assert result.binary[1] == 1
    assert result.binary[0] == 0
    df = result.to_dataframe()
    assert list(df.columns) == ["position", "aa_name", "waltz_score", "waltz_bin"]


def test_pasta_parser(pasta_energy: Path) -> None:
    result = PASTAParser().parse(pasta_energy, protein_id="t", sequence=SEQUENCE)
    assert result.binary[1] == 1  # -6 < -5
    assert result.binary[4] == 1  # -7 < -5


def test_aggreprot_parser(aggreprot_csv: Path) -> None:
    result = AggreProtParser().parse(aggreprot_csv, protein_id="t", sequence=SEQUENCE)
    assert result.binary[1] == 1  # 0.3 >= 0.25
    assert result.binary[3] == 1  # 0.5 >= 0.25


def test_archcandy_parser(archcandy_csv: Path) -> None:
    result = ArchCandyParser().parse(archcandy_csv, protein_id="t", sequence=SEQUENCE)
    assert result.scores[1] == 0.85
    assert result.binary[1] == 1
    assert result.binary[0] == 0


def test_crossbeta_parser(crossbeta_json: Path) -> None:
    result = CrossBetaParser(confidence_threshold=0.5).parse(
        crossbeta_json, protein_id="t", sequence=SEQUENCE
    )
    assert result.binary[1] == 1
    assert result.binary[3] == 1


def test_merge_wide_format(waltz_dat: Path, pasta_energy: Path) -> None:
    waltz = WALTZParser().parse(waltz_dat, protein_id="t", sequence=SEQUENCE)
    pasta = PASTAParser().parse(pasta_energy, protein_id="t", sequence=SEQUENCE)
    wide = merge_predictor_tables([waltz, pasta])
    assert list(wide.columns[:2]) == ["position", "aa_name"]
    assert wide.iloc[0]["position"] == 1
    assert wide.iloc[0]["aa_name"] == "M"
    assert "waltz_score" in wide.columns
    assert "pasta_score" in wide.columns


def test_merge_rejects_duplicate_predictor(waltz_dat: Path, pasta_energy: Path) -> None:
    waltz = WALTZParser().parse(waltz_dat, protein_id="t", sequence=SEQUENCE)
    with pytest.raises(ValueError, match="Duplicate predictor"):
        merge_predictor_tables([waltz, waltz])


def test_crossbeta_length_mismatch(crossbeta_json: Path) -> None:
    with pytest.raises(ValueError, match="does not match sequence length"):
        CrossBetaParser().parse(crossbeta_json, protein_id="t", sequence="MA")


def test_reference_bht_columns_match_registry() -> None:
    """Historical BHT all/RPS2_human_all.csv score columns match registry."""
    from amyloid_wrappers.paths import bht_reference_root

    ref_path = bht_reference_root() / "all" / "RPS2_human_all.csv"
    if not ref_path.exists():
        pytest.skip("Reference CSV not available")
    ref = pd.read_csv(ref_path, index_col=0)
    expected_score_cols = {spec.score_column for spec in map(get_predictor_spec, (
        "aggreprot", "aggrescan", "appnn", "archcandy", "crossbeta", "pasta", "path", "waltz"
    ))}
    missing = expected_score_cols - set(ref.columns)
    assert not missing, f"Reference table missing columns: {missing}"
