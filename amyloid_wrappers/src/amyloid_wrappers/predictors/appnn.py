"""APPNN CSV (from appnn_converter.R) → standard table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amyloid_wrappers.core.schema import PredictorResult, get_predictor_spec
from amyloid_wrappers.predictors.base import BasePredictorParser


class APPNNParser(BasePredictorParser):
    spec = get_predictor_spec("appnn")

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        score_threshold: float = 0.5,
        **kwargs,
    ) -> PredictorResult:
        df = pd.read_csv(source)

        position_col = _pick_column(df, ("aminoacid_position", "position"))
        score_col = _pick_column(df, ("aminoacid_score", "score", "APPNN_score"))
        aa_col = _pick_column(df, ("aminoacid", "aa_name"), required=False)

        ordered = df.sort_values(position_col)

        if aa_col is not None:
            seq_from_file = "".join(str(a) for a in ordered[aa_col])
            if len(seq_from_file) != len(sequence):
                raise ValueError(
                    f"APPNN row count ({len(seq_from_file)}) "
                    f"does not match sequence length ({len(sequence)})"
                )
            if seq_from_file != sequence:
                raise ValueError("APPNN amino acids disagree with supplied sequence")
            sequence = seq_from_file

        scores = [0.0] * len(sequence)
        binary = [0] * len(sequence)

        for _, row in ordered.iterrows():
            pos = int(row[position_col])
            idx = pos - 1
            if not 0 <= idx < len(sequence):
                raise ValueError(f"APPNN position out of range: {pos}")
            score = float(row[score_col])
            scores[idx] = score
            hotspot = row.get("hotspot_region", 0)
            if pd.notna(hotspot) and int(hotspot) == 1:
                binary[idx] = 1
            elif score >= score_threshold:
                binary[idx] = 1

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            metadata={"score_threshold": score_threshold},
        )


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], *, required: bool = True) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"Expected one of {candidates}, got {list(df.columns)}")
    return None
