"""WALTZ .dat output → standard table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amyloid_wrappers.core.schema import PredictorResult, get_predictor_spec
from amyloid_wrappers.predictors.base import BasePredictorParser


class WALTZParser(BasePredictorParser):
    spec = get_predictor_spec("waltz")

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        raw = pd.read_csv(source, sep="\t", header=None, comment="#")
        raw = raw.dropna(how="all")
        if raw.empty:
            raise ValueError(f"WALTZ file is empty: {source}")

        scores = [0.0] * len(sequence)
        binary = [0] * len(sequence)

        for _, row in raw.iterrows():
            if len(row) < 2:
                continue
            pos = int(row[0])
            score = float(row[1])
            idx = pos - 1
            if not 0 <= idx < len(sequence):
                raise ValueError(f"WALTZ position out of range: {pos}")
            scores[idx] = score
            binary[idx] = 1 if score != 0 else 0

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
        )
