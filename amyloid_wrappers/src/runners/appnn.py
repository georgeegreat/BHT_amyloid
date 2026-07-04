"""APPNN R runner + parse pipeline."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from amyloid_wrappers.core.fasta import read_first_sequence
from amyloid_wrappers.core.schema import PredictorResult
from amyloid_wrappers.paths import DEFAULT_APPNN_SCRIPT
from amyloid_wrappers.predictors.appnn import APPNNParser
from amyloid_wrappers.runners.base import BasePredictorRunner


class APPNNRunner(BasePredictorRunner):
    """
    Run ``legacy/appnn_converter.R`` via Rscript, locate ``APPNN_parsed/*.csv``,
    then parse into standard columns.
    """

    def __init__(
        self,
        *,
        rscript: str = "Rscript",
        converter_script: str | Path | None = None,
        output_dir: str = "APPNN_parsed",
        score_threshold: float = 0.5,
        timeout_seconds: int | None = None,
    ) -> None:
        self.rscript = rscript
        script_path = converter_script or os.environ.get("AMYLOID_APPNN_SCRIPT") or DEFAULT_APPNN_SCRIPT
        self.converter_script = Path(script_path)
        self.output_dir = output_dir
        self.score_threshold = score_threshold
        self.timeout_seconds = timeout_seconds or None

    def run(
        self,
        *,
        fasta: str | Path,
        protein_id: str | None = None,
        work_dir: str | Path | None = None,
        raw_csv: str | Path | None = None,
        skip_run: bool = False,
        **kwargs,
    ) -> PredictorResult:
        fasta_path = Path(fasta)
        resolved_id, sequence = read_first_sequence(fasta_path)
        protein_id = protein_id or resolved_id

        if raw_csv is not None:
            appnn_csv = Path(raw_csv)
        elif skip_run:
            raise ValueError("Provide --input when --skip-run is set")
        else:
            appnn_csv = self._execute_appnn(fasta_path, work_dir, protein_id)

        parser = APPNNParser()
        return parser.parse(
            appnn_csv,
            protein_id=protein_id,
            sequence=sequence,
            score_threshold=self.score_threshold,
        )

    def _execute_appnn(
        self,
        fasta_path: Path,
        work_dir: str | Path | None,
        protein_id: str,
    ) -> Path:
        if not self.converter_script.is_file():
            raise FileNotFoundError(f"APPNN converter not found: {self.converter_script}")

        cwd = Path(work_dir) if work_dir is not None else fasta_path.parent
        cwd.mkdir(parents=True, exist_ok=True)

        cmd = [self.rscript, str(self.converter_script.resolve()), str(fasta_path.resolve())]
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            msg = exc.stderr or exc.stdout or str(exc)
            raise RuntimeError(f"APPNN R script failed: {msg}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"APPNN timed out after {self.timeout_seconds}s") from exc

        return self._find_appnn_csv(cwd, protein_id)

    def _find_appnn_csv(self, cwd: Path, protein_id: str) -> Path:
        out_root = cwd / self.output_dir
        if not out_root.is_dir():
            raise FileNotFoundError(
                f"APPNN output directory missing: {out_root}. "
                "Ensure R package 'appnn' is installed."
            )

        clean_id = re.sub(r"[^A-Za-z0-9_]", "_", protein_id)
        candidates = [
            out_root / f"{clean_id}_APPNN.csv",
            out_root / f"{protein_id}_APPNN.csv",
        ]
        for path in candidates:
            if path.is_file():
                return path

        matches = sorted(out_root.glob("*_APPNN.csv"))
        if len(matches) == 1:
            return matches[0]
        if matches:
            for path in matches:
                if clean_id in path.stem or protein_id in path.stem:
                    return path
            return matches[0]

        raise FileNotFoundError(f"No APPNN CSV found under {out_root}")
