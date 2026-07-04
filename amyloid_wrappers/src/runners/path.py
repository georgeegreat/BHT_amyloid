"""PATH threading runner + parse pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from amyloid_wrappers.core.fasta import read_first_sequence
from amyloid_wrappers.core.schema import PredictorResult
from amyloid_wrappers.predictors.path import PATHParser
from amyloid_wrappers.runners.base import BasePredictorRunner


class PATHRunner(BasePredictorRunner):
    """
    Run the PATH tool (``python path1.1py -f FASTA -o work_dir``) then parse
    ``results.csv`` into standard columns.

    Configure via ``[runners.path]`` in predictors.toml or env
    ``AMYLOID_PATH_SCRIPT``.
    """

    def __init__(
        self,
        *,
        script: str | None = None,
        python: str = "python3",
        results_filename: str = "results.csv",
        threshold_percentile: float = 75.0,
        timeout_seconds: int | None = None,
    ) -> None:
        self.script = script or os.environ.get("AMYLOID_PATH_SCRIPT", "")
        self.python = python
        self.results_filename = results_filename
        self.threshold_percentile = threshold_percentile
        self.timeout_seconds = timeout_seconds or None

    def run(
        self,
        *,
        fasta: str | Path,
        protein_id: str | None = None,
        work_dir: str | Path | None = None,
        results_csv: str | Path | None = None,
        skip_run: bool = False,
        **kwargs,
    ) -> PredictorResult:
        fasta_path = Path(fasta)
        resolved_id, sequence = read_first_sequence(fasta_path)
        protein_id = protein_id or resolved_id

        if results_csv is not None:
            raw_path = Path(results_csv)
        elif skip_run:
            raise ValueError("Provide --results when --skip-run is set")
        else:
            raw_path = self._execute_path(fasta_path, work_dir)

        parser = PATHParser(threshold_percentile=self.threshold_percentile)
        return parser.parse(
            raw_path,
            protein_id=protein_id,
            sequence=sequence,
        )

    def _execute_path(
        self,
        fasta_path: Path,
        work_dir: str | Path | None,
    ) -> Path:
        if not self.script:
            raise RuntimeError(
                "PATH script not configured. Set [runners.path].script in predictors.toml "
                "or AMYLOID_PATH_SCRIPT to path1.1py from https://github.com/KubaWojciechowski/PATH"
            )

        script_path = Path(self.script)
        if not script_path.is_file():
            raise FileNotFoundError(f"PATH script not found: {script_path}")

        cleanup = False
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="amyloid_path_"))
            cleanup = True
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        cmd = [self.python, str(script_path), "-f", str(fasta_path), "-o", str(work_dir)]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            msg = exc.stderr or exc.stdout or str(exc)
            raise RuntimeError(f"PATH failed: {msg}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"PATH timed out after {self.timeout_seconds}s "
                "(increase [runners.path].timeout_seconds or use --skip-run)"
            ) from exc

        results_path = self._find_results_csv(work_dir)
        if cleanup:
            # Keep results in work_dir when caller provided it; for temp dirs copy out
            dest = fasta_path.parent / f"{fasta_path.stem}_{self.results_filename}"
            shutil.copy2(results_path, dest)
            shutil.rmtree(work_dir, ignore_errors=True)
            return dest
        return results_path

    def _find_results_csv(self, work_dir: Path) -> Path:
        direct = work_dir / self.results_filename
        if direct.is_file():
            return direct

        matches = sorted(work_dir.rglob(self.results_filename))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"PATH output missing {self.results_filename!r} under {work_dir}. "
            "Check PATH installation or pass --results manually."
        )
