"""``amyloid-wrappers batch`` — multifasta pipeline orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from amyloid_wrappers.batch.pipeline import run_multifasta_pipeline
from amyloid_wrappers.predictors.registry import list_parsers
from amyloid_wrappers.runners.registry import list_runners


def build_parser() -> argparse.ArgumentParser:
    available = sorted(set(list_runners()) | set(list_parsers()))
    parser = argparse.ArgumentParser(
        prog="amyloid-wrappers batch",
        description=(
            "Run amyloid predictors for every sequence in a multifasta file, "
            "parse outputs, and write one wide merged CSV per protein."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_BATCH_EPILOG,
    )
    parser.add_argument(
        "fasta",
        type=Path,
        help="Input multifasta file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output root (creates PATH/, APPNN/, merged/ subdirs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Sequences per external tool invocation for batch-capable runners "
            f"({', '.join(sorted(_batch_capable()))}); PATH always uses 1"
        ),
    )
    parser.add_argument(
        "--predictors",
        default="path,appnn",
        help=f"Comma-separated predictors (default: path,appnn). Available: {', '.join(available)}",
    )
    parser.add_argument(
        "--config",
        help="Path to config.cfg",
    )
    parser.add_argument(
        "--save-raw-files",
        type=Path,
        metavar="DIR",
        help=(
            "Optional directory to archive raw predictor outputs "
            "({protein_id}/{predictor}.*). Also used as input for parse-only tools."
        ),
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep cache/ after run (default: remove cache/)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help=(
            "Do not execute external tools; parse existing raw from "
            "{PREDICTOR}/work/ or --save-raw-files archive"
        ),
    )
    return parser


def _batch_capable() -> list[str]:
    from amyloid_wrappers.batch.pipeline import BATCH_CAPABLE_RUNNERS

    return list(BATCH_CAPABLE_RUNNERS)


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (0, None) else int(code)

    if not args.fasta.is_file():
        print(f"Error: FASTA not found: {args.fasta}", file=sys.stderr)
        return 2

    try:
        predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
        merged = run_multifasta_pipeline(
            args.fasta,
            args.output,
            batch_size=args.batch_size,
            predictors=predictors,
            config_path=args.config,
            skip_run=args.skip_run,
            save_raw_files=args.save_raw_files,
            keep_cache=args.keep_cache,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Finished {len(merged)} protein(s). Merged tables:")
    for protein_id, path in merged.items():
        print(f"  {protein_id}\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


_BATCH_EPILOG = """
examples:
  python -m amyloid_wrappers batch proteins.fasta -o results/
  python -m amyloid_wrappers batch proteins.fasta -o results/ --batch-size 5
  python -m amyloid_wrappers batch proteins.fasta -o results/ --skip-run
  python wrappers_run.py proteins.fasta -o results/ --predictors path,appnn,waltz \\
      --save-raw-files ./raw_archive/

layout:
  results/PATH/parsed/{id}_PATH.csv
  results/APPNN/parsed/{id}_APPNN.csv
  results/merged/{id}_merged.csv
  results/{PREDICTOR}/work/     temporary raw outputs (removed after successful run)

notes:
  Parsed CSVs are kept under {output}/{PREDICTOR}/parsed/.
  Per-predictor work/ dirs are removed after a successful run unless --skip-run.
  cache/ is removed after run unless --keep-cache.
  --save-raw-files copies an extra archive per protein/predictor (required for
  --skip-run after a live run that cleaned work/).
  --skip-run re-parses from {PREDICTOR}/work/ or --save-raw-files archive.
  Parse-only tools (waltz, pasta, …) read raw from --save-raw-files or {PREDICTOR}/work/.
"""
