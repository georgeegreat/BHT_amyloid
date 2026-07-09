"""Top-level CLI: ``python -m amyloid_wrappers`` and ``amyloid-wrappers``."""

from __future__ import annotations

import argparse
import sys

from amyloid_wrappers import __version__


def build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amyloid_wrappers",
        description=(
            "Unified parsers and merge tools for amyloidogenicity predictors. "
            "Normalises per-residue outputs into a shared CSV schema."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ROOT_EPILOG,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"amyloid-wrappers {__version__}",
    )
    return parser


def print_root_help() -> None:
    build_root_parser().print_help()


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    if not argv or argv == ["-h"] or argv == ["--help"]:
        print_root_help()
        return 0

    if argv[0] in ("-h", "--help"):
        print_root_help()
        return 0

    command, *rest = argv

    if command == "parse":
        from amyloid_wrappers.cli.parse import main as parse_main

        return parse_main(rest)

    if command == "merge":
        from amyloid_wrappers.cli.merge import main as merge_main

        return merge_main(rest)

    if command == "run":
        from amyloid_wrappers.cli.run import main as run_main

        return run_main(rest)

    if command == "widemerge":
        from amyloid_wrappers.cli.widemerge import main as widemerge_main

        return widemerge_main(rest)

    if command == "batch":
        from amyloid_wrappers.cli.batch import main as batch_main

        return batch_main(rest)

    print(f"amyloid_wrappers: unknown command {command!r}", file=sys.stderr)
    print("Run 'python -m amyloid_wrappers --help' for available commands.", file=sys.stderr)
    return 2


_ROOT_EPILOG = """
commands (also available as standalone scripts):
  parse      amyloid-parse       Raw predictor file → standard per-residue CSV
  merge      amyloid-merge       Standard CSVs → wide merged table
  run        amyloid-run         Execute PATH/APPNN → standard CSV (Phase 1)
  widemerge  amyloids-widemerge  Merge + optional BHT reference validation
  batch      wrappers_run.py     Multifasta → predictors → wide CSV per protein

usage:
  python -m amyloid_wrappers --help
  python -m amyloid_wrappers parse --help
  python -m amyloid_wrappers merge --help
  python -m amyloid_wrappers run --help
  python -m amyloid_wrappers widemerge --help
  python -m amyloid_wrappers batch --help

  amyloid-run appnn --fasta protein.fasta -o APPNN.csv
  python -m amyloid_wrappers batch proteins.fasta -o results/ --batch-size 5
  amyloid-parse waltz --input protein.dat --fasta protein.fasta -o out.csv
  amyloid-merge parsed/*.csv -o merged.csv --fasta protein.fasta
  amyloids-widemerge --input-dir parsed/ -o merged.csv --fasta protein.fasta \\
      --reference ../all/RPS2_human_all.csv

configuration:
  config.cfg                 weights, thresholds, cache, runners (override via --config)
  AMYLOID_WRAPPERS_CONFIG     environment variable for config path

documentation:
  See README.md in the package root for predictor-specific input formats.
"""


if __name__ == "__main__":
    raise SystemExit(main())
