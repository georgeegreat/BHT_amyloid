"""``amyloid-run`` — execute PATH/APPNN and write standard CSV."""

from __future__ import annotations

import argparse
import sys

from amyloid_wrappers.core.cache import store_raw_cache
from amyloid_wrappers.core.config import load_config
from amyloid_wrappers.runners.registry import get_runner, list_runners


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amyloid-run",
        description=(
            "Run external amyloid predictors (PATH, APPNN) and write standard "
            "per-residue CSV (position, aa_name, {Tool}_score, {Tool}_bin)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_RUN_EPILOG,
    )
    parser.add_argument(
        "predictor",
        choices=list_runners(),
        metavar="PREDICTOR",
        help="predictor to run (%(choices)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output standard CSV path",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA (single protein recommended for Phase 1)",
    )
    parser.add_argument(
        "--config",
        help="Path to predictors.toml",
    )
    parser.add_argument(
        "--protein-id",
        help="Protein identifier (default: first FASTA header)",
    )
    parser.add_argument(
        "--work-dir",
        help="Working directory for tool output (default: FASTA parent dir for APPNN)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Parse existing raw output only (requires --results or --input)",
    )
    parser.add_argument(
        "--results",
        help="PATH results.csv (with --skip-run for path)",
    )
    parser.add_argument(
        "--input",
        dest="raw_input",
        help="APPNN per-protein CSV (with --skip-run for appnn)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not copy raw tool output to cache/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override parser threshold for this run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (0, None) else int(code)

    cfg = load_config(args.config)
    runner_kwargs = _runner_kwargs(args)
    runner = get_runner(args.predictor, config_path=args.config, **runner_kwargs)

    run_kwargs: dict = {"fasta": args.fasta, "skip_run": args.skip_run}
    if args.protein_id:
        run_kwargs["protein_id"] = args.protein_id
    if args.work_dir:
        run_kwargs["work_dir"] = args.work_dir
    if args.predictor == "path" and args.results:
        run_kwargs["results_csv"] = args.results
    if args.predictor == "appnn" and args.raw_input:
        run_kwargs["raw_csv"] = args.raw_input

    result = runner.run(**run_kwargs)
    result.to_csv(args.output)

    if not args.no_cache:
        raw_source = args.results or args.raw_input
        if raw_source:
            cached = store_raw_cache(
                result.protein_id,
                args.predictor,
                raw_source,
                config=cfg,
            )
            if cached:
                print(f"Cached raw input → {cached}")

    print(f"Wrote {args.output} ({result.length} residues, {result.spec.display_name})")
    return 0


def _runner_kwargs(args: argparse.Namespace) -> dict:
    if args.threshold is None:
        return {}

    mapping = {
        "path": {"threshold_percentile": args.threshold},
        "appnn": {"score_threshold": args.threshold},
    }
    return mapping.get(args.predictor, {})


if __name__ == "__main__":
    raise SystemExit(main())


_RUN_EPILOG = """
examples:
  amyloid-run appnn --fasta RPS2.fasta -o RPS2_APPNN.csv
  amyloid-run path --fasta RPS2.fasta -o RPS2_PATH.csv --work-dir ./path_out

  # Parse precomputed raw files (no external tool execution):
  amyloid-run path --skip-run --results results.csv --fasta RPS2.fasta -o out.csv
  amyloid-run appnn --skip-run --input APPNN_parsed/RPS2_APPNN.csv --fasta RPS2.fasta -o out.csv

configuration ([runners.path] / [runners.appnn] in predictors.toml):
  path.script          path to path1.1py (or AMYLOID_PATH_SCRIPT)
  appnn.converter_script   defaults to package legacy/appnn_converter.R

notes:
  PATH threading is slow — configure timeout_seconds or use --skip-run for tests.
  APPNN requires R with the 'appnn' CRAN/Bioconductor package installed.
"""
