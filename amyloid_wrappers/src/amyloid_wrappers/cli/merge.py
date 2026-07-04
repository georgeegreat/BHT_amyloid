"""``amyloid-merge`` — combine standard CSVs into a wide table."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from amyloid_wrappers.core.merge import write_merge_csv
from amyloid_wrappers.core.schema import get_predictor_spec, read_standard_csv, resolve_predictor_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amyloid-merge",
        description=(
            "Merge standard per-predictor CSV files into one wide table: "
            "position, aa_name, {predictor}_score, {predictor}_bin, ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_MERGE_EPILOG,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Standard CSV files (or directories containing them)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--protein-id",
        default="protein",
        help="Protein identifier stored in metadata",
    )
    parser.add_argument(
        "--fasta",
        help="Optional FASTA to validate / supply sequence",
    )
    parser.add_argument(
        "--predictor",
        action="append",
        dest="predictors",
        help="Predictor key for the next input (auto-detected from filename if omitted)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (0, None) else int(code)

    input_paths = _expand_inputs(args.inputs)
    if args.predictors and len(args.predictors) != len(input_paths):
        print("Error: --predictor count must match number of input files", file=sys.stderr)
        return 2

    specs = [
        get_predictor_spec(key or _guess_predictor(path))
        for key, path in zip(args.predictors or [None] * len(input_paths), input_paths, strict=True)
    ]

    sequence = None
    if args.fasta:
        from amyloid_wrappers.core.fasta import read_first_sequence

        _, sequence = read_first_sequence(args.fasta)

    results = [
        read_standard_csv(path, spec, protein_id=args.protein_id, sequence=sequence)
        for path, spec in zip(input_paths, specs, strict=True)
    ]

    output = Path(args.output)
    write_merge_csv(results, output)

    print(f"Merged {len(results)} predictors → {output}")
    return 0


def _expand_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
        else:
            paths.append(path)
    if not paths:
        raise SystemExit("No CSV inputs found")
    return paths


def _guess_predictor(path: Path) -> str:
    """Infer predictor from filename tokens (avoids false matches like *apath*)."""
    stem = path.stem.lower()
    tokens = set(re.split(r"[_\-.]+", stem))

    token_map = {
        "crossbeta": "crossbeta",
        "archcandy": "archcandy",
        "aggreprot": "aggreprot",
        "aggrescan": "aggrescan",
        "appnn": "appnn",
        "pasta": "pasta",
        "path": "path",
        "waltz": "waltz",
    }
    for token, key in token_map.items():
        if token in tokens:
            return resolve_predictor_key(key)

    if "cross-beta-predictor" in stem or "cross-beta" in stem:
        return resolve_predictor_key("crossbeta")

    raise SystemExit(f"Cannot infer predictor from filename: {path.name}. Use --predictor.")


if __name__ == "__main__":
    raise SystemExit(main())


_MERGE_EPILOG = """
examples:
  amyloid-merge RPS2_PATH.csv RPS2_APPNN.csv -o RPS2_merged.csv --fasta RPS2.fasta
  amyloid-merge parsed/*.csv -o merged.csv --fasta protein.fasta

output columns:
  position, aa_name, {predictor}_score, {predictor}_bin, ...

notes:
  Inputs must describe the same protein sequence (--fasta validates this).
  Predictor type is inferred from filenames; use --predictor to force (once per file).
"""
