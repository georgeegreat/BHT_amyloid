#!/usr/bin/env python3
"""
Multifasta batch pipeline — root convenience entry point.

Equivalent to:
  python -m amyloid_wrappers batch FASTA -o OUTPUT [options]
  amyloid-wrappers batch FASTA -o OUTPUT [options]

Examples:
  python wrappers_run.py vendor/PATH/test.fasta -o output_dir
  python wrappers_run.py proteins.fasta -o output_dir --predictors path,appnn --batch-size 5
  python wrappers_run.py proteins.fasta -o output_dir --skip-run --save-raw-files ./raw/
"""

from __future__ import annotations

import sys

from amyloid_wrappers.cli.batch import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
