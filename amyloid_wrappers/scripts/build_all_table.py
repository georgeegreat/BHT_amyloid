#!/usr/bin/env python3
"""Deprecated: use ``amyloids-widemerge`` instead."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "scripts/build_all_table.py is deprecated; use: amyloids-widemerge --help",
        file=sys.stderr,
    )
    from amyloid_wrappers.cli.widemerge import main as widemerge_main

    return widemerge_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
