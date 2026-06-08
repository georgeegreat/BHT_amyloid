"""
AGGRESSOR: Aggregation-Guided Generation of REgion-Specific
Substitution ORiented mutations.

A rule-based in silico mutagenesis toolkit for protein sequences with
multi-point mutation support.

The command-line entry point is :func:`aggressor.cli.main`, exposed both
as the ``aggressor`` console script (after installation) and via
``python -m aggressor``.
"""

__version__ = "1.0.0"

from .cli import main

__all__ = ["main", "__version__"]
