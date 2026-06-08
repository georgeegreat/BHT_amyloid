#!/usr/bin/env python3
"""
AGGRESSOR: Aggregation-Guided Generation of REgion-Specific Substitution ORiented mutations

Convenience launcher so the tool can still be run as a plain script from
inside the package directory:

    python main.py <input_file> [options]

It bootstraps the package onto sys.path and delegates to the package
entry point. For installed usage prefer the ``aggressor`` command or
``python -m aggressor``.

References:
- Rousseau et al., J Mol Biol 2006 (gatekeeper hypothesis)
- Beerten et al., FEBS Lett 2012 (APR boundary effects)
- Tartaglia et al., J Mol Biol 2008 (aggregation propensity scale)
"""
import logging
import os
import sys

# Make the package importable regardless of the current working directory.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PACKAGE_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from aggressor.cli import main

if __name__ == '__main__':
    # Prevent duplicate log handlers when running as script
    logging.getLogger().handlers.clear()
    main()
