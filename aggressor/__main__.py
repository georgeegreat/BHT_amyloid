"""
Module entry point so the package can be run with ``python -m aggressor``.
"""
import logging

from .cli import main

if __name__ == '__main__':
    # Prevent duplicate log handlers on repeated runs / auto-reload
    logging.getLogger().handlers.clear()
    main()
