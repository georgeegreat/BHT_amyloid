"""Amyloid predictor wrappers — unified parsing and aggregation."""

__version__ = "0.2.0"

from amyloid_wrappers.core.config import load_config
from amyloid_wrappers.core.merge import merge_predictor_tables, write_merge_csv
from amyloid_wrappers.core.schema import PredictorResult, get_predictor_spec

__all__ = [
    "__version__",
    "PredictorResult",
    "get_predictor_spec",
    "load_config",
    "merge_predictor_tables",
    "write_merge_csv",
]
