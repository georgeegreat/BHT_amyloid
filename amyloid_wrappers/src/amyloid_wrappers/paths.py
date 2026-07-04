"""Package root paths (config, legacy scripts)."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PACKAGE_ROOT / "legacy"
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "predictors.toml"
DEFAULT_APPNN_SCRIPT = LEGACY_ROOT / "appnn_converter.R"
