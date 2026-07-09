"""CLI --help smoke tests."""

from __future__ import annotations

import subprocess
import sys


def test_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "amyloid_wrappers", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "amyloid-parse" in result.stdout
    assert "amyloid-merge" in result.stdout
    assert "amyloids-widemerge" in result.stdout
    assert "batch" in result.stdout
    assert "parse" in result.stdout and "merge" in result.stdout


def test_parse_help() -> None:
    from amyloid_wrappers.cli.parse import main

    assert main(["--help"]) == 0


def test_merge_help() -> None:
    from amyloid_wrappers.cli.merge import main

    assert main(["--help"]) == 0


def test_app_main_help() -> None:
    from amyloid_wrappers.cli.app import main

    assert main(["--help"]) == 0
    assert main([]) == 0


def test_app_dispatch_parse_help() -> None:
    from amyloid_wrappers.cli.app import main

    assert main(["parse", "--help"]) == 0


def test_app_dispatch_run_help() -> None:
    from amyloid_wrappers.cli.app import main

    assert main(["run", "--help"]) == 0


def test_widemerge_help() -> None:
    from amyloid_wrappers.cli.widemerge import main

    assert main(["--help"]) == 0


def test_app_dispatch_widemerge_help() -> None:
    from amyloid_wrappers.cli.app import main

    assert main(["widemerge", "--help"]) == 0
