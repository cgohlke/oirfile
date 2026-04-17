# oirfile/tests/conftest.py

"""Pytest configuration."""

import os
import sys
from pathlib import Path

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


# Ensure local source tree is importable when running tests from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def pytest_report_header(config: object) -> str:
    """Return pytest report header."""
    try:
        import oirfile

        return (
            f'Python {sys.version.splitlines()[0]}\n'
            f'packagedir: {oirfile.__path__[0]}\n'
            f'version: oirfile {oirfile.__version__}'
        )
    except Exception as exc:
        return f'pytest_report_header failed: {exc!s}'
