#!/usr/bin/env python3
"""CI gate: ensure docs/index.html version matches pyproject.toml.

Exits 0 if synced, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
DOCS_INDEX = REPO_ROOT / "docs" / "index.html"

VERSION_PATTERN = re.compile(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', re.MULTILINE)
DOCS_VERSION_PATTERN = re.compile(r'"softwareVersion"\s*:\s*"(\d+\.\d+\.\d+)"')


def main() -> int:
    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    match_py = VERSION_PATTERN.search(pyproject_text)
    if not match_py:
        print(f"ERROR: No version found in {PYPROJECT}", file=sys.stderr)
        return 1

    docs_text = DOCS_INDEX.read_text(encoding="utf-8")
    match_docs = DOCS_VERSION_PATTERN.search(docs_text)
    if not match_docs:
        print(f"ERROR: No softwareVersion found in {DOCS_INDEX}", file=sys.stderr)
        return 1

    py_version = match_py.group(1)
    docs_version = match_docs.group(1)

    if py_version != docs_version:
        print(
            f"ERROR: Version mismatch! "
            f"pyproject.toml={py_version}, docs/index.html={docs_version}",
            file=sys.stderr,
        )
        print("Run: python scripts/bump_version.py patch", file=sys.stderr)
        return 1

    print(f"OK: versions synced at {py_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
