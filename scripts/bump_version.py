#!/usr/bin/env python3
"""Bump version across the project atomically.

Reads pyproject.toml, increments patch/minor/major, and syncs docs/index.html.
Usage:
    python scripts/bump_version.py patch   # 0.1.10 -> 0.1.11
    python scripts/bump_version.py minor   # 0.1.10 -> 0.2.0
    python scripts/bump_version.py major   # 0.1.10 -> 1.0.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
DOCS_INDEX = REPO_ROOT / "docs" / "index.html"

VERSION_PATTERN = re.compile(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', re.MULTILINE)
PLACEHOLDER = "{{VERSION}}"


def read_current_version() -> tuple[int, int, int]:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(text)
    if not match:
        print(f"Could not find version in {PYPROJECT}", file=sys.stderr)
        sys.exit(1)
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def bump(current: tuple[int, int, int], part: str) -> tuple[int, int, int]:
    major, minor, patch = current
    if part == "major":
        return major + 1, 0, 0
    if part == "minor":
        return major, minor + 1, 0
    return major, minor, patch + 1


def write_version(major: int, minor: int, patch: int) -> str:
    new_version = f"{major}.{minor}.{patch}"

    # Update pyproject.toml
    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    new_pyproject, count = VERSION_PATTERN.subn(
        f'version = "{new_version}"', pyproject_text, count=1
    )
    if count != 1:
        print("Failed to substitute version in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    PYPROJECT.write_text(new_pyproject, encoding="utf-8")

    # Update docs/index.html — replace placeholder or hard-coded version
    docs_text = DOCS_INDEX.read_text(encoding="utf-8")
    if PLACEHOLDER in docs_text:
        new_docs = docs_text.replace(PLACEHOLDER, new_version)
    else:
        # Fallback: replace any hard-coded X.Y.Z in softwareVersion line
        docs_version_pattern = re.compile(
            r'("softwareVersion"\s*:\s*")\d+\.\d+\.\d+(")'
        )
        new_docs, count = docs_version_pattern.subn(
            f'\\g<1>{new_version}\\g<2>', docs_text
        )
        if count != 1:
            print("Failed to substitute version in docs/index.html", file=sys.stderr)
            sys.exit(1)
    DOCS_INDEX.write_text(new_docs, encoding="utf-8")

    return new_version


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump project version")
    parser.add_argument(
        "part",
        choices=["patch", "minor", "major"],
        help="Which part of the version to increment",
    )
    args = parser.parse_args()

    current = read_current_version()
    new = bump(current, args.part)
    new_version = write_version(*new)

    print(f"Bumped {'.'.join(map(str, current))} -> {new_version}")
    print(f"  Updated {PYPROJECT}")
    print(f"  Updated {DOCS_INDEX}")


if __name__ == "__main__":
    main()
