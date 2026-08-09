"""Run the full quality gate locally - the same checks CI runs.

    python scripts/check.py [--fast]

``--fast`` skips the slower end-to-end tests and the type checker.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

Check = tuple[str, list[str]]

FULL: list[Check] = [
    ("ruff (lint)", [sys.executable, "-m", "ruff", "check", "app", "tests"]),
    ("ruff (format)", [sys.executable, "-m", "ruff", "format", "--check", "app", "tests"]),
    ("mypy", [sys.executable, "-m", "mypy", "app"]),
    ("bandit", [sys.executable, "-m", "bandit", "-r", "app", "-ll", "-ii", "-q"]),
    ("pytest", [sys.executable, "-m", "pytest", "--cov=app", "--cov-report=term:skip-covered"]),
]

FAST: list[Check] = [
    ("ruff (lint)", [sys.executable, "-m", "ruff", "check", "app", "tests"]),
    ("ruff (format)", [sys.executable, "-m", "ruff", "format", "--check", "app", "tests"]),
    ("pytest (unit)", [sys.executable, "-m", "pytest", "tests/unit", "-q"]),
]


def run(name: str, command: list[str]) -> bool:
    print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, check=False)  # noqa: S603 - fixed command list
    elapsed = time.perf_counter() - started
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"--- {name}: {status} ({elapsed:.1f}s)")
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the project quality gate.")
    parser.add_argument("--fast", action="store_true", help="skip mypy, bandit and slow tests")
    args = parser.parse_args()

    checks = FAST if args.fast else FULL
    failures = [name for name, command in checks if not run(name, command)]

    print("\n" + "=" * 68)
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print(f"All {len(checks)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
