"""Hydra multirun sweeper: turn a grid defined in a small YAML file into a
`python scripts/evaluate.py -m ...` invocation using Hydra's built-in basic
sweeper, rather than reimplementing grid iteration.

Why a wrapper instead of typing the multirun command by hand: the grid and
its fixed overrides live in one reviewable file (config/sweep/*.yaml,
version-controlled, diffable in a PR) instead of a shell one-liner someone
has to reconstruct from a paper's hyperparameter table.

Usage:
    python scripts/sweep.py --sweep-config config/sweep/locomo_grid.yaml
    python scripts/sweep.py --sweep-config config/sweep/locomo_grid.yaml --dry-run
"""

from __future__ import annotations

from iterret.utils.logger import get_logger

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Hydra multirun sweep for IterRet evaluate.py."
    )
    parser.add_argument(
        "--sweep-config",
        required=True,
        help="Path to a sweep YAML (see config/sweep/locomo_grid.yaml for the schema).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resulting Hydra multirun command without executing it.",
    )
    return parser.parse_args()


def build_overrides(sweep_config: dict) -> list[str]:
    """Turn `grid: {key: [v1, v2, ...]}` into Hydra multirun overrides
    (`key=v1,v2,...` with `-m`) plus flat `fixed_overrides` entries."""
    grid = sweep_config.get("grid", {})
    if not grid:
        raise ValueError("Sweep config has no 'grid' section -- nothing to sweep over.")

    overrides = [f"{key}={','.join(str(v) for v in values)}" for key, values in grid.items()]
    for key, value in (sweep_config.get("fixed_overrides") or {}).items():
        overrides.append(f"{key}={value}")
    return overrides


def main() -> None:
    args = parse_args()
    sweep_path = Path(args.sweep_config)
    if not sweep_path.is_absolute():
        sweep_path = _REPO_ROOT / sweep_path
    with sweep_path.open("r", encoding="utf-8") as fh:
        sweep_config = yaml.safe_load(fh)

    overrides = build_overrides(sweep_config)
    command = [sys.executable, str(_REPO_ROOT / "scripts" / "evaluate.py"), "-m", *overrides]

    print("[sweep] running:", " ".join(command))
    if args.dry_run:
        return

    result = subprocess.run(command, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
