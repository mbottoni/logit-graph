#!/usr/bin/env python3
"""Execute a platform notebook's code cells (for CI / batch runs)."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

try:
    from IPython.display import display
except ImportError:
    display = print  # noqa: A001


def run_notebook(nb_path: Path, max_networks: int | None = None) -> None:
    nb = json.loads(nb_path.read_text())
    g: dict = {"display": display}
    print(f"\n{'=' * 60}\nRunning {nb_path.name}\n{'=' * 60}")

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if max_networks is not None and "for i, edge_path in enumerate(graph_files" in src:
            src = src.replace(
                "for i, edge_path in enumerate(graph_files, start=1):",
                f"for i, edge_path in enumerate(graph_files[:{max_networks}], start=1):",
            )
            print(f"  [cell {i}] loop capped at {max_networks} networks")
        else:
            print(f"  [cell {i}]")
        exec(src, g)  # noqa: S102


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path, help="Path to .ipynb")
    parser.add_argument(
        "--max-networks",
        type=int,
        default=None,
        help="Cap the fit loop (for smoke tests)",
    )
    args = parser.parse_args()
    try:
        run_notebook(args.notebook.resolve(), args.max_networks)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
