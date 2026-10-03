"""Re-execute the notebooks in place so their committed outputs are real.

Usage:
    python scripts/run_notebooks.py [notebook ...]

The notebooks are executed with nbclient against the same interpreter that runs
the pipeline, in a temporary kernel so that a failure cannot leave a partially
written notebook behind.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
TIMEOUT_SECONDS = 900


def run_notebook(path: Path) -> None:
    """Execute one notebook and write the executed copy back to disk."""
    # The pipeline calls matplotlib.use("Agg") for headless CLI runs. In a
    # notebook that suppresses every figure, so switch to the inline backend for
    # the kernel: it captures pyplot figures into display_data outputs.
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    print(f"executing {path.relative_to(REPO_ROOT)} ...", flush=True)
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
        allow_errors=False,
    )
    client.execute()
    # Normalise before writing: nbformat 4.5+ requires a stable cell id, and the
    # generated sources do not carry one.
    _, notebook = nbformat.validator.normalize(notebook)
    nbformat.write(notebook, path)
    print(f"  done: {path.name}", flush=True)


def main(argv: list[str]) -> int:
    targets = [NOTEBOOK_DIR / a for a in argv[1:]] or sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not targets:
        print("no notebooks found", file=sys.stderr)
        return 1
    for target in targets:
        run_notebook(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
