"""Validate saved tutorial outputs and optionally rerun a downloaded notebook."""

import argparse
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient


def inspect(path):
    """Fail if a tutorial download silently loses its executed code or figures."""
    path = Path(path)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    cells = [
        cell for cell in notebook.cells
        if cell.cell_type == "code" and "metis-setup" not in cell.metadata.get("tags", [])
    ]
    if not cells:
        raise ValueError(f"{path}: no runnable tutorial cells")
    missing = [index for index, cell in enumerate(cells, 1) if cell.execution_count is None]
    if missing:
        raise ValueError(f"{path}: tutorial cells were not executed: {missing}")
    outputs = [output for cell in cells for output in cell.outputs]
    if any(output.output_type == "error" for output in outputs):
        raise ValueError(f"{path}: saved notebook contains an execution error")
    figures = sum(
        bool(set(output.get("data", {})) & {"image/png", "image/svg+xml", "image/jpeg"})
        for output in outputs
    )
    if not figures:
        raise ValueError(f"{path}: expected at least one displayed tutorial figure")
    return {"path": str(path), "code_cells": len(cells), "figures": figures}


def check(path, *, timeout=300):
    summary = inspect(path)
    notebook = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="metis-notebook-") as directory:
        NotebookClient(
            notebook, timeout=timeout, kernel_name="python3",
            resources={"metadata": {"path": directory}},
        ).execute()
        if Path(path).stem == "saving-and-loading":
            if not (Path(directory) / "sirds_surrogate.pt").exists():
                raise ValueError("The saving tutorial did not write its surrogate artifact")
    print(f"Executed {summary['code_cells']} tutorial cells from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="quarto/_site/basics/saving-and-loading.ipynb")
    parser.add_argument("--outputs-only", action="store_true")
    args = parser.parse_args()
    target = Path(args.path)
    paths = sorted(target.rglob("*.ipynb")) if target.is_dir() else [target]
    if not paths:
        raise ValueError(f"No notebooks found under {target}")
    for path in paths:
        if args.outputs_only:
            print(inspect(path))
        else:
            check(path)
