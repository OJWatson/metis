"""Execute the generated beginner notebook in a fresh temporary directory."""

from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient


def check(path):
    notebook = nbformat.read(path, as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if len(cells) < 2:
        raise ValueError("The beginner notebook must contain runnable training and reload cells")
    with tempfile.TemporaryDirectory(prefix="metis-notebook-") as directory:
        NotebookClient(
            notebook, timeout=120, kernel_name="python3",
            resources={"metadata": {"path": directory}},
        ).execute()
        if not (Path(directory) / "surrogate.pt").exists():
            raise ValueError("The beginner notebook did not save a surrogate")
        if not (Path(directory) / "evaluation.json").exists():
            raise ValueError("The beginner notebook did not save an evaluation")
    print(f"Executed {len(cells)} notebook code cells, including training and reload")


if __name__ == "__main__":
    check(sys.argv[1] if len(sys.argv) > 1 else "quarto/_site/package/quickstart.ipynb")
