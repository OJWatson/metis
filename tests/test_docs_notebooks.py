"""The downloadable tutorials must retain the results readers see on the site."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def saved_notebook(tmp_path, *, executed=True, figure=True, error=False):
    output = nbformat.v4.new_output("display_data", data={"image/png": "aW1hZ2U="})
    if error:
        output = nbformat.v4.new_output("error", ename="ValueError", evalue="bad", traceback=[])
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("setup()", metadata={"tags": ["metis-setup"]}),
        nbformat.v4.new_code_cell(
            "plot()", execution_count=1 if executed else None,
            outputs=[output] if figure or error else [],
        ),
    ])
    path = tmp_path / "tutorial.ipynb"
    nbformat.write(notebook, path)
    return path


def test_download_retains_executed_figures(tmp_path):
    checker = load_script("check_tutorial_notebook")
    result = checker.inspect(saved_notebook(tmp_path))
    assert result["figures"] == 1
    assert result["code_cells"] == 1


@pytest.mark.parametrize("options, message", [
    ({"executed": False}, "not executed"),
    ({"figure": False}, "displayed tutorial figure"),
    ({"error": True}, "execution error"),
])
def test_download_rejects_missing_or_failed_results(tmp_path, options, message):
    checker = load_script("check_tutorial_notebook")
    with pytest.raises(ValueError, match=message):
        checker.inspect(saved_notebook(tmp_path, **options))


def test_export_preserves_outputs_and_pins_bootstrap(tmp_path):
    builder = load_script("build_docs")
    path = saved_notebook(tmp_path)
    source = nbformat.read(path, as_version=4)
    source.cells = [
        nbformat.v4.new_markdown_cell('---\ntitle: "A tutorial"\n---'),
        nbformat.v4.new_markdown_cell("Read [setup](../basics/installing_metis.qmd)."),
        source.cells[-1],
    ]
    nbformat.write(source, path)
    destination = tmp_path / "download.ipynb"
    builder.prepare_notebook(
        path, destination, relative=Path("package/tutorial.qmd"),
        revision="a" * 40, setup_source="def prepare_colab(revision): pass",
    )
    notebook = nbformat.read(destination, as_version=4)
    assert notebook.cells[0].source == "# A tutorial"
    assert notebook.cells[-1].outputs == source.cells[-1].outputs
    assert notebook.metadata.metis.source_revision == "a" * 40
    assert "https://ojwatson.github.io/metis/basics/installing_metis.html" in notebook.cells[3].source
    assert f'prepare_colab("{"a" * 40}")' in notebook.cells[2].source
    assert notebook.cells[2].metadata.cellView == "form"


def test_export_retains_intro_and_scientific_figure_captions(tmp_path):
    builder = load_script("build_docs")
    path = saved_notebook(tmp_path)
    source = nbformat.read(path, as_version=4)
    code = source.cells[-1]
    code.source = '#| label: fig-test\n#| fig-cap: "SIRDS, held-out test groups; pointwise samples."\nplot()'
    source.cells = [
        nbformat.v4.new_markdown_cell('---\ntitle: "A tutorial"\n---\n\nEssential introduction.\n\n## The data\n\nAll 500 runs.'),
        code,
    ]
    nbformat.write(source, path)
    destination = tmp_path / "download.ipynb"
    builder.prepare_notebook(
        path, destination, relative=Path("package/tutorial.qmd"),
        revision="a" * 40, setup_source="def prepare_colab(revision): pass",
    )
    notebook = nbformat.read(destination, as_version=4)
    assert notebook.cells[0].source == "# A tutorial"
    assert notebook.cells[3].source == "Essential introduction.\n\n## The data\n\nAll 500 runs."
    assert notebook.cells[-1].source == "*SIRDS, held-out test groups; pointwise samples.*"
    assert notebook.cells[-2].outputs == code.outputs


def test_colab_link_follows_jupyter_download(tmp_path):
    builder = load_script("build_docs")
    page = tmp_path / "tutorial.html"
    page.write_text('<ul><li><a href="tutorial.ipynb" download>Jupyter</a></li></ul>')
    builder.add_colab_link(page, Path("package/tutorial.qmd"))
    content = page.read_text()
    assert content.index("Jupyter") < content.index("Open in Colab")
    assert "https://colab.research.google.com/github/OJWatson/metis/blob/notebooks/package/tutorial.ipynb" in content


def test_colab_setup_does_not_mutate_a_local_environment(monkeypatch):
    setup = load_script("colab_setup")
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(setup, "_installed_version", lambda name: "0.3.0")
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not install locally"))
    setup.prepare_colab("a" * 40)


def test_local_colab_package_alone_does_not_enable_installation(monkeypatch):
    setup = load_script("colab_setup")
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object())
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_BACKEND_VERSION", raising=False)
    monkeypatch.setattr(setup, "_installed_version", lambda name: "0.3.0")
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not install locally"))
    setup.prepare_colab("a" * 40)


def test_colab_setup_preserves_compatible_binary_versions(monkeypatch):
    setup = load_script("colab_setup")
    monkeypatch.setattr(setup, "_installed_version", lambda name: {"numpy": "2.2.4", "pandas": "2.0.0"}.get(name))
    source = setup._project_text("a" * 40)
    assert "numpy==2.2.4" in source
    assert "pandas==2.0.0" not in source
    assert "torch==2.10.0" in source
    assert "a" * 40 in source


def test_colab_setup_retains_runtime_packages_and_requires_restart_if_loaded_library_changes(tmp_path, monkeypatch):
    setup = load_script("colab_setup")
    installed = {"numpy": "1.26.4", "metis-surrogate": "0.3.0"}
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test-colab-runtime")
    monkeypatch.setattr(setup, "Path", lambda value: tmp_path)
    monkeypatch.setattr(setup, "_uv_executable", lambda directory: "uv")
    monkeypatch.setattr(setup, "_installed_version", installed.get)
    monkeypatch.setattr(setup, "_metis_origin", lambda: None)
    monkeypatch.setitem(sys.modules, "numpy", object())

    def install(command, **kwargs):
        assert "--inexact" in command
        assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == sys.prefix
        assert command[-1] == sys.executable
        installed["numpy"] = "2.2.4"
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(setup.subprocess, "run", install)
    with pytest.raises(RuntimeError, match="Restart session"):
        setup.prepare_colab("a" * 40)
