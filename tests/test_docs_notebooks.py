"""The downloadable tutorials must retain the results readers see on the site."""

import importlib.util
from pathlib import Path
import sys

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
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_BACKEND_VERSION", raising=False)
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not install locally"))
    setup.prepare_colab("a" * 40)


def test_colab_sync_uses_current_interpreter_and_keeps_unrelated_packages(monkeypatch):
    setup = load_script("colab_setup")
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test-colab-runtime")
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    setup.prepare_colab("a" * 40)
    command = calls[0][0][0]
    assert command[:2] == ["bash", "-c"]
    script = command[2]
    assert "uv sync" in script
    assert "--inexact" in script
    assert "--locked" in script
    assert "--no-editable" in script
    assert "--extra docs" in script
    assert setup.shlex.quote(sys.executable) in script
    assert "checkout -q --detach " + "a" * 40 in script
    assert calls[0][1]["env"]["UV_PROJECT_ENVIRONMENT"] == sys.prefix
    assert calls[0][1]["check"] is True


def test_colab_setup_rejects_non_revision_input(monkeypatch):
    setup = load_script("colab_setup")
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test-colab-runtime")
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("must reject before shell"))
    with pytest.raises(ValueError, match="source revision"):
        setup.prepare_colab("main; echo should-not-execute")


def test_build_rejects_unexecuted_plot_listings():
    builder = load_script("build_docs")
    for block in [
        '```python\nplt.plot(x)\n```',
        '```{python}\n#| eval: false\nplt.plot(x)\n```',
        '```{python}\n#| include: false\nplt.plot(x)\n```',
    ]:
        with pytest.raises(ValueError, match="plot-producing code"):
            builder.check_plot_cells("tutorial.qmd", block)
    builder.check_plot_cells("tutorial.qmd", '```{python}\nplt.plot(x)\n```')
    builder.check_plot_cells("reference.qmd", '```python\nmodel = Surrogate("mlp")\n```')


def test_each_captioned_figure_must_have_its_own_output(tmp_path):
    path = saved_notebook(tmp_path)
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell(
        '#| fig-cap: "Missing result"\nplt.plot(x)', execution_count=2,
    ))
    nbformat.write(notebook, path)
    with pytest.raises(ValueError, match="captioned figure missing"):
        load_script("check_tutorial_notebook").inspect(path)
