"""Execute the docs once, then publish the same results as HTML and notebooks."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote, urljoin, urlsplit

import nbformat
import yaml

from check_tutorial_notebook import inspect


SITE_URL = "https://ojwatson.github.io/metis/"
COLAB_URL = "https://colab.research.google.com/github/OJWatson/metis/blob/notebooks/"
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", flags=re.DOTALL)


def frontmatter(text):
    match = FRONTMATTER.match(text)
    return (yaml.safe_load(match[1]) or {}) if match else {}


def check_plot_cells(source, text):
    """Do not publish plot-generating Python as an unexecuted code listing."""
    for match in re.finditer(r"^```(python|\{python\})\s*\n(.*?)^```", text, re.MULTILINE | re.DOTALL):
        language, code = match.groups()
        plots = re.search(r"\.(?:plot|hist|scatter|imshow|pairplot|subplots|show)\s*\(", code)
        disabled = re.search(r"^#\|\s*(?:eval|include):\s*false\b", code, re.MULTILINE)
        if plots and (language == "python" or disabled):
            line = text[:match.start()].count("\n") + 1
            raise ValueError(f"{source}:{line}: plot-producing code must be executed and displayed")


def portable_markdown(text, relative):
    """Keep doc links and static illustrations usable after a notebook download."""
    folder = "" if relative.parent == Path(".") else relative.parent.as_posix() + "/"
    base = urljoin(SITE_URL, folder)

    def absolute(raw):
        value = raw.strip("<>")
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path or value.startswith("#"):
            return raw
        value = re.sub(r"\.qmd(?=[?#]|$)", ".html", value)
        return urljoin(base, value)

    text = re.sub(r"(?<=\]\()([^\s)]+)(?=\))", lambda match: absolute(match[1]), text)
    text = re.sub(r'(?<=src=")([^\"]+)(?=")', lambda match: absolute(match[1]), text)

    def plain_image(match):
        caption, target, attributes = match.groups()
        description = re.search(r'''\bfig-alt\s*=\s*(["'])(.*?)\1''', attributes)
        alt = description[2] if description else caption
        if description:
            alt = re.sub(r"([\\\[\]])", r"\\\1", alt)
        image = f"![{alt}]({target})"
        # Quarto uses the bracket text as a visible caption; Markdown only uses it as alt text.
        return image + (f"\n\n{caption}" if caption else "")

    text = re.sub(r"!\[([^\]]*)\]\(([^)\n]+)\)\{([^}\n]*)\}", plain_image, text)

    def plain_div(match):
        attributes = match[1] or ""
        title = re.search(r'title="([^"]+)"', attributes)
        return f"**{title[1]}**" if title else ""

    # Colab supports ordinary Markdown, not Quarto's fenced div syntax.
    return re.sub(r"^\s*:{3,}\s*(?:\{([^}]*)\})?\s*$", plain_div, text, flags=re.MULTILINE)


def prepare_notebook(source, destination, *, relative, revision, setup_source):
    notebook = nbformat.read(source, as_version=4)
    cells = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cells.append(cell)
            options = "\n".join(line[3:] for line in cell.source.splitlines() if line.startswith("#| "))
            caption = (yaml.safe_load(options) or {}).get("fig-cap") if options else None
            if isinstance(caption, str) and caption.strip():
                cells.append(nbformat.v4.new_markdown_cell(f"*{caption}*"))
            continue
        if cell.cell_type != "markdown":
            cells.append(cell)
            continue
        metadata = frontmatter(cell.source)
        if metadata:
            title = metadata.get("title", relative.stem.replace("-", " ").title())
            match = FRONTMATTER.match(cell.source)
            body = cell.source[match.end():].strip()
            cell.source = f"# {title}"
            cells.append(cell)
            if body:
                cells.append(nbformat.v4.new_markdown_cell(portable_markdown(body, relative)))
        else:
            cell.source = portable_markdown(cell.source, relative)
            cells.append(cell)
    notebook.cells = cells
    introductory = nbformat.v4.new_markdown_cell(
        "**Run this notebook:** in Colab, choose **Runtime → Run all**. The first cell "
        "prepares METIS automatically using uv; a fresh runtime needs this setup again. "
        "These tutorials use a CPU and the full original SIRDS bank. Locally, select your "
        f"[METIS Python environment]({SITE_URL}basics/installing_metis.html).\n\n"
        "The outputs below were generated when the documentation was built. "
        f"[Source revision `{revision[:12]}`](https://github.com/OJWatson/metis/tree/{revision})."
    )
    setup = nbformat.v4.new_code_cell(
        "#@title Prepare METIS (run once per runtime)\n"
        + setup_source + f'\n\nprepare_colab("{revision}")\n',
        metadata={"tags": ["metis-setup"], "cellView": "form",
                  "jupyter": {"source_hidden": True}},
    )
    index = 1 if notebook.cells and notebook.cells[0].cell_type == "markdown" else 0
    notebook.cells[index:index] = [introductory, setup]
    notebook.metadata["metis"] = {"source_revision": revision, "source_page": relative.as_posix()}
    notebook.metadata["colab"] = {"name": destination.name, "provenance": []}
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    return inspect(destination)


def add_colab_link(page, relative):
    text = page.read_text(encoding="utf-8")
    address = COLAB_URL + quote(relative.with_suffix(".ipynb").as_posix(), safe="/")
    link = (
        '<li class="metis-colab"><a class="metis-colab-button" '
        f'href="{html.escape(address, quote=True)}" target="_blank" rel="noopener">'
        'Open in Colab</a></li>'
    )
    text, count = re.subn(
        r'(<li>\s*<a\b[^>]*href="[^"]+\.ipynb"[^>]*>.*?</a>\s*</li>)',
        lambda match: match[1] + link, text, count=1, flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError(f"{relative}: no Jupyter download link to attach its Colab button")
    page.write_text(text, encoding="utf-8")


def build(project, *, revision=None):
    project = Path(project).resolve()
    repository = Path(__file__).resolve().parents[1]
    revision = revision or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
    ).strip()
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Use an exact 40-character Git source revision")
    pages = [
        page.relative_to(project) for page in sorted(project.rglob("*.qmd"))
        if "_site" not in page.parts and frontmatter(page.read_text(encoding="utf-8")).get("metis-notebook")
    ]
    for page in project.rglob("*.qmd"):
        if "_site" not in page.parts:
            check_plot_cells(page.relative_to(project), page.read_text(encoding="utf-8"))
    if not pages:
        raise ValueError("No tutorial pages declare metis-notebook: true")
    setup_source = (repository / "scripts" / "colab_setup.py").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="metis-docs-") as directory:
        staged = Path(directory) / "quarto"
        shutil.copytree(project, staged, ignore=shutil.ignore_patterns(
            "_site", ".quarto", "_freeze", "*.quarto_ipynb", "__pycache__",
        ))
        configuration = staged / "_quarto.yml"
        settings = yaml.safe_load(configuration.read_text(encoding="utf-8"))
        settings.setdefault("execute", {}).update({"freeze": False, "keep-ipynb": True})
        configuration.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        environment = os.environ.copy()
        environment["QUARTO_PYTHON"] = sys.executable
        subprocess.run(
            ["quarto", "render", str(staged), "--to", "html", "--execute", "--no-cache"],
            env=environment, check=True,
        )
        site = staged / "_site"
        manifest = {"source_revision": revision, "notebooks": []}
        for relative in pages:
            source = staged / relative.with_suffix(".quarto_ipynb")
            if not source.exists():
                raise ValueError(f"{relative}: Quarto did not retain an executed notebook")
            destination = site / relative.with_suffix(".ipynb")
            summary = prepare_notebook(
                source, destination, relative=relative, revision=revision, setup_source=setup_source,
            )
            summary["path"] = destination.relative_to(site).as_posix()
            summary["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
            manifest["notebooks"].append(summary)
            add_colab_link(site / relative.with_suffix(".html"), relative)
        (site / "notebook-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        destination_site = project / "_site"
        if destination_site.exists():
            shutil.rmtree(destination_site)
        shutil.copytree(site, destination_site)
    print(f"Published {len(pages)} executed tutorial notebooks to {destination_site}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?", default="quarto")
    parser.add_argument("--revision", help="Exact source commit to install in Colab (defaults to HEAD)")
    arguments = parser.parse_args()
    build(arguments.project, revision=arguments.revision)
