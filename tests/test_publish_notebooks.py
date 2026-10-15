"""Exercise publication against local Git only, including checkout v6 auth."""

import os
from pathlib import Path
import runpy
import subprocess


PUBLISH = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "publish_notebooks.py")
)["publish"]


def test_publish_initial_update_and_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    remote = tmp_path / "remote.git"
    checkout = tmp_path / "checkout"
    source = tmp_path / "executed"
    destination = tmp_path / "publication"
    source.mkdir()
    (source / "package").mkdir()

    def git(*args, cwd=None):
        return subprocess.check_output(
            ["git", *map(str, args)], cwd=cwd, text=True,
        ).strip()

    git("init", "--bare", remote)
    git("init", "--initial-branch=main", checkout)
    (checkout / "source.txt").write_text("Keep the source checkout unchanged.\n")
    git("add", ".", cwd=checkout)
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-m", "Source", cwd=checkout)
    revision = git("rev-parse", "HEAD", cwd=checkout)
    git("remote", "add", "origin", remote.as_uri(), cwd=checkout)
    git("push", "origin", "main", cwd=checkout)
    monkeypatch.chdir(checkout)

    # Reproduce checkout v6: a conditional include outside .git/config. No real
    # credential is needed or read. Publication must see the include in its
    # worktree without copying the value into command-line arguments.
    header_key = "http.https://github.com/.extraheader"
    dummy = "AUTHORIZATION: basic test-placeholder"
    credentials = tmp_path / "git-credentials-test.config"
    credentials.write_text(f'[http "https://github.com/"]\n\textraheader = {dummy}\n')
    for directory in (str(checkout / ".git"), str(checkout / ".git/worktrees/*")):
        git("config", f"includeIf.gitdir:{directory}.path", credentials)
    original = subprocess.check_output
    authenticated_worktree_calls = []

    def checked_output(command, **kwargs):
        assert dummy not in command
        if command[:3] == ["git", "-C", str(destination)]:
            inherited = original(
                ["git", "-C", str(destination), "config", "--get", header_key],
                text=True,
            ).strip()
            assert inherited == dummy
            authenticated_worktree_calls.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", checked_output)
    (source / "package" / "old.ipynb").write_text('{"cells": []}\n')
    PUBLISH(source, destination, revision)
    first = git("--git-dir", remote, "rev-parse", "notebooks")
    assert git("--git-dir", remote, "ls-tree", "-r", "--name-only", first).splitlines() == [
        "README.md", "package/old.ipynb",
    ]
    assert not git("--git-dir", remote, "show", "-s", "--format=%P", first)
    assert not destination.exists()

    (source / "package" / "old.ipynb").unlink()
    (source / "package" / "new.ipynb").write_text('{"cells": [], "metadata": {}}\n')
    PUBLISH(source, destination, revision)
    second = git("--git-dir", remote, "rev-parse", "notebooks")
    assert second != first
    assert git("--git-dir", remote, "show", "-s", "--format=%P", second) == first
    assert git("--git-dir", remote, "ls-tree", "-r", "--name-only", second).splitlines() == [
        "README.md", "package/new.ipynb",
    ]
    assert git("--git-dir", remote, "show", f"{second}:package/new.ipynb") == '{"cells": [], "metadata": {}}'

    PUBLISH(source, destination, revision)
    assert git("--git-dir", remote, "rev-parse", "notebooks") == second
    assert git("rev-parse", "HEAD") == revision
    assert not git("status", "--porcelain")
    assert not destination.exists()
    assert authenticated_worktree_calls
