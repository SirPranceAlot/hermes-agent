"""Tests for ``hermes update-fork`` — all git operations run on local bare repos."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from hermes_cli.subcommands import update_fork


def _git(repo: Path, *argv: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", "-C", str(repo), *argv],
                          capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(argv)} failed: {proc.stderr}")
    return proc


def _commit(repo: Path, filename: str, content: str, msg: str) -> None:
    (repo / filename).write_text(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", msg)


@pytest.fixture()
def upstream(tmp_path: Path) -> Path:
    """A bare upstream repo with one commit on main, plus its source worktree."""
    work = tmp_path / "upstream_work"
    bare = tmp_path / "upstream.git"
    _git(tmp_path, "init", "-b", "main", str(work))
    _git(work, "config", "user.email", "t@example.com")
    _git(work, "config", "user.name", "T")
    _commit(work, "shared.txt", "base\n", "base")
    _git(work, "clone", "--bare", str(work), str(bare))
    # the worktree needs a remote so tests can advance the bare repo from it
    _git(work, "remote", "add", "origin", str(bare))
    return bare


@pytest.fixture()
def fork(tmp_path: Path, upstream: Path) -> Path:
    """A local clone of the upstream bare repo (origin = the 'user's' remote)."""
    fork = tmp_path / "fork"
    _git(tmp_path, "clone", str(upstream), str(fork))
    _git(fork, "config", "user.email", "t@example.com")
    _git(fork, "config", "user.name", "T")
    return fork


def _args(repo: Path, *, push: bool = False, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(repo=str(repo), push=push, dry_run=dry_run)


def _backup_branches(fork: Path) -> list[str]:
    out = _git(fork, "branch", "--list", "backup-before-update-*").stdout
    return [line.strip().split()[0].lstrip("* ") for line in out.splitlines() if line.strip()]


def test_success_merge_preserves_both_sides(fork, upstream, monkeypatch, capsys):
    monkeypatch.setattr(update_fork, "UPSTREAM_URL", f"file://{upstream}")
    _commit(fork, "mine.txt", "my change\n", "local work")
    head_before = _git(fork, "rev-parse", "HEAD").stdout.strip()
    uw = upstream.parent / "upstream_work"
    _commit(uw, "theirs.txt", "their change\n", "upstream work")
    _git(uw, "push", "origin", "main")

    rc = update_fork.run_update_fork(_args(fork))
    assert rc == 0
    log = _git(fork, "log", "--oneline").stdout
    assert "local work" in log and "upstream work" in log
    backups = _backup_branches(fork)
    assert backups, "a backup branch must exist"
    assert _git(fork, "rev-parse", backups[0]).stdout.strip() == head_before
    assert (fork / "mine.txt").read_text() == "my change\n"
    assert (fork / "theirs.txt").read_text() == "their change\n"
    out = capsys.readouterr().out
    assert "merge completed" in out


def test_push_success_to_fork_origin(fork, upstream, monkeypatch):
    monkeypatch.setattr(update_fork, "UPSTREAM_URL", f"file://{upstream}")
    _commit(fork, "mine.txt", "my change\n", "local work")
    uw = upstream.parent / "upstream_work"
    _commit(uw, "theirs.txt", "their change\n", "upstream work")
    _git(uw, "push", "origin", "main")

    rc = update_fork.run_update_fork(_args(fork, push=True))
    assert rc == 0
    # the fork's origin (the local bare repo) must now contain the merged work
    bare_tip = _git(upstream, "rev-parse", "refs/heads/main").stdout.strip()
    fork_tip = _git(fork, "rev-parse", "HEAD").stdout.strip()
    assert bare_tip == fork_tip


def test_conflict_returns_1_and_keeps_recoverable_state(fork, upstream, monkeypatch):
    monkeypatch.setattr(update_fork, "UPSTREAM_URL", f"file://{upstream}")
    _commit(fork, "shared.txt", "my line\n", "local edit")
    head_before = _git(fork, "rev-parse", "HEAD").stdout.strip()
    uw = upstream.parent / "upstream_work"
    _commit(uw, "shared.txt", "their line\n", "upstream edit")
    _git(uw, "push", "origin", "main")

    rc = update_fork.run_update_fork(_args(fork))
    assert rc == 1
    git_dir = Path(_git(fork, "rev-parse", "--absolute-git-dir").stdout.strip())
    assert (git_dir / "MERGE_HEAD").exists(), "conflict must leave recoverable state"
    unmerged = _git(fork, "diff", "--name-only", "--diff-filter=U").stdout
    assert "shared.txt" in unmerged
    backups = _backup_branches(fork)
    assert backups
    assert _git(fork, "rev-parse", backups[0]).stdout.strip() == head_before
    _git(fork, "merge", "--abort")
    assert (fork / "shared.txt").read_text() == "my line\n"


def test_dirty_tree_refuses_without_changes(fork, capsys):
    (fork / "dirty.txt").write_text("uncommitted\n")
    rc = update_fork.run_update_fork(_args(fork))
    assert rc == 1
    assert "upstream" not in _git(fork, "remote").stdout  # nothing was added
    assert not _backup_branches(fork)
    assert "uncommitted changes" in capsys.readouterr().err


def test_detached_head_refuses(fork, capsys):
    sha = _git(fork, "rev-parse", "HEAD").stdout.strip()
    _git(fork, "checkout", "--quiet", sha)
    rc = update_fork.run_update_fork(_args(fork))
    assert rc == 1
    assert "detached HEAD" in capsys.readouterr().err


def test_existing_nonofficial_upstream_never_overwritten(fork, upstream, monkeypatch,
                                                        tmp_path, capsys):
    other = tmp_path / "other.git"
    _git(tmp_path, "clone", "--bare", str(upstream.parent / "upstream_work"), str(other))
    _git(fork, "remote", "add", "upstream", str(other))
    rc = update_fork.run_update_fork(_args(fork))
    assert rc == 1
    assert _git(fork, "remote", "get-url", "upstream").stdout.strip() == str(other)
    assert "refusing to overwrite" in capsys.readouterr().err


def test_existing_official_upstream_is_used(fork, upstream, monkeypatch):
    monkeypatch.setattr(update_fork, "UPSTREAM_URL", f"file://{upstream}")
    # accept the local bare path as 'official' for this test
    monkeypatch.setattr(update_fork, "_is_official",
                        lambda url: url.rstrip("/") == f"file://{upstream}".rstrip("/")
                        or url == str(upstream))
    _git(fork, "remote", "add", "upstream", f"file://{upstream}")
    _commit(fork, "mine.txt", "my change\n", "local work")
    uw = upstream.parent / "upstream_work"
    _commit(uw, "theirs.txt", "their change\n", "upstream work")
    _git(uw, "push", "origin", "main")

    rc = update_fork.run_update_fork(_args(fork))
    assert rc == 0
    assert "upstream work" in _git(fork, "log", "--oneline").stdout


def test_dry_run_changes_nothing(fork, upstream, monkeypatch, capsys):
    monkeypatch.setattr(update_fork, "UPSTREAM_URL", f"file://{upstream}")
    refs_before = _git(fork, "for-each-ref").stdout
    remotes_before = _git(fork, "remote").stdout
    rc = update_fork.run_update_fork(_args(fork, dry_run=True))
    assert rc == 0
    assert _git(fork, "for-each-ref").stdout == refs_before
    assert _git(fork, "remote").stdout == remotes_before
    out = capsys.readouterr().out
    assert "dry run" in out
    assert f"(would add) file://{upstream}" in out  # plan shows the would-be add


def test_push_refuses_official_origin_before_any_mutation(fork, upstream, monkeypatch,
                                                          capsys):
    monkeypatch.setattr(update_fork, "UPSTREAM_URL", f"file://{upstream}")
    monkeypatch.setattr(update_fork, "OFFICIAL_REPO", ("testorg", "testrepo"))
    official_url = "https://github.com/testorg/testrepo.git"
    _git(fork, "remote", "set-url", "origin", official_url)
    rc = update_fork.run_update_fork(_args(fork, push=True))
    assert rc == 1
    assert not _backup_branches(fork)
    assert "upstream" not in _git(fork, "remote").stdout
    assert "refusing to push" in capsys.readouterr().err


def test_push_refuses_all_push_urls(fork, upstream, monkeypatch, tmp_path):
    monkeypatch.setattr(update_fork, "UPSTREAM_URL", f"file://{upstream}")
    monkeypatch.setattr(update_fork, "OFFICIAL_REPO", ("testorg", "testrepo"))
    # origin stays a fork, but a configured pushurl targets the official repo
    _git(fork, "remote", "set-url", "--push", "origin",
         "https://github.com/testorg/testrepo.git")
    rc = update_fork.run_update_fork(_args(fork, push=True))
    assert rc == 1
    assert not _backup_branches(fork)


def test_discovery_failure_fails_cleanly(fork, upstream, monkeypatch, capsys):
    monkeypatch.setattr(update_fork, "UPSTREAM_URL", f"file://{upstream}")
    monkeypatch.setattr(update_fork, "_discover_upstream_branch", lambda repo: "")
    rc = update_fork.run_update_fork(_args(fork))
    assert rc == 1
    err = capsys.readouterr().err
    assert "could not discover" in err
    assert "backup-before-update-" in err  # message must reflect the created backup
    assert _backup_branches(fork), "backup is created before discovery"


def test_url_normalization():
    n = update_fork._normalize_github
    assert n("https://github.com/NousResearch/hermes-agent.git") == "nousresearch/hermes-agent"
    assert n("git@github.com:NousResearch/hermes-agent") == "nousresearch/hermes-agent"
    assert n("https://github.com/nousresearch/hermes-agent/") == "nousresearch/hermes-agent"
    assert n("file:///tmp/upstream.git") is None
    assert n("https://gitlab.com/foo/bar") is None
    assert update_fork._is_official("HTTPS://GitHub.com/NousResearch/Hermes-Agent.GIT")
    assert not update_fork._is_official("https://github.com/testorg/testrepo.git")


def test_subcommand_registers():
    sentinel = object()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    update_fork.build_update_fork_parser(sub, cmd_update_fork=lambda a: sentinel)
    args = parser.parse_args(["update-fork", "--dry-run"])
    assert args.dry_run is True
    assert args.push is False
    assert args.repo is None
    assert args.func(None) is sentinel
