"""``hermes update-fork`` subcommand — safe sync of a fork's checkout with upstream.

Updates a **git checkout** of a fork (e.g. SirPranceAlot/hermes-agent) with the latest
upstream changes while preserving local work:

1. Refuses to run on a dirty checkout, a detached HEAD, or an in-progress merge,
   rebase, cherry-pick, revert, or sequencer operation.
2. Ensures the ``upstream`` remote points at the official repository (adds it only when
   absent and only outside dry-run; never overwrites an existing remote).
3. Validates any ``--push`` target **before** mutating the repository.
4. Creates a timestamped backup branch at the current HEAD before touching anything.
   A backup preserves tracked, committed history only — not ignored files or secrets.
5. Discovers the upstream default branch via ``git ls-remote --symref`` (failing clearly
   rather than assuming ``main``), fetches exactly that branch with an explicit refspec,
   and merges the fetched commit ID into the current branch.
6. On a merge conflict, stops with a nonzero exit, preserves the backup, and explains
   resolution; any other git failure is reported as-is.

This never runs ``git reset --hard``, never discards changes, and never force-pushes.
Note: this updates the checkout only — it does not update a managed installation
(``~/.hermes/installs/...`` stays untouched until rebuilt/reinstalled).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

OFFICIAL_REPO = ("nousresearch", "hermes-agent")  # compared case-insensitively
UPSTREAM_URL = "https://github.com/NousResearch/hermes-agent.git"

_GITHUB_PREFIXES = ("https://github.com/", "git@github.com:", "ssh://git@github.com/")


class ForkUpdateError(RuntimeError):
    """User-facing failure of ``update-fork``."""


def _git(repo: Path, *argv: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in ``repo`` using an argument list (no shell interpolation)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *argv],
            capture_output=True, text=True,
        )
    except OSError as exc:
        raise ForkUpdateError(f"failed to run git: {exc}") from exc
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise ForkUpdateError(f"git {' '.join(argv)} failed: {detail}")
    return proc


def _rev_parse(repo: Path, rev: str) -> str | None:
    """Return the commit id for ``rev``, or None if it does not resolve."""
    proc = _git(repo, "rev-parse", "--verify", "--quiet", rev, check=False)
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def _resolve_repo(explicit: str | None) -> Path:
    """Resolve the checkout: explicit --repo path (with ~ expansion), else the cwd."""
    repo = Path(explicit).expanduser().resolve() if explicit else Path.cwd().resolve()
    probe = repo
    while probe != probe.parent:
        if (probe / ".git").exists():
            return probe
        probe = probe.parent
    raise ForkUpdateError(f"{repo} is not inside a git checkout")


def _current_branch(repo: Path) -> str:
    branch = _git(repo, "branch", "--show-current").stdout.strip()
    if not branch:
        raise ForkUpdateError("not on a branch (detached HEAD) — refusing to update")
    return branch


def _assert_safe(repo: Path) -> None:
    """Refuse dirty checkouts and in-progress merge/rebase/cherry-pick/revert state."""
    status = _git(repo, "status", "--porcelain", check=False)
    if status.returncode != 0:
        raise ForkUpdateError(
            "could not read the working-tree status: "
            + (status.stderr.strip() or "unknown git error")
        )
    if status.stdout.strip():
        raise ForkUpdateError(
            "working tree has uncommitted changes — commit or stash them first "
            "(never update a fork on top of uncommitted work)"
        )
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir").stdout.strip())
    for marker in ("MERGE_HEAD", "REBASE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD",
                   "rebase-merge", "rebase-apply", "sequencer"):
        if (git_dir / marker).exists():
            raise ForkUpdateError(
                "a merge, rebase, cherry-pick, or revert is already in progress — "
                "resolve it before updating the fork"
            )


def _upstream_url(repo: Path) -> str | None:
    """Return the upstream remote URL, or None when the remote does not exist."""
    proc = _git(repo, "remote", "get-url", "upstream", check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _normalize_github(url: str) -> str | None:
    """Reduce an https/ssh GitHub URL to the lowercased owner/repo pair.

    Returns None for anything that is not a recognized GitHub URL.
    """
    cleaned = url.strip()
    if cleaned.endswith("/") and "://" not in cleaned and "@" not in cleaned:
        cleaned = cleaned.rstrip("/")
    for prefix in _GITHUB_PREFIXES:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
            if cleaned.lower().endswith(".git"):
                cleaned = cleaned[:-4]
            return cleaned.strip("/").lower()
    return None


def _is_official(url: str) -> bool:
    pair = _normalize_github(url)
    return pair is not None and pair == "/".join(OFFICIAL_REPO)


def _ensure_upstream(repo: Path, *, dry_run: bool) -> tuple[str, bool]:
    """Ensure an ``upstream`` remote points at the official repo.

    Returns (url_or_description, was_added). Adds the remote only outside dry-run
    and only when absent; never modifies an existing remote. Prints the addition
    itself so the caller cannot report it twice.
    """
    existing = _upstream_url(repo)
    if existing is not None:
        if not _is_official(existing):
            raise ForkUpdateError(
                f"the 'upstream' remote points at {existing!r}, not the official "
                "repository — refusing to overwrite it; fix it manually if intended"
            )
        return existing, False
    if dry_run:
        return "(would add) " + UPSTREAM_URL, False
    _git(repo, "remote", "add", "upstream", UPSTREAM_URL)
    print(f"added upstream remote -> {UPSTREAM_URL}")
    return UPSTREAM_URL, True


def _discover_upstream_branch(repo: Path) -> str:
    """Discover the upstream default branch via ``git ls-remote --symref`` (no fetch).

    Returns "" when discovery fails — callers must fail rather than guess ``main``.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "ls-remote", "--symref", "upstream", "HEAD"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if line.startswith("ref:"):
                # e.g. "ref: refs/heads/main\tHEAD"
                ref = line.split("\t", 1)[0].split()[-1]
                if ref.startswith("refs/heads/"):
                    return ref[len("refs/heads/"):]
    return ""


def _fetch_upstream_head(repo: Path, branch: str) -> str:
    """Fetch exactly ``refs/heads/<branch>`` and return the fetched commit ID."""
    _git(repo, "fetch", "--no-tags", "upstream",
         f"refs/heads/{branch}:refs/remotes/upstream/{branch}")
    commit = _rev_parse(repo, f"refs/remotes/upstream/{branch}")
    if not commit:
        raise ForkUpdateError(
            f"fetched upstream but refs/remotes/upstream/{branch} did not resolve"
        )
    return commit


def _unique_backup_name(repo: Path, base: str) -> str:
    """Return a backup branch name that does not already exist (timestamp + counter)."""
    name = base
    counter = 1
    while _rev_parse(repo, name):
        counter += 1
        name = f"{base}_{counter}"
    return name


def _assert_push_safe(repo: Path) -> None:
    """--push must target the user's fork (origin), never the official repository.

    Checks every push URL configured for origin, not just the first.
    """
    proc = _git(repo, "remote", "get-url", "--push", "--all", "origin", check=False)
    if proc.returncode != 0:
        raise ForkUpdateError("no 'origin' remote — cannot --push")
    urls = [u.strip() for u in proc.stdout.splitlines() if u.strip()]
    official = [u for u in urls if _is_official(u)]
    if official:
        raise ForkUpdateError(
            "origin push URL points at the official repository "
            f"({official[0]}) — refusing to push to it"
        )


def run_update_fork(args: argparse.Namespace) -> int:
    """Handler for ``hermes update-fork``. Returns a process exit code."""
    try:
        repo = _resolve_repo(args.repo)
        branch = _current_branch(repo)
        _assert_safe(repo)
        if args.push:
            _assert_push_safe(repo)
        head = _git(repo, "rev-parse", "HEAD").stdout.strip()
        upstream_desc, _ = _ensure_upstream(repo, dry_run=args.dry_run)
        backup = _unique_backup_name(
            repo, f"backup-before-update-{time.strftime('%Y%m%d_%H%M%S')}"
        )

        if args.dry_run:
            print(f"repo:     {repo}")
            print(f"branch:   {branch}")
            print(f"backup:   {backup}  (at {head[:12]})")
            print(f"upstream: {upstream_desc}")
            print(f"git branch {backup} {head[:12]}")
            print(f"git ls-remote --symref upstream HEAD   # find the default branch")
            print(f"git fetch --no-tags upstream refs/heads/<branch>:refs/remotes/upstream/<branch>")
            print("git merge --no-edit --no-autostash <fetched commit>")
            if args.push:
                print(f"git push origin refs/heads/{branch}:refs/heads/{branch}")
            print("(dry run — nothing was changed)")
            return 0

        _git(repo, "branch", backup, head)
        print(f"backup branch created: {backup}")
        upstream_branch = _discover_upstream_branch(repo)
        if not upstream_branch:
            raise ForkUpdateError(
                "could not discover the upstream default branch "
                "(git ls-remote --symref upstream HEAD failed) — check your network "
                "and the upstream remote; no commits or working files were changed. "
                f"A backup branch '{backup}' was created and the 'upstream' remote "
                "may have been added."
            )
        try:
            fetched = _fetch_upstream_head(repo, upstream_branch)
        except ForkUpdateError:
            print(
                f"\nThe fetch failed. Your pre-merge state is untouched; the backup "
                f"branch '{backup}' is already in place. No merge was attempted.",
                file=sys.stderr,
            )
            raise
        print(f"merging upstream/{upstream_branch} ({fetched[:12]}) into {branch} ...")
        merge = _git(repo, "merge", "--no-edit", "--no-autostash", fetched, check=False)
        if merge.returncode != 0:
            unmerged = _git(
                repo, "diff", "--name-only", "--diff-filter=U", check=False
            ).stdout
            git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir").stdout.strip())
            conflicted = bool(unmerged.strip()) or (git_dir / "MERGE_HEAD").exists()
            print(merge.stdout + merge.stderr, file=sys.stderr)
            if conflicted:
                print("\nConflicted files:", file=sys.stderr)
                print(unmerged, file=sys.stderr)
                print(
                    "\nResolve the conflicts, then 'git add' them and 'git commit'.\n"
                    f"Your pre-merge state is safe on branch '{backup}'.\n"
                    "To cancel instead: git merge --abort",
                    file=sys.stderr,
                )
            else:
                print(
                    "\nThe merge failed for a non-conflict reason (see the git output "
                    "above). Check 'git status' to see whether any merge state was "
                    f"left behind; your pre-merge commit is preserved on '{backup}'.",
                    file=sys.stderr,
                )
            return 1

        merged = _git(repo, "log", "--oneline", f"{head}..HEAD").stdout.splitlines()
        print("merge completed:")
        for line in merged[:10]:
            print(f"  {line}")
        if len(merged) > 10:
            print(f"  ... and {len(merged) - 10} more")
        print(f"\n{branch} is now {len(merged)} commit(s) ahead of the pre-merge HEAD")
        print("(this count includes the merge commit itself).")
        print(f"Pre-merge state preserved on '{backup}' — delete it once verified.")
        print("Reminder: this updated the git checkout only; a managed installation")
        print("(~/.hermes/installs/...) is unchanged until it is rebuilt/reinstalled.")
        if args.push:
            _git(repo, "push", "origin", f"refs/heads/{branch}:refs/heads/{branch}")
            print(f"pushed {branch} to origin")
    except ForkUpdateError as exc:
        print(f"update-fork: {exc}", file=sys.stderr)
        return 1
    return 0


def build_update_fork_parser(subparsers, *, cmd_update_fork) -> None:
    """Attach the ``update-fork`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "update-fork",
        help="Update a fork checkout with the latest upstream changes",
        description="Sync a git checkout of a Hermes fork with the official upstream "
            "repository, preserving local committed work. Creates a timestamped backup "
            "branch, fetches upstream, and merges the upstream default branch into the "
            "current branch. Refuses to run on a dirty checkout or detached HEAD.",
        epilog="Examples:\n"
            "  hermes update-fork                    update the checkout you are in\n"
            "  hermes update-fork --repo ~/code/hermes-agent --dry-run\n"
            "  hermes update-fork --push             also push the result to origin\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=None, metavar="PATH",
                        help="Path to the fork checkout (default: current directory)")
    parser.add_argument("--push", action="store_true",
                        help="After a successful merge, push the current branch to origin (no force)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned git operations without running them")
    parser.set_defaults(func=cmd_update_fork)
