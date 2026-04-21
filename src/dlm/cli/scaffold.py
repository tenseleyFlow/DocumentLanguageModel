"""`dlm train <dir>` auto-scaffold — zero-ceremony directory training.

When the CLI arg points at a directory rather than a `.dlm` file,
`scaffold_train_target` resolves or creates the `.dlm` file that will
drive the train:

1. Look for `<dir>/.dlm/*.dlm`. Exactly one match → reuse it.
   Multiple matches + no `--name` → refuse with the candidates listed.
   `--name <n>` narrows to `<dir>/.dlm/<n>.dlm`.
2. No match → require `--base` (no silent default), mint a fresh
   ULID, write `<dir>/.dlm/corpus.dlm` (or `<name>.dlm`) with
   `training.sources` built from the CLI flags, and return its
   path. The caller then proceeds as if the `.dlm` was passed
   directly.

`--rescaffold` rewrites an existing scaffolded `.dlm` in place (same
ULID kept so the store stays intact). Without it, re-running with
frontmatter-editing flags refuses to shadow-edit the on-disk config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dlm.doc.errors import DlmParseError
from dlm.doc.parser import parse_file
from dlm.doc.schema import CURRENT_SCHEMA_VERSION
from dlm.io.atomic import write_text as atomic_write_text
from dlm.io.ulid import mint_ulid

_LOG = logging.getLogger(__name__)

_SCAFFOLD_DIR = ".dlm"
_DEFAULT_NAME = "corpus"


Policy = Literal["permissive", "strict"]


class ScaffoldError(DlmParseError):
    """Scaffold-mode failure surfaced through the CLI reporter.

    Subclass of DlmParseError so it gets the uniform `file:line:col`
    treatment even though scaffold errors don't have line/col info.
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message, path=path)


@dataclass(frozen=True)
class ScaffoldResult:
    """What the CLI command reports after resolving or creating."""

    dlm_path: Path
    scaffolded: bool
    """True when we wrote a new file on this invocation (first run
    or --rescaffold). False when we reused an existing one."""
    dlm_id: str


def scaffold_train_target(
    target: Path,
    *,
    base: str | None,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    recursive: bool,
    name: str,
    policy: Policy,
    rescaffold: bool = False,
) -> ScaffoldResult:
    """Resolve or scaffold a `.dlm` file anchoring `target` (a directory).

    Returns a `ScaffoldResult` pointing at the `.dlm` the trainer
    should consume. Raises `ScaffoldError` when scaffolding is needed
    but required inputs (most commonly `--base`) are missing, or when
    an ambiguous multi-`.dlm` tree was passed without `--name`.
    """
    if not target.exists():
        raise ScaffoldError(f"target does not exist: {target}", path=target)
    if not target.is_dir():
        raise ScaffoldError(
            f"scaffold expects a directory, got file: {target}", path=target
        )

    dlm_dir = target / _SCAFFOLD_DIR
    existing = sorted(dlm_dir.glob("*.dlm")) if dlm_dir.is_dir() else []
    named_match = next((c for c in existing if c.stem == name), None)
    name_is_default = name == _DEFAULT_NAME

    # --- Resume: reuse an existing .dlm when appropriate --------------
    # (1) Explicit --name matches an existing file: reuse that one.
    # (2) Default --name (corpus), single existing file: reuse it as a
    #     convenience (user passed no explicit name, we assume they
    #     want the one `.dlm` that's there).
    # (3) Default --name, multiple existing files: refuse (ambiguous
    #     without --name disambiguation).
    if not rescaffold:
        if named_match is not None:
            dlm_id = _dlm_id_from_file(named_match)
            return ScaffoldResult(dlm_path=named_match, scaffolded=False, dlm_id=dlm_id)
        if name_is_default and len(existing) == 1:
            dlm_id = _dlm_id_from_file(existing[0])
            return ScaffoldResult(
                dlm_path=existing[0], scaffolded=False, dlm_id=dlm_id
            )
        if name_is_default and len(existing) > 1:
            listing = "\n".join(
                f"  dlm train {target} --name {c.stem}" for c in existing
            )
            raise ScaffoldError(
                f"multiple .dlm files found under {target / _SCAFFOLD_DIR}; "
                f"pass --name to pick one:\n{listing}",
                path=target,
            )

    # --- Scaffold or rescaffold path ----------------------------------
    if base is None:
        raise ScaffoldError(
            "first-run scaffold requires --base <key>. Pick from the base "
            "registry (e.g. smollm2-135m, qwen2.5-coder-1.5b) or pass "
            "--base hf:<org>/<name> for an off-registry model.",
            path=target,
        )

    dlm_path = dlm_dir / f"{name}.dlm"
    existing_id = (
        _dlm_id_from_file(dlm_path)
        if rescaffold and dlm_path.is_file()
        else None
    )

    dlm_id = existing_id or mint_ulid()
    dlm_dir.mkdir(parents=True, exist_ok=True)
    _write_scaffold(
        dlm_path=dlm_path,
        dlm_id=dlm_id,
        base=base,
        include=include,
        exclude=exclude,
        recursive=recursive,
        policy=policy,
        target=target,
    )
    _LOG.info(
        "scaffold: wrote %s (dlm_id=%s, base=%s)", dlm_path, dlm_id, base
    )
    return ScaffoldResult(dlm_path=dlm_path, scaffolded=True, dlm_id=dlm_id)


def _dlm_id_from_file(path: Path) -> str:
    """Extract `dlm_id` from an existing `.dlm` by parsing its frontmatter."""
    parsed = parse_file(path)
    return parsed.frontmatter.dlm_id


def _write_scaffold(
    *,
    dlm_path: Path,
    dlm_id: str,
    base: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    recursive: bool,
    policy: Policy,
    target: Path,
) -> None:
    """Serialize a minimal `.dlm` frontmatter + body to `dlm_path`.

    Writes directly as YAML-frontmatter text rather than round-tripping
    through the `DlmFrontmatter` serializer — the scaffold is small,
    the frontmatter is a fixed shape, and keeping the write path
    independent of the full parser simplifies bootstrapping for
    directory-first users.

    The `path:` field is the absolute resolved `target`, not `"."`. A
    relative `"."` would anchor on `dlm_path.parent` (the `.dlm/`
    directory), which doesn't contain the user's files and is
    default-excluded by the descent protocol — the first scaffolded
    train would ingest zero content.
    """
    effective_include = _build_include_globs(include, recursive=recursive)
    lines: list[str] = [
        "---",
        f"dlm_id: {dlm_id}",
        f"dlm_version: {CURRENT_SCHEMA_VERSION}",
        f"base_model: {base}",
        "training:",
        f"  sources_policy: {policy}",
        "  sources:",
        f'    - path: "{target.resolve().as_posix()}"',
        "      include:",
    ]
    for pat in effective_include:
        lines.append(f'        - "{pat}"')
    if exclude:
        lines.append("      exclude:")
        for pat in exclude:
            lines.append(f'        - "{pat}"')
    lines.extend(
        [
            "---",
            "",
            "# Auto-scaffolded by `dlm train`. Edit the frontmatter above "
            "to refine training.",
            "",
        ]
    )
    atomic_write_text(dlm_path, "\n".join(lines))


def _build_include_globs(
    include: tuple[str, ...], *, recursive: bool
) -> tuple[str, ...]:
    """Map `--include` flags + `--recursive` to frontmatter globs.

    Empty `--include` + `--recursive` → `["**/*"]`: train on every
    file the descent protocol approves.
    Empty `--include` + `--no-recursive` → `["*"]`: top-level files
    only.
    Explicit `--include` globs are passed through unchanged; the
    `--recursive` flag doesn't transform user-supplied patterns (users
    writing `*.py` know what they want).
    """
    if include:
        return include
    return ("**/*",) if recursive else ("*",)
