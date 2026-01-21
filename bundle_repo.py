# scripts/bundle_repo.py
"""
Repository bundler (source-of-truth snapshot) — snapshot-grade, fail-fast, low-noise.

Includes:
- Source: .py, .pyi
- Config/packaging/docs: .yaml/.yml/.toml/.json/.ini/.cfg/.md/.txt
- Scripts: .ps1/.sh/.bat
- repo files (if present): requirements.txt, pyproject.toml, .gitignore, README.md, LICENSE,
  .pre-commit-settings.yaml, ruff.toml, pytest.ini, tox.ini, Makefile, .editorconfig, mypy.ini, setup.cfg, CHANGELOG.md

Excludes noise:
- .git, .venv/venv, __pycache__, IDE/caches, dist/build
- outputs/, logs/, images/archives/binaries
- secrets by filename patterns (.env*, *.key/*.pem/*.pfx/*.kdbx, etc.)

Output:
- outputs/bundles/full_repository_bundle.txt

Adds:
- REPO STRUCTURE TREE (dirs-only, pruned) for architecture visibility
- DIRECTORY TREE (only included files + directories)
- MANIFEST with bytes + sha256 + flags
- FAIL-FAST validation:
  - read errors
  - non-utf8 replacements (unless allowlisted)
  - truncation due to MAX_FILE_BYTES (unless disabled)
  - empty python files (except __init__.py; others allowlisted)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# --- bundle policy knobs ---
MAX_FILE_BYTES = 2_000_000  # safety cap (prevents accidental huge dumps)
FAIL_ON_REPLACEMENT = True  # fail if non-utf8 decode needed replacement
FAIL_ON_TRUNCATION = True   # fail if any file exceeds MAX_FILE_BYTES
FAIL_ON_EMPTY_PY = True     # fail if any .py/.pyi is empty (except __init__.py; unless allowlisted)


# allowlist for intentionally-empty python files (keep tight)
# NOTE: __init__.py is always allowed to be empty without listing here.
ALLOW_EMPTY_PY = {
    # "tests/unit/test_config.py",
}

# allowlist for files that may contain non-utf8 bytes but are still "texty" (keep tight)
ALLOW_REPLACEMENT = {
    # "docs/legacy.txt",
}


def _rel(root: Path, p: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/")


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def find_repo_root(start: Path) -> Path:
    """
    Walk upwards until we find a marker for repo-root.
    Markers: .git OR requirements.txt OR pyproject.toml
    """
    cur = start.resolve()
    for _ in range(30):
        if (cur / ".git").exists() or (cur / "requirements.txt").exists() or (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def should_exclude_dir(name: str) -> bool:
    excluded = {
        ".git", ".venv", "venv", "__pycache__", ".idea",
        ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "outputs", "logs", "log", "backtest_png",
        "dist", "build", ".tox", ".nox", ".eggs",
    }
    return name in excluded


def _looks_like_secret_name(name: str) -> bool:
    n = name.lower()

    if n == ".env" or n.startswith(".env."):
        return True

    secret_ext = (".key", ".pem", ".pfx", ".p12", ".kdbx", ".crt", ".cer")
    if n.endswith(secret_ext):
        return True

    if "secret" in n or "secrets" in n:
        return True

    # databases often contain credentials or tokens in local dev
    if n.endswith(".sqlite") or n.endswith(".db"):
        return True

    return False


def should_include_file(path: Path) -> bool:
    # never include secrets (even if extension is "allowed")
    if _looks_like_secret_name(path.name):
        return False

    # only include env templates, never real envs
    if path.name == ".env.example":
        return True
    if path.name.lower().startswith(".env"):
        return False

    include_ext = {
        ".py", ".pyi",
        ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg",
        ".md", ".txt",
        ".ps1", ".sh", ".bat",
    }

    include_names = {
        "requirements.txt",
        "pyproject.toml",
        ".gitignore",
        "README.md",
        "LICENSE",
        ".pre-commit-settings.yaml",
        "ruff.toml",
        "pytest.ini",
        "tox.ini",
        "Makefile",
        ".editorconfig",
        "mypy.ini",
        "setup.cfg",
        "CHANGELOG.md",
    }

    exclude_ext = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp",
        ".zip", ".7z", ".rar",
        ".pdf",
        ".exe", ".dll",
        ".pkl", ".pickle",
        ".parquet", ".feather",
    }

    if path.name in include_names:
        return True

    if path.suffix.lower() in exclude_ext:
        return False

    return path.suffix.lower() in include_ext


def _read_text_with_meta(root: Path, p: Path) -> tuple[str, dict]:
    raw = p.read_bytes()
    rel = _rel(root, p)

    meta = {
        "rel": rel,
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
        "truncated": False,
        "had_replacement": False,
    }

    if len(raw) > MAX_FILE_BYTES:
        meta["truncated"] = True
        raw = raw[:MAX_FILE_BYTES]

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        meta["had_replacement"] = True

    # stabilize diffs: always end with newline in dump
    if text and not text.endswith("\n"):
        text += "\n"

    return text, meta


@dataclass(frozen=True)
class FileRec:
    path: Path
    rel: str
    text: str
    meta: dict


def _collect_files(root: Path, out_path: Path) -> list[Path]:
    files_to_dump: list[Path] = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
        base_p = Path(base)
        for f in files:
            p = base_p / f
            if p.resolve() == out_path.resolve():
                continue
            if should_include_file(p):
                files_to_dump.append(p)
    return sorted(set(files_to_dump), key=lambda p: str(p).lower())


def write_pruned_tree_dirs_only(root: Path, out) -> None:
    """
    Dirs-only tree for architecture visibility (incl. empty dirs),
    pruned using should_exclude_dir() to avoid noise (.venv/.git/etc.).
    """
    out.write("### REPO STRUCTURE TREE (dirs-only, pruned)\n")
    out.write(f"### ROOT: {root}\n")
    out.write("### NOTE: same exclude rules as bundler; shows directories even if empty\n")
    out.write("=" * 80 + "\n")

    all_dirs: set[Path] = {root}
    for base, dirs, _files in os.walk(root):
        dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
        base_p = Path(base)
        for d in dirs:
            all_dirs.add(base_p / d)

    ordered = sorted(all_dirs, key=lambda p: str(p).lower())

    for d in ordered:
        rel = "." if d == root else _rel(root, d)
        depth = 0 if d == root else rel.count("/")
        indent = "  " * depth
        name = f"{root.name}/" if d == root else f"{d.name}/"
        out.write(f"{indent}{name}\n")

    out.write("=" * 80 + "\n\n")


def write_tree(root: Path, out, included_files: Iterable[Path]) -> None:
    """
    Tree shows only directories + files that are actually included in the dump.
    This prevents tree noise and avoids leaking secret-ish filenames.
    """
    out.write("### DIRECTORY TREE\n")
    out.write(f"### ROOT: {root}\n")
    out.write("### NOTE: outputs/ and venv/caches are excluded\n")
    out.write("=" * 80 + "\n")

    dirs: set[Path] = set()
    files: set[Path] = set()

    for p in included_files:
        files.add(p)
        parent = p.parent
        while parent != root and root in parent.parents:
            dirs.add(parent)
            parent = parent.parent
        dirs.add(root)

    children: dict[Path, list[Path]] = {d: [] for d in dirs}
    for d in dirs:
        if d != root and d.parent in dirs:
            children[d.parent].append(d)
    for f in files:
        if f.parent in dirs:
            children[f.parent].append(f)

    def _sort_key(p: Path) -> str:
        return str(p).lower()

    for d in children:
        children[d] = sorted(children[d], key=_sort_key)

    def _emit_dir(cur: Path, depth: int) -> None:
        if cur != root:
            out.write(f"{'  '*depth}{cur.name}/\n")

        for ch in children.get(cur, []):
            if ch.is_dir():
                _emit_dir(ch, depth + (0 if cur == root else 1))
            else:
                out.write(f"{'  '*(depth + (0 if cur == root else 1))}{ch.name}\n")

    _emit_dir(root, 0)
    out.write("=" * 80 + "\n\n")


def bundle(output_filename: str = "full_repository_bundle.txt") -> Path:
    script_dir = Path(__file__).resolve().parent
    root = find_repo_root(script_dir)

    out_dir = root / "outputs" / "bundles"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename

    files_to_dump = _collect_files(root, out_path)

    records: list[FileRec] = []
    errors: list[str] = []

    for p in files_to_dump:
        rel = _rel(root, p)
        try:
            text, meta = _read_text_with_meta(root, p)
            records.append(FileRec(path=p, rel=rel, text=text, meta=meta))

            if FAIL_ON_TRUNCATION and meta["truncated"]:
                errors.append(f"TRUNCATED_FILE: {rel} (>{MAX_FILE_BYTES} bytes cap)")

            if meta["had_replacement"] and FAIL_ON_REPLACEMENT and rel not in ALLOW_REPLACEMENT:
                errors.append(f"NON_UTF8_REPLACEMENT: {rel}")

            if FAIL_ON_EMPTY_PY and p.suffix.lower() in {".py", ".pyi"} and meta["bytes"] == 0:
                if rel.endswith("/__init__.py"):
                    pass
                elif rel in ALLOW_EMPTY_PY:
                    pass
                else:
                    errors.append(f"EMPTY_PY: {rel}")

        except Exception as e:
            errors.append(f"READ_ERROR: {rel}: {e}")

    with out_path.open("w", encoding="utf-8") as out:
        out.write("### REPOSITORY BUNDLE\n")
        out.write(f"### GENERATED_UTC: {_utc_now_z()}\n")
        out.write(f"### ROOT: {root}\n\n")

        write_pruned_tree_dirs_only(root, out)
        write_tree(root, out, [r.path for r in records])

        out.write("### MANIFEST (relpath | bytes | sha256 | truncated | replacement)\n")
        out.write("=" * 80 + "\n")
        for r in sorted(records, key=lambda r: r.rel.lower()):
            truncated = int(r.meta["truncated"])
            repl = int(r.meta["had_replacement"])
            out.write(f"{r.rel} | {r.meta['bytes']} | {r.meta['sha256']} | {truncated} | {repl}\n")
        out.write("=" * 80 + "\n\n")

        out.write("### FILE CONTENTS\n")
        out.write("=" * 80 + "\n")
        for r in sorted(records, key=lambda r: r.rel.lower()):
            out.write(f"\n\n### FILE: {r.rel}\n")
            out.write("-" * 80 + "\n")
            out.write(r.text)

    if errors:
        msg = "BUNDLE_FAILED:\n" + "\n".join(errors) + f"\n\nOUTPUT_PARTIAL: {out_path}"
        raise SystemExit(msg)

    return out_path


if __name__ == "__main__":
    out_path = bundle()
    print(f"OK: wrote {out_path}")
