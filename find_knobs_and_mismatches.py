# tools/find_knobs_and_mismatches.py
from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple


@dataclass
class CliArg:
    file: str
    lineno: int
    flags: List[str]
    dest: str | None
    kwargs: Dict[str, Any]


@dataclass
class GetAttrUsage:
    file: str
    lineno: int
    container: str
    attr_name: str
    default_value: Any


def _safe_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{_safe_literal(node.value)}.{node.attr}"
        if isinstance(node, ast.Call):
            return f"{_safe_literal(node.func)}(...)"
        return "<non-literal>"


def _iter_py_files(root: str, exclude_dirs: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
    return out


def _read(fp: str) -> str:
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()


def _relpath(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


class Visitor(ast.NodeVisitor):
    def __init__(self, file_path: str, source: str):
        self.file_path = file_path
        self.source = source
        self.cli_args: List[CliArg] = []
        self.getattrs: List[GetAttrUsage] = []
        self._stack: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        # argparse: *.add_argument(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags: List[str] = []
            kwargs: Dict[str, Any] = {}
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    flags.append(a.value)
                else:
                    flags.append(str(_safe_literal(a)))

            dest = None
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                val = _safe_literal(kw.value)
                kwargs[kw.arg] = val
                if kw.arg == "dest" and isinstance(val, str):
                    dest = val

            # If no explicit dest, infer from longest flag like --ema-period -> ema_period
            if dest is None:
                long_flags = [f for f in flags if isinstance(f, str) and f.startswith("--")]
                if long_flags:
                    dest = long_flags[-1].lstrip("-").replace("-", "_")

            self.cli_args.append(
                CliArg(
                    file=self.file_path,
                    lineno=node.lineno,
                    flags=flags,
                    dest=dest,
                    kwargs=kwargs,
                )
            )

        # getattr(args, "x", default)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2:
                obj = node.args[0]
                name = node.args[1]
                if isinstance(obj, ast.Name) and obj.id == "args":
                    if isinstance(name, ast.Constant) and isinstance(name.value, str):
                        default = None
                        if len(node.args) >= 3:
                            default = _safe_literal(node.args[2])
                        container = ".".join(self._stack) if self._stack else "<module>"
                        self.getattrs.append(
                            GetAttrUsage(
                                file=self.file_path,
                                lineno=node.lineno,
                                container=container,
                                attr_name=name.value,
                                default_value=default,
                            )
                        )

        self.generic_visit(node)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--json", default="knobs_and_mismatches.json")
    ap.add_argument(
        "--exclude",
        default=".venv,.git,__pycache__,.pytest_cache,.mypy_cache,_export,_llm_export",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    exclude_dirs = tuple(d.strip() for d in args.exclude.split(",") if d.strip())

    files = _iter_py_files(root, exclude_dirs)

    all_cli: List[CliArg] = []
    all_get: List[GetAttrUsage] = []

    for fp in files:
        try:
            src = _read(fp)
            tree = ast.parse(src, filename=fp)
            v = Visitor(_relpath(fp, root), src)
            v.visit(tree)
            all_cli.extend(v.cli_args)
            all_get.extend(v.getattrs)
        except Exception:
            continue

    cli_dests = {c.dest for c in all_cli if c.dest}
    get_names = {g.attr_name for g in all_get}

    cli_not_read = sorted([d for d in cli_dests if d not in get_names])
    read_not_cli = sorted([n for n in get_names if n not in cli_dests])

    print("\n" + "=" * 100)
    print("CLI DESTS FOUND")
    print("=" * 100)
    for d in sorted(cli_dests):
        print(d)

    print("\n" + "=" * 100)
    print("ARGS READ VIA getattr(args, ...)")
    print("=" * 100)
    for n in sorted(get_names):
        print(n)

    print("\n" + "=" * 100)
    print("MISMATCHES")
    print("=" * 100)
    print("CLI exists but never read (likely ignored / mapping bug):")
    for d in cli_not_read:
        print(f"  - {d}")

    print("\nArgs read but no CLI flag provides them (only defaults / programmatic set):")
    for n in read_not_cli:
        print(f"  - {n}")

    report = {
        "root": root,
        "cli_args": [asdict(x) for x in all_cli],
        "getattr_usages": [asdict(x) for x in all_get],
        "cli_dests": sorted(list(cli_dests)),
        "getattr_names": sorted(list(get_names)),
        "mismatches": {
            "cli_exists_but_not_read": cli_not_read,
            "read_but_no_cli": read_not_cli,
        },
    }

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
