# Script: find_knobs_and_mismatches.py
# Module: tools.find_knobs_and_mismatches
# Location: tools/find_knobs_and_mismatches.py

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CliArg:
    file: str
    lineno: int
    flags: list[str]
    dest: str | None
    kwargs: dict[str, Any]


@dataclass
class ArgsReadUsage:
    file: str
    lineno: int
    container: str
    attr_name: str
    kind: str  # "getattr" | "attribute"
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


def _iter_py_files(root: str, exclude_dirs: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
    return out


def _read(fp: str) -> str:
    with open(fp, encoding="utf-8") as f:
        return f.read()


def _relpath(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


class Visitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.cli_args: list[CliArg] = []
        self.args_reads: list[ArgsReadUsage] = []
        self._stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def _container(self) -> str:
        return ".".join(self._stack) if self._stack else "<module>"

    def visit_Call(self, node: ast.Call) -> Any:
        # argparse: *.add_argument(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags: list[str] = []
            kwargs: dict[str, Any] = {}

            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    flags.append(a.value)
                else:
                    flags.append(str(_safe_literal(a)))

            dest: str | None = None
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                val = _safe_literal(kw.value)
                kwargs[kw.arg] = val
                if kw.arg == "dest" and isinstance(val, str):
                    dest = val

            # Infer dest from last long flag (argparse behavior is more complex, but good enough)
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
                        default: Any = None
                        if len(node.args) >= 3:
                            default = _safe_literal(node.args[2])
                        self.args_reads.append(
                            ArgsReadUsage(
                                file=self.file_path,
                                lineno=node.lineno,
                                container=self._container(),
                                attr_name=name.value,
                                kind="getattr",
                                default_value=default,
                            )
                        )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # args.foo usage
        if isinstance(node.value, ast.Name) and node.value.id == "args":
            self.args_reads.append(
                ArgsReadUsage(
                    file=self.file_path,
                    lineno=node.lineno,
                    container=self._container(),
                    attr_name=node.attr,
                    kind="attribute",
                    default_value=None,
                )
            )
        self.generic_visit(node)


def _load_schema_fields(repo_root: str) -> set[str]:
    """
    Loads StrategyConfig fields if available. If import fails, returns empty set.
    """
    try:
        import sys

        sys.path.insert(0, repo_root)
        from trading_framework.config.schema import StrategyConfig  # type: ignore

        fields = set(StrategyConfig.model_fields.keys())
        return fields
    except Exception:
        return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--json", default="knobs_and_mismatches.json")
    ap.add_argument(
        "--exclude",
        default=".venv,.git,__pycache__,.pytest_cache,.mypy_cache,_export,_llm_export,outputs,logs",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    exclude_dirs = tuple(d.strip() for d in args.exclude.split(",") if d.strip())

    files = _iter_py_files(root, exclude_dirs)

    all_cli: list[CliArg] = []
    all_reads: list[ArgsReadUsage] = []

    for fp in files:
        try:
            src = _read(fp)
            tree = ast.parse(src, filename=fp)
            v = Visitor(_relpath(fp, root))
            v.visit(tree)
            all_cli.extend(v.cli_args)
            all_reads.extend(v.args_reads)
        except Exception:
            continue

    cli_dests = {c.dest for c in all_cli if c.dest}
    read_names = {r.attr_name for r in all_reads}

    cli_not_read = sorted([d for d in cli_dests if d not in read_names])
    read_not_cli = sorted([n for n in read_names if n not in cli_dests])

    schema_fields = _load_schema_fields(root)
    not_in_schema = sorted(
        [n for n in (cli_dests | read_names) if schema_fields and n not in schema_fields]
    )
    schema_not_referenced = (
        sorted([f for f in schema_fields if f not in (cli_dests | read_names)])
        if schema_fields
        else []
    )

    report = {
        "root": root,
        "cli_args": [asdict(x) for x in all_cli],
        "args_reads": [asdict(x) for x in all_reads],
        "cli_dests": sorted(list(cli_dests)),
        "read_names": sorted(list(read_names)),
        "schema_fields": sorted(list(schema_fields)),
        "mismatches": {
            "cli_exists_but_not_read": cli_not_read,
            "read_but_no_cli": read_not_cli,
            "names_not_in_strategy_schema": not_in_schema,
            "schema_fields_not_referenced_by_cli_or_args": schema_not_referenced,
        },
    }

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 100)
    print("MISMATCH SUMMARY")
    print("=" * 100)
    print(
        f"CLI dests: {len(cli_dests)} | args reads: {len(read_names)} | schema fields: {len(schema_fields)}"
    )
    print(f"CLI exists but not read: {len(cli_not_read)}")
    print(f"Read but no CLI: {len(read_not_cli)}")
    print(f"Names not in StrategyConfig schema: {len(not_in_schema)}")
    print(f"Schema fields not referenced by CLI/args: {len(schema_not_referenced)}")
    print(f"\nWrote: {os.path.abspath(args.json)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
