# Original to blis-substrate.
"""Write code_paths.json: the engine code each host, scheduler and graph-launch cost depends on, with its CodeID.

A host, scheduler or graph-launch coefficient prices work that an engine's own code does. Its value
is valid for that code, not for a version string: a release that leaves the code alone keeps the
coefficient and one that changes a function the cost depends on does not (design 9.2, 12.4). The
engine publishes the identity of that code as code_paths.json. Its type, validator and the rule
that turns function hashes into a CodeID are blis-substrate's contracts/spec/codepaths; a Go test
in each engine loads the committed file with that loader, which recomputes every CodeID, so this
script and the rule cannot drift apart.

WHAT PEOPLE EDIT, AND WHAT THIS SCRIPT DOES. An engine keeps a spec (code_paths/spec.json) that
names the release (kind, version, commit) and, for each code path, the functions the path's cost
depends on, each with the reason. That list is the closure: the functions the cost's work runs
through, chosen by reading the release's code, not discovered by a call graph. This script
  - refuses a source tree at another commit than the spec's release;
  - refuses a function the spec names that does not exist in the source (a renamed or deleted
    function is a change of code a person must look at), or exists more than once;
  - hashes each function's code (below) and writes the table.
A function the engine adds to a path is a spec edit and changes the CodeID; a function whose code
changes in a release changes it by itself.

WHAT A FUNCTION'S HASH COVERS. The SHA-256 of a canonical text of the function's AST, so that the
hash follows what the function does and not how it is written. Comments and formatting are not in
the AST. The canonical text also leaves out docstrings, type annotations (a parameter's, a return's
and an annotated assignment's, which keeps its value as a plain assignment), and statements that
only log (logger.info(...), logging.warning(...), logger.debug_once(...)). Decorators, default
values, every expression and every other statement are in it. The text is written by this
script (not ast.dump, whose layout varies by Python release) from node class names and fields,
skipping fields that are None or empty, so a Python that adds an empty field leaves it unchanged;
it is defined for Python 3.9 and later, whose AST for subscripts and slices is the one it names.

Usage (regen.py of an engine does all of it, fetching the source):
    python3 -I gen_code_paths.py <source-tree> --spec spec.json > code_paths.json
    python3 -I gen_code_paths.py <source-tree> --spec spec.json --check code_paths.json
"""

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys

HASH_VERSION = "blis code path hash v1"
PATHS = ("host", "graphs")
LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log"}
LOG_NAMES = {"logger", "logging", "log"}
SKIPPED_FIELDS = {"ctx", "type_comment", "kind", "lineno", "col_offset", "end_lineno", "end_col_offset"}


class Refused(Exception):
    """The spec and the source disagree, or the source is not the release the spec names."""


# ------------------------------------------------------------------------------------------------
# The canonical text of a function.


def _is_docstring(stmt):
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _terminal_name(node):
    """The last identifier of a dotted expression (self.logger -> logger), or None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_log_statement(stmt):
    """An expression statement that only logs: logger.info(...), logging.warning(...), logger.debug_once(...)."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    f = stmt.value.func
    if not isinstance(f, ast.Attribute):
        return False
    method = f.attr[:-5] if f.attr.endswith("_once") else f.attr
    base = _terminal_name(f.value)
    return method in LOG_METHODS and base is not None and base.strip("_").lower() in LOG_NAMES


def _clean_body(stmts, keep_docstring=False):
    """A statement list as the canonical text sees it: no docstring, no logging statement, an
    annotated assignment as a plain assignment (or nothing when it assigns no value)."""
    out = []
    for i, s in enumerate(stmts):
        if i == 0 and not keep_docstring and _is_docstring(s):
            continue
        if _is_log_statement(s):
            continue
        if isinstance(s, ast.AnnAssign):
            if s.value is None:
                continue
            s = ast.Assign(targets=[s.target], value=s.value)
        out.append(s)
    return out or [ast.Pass()]


def canonical(node):
    """The canonical text of an AST node, a list of nodes or a plain value."""
    if isinstance(node, list):
        return "[" + ",".join(canonical(x) for x in node) + "]"
    if isinstance(node, ast.AST):
        parts = []
        for name in node._fields:
            if name in SKIPPED_FIELDS:
                continue
            value = getattr(node, name, None)
            if name in ("annotation", "returns") or value is None or value == []:
                continue
            if name in ("body", "orelse", "finalbody") and isinstance(value, list) and value \
                    and isinstance(value[0], ast.stmt):
                is_def = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
                value = _clean_body(value) if name == "body" and is_def else _clean_stmts(value)
            parts.append(f"{name}={canonical(value)}")
        return f"{type(node).__name__}({','.join(parts)})"
    return repr(node)


def _clean_stmts(stmts):
    """Statements of a nested block (an if's body, a loop's): logging and annotations as above."""
    return _clean_body(stmts, keep_docstring=True) if stmts else stmts


def function_hash(node):
    """The SHA-256 of a function definition's canonical text."""
    return hashlib.sha256(canonical(node).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------
# Finding functions.


def find_function(tree, qualname, where):
    """The one (async) function definition named qualname (Class.method, outer.inner) in a module."""
    scope = [tree]
    node = None
    for name in qualname.split("."):
        matches = [n for s in scope for n in ast.iter_child_nodes(s)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name
                   and not _is_overload(n)]
        if len(matches) != 1:
            raise Refused(f"{where}: {qualname!r} names {len(matches)} definitions ({name!r}); it must name exactly one")
        node = matches[0]
        scope = [node]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise Refused(f"{where}: {qualname!r} is not a function")
    return node


def _is_overload(node):
    for d in getattr(node, "decorator_list", []):
        if _terminal_name(d) == "overload":
            return True
    return False


def source_commit(root):
    r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        raise Refused(f"{root} is not a git checkout: {r.stderr.strip()}")
    return r.stdout.strip()


# ------------------------------------------------------------------------------------------------
# The table.


def code_id(path, functions):
    """blis-substrate's codepaths.CodeID, restated; a Go test holds the two together."""
    lines = [HASH_VERSION, f"path {path}"]
    lines += [f"function {f['id']} {f['hash']}" for f in sorted(functions, key=lambda f: f["id"].encode())]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def load_spec(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    for key in ("engine", "paths"):
        if key not in spec:
            raise Refused(f"{path} has no {key!r}")
    if set(spec["paths"]) != set(PATHS):
        raise Refused(f"{path} must name the paths {', '.join(PATHS)}, got {', '.join(sorted(spec['paths']))}")
    return spec


def generate(root, spec):
    """The table for the release in root, as a dict in the order the Go type declares."""
    engine = spec["engine"]
    commit = source_commit(root)
    if commit != engine["commit"]:
        raise Refused(f"{root} is at {commit}, not the spec's release {engine['kind']} {engine['version']} at {engine['commit']}")
    trees = {}
    paths = []
    for path in PATHS:
        functions = []
        seen = set()
        for item in spec["paths"][path]:
            fid = item["id"]
            if fid in seen:
                raise Refused(f"{path}: {fid} is listed twice")
            seen.add(fid)
            if "::" not in fid:
                raise Refused(f"{path}: {fid!r} must be <source path>::<qualified name>")
            rel, qual = fid.split("::", 1)
            if rel not in trees:
                full = os.path.join(root, rel)
                if not os.path.isfile(full):
                    raise Refused(f"{path}: {rel} does not exist in {engine['kind']} {engine['version']}")
                with open(full, encoding="utf-8") as f:
                    trees[rel] = ast.parse(f.read(), filename=rel)
            node = find_function(trees[rel], qual, f"{path}: {fid}")
            functions.append({"id": fid, "source": f"{rel}:{node.lineno}@{commit[:8]}", "hash": function_hash(node)})
        functions.sort(key=lambda f: f["id"].encode())
        paths.append({"path": path, "code_id": code_id(path, functions), "functions": functions})
    return {"kind": "CodePathTable",
            "engine": {"kind": engine["kind"], "version": engine["version"], "commit": engine["commit"]},
            "paths": paths}


def render(table):
    """The canonical bytes the Go side's Marshal writes."""
    return json.dumps(table, indent=2, ensure_ascii=False) + "\n"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help="a checkout of the engine release the spec names")
    ap.add_argument("--spec", required=True, help="the engine's code_paths/spec.json")
    ap.add_argument("--check", metavar="FILE", help="fail when FILE is not the table instead of writing it")
    args = ap.parse_args(argv)
    try:
        out = render(generate(args.source, load_spec(args.spec)))
    except Refused as e:
        print(f"gen_code_paths: {e}", file=sys.stderr)
        return 1
    if args.check:
        with open(args.check, encoding="utf-8") as f:
            if f.read() != out:
                print(f"gen_code_paths: {args.check} is not current; regenerate it with the engine's code_paths/regen.py",
                      file=sys.stderr)
                return 1
        return 0
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
