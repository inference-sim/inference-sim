# Original to blis-sglang.
"""Write fields.json: every configuration field and environment variable SGLang reads, with its status.

The field NAMES come from SGLang's own source at the pinned commit, read with the ast module
(SGLang is not imported). The STATUS of each name, its evidence and its estimation rule come from
fields/status.json, which people edit. The generator refuses to write a list in which a derived
name has no status entry, or a status entry names a field SGLang does not have, so the list is
complete by construction (SGL-3, under the rules of VLM-3) and cannot carry a stale entry. The
method is blis-vllm's fields/gen_fields.py, applied to SGLang's sources.

Usage:
    python3 -I fields/gen_fields.py <sglang-source> fields/status.json > fields.json
    python3 -I fields/gen_fields.py <sglang-source> fields/status.json --value-classes fields/value_classes.json --check fields.json
    python3 -I fields/gen_fields.py <sglang-source> --declarations > fields/declarations.json
    python3 -I fields/gen_fields.py <sglang-source> --declarations --check fields/declarations.json
    python3 -I fields/gen_fields.py <sglang-source> --names           (the derived names, one per line)
    python3 -I fields/gen_fields.py <sglang-source> --env-reads       (each variable read, and where)

<sglang-source> is the root of an SGLang checkout (it holds python/sglang/).

Name spaces. A configuration field is named as ServerArgs names it:
  - every field ServerArgs collects (python/sglang/srt/server_args.py): the annotated fields of
    each class in its _INPUT_NAMESPACES list (the msgspec Structs of arg_groups/fields/*.py), and
    any annotated field of the ServerArgs class itself. A class attribute bound to Derived(...)
    is computed by SGLang after resolution, not set by a user, and is not a field;
  - every flag ServerArgs.add_cli_args registers by hand whose dest is not such a field
    (--config, and deprecated flags that store into no field).
A field is listed whether or not it has a flag. NO_CLI RULE: add_cli_args_from_dataclass
(arg_groups/arg_utils.py) gives a field a flag only when its annotation is A[T, "help"] or
A[T, Arg(...)] and the Arg is not no_cli=True. A field without one is still a setting: the
Python Engine(**kwargs) API, and ServerArgs(...) itself, take every field. fields/declarations.json
records, for each field, its flags (none for a no_cli field, plus any flag add_cli_args registers
by hand for it, such as the deprecated --disable-cuda-graph for disable_cuda_graph), its type,
default, choices and resolution metadata, as the declaration states them.

An environment variable is named as it is read. The list holds every field of the Envs class
(python/sglang/srt/environ.py: each NAME = Env*(...)), the deprecated_name of an Env*WithAlias
field, every key of _DEPRECATED_ENVS, and every variable SGLang reads by a literal name (READ
RULE below). status.json's "extra_env" adds a variable a reviewer decides belongs in the list
although no literal read names it. environ.py also rewrites every SGL_<X> variable to SGLANG_<X>
at import (_handle_deprecated_envs); that names no variable of its own, so each SGLANG_ entry
stands for both spellings.

READ RULE. A read is one of these, its name a string literal or a constant bound to exactly one
string in the same file:
  Python, every .py file under SCANNED_PY (read with ast):
    os.environ.get("X") and os.getenv("X"), also as imported environ / getenv, and through a
      name a function binds to os.environ;
    get_bool_env_var("X") and get_int_env_var("X") (python/sglang/srt/utils/common.py), which
      read os.getenv;
    os.environ["X"] as a value (not as an assignment or del target);
    "X" in os.environ and "X" not in os.environ;
    os.environ.setdefault("X", ...), since a value already set wins;
    os.environ.pop("X", ...) when its value is used;
    envs.X, which must be a field of Envs, or the generator stops.
  C and CUDA, every .c, .cc, .cpp, .cu, .cuh, .h, .hpp and .inl file under SCANNED_CPP:
    getenv("X") and std::getenv("X"), or either applied to a const char* constant.
  Rust, every .rs file under SCANNED_RS: env::var("X") and env::var_os("X"), or either applied
    to a const &str. Code from a file's first #[cfg(test)] on is test code and is not read.
A variable SGLang only writes (os.environ["X"] = v, del, or a pop whose value is dropped) is not
a read. NCCL_*, NVSHMEM_* and UCX_* are the exception, as in VLM-3: those libraries read what
SGLang sets, so each of their variables SGLang reads or sets is listed.
A name is also resolved through a loop over a literal collection and through a conditional
expression of literals. A read whose name is still not resolved is listed in status.json's
"dynamic_reads", keyed "<path>: <call>", with the reason, and with the variables it can read
("names") when a reviewer has traced them; those are added to the list. A dynamic read not listed
there, or a listed one SGLang no longer makes, stops the generator. Reading the whole environment
(os.environ.copy(), dict(os.environ), .items()) names no variable.

SCANNED, by this rule and no other: what the serving process (`sglang serve`,
python -m sglang.launch_server, and the scheduler, tokenizer, detokenizer and worker processes
they start) runs.
  Python: python/sglang/srt/, python/sglang/kernels/, python/sglang/utils.py (srt imports it),
    python/sglang/launch_server.py and python/sglang/cli/ except cli/killall.py (`sglang killall`,
    a separate command).
  C and CUDA: python/sglang/kernels/ (the JIT and AOT kernel sources).
  Rust: rust/, except rust/sglang-bench/ (a benchmark client), rust/sglang-api-codegen/ (a build
    tool) and every build.rs.
Not scanned: python/sglang/test/, python/sglang/benchmark/, python/sglang/bench_*.py,
python/sglang/lang/ (the frontend language, a client), python/sglang/multimodal_gen/ (the
diffusion server, `sglang generate`, another product), python/sglang/check_env.py and
python/sglang/profiler.py (commands that print or profile), sgl-router/ and sgl-model-gateway/
(the router, which llm-d replaces), and everything outside python/ and rust/.
Inside the scanned trees nothing is excluded: a module for another platform is read like any
other. A variable read only behind a check that is false on CUDA can be proven inert by a reviewed
static argument (design §7.4); that is work for its issue, not a reason to leave it off.
"""

import argparse
import ast
import difflib
import json
import os
import re
import sys

ENGINE_KIND = "sglang"
SRT = "python/sglang/srt"
SERVER_ARGS = SRT + "/server_args.py"
ENVIRON_PY = SRT + "/environ.py"
FIELDS_PKG = "sglang.srt.arg_groups.fields."

SCANNED_PY = (SRT, "python/sglang/kernels", "python/sglang/utils.py", "python/sglang/launch_server.py",
              "python/sglang/cli")
SCANNED_CPP = ("python/sglang/kernels",)
SCANNED_RS = ("rust",)
EXCLUDED = (
    "python/sglang/cli/killall.py",
    "rust/sglang-bench/",
    "rust/sglang-api-codegen/",
)


def parse(path):
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def rel_of(src, path):
    return os.path.relpath(path, src).replace(os.sep, "/")


# ---------------------------------------------------------------- ServerArgs declarations

def module_path(src, module):
    base = os.path.join(src, "python", *module.split("."))
    for cand in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(cand):
            return cand
    return None


def imports_of(tree):
    """{local name: (module, name)} for every `from M import N [as L]` at module level."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            for a in n.names:
                out[a.asname or a.name] = (n.module, a.name)
    return out


def input_namespaces(src):
    """The classes ServerArgs collects its fields from, in order, as (class name, file)."""
    path = os.path.join(src, SERVER_ARGS)
    tree = parse(path)
    imports = imports_of(tree)
    names = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "_INPUT_NAMESPACES":
            if not isinstance(n.value, ast.List) or not all(isinstance(e, ast.Name) for e in n.value.elts):
                raise SystemExit("gen_fields: %s: _INPUT_NAMESPACES is not a list of class names" % SERVER_ARGS)
            names = [e.id for e in n.value.elts]
    if not names:
        raise SystemExit("gen_fields: %s has no _INPUT_NAMESPACES list" % SERVER_ARGS)
    out = []
    for name in names:
        if name not in imports or not imports[name][0].startswith(FIELDS_PKG):
            raise SystemExit("gen_fields: %s: namespace %s is not imported from %s*" % (SERVER_ARGS, name, FIELDS_PKG))
        mod, cls = imports[name]
        p = module_path(src, mod)
        if p is None:
            raise SystemExit("gen_fields: namespace %s: module %s is not in the source tree" % (name, mod))
        out.append((cls, p))
    return out


def class_node(tree, name):
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


def literal(node):
    """(True, value) for a node ast.literal_eval reads, else (False, None)."""
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def module_constant(src, tree, name, depth=0):
    """The literal a module-level name is bound to, following `from M import name`; None if not
    a literal list, tuple or set of literals (set(X) and frozenset(X) of one are read through)."""
    if depth > 4:
        return None
    for n in tree.body:
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) and n.value else []
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                v = n.value
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("set", "frozenset", "list", "tuple") \
                        and len(v.args) == 1:
                    v = v.args[0]
                    if isinstance(v, ast.Name):
                        return module_constant(src, tree, v.id, depth + 1)
                ok, val = literal(v)
                if ok and isinstance(val, (list, tuple, set, frozenset)):
                    return sorted(val, key=repr) if isinstance(val, (set, frozenset)) else list(val)
                return None
    imp = imports_of(tree)
    if name in imp:
        p = module_path(src, imp[name][0])
        if p is not None:
            return module_constant(src, parse(p), imp[name][1], depth + 1)
    return None


ARG_KEYS = ("help", "choices", "aliases", "cli_name", "type_parser", "nargs", "required", "action",
            "action_kwargs", "const", "no_cli", "resolvable", "fallback")


def arg_meta(src, tree, ann):
    """(inner type text, Arg keywords as a dict, has_arg) of an A[T, ...] annotation; has_arg is
    False when the annotation carries neither a help string nor an Arg (no flag, NO_CLI RULE)."""
    if not (isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) and ann.value.id in ("A", "Annotated")):
        return ast.unparse(ann), {}, False
    elts = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
    inner, rest = elts[0], elts[1:]
    meta, has = {}, False
    for e in rest:
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "Arg":
            has = True
            for kw in e.keywords:
                if kw.arg not in ARG_KEYS:
                    raise SystemExit("gen_fields: Arg keyword %r is not one gen_fields reads" % kw.arg)
                meta[kw.arg] = kw.value
            break
        if isinstance(e, (ast.Constant, ast.JoinedStr, ast.BinOp)) and not (isinstance(e, ast.Constant) and not isinstance(e.value, str)):
            has = True
            meta["help"] = e
            break
    return ast.unparse(inner), meta, has


def literal_choices(ann):
    """The values of the Literal[...] an A[T, ...] annotation's T is (or Optional of), else None:
    add_cli_args_from_dataclass takes a Literal's values as the flag's choices."""
    inner = ann.slice.elts[0] if isinstance(ann, ast.Subscript) and isinstance(ann.slice, ast.Tuple) else ann
    if isinstance(inner, ast.Subscript) and ast.unparse(inner.value) == "Optional":
        inner = inner.slice
    if isinstance(inner, ast.Subscript) and ast.unparse(inner.value) in ("Literal", "typing.Literal"):
        elts = inner.slice.elts if isinstance(inner.slice, ast.Tuple) else [inner.slice]
        ok, val = literal(ast.List(elts=elts, ctx=ast.Load()))
        return val if ok else None
    return None


def flag_names(name, meta):
    if meta.get("no_cli") is not None and literal(meta["no_cli"]) == (True, True):
        return []
    cli = literal(meta["cli_name"])[1] if "cli_name" in meta else None
    first = cli or "--" + name.replace("_", "-")
    aliases = literal(meta["aliases"])[1] if "aliases" in meta else None
    return [first] + list(aliases or [])


def declaration(src, tree, rel, stmt, namespace):
    name = stmt.target.id
    tp, meta, has_arg = arg_meta(src, tree, stmt.annotation)
    d = {"namespace": namespace, "site": "%s:%d" % (rel, stmt.lineno), "type": tp}
    if stmt.value is None:
        d["required"] = True
    else:
        ok, val = literal(stmt.value)
        if ok and not isinstance(val, (set, frozenset, bytes, complex)):
            d["default"] = val
        else:
            d["default_expr"] = ast.unparse(stmt.value)
    no_cli = "no_cli" in meta and literal(meta["no_cli"]) == (True, True)
    d["flags"] = flag_names(name, meta) if has_arg and not no_cli else []
    if no_cli:
        d["no_cli"] = True
    elif not has_arg:
        d["no_arg"] = True
    if "choices" in meta:
        c = meta["choices"]
        ok, val = literal(c)
        if not ok and isinstance(c, ast.Name):
            val = module_constant(src, tree, c.id)
            ok = val is not None
        if ok:
            d["choices"] = list(val)
        else:
            d["choices_expr"] = ast.unparse(c)
    lit = literal_choices(stmt.annotation)
    if lit is not None and "choices" not in d and "choices_expr" not in d:
        d["choices"] = lit
    for k in ("resolvable", "fallback", "nargs", "const"):
        if k in meta:
            ok, val = literal(meta[k])
            if ok:
                d[k] = val
            else:
                d[k + "_expr"] = ast.unparse(meta[k])
    for k in ("action", "type_parser"):
        if k in meta:
            d[k] = ast.unparse(meta[k])
    return name, d


def is_derived(value):
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Derived"


def manual_flags(src):
    """{dest: [(flag, site, deprecated)]} for each add_argument call in ServerArgs.add_cli_args."""
    tree = parse(os.path.join(src, SERVER_ARGS))
    sa = class_node(tree, "ServerArgs")
    fn = [n for n in sa.body if isinstance(n, ast.FunctionDef) and n.name == "add_cli_args"] if sa else []
    if not fn:
        raise SystemExit("gen_fields: %s: no ServerArgs.add_cli_args" % SERVER_ARGS)
    out = {}
    for node in ast.walk(fn[0]):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not flags:
            raise SystemExit("gen_fields: %s:%d: add_argument with no literal flag" % (SERVER_ARGS, node.lineno))
        dest = [k.value.value for k in node.keywords if k.arg == "dest" and isinstance(k.value, ast.Constant)]
        d = dest[0] if dest else flags[0].lstrip("-").replace("-", "_")
        action = [ast.unparse(k.value) for k in node.keywords if k.arg == "action"]
        deprecated = bool(action) and "Deprecated" in action[0]
        out.setdefault(d, []).append((flags, "%s:%d" % (SERVER_ARGS, node.lineno), deprecated))
    return out


def server_args_declarations(src):
    """{field: declaration} for every ServerArgs field, and {name: flag entry} for every flag
    add_cli_args registers by hand that stores into no field."""
    decls = {}
    sources = [(cls, p) for cls, p in input_namespaces(src)]
    sources.append(("ServerArgs", os.path.join(src, SERVER_ARGS)))
    for cls, path in sources:
        tree = parse(path)
        node = class_node(tree, cls)
        if node is None:
            raise SystemExit("gen_fields: %s defines no class %s" % (rel_of(src, path), cls))
        ns = None
        for s in node.body:
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) \
                    and s.targets[0].id == "_NS_PATH":
                ns = literal(s.value)[1]
        if ns is None and cls != "ServerArgs":
            raise SystemExit("gen_fields: %s.%s has no _NS_PATH" % (rel_of(src, path), cls))
        for s in node.body:
            if not (isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)):
                continue
            if s.target.id.startswith("_") or "ClassVar" in ast.unparse(s.annotation) or (s.value is not None and is_derived(s.value)):
                continue
            name, d = declaration(src, tree, rel_of(src, path), s, ns or "")
            if name in decls:
                raise SystemExit("gen_fields: %s is declared by both %s and %s" % (name, decls[name]["site"], d["site"]))
            decls[name] = d
    cli_only = {}
    for dest, regs in sorted(manual_flags(src).items()):
        for flags, site, deprecated in regs:
            if dest in decls:
                d = decls[dest]
                d.setdefault("manual_flags", []).append({"flags": flags, "site": site, "deprecated": deprecated})
            else:
                cli_only.setdefault(dest, []).append({"flags": flags, "site": site, "deprecated": deprecated})
    return decls, cli_only


def config_names(src):
    decls, cli_only = server_args_declarations(src)
    return sorted(set(decls) | set(cli_only))


# ---------------------------------------------------------------- environment variables

def envs_declarations(src):
    """{name: declaration} for each Envs field of environ.py, its WithAlias deprecated names, and
    the keys of _DEPRECATED_ENVS."""
    tree = parse(os.path.join(src, ENVIRON_PY))
    envs = class_node(tree, "Envs")
    if envs is None:
        raise SystemExit("gen_fields: %s has no class Envs" % ENVIRON_PY)
    out = {}
    for s in envs.body:
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)):
            continue
        v = s.value
        if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id.startswith("Env")):
            continue
        name = s.targets[0].id
        d = {"site": "%s:%d" % (ENVIRON_PY, s.lineno), "type": v.func.id}
        if v.args:
            ok, val = literal(v.args[0])
            if ok:
                d["default"] = val
            else:
                d["default_expr"] = ast.unparse(v.args[0])
        for kw in v.keywords:
            if kw.arg == "deprecated_name":
                d["deprecated_name"] = literal(kw.value)[1]
        out[name] = d
    for s in envs.body:
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id.isupper():
            raise SystemExit("gen_fields: %s:%d: Envs.%s is annotated; gen_fields reads NAME = Env*(...)"
                             % (ENVIRON_PY, s.lineno, s.target.id))
    if not out:
        raise SystemExit("gen_fields: %s: class Envs has no Env* field" % ENVIRON_PY)
    for name, d in list(out.items()):
        old = d.get("deprecated_name")
        if old:
            out.setdefault(old, {"site": d["site"], "type": "deprecated alias", "alias_of": name})
    for n in tree.body:
        target = n.target if isinstance(n, ast.AnnAssign) else (n.targets[0] if isinstance(n, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "_DEPRECATED_ENVS" and isinstance(n.value, ast.Dict):
            for k, v in zip(n.value.keys, n.value.values):
                ok, key = literal(k)
                if ok and isinstance(key, str):
                    rep = [kw.value for kw in v.keywords if kw.arg == "replacement"] if isinstance(v, ast.Call) else []
                    entry = {"site": "%s:%d" % (ENVIRON_PY, k.lineno), "type": "deprecated"}
                    if rep:
                        entry["replacement"] = literal(rep[0])[1]
                    out.setdefault(key, entry)
    return out


def envs_keys(src):
    tree = parse(os.path.join(src, ENVIRON_PY))
    envs = class_node(tree, "Envs")
    return {s.targets[0].id for s in envs.body
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
            and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name) and s.value.func.id.startswith("Env")}


def env_names(src, extra, dynamic=None):
    """Envs' fields and deprecated names, every variable read by a literal name (READ RULE), and extra_env."""
    names = set(envs_declarations(src)) | set(extra)
    reads, dyn = literal_env_reads(src)
    names |= set(reads)
    for entry in (dynamic or {}).values():
        names |= set(entry.get("names", []))
    if dynamic is not None:
        unlisted = sorted(set(dyn) - set(dynamic))
        gone = sorted(set(dynamic) - set(dyn))
        if unlisted or gone:
            msg = []
            if unlisted:
                msg.append("reads whose name is not a literal, not in status.json dynamic_reads:\n  " + "\n  ".join(unlisted))
            if gone:
                msg.append("dynamic_reads entries SGLang no longer makes:\n  " + "\n  ".join(gone))
            raise SystemExit("gen_fields: " + "\n".join(msg))
    return sorted(names)


COMM_ENV = re.compile(r"^(NCCL|NVSHMEM|UCX)_[A-Z0-9_]+$")
ENVIRON = re.compile(r"^(?:os\.)?environ$")
GET_CALL = re.compile(r"^(?:os\.)?(?:environ\.get|getenv)$|^(?:\w+\.)*get_(?:bool|int)_env_var$")
CPP_EXT = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inl")
CPP_CONST = re.compile(r'\bconst\s+char\s*\*\s*(?:const\s+)?(\w+)\s*=\s*"([^"]+)"')
CPP_READ = re.compile(r'\b(?:std::)?getenv\s*\(\s*(?:"([^"]+)"|(\w+))\s*\)')
RS_CONST = re.compile(r'\bconst\s+(\w+)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"')
RS_READ = re.compile(r'\benv::var(?:_os)?\s*\(\s*(?:"([^"]+)"|(\w+))\s*\)')


def excluded(rel):
    if os.path.basename(rel) == "build.rs":
        return True
    return any(rel == e or (e.endswith("/") and rel.startswith(e)) for e in EXCLUDED)


def walk(src, tops, exts):
    for top in tops:
        p = os.path.join(src, top)
        if os.path.isfile(p):
            if p.endswith(exts) and not excluded(top):
                yield p, top
            continue
        if not os.path.isdir(p):
            raise SystemExit("gen_fields: scanned path %s is not in the source tree" % top)
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for f in sorted(files):
                fp = os.path.join(root, f)
                rel = rel_of(src, fp)
                if f.endswith(exts) and not excluded(rel):
                    yield fp, rel


def literal_strings(node, consts):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name) and node.id in consts:
        return set(consts[node.id])
    if isinstance(node, ast.IfExp):
        a, b = literal_strings(node.body, consts), literal_strings(node.orelse, consts)
        return a | b if a is not None and b is not None else None
    return None


def literal_collection(node, consts, collections):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("keys", "items") and not node.args:
        node = node.func.value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("frozenset", "set", "tuple", "list") and len(node.args) == 1):
        node = node.args[0]
    if isinstance(node, ast.Name) and node.id in collections:
        return collections[node.id]
    elts = node.keys if isinstance(node, ast.Dict) else node.elts if isinstance(node, (ast.List, ast.Tuple, ast.Set)) else None
    if elts is None:
        return None
    out = set()
    for e in elts:
        v = literal_strings(e, consts) if e is not None else None
        if v is None:
            return None
        out |= v
    return out


SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def scope_nodes(scope):
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        n = todo.pop()
        yield n
        if not isinstance(n, SCOPES):
            todo.extend(ast.iter_child_nodes(n))


def string_constants(scope, outer=({}, {})):
    assigns = []
    for n in scope_nodes(scope):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Name):
                    assigns.append((t.id, n.value))
    consts, colls = {}, {}
    seen_c, seen_k = dict(outer[0]), dict(outer[1])
    while True:
        before = (len(consts), len(colls))
        for name, value in assigns:
            v = literal_strings(value, {**seen_c, **consts})
            if v is not None:
                consts.setdefault(name, set()).update(v)
                continue
            c = literal_collection(value, {**seen_c, **consts}, {**seen_k, **colls})
            if c is not None:
                colls.setdefault(name, set()).update(c)
        if (len(consts), len(colls)) == before:
            break
    allc, allk = {**seen_c, **consts}, {**seen_k, **colls}
    other = {name for name, value in assigns
             if literal_strings(value, allc) is None and literal_collection(value, allc, allk) is None}
    for name in other:
        seen_c.pop(name, None)
        seen_k.pop(name, None)
    seen_c.update({k: v for k, v in consts.items() if k not in other})
    seen_k.update({k: v for k, v in colls.items() if k not in other})
    return seen_c, seen_k


def python_env_reads(path, rel, keys):
    """(reads, writes, dynamic) for one module: reads and writes map a name to its lines."""
    tree = parse(path)
    parent = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[c] = n
    bound = {}

    def scoped(node):
        chain, p = [], parent.get(node)
        while p is not None:
            if isinstance(p, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                chain.append(p)
            p = parent.get(p)
        env = ({}, {})
        for scope in reversed(chain):
            if scope not in bound:
                bound[scope] = string_constants(scope, env)
            env = bound[scope]
        return env

    def loop_names(name_node, consts, colls):
        p = parent.get(name_node)
        while p is not None:
            gens = [(p.target, p.iter)] if isinstance(p, (ast.For, ast.AsyncFor)) else \
                [(g.target, g.iter) for g in p.generators] if isinstance(p, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) else []
            for target, it in gens:
                first = target.elts[0] if isinstance(target, ast.Tuple) and target.elts else target
                if isinstance(first, ast.Name) and first.id == name_node.id:
                    return literal_collection(it, consts, colls)
            p = parent.get(p)
        return None

    def names_of(a):
        consts, colls = scoped(a)
        v = literal_strings(a, consts)
        if v is None and isinstance(a, ast.Name):
            v = loop_names(a, consts, colls)
        return v

    def comm_setting(const):
        p = parent.get(const)
        if isinstance(p, ast.Dict) and any(k is const for k in p.keys):
            return True
        return (isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute)
                and p.func.attr == "setdefault" and p.args and p.args[0] is const)

    def environ_aliases(scope):
        out = set()
        for a in scope_nodes(scope):
            if isinstance(a, ast.Assign) and len(a.targets) == 1 and isinstance(a.targets[0], ast.Name):
                v = a.value
                branches = [v.body, v.orelse] if isinstance(v, ast.IfExp) else [v]
                if any(ENVIRON.match(ast.unparse(b)) for b in branches):
                    out.add(a.targets[0].id)
        return out

    alias_of = {}

    def is_environ(node):
        if ENVIRON.match(ast.unparse(node)):
            return True
        if not isinstance(node, ast.Name):
            return False
        p = parent.get(node)
        while p is not None and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            p = parent.get(p)
        if p not in alias_of:
            alias_of[p] = environ_aliases(p)
        return node.id in alias_of[p]

    reads, writes, dynamic, dyn_nodes = {}, {}, [], []
    for n in ast.walk(tree):
        names, is_read = None, True
        if isinstance(n, ast.Call) and n.args:
            fn = ast.unparse(n.func)
            method = n.func.attr if isinstance(n.func, ast.Attribute) else None
            on_environ = method is not None and is_environ(n.func.value)
            if GET_CALL.match(fn) or (on_environ and method in ("get", "setdefault")):
                pass
            elif on_environ and method == "pop":
                is_read = not isinstance(parent.get(n), ast.Expr)
            else:
                continue
            names = names_of(n.args[0])
            if names is None:
                if is_read:
                    dyn_nodes.append((n, n.args[0]))
                continue
        elif isinstance(n, ast.Subscript) and is_environ(n.value):
            is_read = isinstance(n.ctx, ast.Load)
            names = names_of(n.slice)
            if names is None:
                if is_read:
                    dyn_nodes.append((n, n.slice))
                continue
        elif (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn))
              and is_environ(n.comparators[0])):
            names = names_of(n.left)
            if names is None:
                dyn_nodes.append((n, n.left))
                continue
        elif isinstance(n, ast.Attribute) and ast.unparse(n.value) in ("envs", "environ.envs") and isinstance(n.ctx, ast.Load):
            if n.attr in keys or not n.attr.isupper():
                continue
            raise SystemExit("gen_fields: %s:%d reads envs.%s, which is not a field of Envs (%s)"
                             % (rel, n.lineno, n.attr, ENVIRON_PY))
        elif (isinstance(n, ast.Constant) and isinstance(n.value, str) and COMM_ENV.match(n.value)
              and comm_setting(n)):
            names, is_read = {n.value}, False
        else:
            continue
        for name in names:
            (reads if is_read else writes).setdefault(name, []).append("%s:%d" % (rel, n.lineno))

    # HELPER RULE. A function of this file whose read takes the variable's name from one of its
    # own parameters (def _bool_env(name, default): os.getenv(name)) is a reader: each call of it
    # in this file is a read of the name its argument gives, and its own read names nothing.
    helpers = {}
    for n, arg in dyn_nodes:
        fn = parent.get(n)
        while fn is not None and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = parent.get(fn)
        if fn is None or not isinstance(arg, ast.Name):
            dynamic.append("%s: %s" % (rel, ast.unparse(n)))
            continue
        params = [a.arg for a in fn.args.posonlyargs + fn.args.args]
        if params and params[0] in ("self", "cls"):
            params = params[1:]
        if arg.id not in params:
            dynamic.append("%s: %s" % (rel, ast.unparse(n)))
            continue
        helpers[fn.name] = (params.index(arg.id), arg.id)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        callee = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else None
        if callee not in helpers:
            continue
        i, pname = helpers[callee]
        a = n.args[i] if i < len(n.args) else next((k.value for k in n.keywords if k.arg == pname), None)
        names = names_of(a) if a is not None else None
        if names is None:
            dynamic.append("%s: %s" % (rel, ast.unparse(n)))
            continue
        for name in names:
            reads.setdefault(name, []).append("%s:%d" % (rel, n.lineno))
    return reads, writes, dynamic


def regex_env_reads(path, rel, const_re, read_re, cut_at=None):
    """(reads, dynamic) of a C, CUDA or Rust file: a read of a literal or of a constant of the
    file is a read; a read of anything else is dynamic."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if cut_at is not None and cut_at in text:
        text = text[:text.index(cut_at)]
    consts = dict(const_re.findall(text))
    reads, dynamic = {}, []
    for m in read_re.finditer(text):
        name = m.group(1) or consts.get(m.group(2))
        line = text.count("\n", 0, m.start()) + 1
        if name is None:
            dynamic.append("%s: %s" % (rel, m.group(0)))
            continue
        reads.setdefault(name, []).append("%s:%d" % (rel, line))
    return reads, dynamic


def literal_env_reads(src):
    """Every variable read by a literal name (READ RULE), with where it is read, and the
    reads whose name is not a literal."""
    keys = envs_keys(src)
    reads, writes, dynamic = {}, {}, []

    def add(into, found):
        for k, v in found.items():
            into.setdefault(k, []).extend(v)

    for p, rel in walk(src, SCANNED_PY, (".py",)):
        r, w, d = python_env_reads(p, rel, keys)
        add(reads, r)
        add(writes, w)
        dynamic += d
    for p, rel in walk(src, SCANNED_CPP, CPP_EXT):
        r, d = regex_env_reads(p, rel, CPP_CONST, CPP_READ)
        add(reads, r)
        dynamic += d
    for p, rel in walk(src, SCANNED_RS, (".rs",)):
        r, d = regex_env_reads(p, rel, RS_CONST, RS_READ, cut_at="#[cfg(test)]")
        add(reads, r)
        dynamic += d
    for name, where in writes.items():
        if COMM_ENV.match(name):
            reads.setdefault(name, []).extend(where)
    return {k: sorted(set(v), key=site_order) for k, v in reads.items()}, sorted(set(dynamic))


def site_order(site):
    path, _, line = site.rpartition(":")
    return path, int(line)


# ---------------------------------------------------------------- the documents

def key(source, name):
    return source + ":" + name


def build(src, status, value_classes=None):
    derived = [("config", n) for n in config_names(src)] + \
        [("env", n) for n in env_names(src, status.get("extra_env", []), status.get("dynamic_reads", {}))]
    entries = status["fields"]
    missing = [key(s, n) for s, n in derived if key(s, n) not in entries]
    stale = sorted(set(entries) - {key(s, n) for s, n in derived})
    if missing or stale:
        msg = []
        if missing:
            msg.append("no status for %d fields SGLang reads:\n  %s" % (len(missing), "\n  ".join(missing)))
        if stale:
            msg.append("status for %d names SGLang does not read:\n  %s" % (len(stale), "\n  ".join(stale)))
        raise SystemExit("gen_fields: " + "\n".join(msg))
    fields = []
    for source, name in sorted(derived, key=lambda x: (x[1], x[0])):
        e = entries[key(source, name)]
        f = {"name": name, "source": source, "status": e["status"], "evidence": e["evidence"], "estimation": e["estimation"]}
        if value_classes is not None and e["status"] in ("modelled", "proven_inert"):
            if name not in value_classes:
                raise SystemExit("gen_fields: %s is %s and fields/value_classes.json has no classes for it" % (name, e["status"]))
            f["value_classes"] = value_classes[name]
        fields.append(f)
    return {"kind": "FieldList", "engine": {"kind": ENGINE_KIND, "version": status["engine_version"]}, "fields": fields}


def declarations(src):
    """fields/declarations.json: what SGLang's own declarations state of each name, for config/
    to build ServerArgs from (type, default, flags, choices) and for a reader to check."""
    decls, cli_only = server_args_declarations(src)
    reads, _ = literal_env_reads(src)
    envs = envs_declarations(src)
    env_out = {}
    for name in sorted(set(envs) | set(reads)):
        d = dict(envs.get(name, {}))
        if name in reads:
            d["read_at"] = reads[name]
        env_out[name] = d
    return {
        "kind": "SGLangDeclarations",
        "generator": "fields/gen_fields.py --declarations",
        "server_args": {k: decls[k] for k in sorted(decls)},
        "cli_only": {k: cli_only[k] for k in sorted(cli_only)},
        "env": env_out,
    }


# Key order of each object, as blis-substrate contracts/spec/fields declares the struct fields;
# FieldList.Marshal writes them in this order.
ORDER = {
    "list": ("kind", "engine", "fields"),
    "engine": ("kind", "version"),
    "field": ("name", "source", "status", "evidence", "value_classes", "estimation"),
    "value_classes": ("type", "values", "covered", "thresholds", "covered_classes"),
    "threshold": ("value", "closed"),
    "evidence": ("traces", "proof", "issue", "note"),
    "estimation": ("rule", "setting", "reason", "preserves", "valid_when", "approximates"),
    "approximates": ("mode", "evidence", "differs", "when"),
    "clause": ("value_in", "value_list_has_any", "value_list_has_none", "value_not_prefixed", "tp_group_nodes",
               "compute_capability"),
    "comparison": ("eq", "gt", "gte", "lt", "lte"),
}


def ordered(obj, kind):
    return {k: obj[k] for k in ORDER[kind] if k in obj}


def canonical(doc):
    out = ordered(doc, "list")
    out["engine"] = ordered(doc["engine"], "engine")
    fields = []
    for f in doc["fields"]:
        g = ordered(f, "field")
        g["evidence"] = ordered(f["evidence"], "evidence")
        if "value_classes" in f:
            vc = ordered(f["value_classes"], "value_classes")
            if "thresholds" in vc:
                vc["thresholds"] = [ordered(t, "threshold") for t in vc["thresholds"]]
            g["value_classes"] = vc
        est = ordered(f["estimation"], "estimation")
        if "setting" in est:
            est["setting"] = json.loads(json.dumps(est["setting"], sort_keys=True))
        if "valid_when" in est:
            est["valid_when"] = [
                {k: (ordered(v, "comparison") if isinstance(v, dict) else v) for k, v in ordered(c, "clause").items()}
                for c in est["valid_when"]]
        if "approximates" in est:
            a = ordered(est["approximates"], "approximates")
            a["mode"] = json.loads(json.dumps(a["mode"], sort_keys=True))
            a["when"] = [
                {k: (ordered(v, "comparison") if isinstance(v, dict) else v) for k, v in ordered(c, "clause").items()}
                for c in a["when"]]
            est["approximates"] = a
        g["estimation"] = est
        fields.append(g)
    out["fields"] = fields
    return out


def render(doc):
    return json.dumps(canonical(doc), indent=2, ensure_ascii=False) + "\n"


def render_declarations(doc):
    return json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=False) + "\n"


def compare(path, out, what):
    with open(path, encoding="utf-8") as f:
        have = f.read()
    if have != out:
        sys.stdout.writelines(difflib.unified_diff(have.splitlines(True), out.splitlines(True), path, "generated"))
        print("gen_fields: %s is not what the generator writes; regenerate it (%s)" % (path, what), file=sys.stderr)
        return 1
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sglang_source")
    ap.add_argument("status", nargs="?")
    ap.add_argument("--check", metavar="FILE")
    ap.add_argument("--value-classes", metavar="VALUE_CLASSES_JSON",
                    help="write each modelled and proven-inert field's value_classes from fields/gen_classes.py's file")
    ap.add_argument("--declarations", action="store_true", help="write fields/declarations.json instead of fields.json")
    ap.add_argument("--names", action="store_true", help="print the derived names and exit")
    ap.add_argument("--env-reads", action="store_true", help="print every variable read (READ RULE) and where, then exit")
    a = ap.parse_args(argv)
    if not os.path.isdir(os.path.join(a.sglang_source, SRT)):
        raise SystemExit("gen_fields: %s has no %s; give the root of an SGLang checkout" % (a.sglang_source, SRT))
    if a.names:
        for n in config_names(a.sglang_source):
            print("config:" + n)
        for n in env_names(a.sglang_source, []):
            print("env:" + n)
        return 0
    if a.env_reads:
        reads, dynamic = literal_env_reads(a.sglang_source)
        for n, where in sorted(reads.items()):
            print(n, " ".join(where))
        for d in dynamic:
            print("dynamic", d)
        return 0
    if a.declarations:
        out = render_declarations(declarations(a.sglang_source))
        if a.check:
            return compare(a.check, out, "--declarations")
        sys.stdout.write(out)
        return 0
    if a.status is None:
        ap.error("status.json is required")
    with open(a.status, encoding="utf-8") as f:
        status = json.load(f)
    classes = None
    if a.value_classes:
        with open(a.value_classes, encoding="utf-8") as f:
            classes = {n: r["value_classes"] for n, r in json.load(f)["fields"].items()}
    out = render(build(a.sglang_source, status, classes))
    if a.check:
        return compare(a.check, out, "fields.json")
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
