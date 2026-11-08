# Original to blis-vllm.
"""Write fields.json: every configuration field and environment variable vLLM reads, with its status.

The field NAMES come from vLLM's own source at the pinned commit, read with the ast module (vLLM is
not imported). The STATUS of each name, its evidence and its estimation rule come from status.json,
which people edit. The generator refuses to write a list in which a derived name has no status
entry, or a status entry names a field vLLM does not have, so the list is complete by construction
(VLM-3) and cannot carry a stale entry.

Usage:
    python3 -I fields/gen_fields.py <vllm-source> fields/status.json > fields.json
    python3 -I fields/gen_fields.py <vllm-source> fields/status.json --check fields.json
    python3 -I fields/gen_fields.py <vllm-source> --names           (the derived names, one per line)
    python3 -I fields/gen_fields.py <vllm-source> --env-reads       (each variable read, and where)

Name spaces. A configuration field is named as `vllm serve` accepts it, in snake case:
  - every field of EngineArgs and AsyncEngineArgs (vllm/engine/arg_utils.py) that has a flag;
  - every field of FrontendArgs and BaseFrontendArgs (vllm/entrypoints/launchers/cli_args.py),
    and the flags make_arg_parser adds there itself (model_tag, headless, api_server_count, ...);
  - for an EngineArgs field whose value is a vLLM config object (its annotation names a class
    decorated with @config under vllm/config/, or one of the four dict-typed fields that
    arg_utils.py turns into a config object, CONFIG_OF_DICT below), one entry per field of that
    class, named <field>.<subfield>, recursively, and no entry for the object itself. A field
    declared init=False, or typed SkipValidation[...] (set by vLLM's own code, not by a user),
    is not a setting and gets no entry.
An environment variable is named as it is read. The list holds every key of vllm/envs.py's
environment_variables and every variable vLLM v0.32.0 reads by a literal name (READ RULE below).
status.json's "extra_env" adds a variable a reviewer decides belongs in the list although no
literal read names it.

READ RULE. A read is one of these, its name a string literal or a constant bound to exactly one
string in the same file:
  Python, every .py file under vllm/ (read with ast):
    os.environ.get("X") and os.getenv("X"), also as imported environ / getenv, and through a
      name a function binds to os.environ (env = os.environ if environ is None else environ);
    os.environ["X"] as a value (not as an assignment or del target);
    "X" in os.environ and "X" not in os.environ;
    os.environ.setdefault("X", ...), since a value already set wins;
    os.environ.pop("X", ...) when its value is used;
    envs.is_set("X");
    envs.X, which must be a key of environment_variables, or the generator stops.
  C and CUDA, every .c, .cc, .cpp, .cu, .cuh, .h, .hpp and .inl file under csrc/:
    getenv("X") and std::getenv("X"), or either applied to a const char* constant.
  Rust, every .rs file under rust/ (the vllm-rs front end, selected by VLLM_USE_RUST_FRONTEND):
    env::var("X") and env::var_os("X"), or either applied to a const &str; and a clap
    attribute #[arg(..., env = "X")], which clap reads when the flag is absent (a commented-out
    attribute is not read). Code from a file's first #[cfg(test)] on is test code and is not
    read.
A variable vLLM only writes (os.environ["X"] = v, del, or a pop whose value is dropped) is not a
read: vLLM sets it for a library or a child process, and a value the user set is overwritten.
NCCL_*, NVSHMEM_* and UCX_* are the exception. VLM-3 names those libraries, which read what vLLM
sets, so each of their variables that vLLM reads or sets is listed: set through os.environ, as a
key of a dict literal (the variables batch_invariant.py pins for NCCL) or by a mapping's
setdefault (a child process's environment).
A name is also resolved through a loop over a literal collection (for v in ["A", "B"],
for k, _ in {"A": ...}.items(), or a name bound to such a collection in the same file) and
through a conditional expression of literals ("A" if cond else "B"). A read whose name is
still not resolved (os.environ.get(name)) is listed in status.json's "dynamic_reads", keyed
"<path>: <call>", with the reason, and with the variables it can read ("names") when a reviewer
has traced them; those are added to the list. A dynamic read not listed there, or a listed one
vLLM no longer makes, stops the generator. Reading the whole environment (os.environ.copy(), dict(os.environ), .items()) to pass
it to a child process, hash it or print it names no variable.

EXCLUDED, by this rule and no other. Each excluded path is a separate command or a build step,
never the serving process (vllm serve, its engine core and its workers):
  vllm/collect_env.py and vllm/entrypoints/cli/collect_env.py: `vllm collect-env`, which prints
    the environment for a bug report;
  vllm/benchmarks/ and vllm/entrypoints/cli/benchmark/: `vllm bench`, a client that sends
    requests to a server;
  vllm/entrypoints/cli/openai.py: `vllm chat` and `vllm complete`, clients;
  rust/src/bench/, the Rust benchmark client, and every build.rs, a build script;
  the .py code generators under csrc/, which run at build time;
  everything outside vllm/, csrc/ and rust/ (tests/, benchmarks/, examples/, docs/, tools/,
    setup.py, cmake/): tests, examples and the build, none of it in the serving process.
Nothing else is excluded. A module named for another platform (platforms/rocm.py, xpu.py,
tpu, cpu, an amd/ kernel directory) is read like any other. Design §7.4 lets a variable read
only behind a platform check that is false on CUDA be proven inert by a reviewed static
argument; that is work for the variable's issue, not a reason to leave it off the list.
"""

import argparse
import ast
import difflib
import json
import os
import re
import sys

ENGINE_KIND = "vllm"

# EngineArgs fields typed dict[str, Any] that arg_utils.py builds into a config class
# (create_speculative_config, create_watermark_config, create_diffusion_config; the
# quantization_config annotation names QuantizationConfigArgs itself).
CONFIG_OF_DICT = {
    "speculative_config": "SpeculativeConfig",
    "watermark_config": "WatermarkConfig",
    "diffusion_config": "DiffusionConfig",
}

def parse(path):
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def class_fields(node):
    """The settable dataclass fields of a class body, in order."""
    out = []
    for s in node.body:
        if not isinstance(s, ast.AnnAssign) or not isinstance(s.target, ast.Name):
            continue
        ann = ast.unparse(s.annotation)
        value = ast.unparse(s.value) if s.value is not None else ""
        if "init=False" in value or "SkipValidation" in ann or "ClassVar" in ann:
            continue
        out.append((s.target.id, ann))
    return out


def config_classes(src):
    """Every class under vllm/config/ decorated with @config or @dataclass, by name."""
    out = {}
    d = os.path.join(src, "vllm", "config")
    for name in sorted(os.listdir(d)):
        if not name.endswith(".py"):
            continue
        for n in parse(os.path.join(d, name)).body:
            if isinstance(n, ast.ClassDef):
                decos = [ast.unparse(x) for x in n.decorator_list]
                if any(x == "config" or x.startswith("config(") or x.startswith("dataclass") for x in decos):
                    out[n.name] = n
    return out


def named_class(ann, classes):
    """The config class an annotation names, or None. Bare-word match on the annotation text."""
    hits = [c for c in classes if re.search(r"\b%s\b" % re.escape(c), ann)]
    return hits[0] if len(hits) == 1 else None


def expand(prefix, cls, classes, seen):
    if cls in seen:
        raise SystemExit("gen_fields: config class %s contains itself" % cls)
    out = []
    for name, ann in class_fields(classes[cls]):
        sub = named_class(ann, classes)
        if sub is not None:
            out += expand(prefix + name + ".", sub, classes, seen | {cls})
        else:
            out.append(prefix + name)
    return out


def flags_in(tree):
    """The dest of every add_argument("--flag", ...) call in a module."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            lits = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            dest = [k.value.value for k in node.keywords if k.arg == "dest" and isinstance(k.value, ast.Constant)]
            if dest:
                out.add(dest[0])
            elif lits and lits[0].startswith("--"):
                out.add(lits[0][2:].replace("-", "_"))
            elif lits and not lits[0].startswith("-"):
                out.add(lits[0])  # a positional, such as model_tag
    return out


def config_names(src):
    classes = config_classes(src)
    au = parse(os.path.join(src, "vllm", "engine", "arg_utils.py"))
    cli = parse(os.path.join(src, "vllm", "entrypoints", "launchers", "cli_args.py"))
    flags = flags_in(au)
    names = []
    for tree, wanted in ((au, ("EngineArgs", "AsyncEngineArgs")), (cli, ("BaseFrontendArgs", "FrontendArgs"))):
        for n in tree.body:
            if not (isinstance(n, ast.ClassDef) and n.name in wanted):
                continue
            for name, ann in class_fields(n):
                if tree is au and name not in flags:
                    # A field with no flag (_api_process_count, mamba_config, ...) is set by
                    # vLLM's own code or reached through another flag; it is not a serve setting.
                    continue
                cls = CONFIG_OF_DICT.get(name) or named_class(ann, classes)
                if cls is not None and cls not in classes:
                    raise SystemExit("gen_fields: EngineArgs.%s maps to %s, which vllm/config/ does not define" % (name, cls))
                if cls is not None:
                    names += expand(name + ".", cls, classes, frozenset())
                else:
                    names.append(name)
    # make_arg_parser's own flags, added beside FrontendArgs and AsyncEngineArgs.
    for n in cli.body:
        if isinstance(n, ast.FunctionDef) and n.name == "make_arg_parser":
            names += sorted(flags_in(n))
    return sorted(set(names))


def envs_keys(src):
    tree = parse(os.path.join(src, "vllm", "envs.py"))
    keys = []
    for n in tree.body:
        target = n.target if isinstance(n, ast.AnnAssign) else (n.targets[0] if isinstance(n, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "environment_variables" and isinstance(n.value, ast.Dict):
            keys = [k.value for k in n.value.keys if isinstance(k, ast.Constant)]
    if not keys:
        raise SystemExit("gen_fields: vllm/envs.py has no environment_variables dict")
    return keys


def env_names(src, extra, dynamic=None):
    """envs.py's keys, every variable read by a literal name (READ RULE), and extra_env."""
    names = set(envs_keys(src)) | set(extra)
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
                msg.append("dynamic_reads entries vLLM no longer makes:\n  " + "\n  ".join(gone))
            raise SystemExit("gen_fields: " + "\n".join(msg))
    return sorted(names)


# Paths, relative to the vLLM checkout, that are not part of the serving process (EXCLUDED in
# the module docstring). A path ending in "/" excludes a directory.
EXCLUDED = (
    "vllm/collect_env.py",
    "vllm/entrypoints/cli/collect_env.py",
    "vllm/benchmarks/",
    "vllm/entrypoints/cli/benchmark/",
    "vllm/entrypoints/cli/openai.py",
    "rust/src/bench/",
)
COMM_ENV = re.compile(r"^(NCCL|NVSHMEM|UCX)_[A-Z0-9_]+$")
ENVIRON = re.compile(r"^(?:os\.)?environ$")
GET_CALL = re.compile(r"^(?:os\.)?(?:environ\.get|getenv)$")
CPP_EXT = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inl")
CPP_CONST = re.compile(r'\bconst\s+char\s*\*\s*(?:const\s+)?(\w+)\s*=\s*"([^"]+)"')
CPP_READ = re.compile(r'\b(?:std::)?getenv\s*\(\s*(?:"([^"]+)"|(\w+))\s*\)')
RS_CONST = re.compile(r'\bconst\s+(\w+)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"')
RS_READ = re.compile(r'\benv::var(?:_os)?\s*\(\s*(?:"([^"]+)"|(\w+))\s*\)')
# A clap attribute, #[arg(long, env = "NAME")]: clap reads the variable when the flag is absent
# (the clap `env` feature, rust/Cargo.toml). Only a literal name is read; a commented-out
# attribute is not code (RS_COMMENT is removed first).
RS_CLAP_ENV = re.compile(r'\benv\s*=\s*"([^"]+)"')
RS_COMMENT = re.compile(r'/\*.*?\*/|//[^\n]*', re.S)


def excluded(rel):
    rel = rel.replace(os.sep, "/")
    if os.path.basename(rel) == "build.rs":
        return True
    if rel.startswith("csrc/") and rel.endswith(".py"):
        return True
    return any(rel == e or (e.endswith("/") and rel.startswith(e)) for e in EXCLUDED)


def walk(src, top, exts):
    for root, dirs, files in os.walk(os.path.join(src, top)):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, src).replace(os.sep, "/")
            if f.endswith(exts) and not excluded(rel):
                yield p, rel


def literal_strings(node, consts):
    """The strings an expression can be when every branch of it is a literal: a string, a
    name bound to strings, or a conditional expression of those. None when it can be other."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name) and node.id in consts:
        return set(consts[node.id])
    if isinstance(node, ast.IfExp):
        a, b = literal_strings(node.body, consts), literal_strings(node.orelse, consts)
        return a | b if a is not None and b is not None else None
    return None


def literal_collection(node, consts, collections):
    """The strings a loop over node visits, when node is a literal list, tuple or set of
    strings, a dict literal (its keys), its .keys() or .items(), or a name bound to one."""
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
    """The nodes of one scope, not descending into a nested function or class."""
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        n = todo.pop()
        yield n
        if not isinstance(n, SCOPES):
            todo.extend(ast.iter_child_nodes(n))


def string_constants(scope, outer=({}, {})):
    """Names one scope binds to string literals (or to names or conditional expressions of
    them), and names it binds to literal collections of strings; a name it also binds to
    anything else is dropped."""
    assigns = []
    for n in scope_nodes(scope):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Name):
                    assigns.append((t.id, n.value))
    consts, colls = {}, {}
    seen_c, seen_k = dict(outer[0]), dict(outer[1])
    while True:  # to a fixed point, so a = b, b = "X" resolves whatever the order
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
    # A name this scope binds hides the enclosing scope's binding of it.
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
        """The constants and collections visible at node: each enclosing function's (and
        lambda's) bindings over the module's. A class body's names are not visible inside its
        methods, as in Python."""
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
        """The strings a loop variable takes, when an enclosing for loops over a literal
        collection (for v in [...], for k, _ in d.items(), {k: ... for k in [...]})."""
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
        """Whether a string is a key of a dict literal, or the first argument of a mapping's
        setdefault: the two ways vLLM hands a NCCL, NVSHMEM or UCX variable to the library or
        a child process other than os.environ[...] = v."""
        p = parent.get(const)
        if isinstance(p, ast.Dict) and any(k is const for k in p.keys):
            return True
        return (isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute)
                and p.func.attr == "setdefault" and p.args and p.args[0] is const)

    def environ_aliases(scope):
        """Names a function binds to os.environ, alone or as one branch of a conditional
        expression (env = os.environ if environ is None else environ)."""
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

    reads, writes, dynamic = {}, {}, []
    for n in ast.walk(tree):
        names, is_read = None, True
        if isinstance(n, ast.Call) and n.args:
            fn = ast.unparse(n.func)
            method = n.func.attr if isinstance(n.func, ast.Attribute) else None
            on_environ = method is not None and is_environ(n.func.value)
            if GET_CALL.match(fn) or fn == "envs.is_set" or (on_environ and method in ("get", "setdefault")):
                pass
            elif on_environ and method == "pop":
                is_read = not isinstance(parent.get(n), ast.Expr)
            else:
                continue
            names = names_of(n.args[0])
            if names is None:
                if is_read:
                    dynamic.append("%s: %s" % (rel, ast.unparse(n)))
                continue
        elif isinstance(n, ast.Subscript) and is_environ(n.value):
            is_read = isinstance(n.ctx, ast.Load)
            names = names_of(n.slice)
            if names is None:
                if is_read:
                    dynamic.append("%s: %s" % (rel, ast.unparse(n)))
                continue
        elif (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn))
              and is_environ(n.comparators[0])):
            names = names_of(n.left)
            if names is None:
                dynamic.append("%s: %s" % (rel, ast.unparse(n)))
                continue
        elif isinstance(n, ast.Attribute) and ast.unparse(n.value) in ("envs", "vllm.envs") and isinstance(n.ctx, ast.Load):
            if n.attr in keys or not n.attr.isupper():
                continue  # a key already listed, or a helper such as envs.is_set
            raise SystemExit("gen_fields: %s:%d reads envs.%s, which is not a key of environment_variables"
                             % (rel, n.lineno, n.attr))
        elif (isinstance(n, ast.Constant) and isinstance(n.value, str) and COMM_ENV.match(n.value)
              and comm_setting(n)):
            # A NCCL, NVSHMEM or UCX variable vLLM sets through a mapping, such as a key of the
            # variables it pins for NCCL or of a child's environment: those libraries read it.
            names, is_read = {n.value}, False
        else:
            continue
        for name in names:
            (reads if is_read else writes).setdefault(name, []).append("%s:%d" % (rel, n.lineno))
    return reads, writes, dynamic


def regex_env_reads(path, rel, const_re, read_re, cut_at=None):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if cut_at is not None and cut_at in text:
        text = text[:text.index(cut_at)]
    consts = dict(const_re.findall(text))
    reads = {}
    for m in read_re.finditer(text):
        name = m.group(1) or consts.get(m.group(2))
        if name is None:
            raise SystemExit("gen_fields: %s:%d reads the environment through %s, which is not a string constant"
                             % (rel, text.count("\n", 0, m.start()) + 1, m.group(2)))
        reads.setdefault(name, []).append("%s:%d" % (rel, text.count("\n", 0, m.start()) + 1))
    return reads


def clap_env_reads(path, rel):
    """The variables a Rust file's clap attributes read, #[arg(env = "NAME")], with where.
    Comments are blanked first, keeping line numbers, so a commented-out attribute is not a
    read. Code from the first #[cfg(test)] on is test code."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if "#[cfg(test)]" in text:
        text = text[:text.index("#[cfg(test)]")]
    text = RS_COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    reads = {}
    for m in RS_CLAP_ENV.finditer(text):
        reads.setdefault(m.group(1), []).append("%s:%d" % (rel, text.count("\n", 0, m.start()) + 1))
    return reads


def literal_env_reads(src):
    """Every variable read by a literal name (READ RULE), with where it is read, and the
    reads whose name is not a literal."""
    keys = set(envs_keys(src))
    reads, writes, dynamic = {}, {}, []

    def add(into, found):
        for k, v in found.items():
            into.setdefault(k, []).extend(v)

    for p, rel in walk(src, "vllm", (".py",)):
        r, w, d = python_env_reads(p, rel, keys)
        add(reads, r)
        add(writes, w)
        dynamic += d
    for p, rel in walk(src, "csrc", CPP_EXT):
        add(reads, regex_env_reads(p, rel, CPP_CONST, CPP_READ))
    for p, rel in walk(src, "rust", (".rs",)):
        add(reads, regex_env_reads(p, rel, RS_CONST, RS_READ, cut_at="#[cfg(test)]"))
        add(reads, clap_env_reads(p, rel))
    for name, where in writes.items():
        if COMM_ENV.match(name):
            reads.setdefault(name, []).extend(where)
    return {k: sorted(set(v), key=site_order) for k, v in reads.items()}, sorted(set(dynamic))


def site_order(site):
    path, _, line = site.rpartition(":")
    return path, int(line)


def key(source, name):
    return source + ":" + name


def build(src, status):
    derived = [("config", n) for n in config_names(src)] + [("env", n) for n in env_names(src, status.get("extra_env", []), status.get("dynamic_reads", {}))]
    entries = status["fields"]
    missing = [key(s, n) for s, n in derived if key(s, n) not in entries]
    stale = sorted(set(entries) - {key(s, n) for s, n in derived})
    if missing or stale:
        msg = []
        if missing:
            msg.append("no status for %d fields vLLM reads:\n  %s" % (len(missing), "\n  ".join(missing)))
        if stale:
            msg.append("status for %d names vLLM does not read:\n  %s" % (len(stale), "\n  ".join(stale)))
        raise SystemExit("gen_fields: " + "\n".join(msg))
    fields = []
    for source, name in sorted(derived, key=lambda x: (x[1], x[0])):
        e = entries[key(source, name)]
        fields.append({
            "name": name,
            "source": source,
            "status": e["status"],
            "evidence": e["evidence"],
            "estimation": e["estimation"],
        })
    return {"kind": "FieldList", "engine": {"kind": ENGINE_KIND, "version": status["engine_version"]}, "fields": fields}


# Key order of each object, as blis-substrate contracts/spec/fields declares the struct fields;
# FieldList.Marshal writes them in this order, and sorts keys only inside a substitute setting.
ORDER = {
    "list": ("kind", "engine", "fields"),
    "engine": ("kind", "version"),
    "field": ("name", "source", "status", "evidence", "estimation"),
    "evidence": ("traces", "proof", "issue", "note"),
    "estimation": ("rule", "setting", "reason"),
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
        est = ordered(f["estimation"], "estimation")
        if "setting" in est:
            est["setting"] = json.loads(json.dumps(est["setting"], sort_keys=True))
        g["estimation"] = est
        fields.append(g)
    out["fields"] = fields
    return out


def render(doc):
    # json.dumps with ensure_ascii=False escapes nothing but what JSON requires; Go's encoder
    # with SetEscapeHTML(false) does the same for these ASCII strings. A setting's nested keys
    # are sorted, as Go sorts map keys.
    return json.dumps(canonical(doc), indent=2, ensure_ascii=False) + "\n"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vllm_source")
    ap.add_argument("status", nargs="?")
    ap.add_argument("--check", metavar="FIELDS_JSON")
    ap.add_argument("--names", action="store_true", help="print the derived names and exit")
    ap.add_argument("--env-reads", action="store_true", help="print every variable read (READ RULE) and where, then exit")
    a = ap.parse_args(argv)
    if a.names:
        for n in config_names(a.vllm_source):
            print("config:" + n)
        for n in env_names(a.vllm_source, []):
            print("env:" + n)
        return 0
    if a.env_reads:
        reads, dynamic = literal_env_reads(a.vllm_source)
        for n, where in sorted(reads.items()):
            print(n, " ".join(where))
        for d in dynamic:
            print("dynamic", d)
        return 0
    if a.status is None:
        ap.error("status.json is required")
    with open(a.status, encoding="utf-8") as f:
        status = json.load(f)
    out = render(build(a.vllm_source, status))
    if a.check:
        with open(a.check, encoding="utf-8") as f:
            have = f.read()
        if have != out:
            sys.stdout.writelines(difflib.unified_diff(
                have.splitlines(True), out.splitlines(True), a.check, "generated"))
            print("gen_fields: %s is not what the generator writes; regenerate it" % a.check, file=sys.stderr)
            return 1
        return 0
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
