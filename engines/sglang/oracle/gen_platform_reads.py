# Original to blis-sglang.
"""List every platform read, clock read and thread start reachable from the code the oracle
drives, and fail unless oracle/mocks.json names the replacement or condition behind each one
(FID-1, FID-8).

The oracle runs SGLang's decision code as released, on a host that is not the CUDA platform it
stands for, with a virtual clock in some modules and not others. A read of the platform, the clock
or a thread that no manifest entry accounts for is a difference between the oracle and the real
run that nobody has looked at. This generator finds them from SGLang's code, not from a hand list.

Reachable code. The decision modules (a module under one of the `decision_prefixes` of
oracle/harness.json) that the scheduler's own files import, directly or through other decision
modules, at any depth, outside `if TYPE_CHECKING:` blocks. The scheduler's files are those
oracle/check_modules.py reads.

Sites. The reading of FID-1 (blis-design#15) is in one place: the root tables, `site_of` (the site
format) and `the_reading` (which references are sites). A change to the reading changes those and
nothing else; the scan (Scan) only resolves names.
- platform: every read of a platform attribute (torch.cuda.*, torch.version.*, torch.npu.* and the
  other device namespaces, hasattr/getattr of one on torch, the stdlib platform module, anything
  of sglang.srt.platforms such as current_platform) and every call or other read of a platform
  predicate. A predicate is a module-level function of a non-decision sglang module whose body
  reads the platform, directly or through another predicate (is_cuda, is_hip, get_device, ...),
  or such a module's class whose __init__ or __post_init__ does.
  A module constant computed from the platform at import (`_is_cuda = is_cuda()`, or a name bound
  under a module-level `if` whose test reads the platform) counts once at its definition, through
  the read that computes it, and again at every read of the constant, in its own module or in
  one that imports it, so a read cannot hide behind a constant.
- clock: every read of the time module (time.*, and a name imported from it) or of
  datetime's now, utcnow and today, and every call of a non-decision helper that reads one.
- thread: every start of a thread or process (threading.Thread and Timer, the concurrent.futures
  executors, multiprocessing.Process), counting a subclass's base, and every call of a
  non-decision helper that starts one.
One site per read or start. A call into a non-decision module is followed through its module-level
functions, its classes' constructors and its constants; a method called on an object is not
followed, since which method runs is not known from the syntax.

Each site is listed in oracle/platform_reads.json as
  "<kind>:<module>:<scope>:<reads>#<n>"
where scope is the enclosing qualified function or class (<module> at import), reads is what is
read (the full attribute chain of a root, or the canonical name of a predicate, helper or
constant), and n counts the same read in the same scope in source order, from 1. So a line moved
inside its scope keeps its id, and an added or removed read changes the list.

The manifest names each site: an entry of oracle/mocks.json lists, in `covers`, the reads it
accounts for, as "<kind>:<reads>". A site is covered by the entry that covers its read; a read of
a platform constant is covered when every read its definition makes is covered. Every site must be
covered by exactly one entry, and every `covers` item must cover a site.

The check (the default) regenerates the list at the checkout and fails on a reachable site the
listing lacks, a listed site that no longer exists or has changed, a commit other than the
listing's, a site no entry covers or two entries cover, and a `covers` item that covers nothing.
--write rewrites oracle/platform_reads.json from the checkout.

Usage: python3 -I oracle/gen_platform_reads.py [--write] <sglang checkout> [oracle/harness.json] [oracle/mocks.json] [oracle/platform_reads.json]
"""
import ast
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEDULER_FILES = ("python/sglang/srt/managers/scheduler*.py", "python/sglang/srt/managers/scheduler_components/*.py")
KINDS = ("platform", "clock", "thread")

# The reading of FID-1 (blis-design#15): the roots of each kind. the_reading applies them, with the
# predicate, helper and constant rules; site_of formats a site.
PLATFORM_ROOTS = ("torch.cuda", "torch.version", "torch.xpu", "torch.npu", "torch.mps", "torch.musa", "torch.hpu",
                  "torch.cpu", "torch.accelerator", "torch.backends.cuda", "torch.backends.cudnn",
                  "torch.backends.mps", "torch_npu", "platform", "sglang.srt.platforms")
CLOCK_ROOTS = ("time",)
CLOCK_NAMES = ("datetime.datetime.now", "datetime.datetime.utcnow", "datetime.datetime.today", "datetime.date.today")
THREAD_NAMES = ("threading.Thread", "threading.Timer", "concurrent.futures.ThreadPoolExecutor",
                "concurrent.futures.ProcessPoolExecutor", "concurrent.futures.thread.ThreadPoolExecutor",
                "concurrent.futures.process.ProcessPoolExecutor", "multiprocessing.Process",
                "torch.multiprocessing.Process", "_thread.start_new_thread")


def under(name: str, roots) -> bool:
    return any(name == r or name.startswith(r + ".") for r in roots)


def root_kind(name: str):
    """The kind of a read of an external name, or None."""
    if under(name, PLATFORM_ROOTS):
        return "platform"
    if under(name, CLOCK_ROOTS) and name not in CLOCK_ROOTS or name in CLOCK_NAMES:
        return "clock"
    if name in THREAD_NAMES:
        return "thread"
    return None


def site_of(kind: str, module: str, scope: str, reads: str, n: int) -> str:
    """The id of the n-th read of `reads` in `scope` of `module`: the one format of a site."""
    return f"{kind}:{module}:{scope}:{reads}#{n}"


def parse_site(site: str) -> tuple:
    """(kind, module, scope, reads) of a site id: the inverse of site_of."""
    kind, module, scope, rest = site.split(":", 3)
    return kind, module, scope, rest.rsplit("#", 1)[0]


def covers_key(kind: str, reads: str) -> str:
    """How a mocks.json `covers` item names a read."""
    return f"{kind}:{reads}"


class Tree:
    """SGLang's Python sources at a checkout: module names to files, parsed once."""

    def __init__(self, root: str, prefixes):
        self.root, self.prefixes = root, tuple(prefixes)
        self._ast, self._bind, self._file, self._canon = {}, {}, {}, {}

    def file(self, module: str):
        if module not in self._file:
            base = os.path.join(self.root, "python", *module.split("."))
            self._file[module] = next((p for p in (base + ".py", os.path.join(base, "__init__.py"))
                                       if os.path.isfile(p)), None)
        return self._file[module]

    def is_package(self, module: str) -> bool:
        return (self.file(module) or "").endswith("__init__.py")

    def decision(self, module: str) -> bool:
        return any(module == p.rstrip(".") or module.startswith(p if p.endswith(".") else p + ".")
                   for p in self.prefixes)

    def tree(self, module: str):
        if module not in self._ast:
            path = self.file(module)
            with open(path) as f:
                self._ast[module] = ast.parse(f.read(), path)
        return self._ast[module]

    def module_of(self, dotted: str):
        """(the longest sglang module prefix of a dotted name that is a file, the rest), or None."""
        parts = dotted.split(".")
        if parts[0] != "sglang":
            return None
        for i in range(len(parts), 0, -1):
            m = ".".join(parts[:i])
            if self.file(m):
                return m, parts[i:]
        return None

    def bindings(self, module: str) -> dict:
        """The module's top-level names: {name: ("import", target) | ("local", None)}, plus the
        modules it star-imports under "*"."""
        if module not in self._bind:
            self._bind[module] = bindings_of(self.tree(module), module, self.is_package(module))
        return self._bind[module]

    def canon(self, dotted: str) -> str:
        """Where an sglang name is defined: re-exports and star imports followed."""
        if dotted not in self._canon:
            self._canon[dotted] = self._follow(dotted, set())
        return self._canon[dotted]

    def _follow(self, dotted: str, seen: set) -> str:
        if dotted in seen:
            return dotted
        seen.add(dotted)
        hit = self.module_of(dotted)
        if not hit or not hit[1]:
            return dotted
        module, rest = hit
        b = self.bindings(module)
        head = rest[0]
        if head in b:
            kind, target = b[head]
            if kind == "import":
                return self._follow(".".join([target] + rest[1:]), seen)
            return dotted
        for star in b["*"]:
            if self.provides(star, head):
                return self._follow(".".join([star] + rest), seen)
        return dotted

    def provides(self, module: str, name: str, seen=None) -> bool:
        """Whether `from module import *` binds name (a module without __all__ exports its names)."""
        seen = seen or set()
        if module in seen or not self.file(module):
            return False
        seen.add(module)
        b = self.bindings(module)
        return name in b or any(self.provides(s, name, seen) for s in b["*"])


def is_type_checking(test) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or \
        (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def runtime_body(body):
    """The statements of a block that run at its level: `if TYPE_CHECKING:` bodies left out (an
    else kept), and if, try and with blocks flattened."""
    for st in body:
        if isinstance(st, ast.If):
            if not is_type_checking(st.test):
                yield from runtime_body(st.body)
            yield from runtime_body(st.orelse)
        elif isinstance(st, ast.Try):
            for b in (st.body, *[h.body for h in st.handlers], st.orelse, st.finalbody):
                yield from runtime_body(b)
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            yield from runtime_body(st.body)
        else:
            yield st


def absolute(module: str, is_pkg: bool, level: int, name):
    """The absolute module a (possibly relative) from-import names."""
    if level == 0:
        return name
    parts = module.split(".")
    base = parts if is_pkg else parts[:-1]
    base = base[: len(base) - (level - 1)]
    return ".".join(base + ([name] if name else []))


def bind_import(st, module, is_pkg, out):
    if isinstance(st, ast.Import):
        for a in st.names:
            if a.asname:
                out[a.asname] = ("import", a.name)
            else:
                top = a.name.split(".")[0]
                out[top] = ("import", top)
    else:
        src = absolute(module, is_pkg, st.level, st.module)
        for a in st.names:
            if a.name == "*":
                out["*"].append(src)
            else:
                out[a.asname or a.name] = ("import", f"{src}.{a.name}")


def scope_nodes(body):
    """Every node of a scope's own code: nested functions, lambdas and classes are not entered, and
    neither are `if TYPE_CHECKING:` bodies."""
    stack = list(body)
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(n, ast.If) and is_type_checking(n.test):
            stack.extend(n.orelse)
            continue
        stack.extend(ast.iter_child_nodes(n))


def bindings_of(node, module: str, is_pkg: bool) -> dict:
    """What a module, function or class body binds: {name: ("import", target) | ("local", None)},
    and under "*" the modules it star-imports. A name both imported and assigned (a fallback in an
    except clause, say) keeps its import."""
    out = {"*": []}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            out[arg.arg] = ("local", None)
    body = node.body if isinstance(node.body, list) else [node.body]
    globals_ = set()
    for n in scope_nodes(body):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            bind_import(n, module, is_pkg, out)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(n.name, ("local", None))
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.setdefault(n.id, ("local", None))
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.setdefault(n.name, ("local", None))
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            globals_.update(n.names)
    for g in globals_:
        out.pop(g, None)
    return out


def chain(node):
    """(the root Name, the attribute names after it) of a pure attribute chain, or None."""
    attrs = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return node, list(reversed(attrs))
    return None


def assigned_names(st) -> tuple:
    """The module-level names an assignment binds."""
    if isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = st.targets if isinstance(st, ast.Assign) else [st.target]
        return tuple(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
    return ()


def bound_names(st) -> tuple:
    """Every module-level name one statement binds."""
    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (st.name,)
    if isinstance(st, ast.Import):
        return tuple(a.asname or a.name.split(".")[0] for a in st.names)
    if isinstance(st, ast.ImportFrom):
        return tuple(a.asname or a.name for a in st.names if a.name != "*")
    return assigned_names(st)


class Scan(ast.NodeVisitor):
    """The references one module's code makes, resolved to qualified names; syntax only, with no
    view on which of them are sites (that is the_reading's).

    refs: [(scope, qualified name, line, statement)], in source order; scope is "<module>" at
    import, else the enclosing qualified function or class. statement indexes stmts for a
    reference made by a module-level statement, else it is None. stmts: [(assigned names, bound
    names, guards)], one per module-level statement or `if` test, where guards are the indexes of
    the module-level `if` tests the statement runs under."""

    def __init__(self, tree: Tree, module: str):
        self.t, self.module = tree, module
        self.is_pkg = tree.is_package(module)
        self.scopes = [("<module>", tree.bindings(module), "module")]
        self.refs, self.stmts, self.stmt = [], [], None

    def resolve(self, name: str):
        """The qualified name a Name refers to here, or None for a local value or a builtin."""
        inner = len(self.scopes) - 1
        for i in range(inner, -1, -1):
            _, b, what = self.scopes[i]
            if what == "class" and i != inner:
                continue  # a class body's names are not visible in its methods
            if name in b:
                kind, target = b[name]
                if kind == "import":
                    return target
                return f"{self.module}.{name}" if what == "module" else None
        for star in self.scopes[0][1]["*"]:
            if self.t.provides(star, name):
                return f"{star}.{name}"
        return None

    def scope(self) -> str:
        return ".".join(s[0] for s in self.scopes[1:]) or "<module>"

    def ref(self, node, dotted: str):
        self.refs.append((self.scope(), dotted, node.lineno, self.stmt if len(self.scopes) == 1 else None))

    def visit_ref(self, node) -> bool:
        """Record a Name or attribute chain that is read; False if node is not a pure chain."""
        c = chain(node)
        if c is None:
            return False
        root, attrs = c
        if isinstance(node.ctx, ast.Load):
            q = self.resolve(root.id)
            if q is not None:
                self.ref(node, ".".join([q] + attrs))
        return True

    def visit_Name(self, node):
        self.visit_ref(node)

    def visit_Attribute(self, node):
        if not self.visit_ref(node):
            self.generic_visit(node)

    def visit_Call(self, node):
        # hasattr(torch, "npu") and getattr(torch, "npu", None) read torch.npu.
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in ("hasattr", "getattr") and self.resolve(fn.id) is None \
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                and isinstance(node.args[1].value, str):
            target = chain(node.args[0])
            q = self.resolve(target[0].id) if target else None
            if q is not None:
                self.ref(node, ".".join([q] + target[1] + [node.args[1].value]))
                for a in node.args[2:]:
                    self.visit(a)
                return
        self.generic_visit(node)

    def _scope(self, node, name, what):
        self.scopes.append((name, bindings_of(node, self.module, self.is_pkg), what))
        for st in (node.body if isinstance(node.body, list) else [node.body]):
            self.visit(st)
        self.scopes.pop()

    def visit_FunctionDef(self, node):
        for d in node.decorator_list + node.args.defaults + [x for x in node.args.kw_defaults if x]:
            self.visit(d)
        self._scope(node, node.name, "function")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        for d in node.args.defaults + [x for x in node.args.kw_defaults if x]:
            self.visit(d)
        self._scope(node, "<lambda>", "function")

    def visit_ClassDef(self, node):
        for d in node.decorator_list + node.bases + [k.value for k in node.keywords]:
            self.visit(d)
        self._scope(node, node.name, "class")

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit(node.value)  # the annotation is not a read

    def visit_arg(self, node):
        pass  # nor is a parameter's

    def visit_If(self, node):
        if is_type_checking(node.test):
            for st in node.orelse:
                self.visit(st)
            return
        self.generic_visit(node)

    def visit_Module(self, node):
        self.module_body(node.body, ())

    def module_body(self, body, guards: tuple):
        for st in body:
            if isinstance(st, ast.If):
                if is_type_checking(st.test):
                    self.module_body(st.orelse, guards)
                    continue
                self.stmt = len(self.stmts)
                self.stmts.append(((), (), guards))
                self.visit(st.test)
                inner = guards + (self.stmt,)
                self.module_body(st.body, inner)
                self.module_body(st.orelse, inner)
            elif isinstance(st, ast.Try):
                for b in (st.body, *[h.body for h in st.handlers], st.orelse, st.finalbody):
                    self.module_body(b, guards)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                self.stmt = len(self.stmts)
                self.stmts.append(((), (), guards))
                for item in st.items:
                    self.visit(item.context_expr)
                self.module_body(st.body, guards)
            else:
                self.stmt = len(self.stmts)
                self.stmts.append((assigned_names(st), bound_names(st), guards))
                self.visit(st)
        self.stmt = None


def imports_of(tree: Tree, module: str) -> set:
    """The sglang modules a module imports at run time, at any depth of its code."""
    out = set()
    is_pkg = tree.is_package(module)
    skip = set()
    for n in ast.walk(tree.tree(module)):
        if isinstance(n, ast.If) and is_type_checking(n.test):
            skip.update(id(x) for b in n.body for x in ast.walk(b))
    for n in ast.walk(tree.tree(module)):
        if id(n) in skip:
            continue
        if isinstance(n, ast.Import):
            for a in n.names:
                parts = a.name.split(".")
                out.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
        elif isinstance(n, ast.ImportFrom):
            src = absolute(module, is_pkg, n.level, n.module)
            if src:
                out.add(src)
                out.update(f"{src}.{a.name}" for a in n.names)
    return {m for m in out if (m == "sglang" or m.startswith("sglang.")) and tree.file(m)}


def reachable(tree: Tree) -> list:
    """The decision modules the scheduler's files import, directly or through decision modules."""
    files = sorted({f for pat in SCHEDULER_FILES for f in glob.glob(os.path.join(tree.root, pat))})
    if not files:
        raise SystemExit(f"{tree.root}: no scheduler files under python/sglang/srt/managers; is it an SGLang checkout?")
    start = [os.path.relpath(f, os.path.join(tree.root, "python"))[:-3].replace(os.sep, ".") for f in files]
    seen, todo = set(), [m for m in start if tree.decision(m)]
    for m in start:
        todo += [d for d in imports_of(tree, m) if tree.decision(d)]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        todo += [d for d in imports_of(tree, m) if tree.decision(d) and d not in seen]
    return sorted(seen)


def is_function(tree: Tree, module: str, name: str) -> bool:
    return any(isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name == name
               for st in runtime_body(tree.tree(module).body))


def is_class(tree: Tree, module: str, name: str) -> bool:
    return any(isinstance(st, ast.ClassDef) and st.name == name for st in runtime_body(tree.tree(module).body))


def scans_of(tree: Tree, mods: list) -> dict:
    """{module: Scan} for the reachable modules and every non-decision sglang module their code
    refers to, at any depth through the latter."""
    scans, todo = {}, list(mods)
    while todo:
        m = todo.pop()
        if m in scans:
            continue
        s = Scan(tree, m)
        s.visit(tree.tree(m))
        scans[m] = s
        for _, dotted, _, _ in s.refs:
            if not dotted.startswith("sglang."):
                continue
            parts = dotted.split(".")
            for i in range(len(parts), 1, -1):
                hit = tree.module_of(tree.canon(".".join(parts[:i])))
                if hit and not tree.decision(hit[0]) and hit[0] not in scans:
                    todo.append(hit[0])
    return scans


def the_reading(tree: Tree, mods: list, scans: dict) -> dict:
    """The sites of the reachable modules, {id: record}, by the reading of FID-1 (blis-design#15).
    This function, the root tables and site_of are the whole reading.

    - A read of a name under a root of a kind is a site of that kind (root_kind).
    - A non-decision module's module-level function stands for each kind its body reads, and so
      does its class, for each kind its __init__ or __post_init__ reads: a call or other read of
      it is a site of that kind. This is how a predicate (is_cuda) or a helper (a function that
      starts a thread) is found.
    - A module-level name computed from the platform at import, in any scanned module, is a
      platform constant: an assignment whose value reads the platform binds one, and so does any
      statement under a module-level `if` whose test reads it. Its definition is a site through
      the read that computes it, and each read of the constant is a site again, with `via`
      naming the reads, other than constants, that it stands for.
    The longest prefix of a qualified name that is a root, a predicate, a helper or a constant
    decides, so torch.cuda.is_available is one read, and mod._is_cuda.x a read of mod._is_cuda.
    Predicates, helpers and constants are found to a fixed point."""
    known = {k: {} for k in KINDS}  # qualified name -> the reads it stands for
    constants = set()

    def classify(dotted: str):
        parts = dotted.split(".")
        if parts[0] == "sglang":
            for i in range(len(parts), 1, -1):
                c = tree.canon(".".join(parts[:i]))
                for kind in KINDS:
                    if c in known[kind]:
                        return kind, c
        k = root_kind(dotted)
        return (k, dotted) if k else None

    def size():
        return sum(len(v) for d in known.values() for v in d.values()) + sum(len(d) for d in known.values())

    while True:
        before = size()
        for m, s in scans.items():
            reads = [(r, classify(r[1])) for r in s.refs]
            stmt_reads = {}
            for (scope, _, _, stmt), hit in reads:
                if not hit:
                    continue
                kind, name = hit
                if stmt is not None and kind == "platform":
                    stmt_reads.setdefault(stmt, set()).add(name)
                if tree.decision(m) or scope == "<module>":
                    continue
                parts = scope.split(".")
                if is_function(tree, m, parts[0]) or (
                        len(parts) > 1 and parts[1] in ("__init__", "__post_init__") and is_class(tree, m, parts[0])):
                    known[kind].setdefault(f"{m}.{parts[0]}", set()).add(name)
            for i, (assigned, bound, guards) in enumerate(s.stmts):
                via = set(stmt_reads.get(i, ()))
                names = set(assigned) if via else set()
                guard_via = set().union(*(stmt_reads.get(g, set()) for g in guards))
                if guard_via:
                    names |= set(bound)
                for n in names:
                    c = f"{m}.{n}"
                    known["platform"].setdefault(c, set()).update(via | guard_via)
                    constants.add(c)
        if size() == before:
            break
    sites, counts = {}, {}
    for m in mods:
        for scope, dotted, line, _ in scans[m].refs:
            hit = classify(dotted)
            if not hit:
                continue
            kind, reads = hit
            key = (kind, m, scope, reads)
            counts[key] = counts.get(key, 0) + 1
            rec = {"line": line}
            if reads in constants:
                rec["via"] = sorted(expand(reads, known, constants))
            sites[site_of(kind, m, scope, reads, counts[key])] = rec
    return sites


def generate(root: str, prefixes) -> tuple:
    """(the reachable decision modules, their sites as {id: record})."""
    tree = Tree(root, prefixes)
    mods = reachable(tree)
    return mods, the_reading(tree, mods, scans_of(tree, mods))


def expand(name: str, known: dict, constants: set, seen=None) -> set:
    """The reads, other than constants, that a platform constant stands for."""
    seen = seen if seen is not None else {name}
    out = set()
    for r in known["platform"].get(name, ()):
        if r in constants:
            if r not in seen:
                seen.add(r)
                out |= expand(r, known, constants, seen)
        else:
            out.add(r)
    return out


def coverage_problems(sites: dict, mocks: list) -> list:
    """What is wrong with how the manifest names the sites: a site no entry covers or two do, and a
    `covers` item that covers no site."""
    bad, owner = [], {}
    for m in mocks:
        for item in m.get("covers", []):
            if item in owner:
                bad.append(f"{item}: covered by both {owner[item]!r} and {m['symbol']!r}; one entry accounts for a read")
            owner.setdefault(item, m["symbol"])
    used = set()
    for site, rec in sorted(sites.items()):
        kind, _, _, reads = parse_site(site)
        keys = [covers_key("platform", v) for v in rec["via"]] if "via" in rec else [covers_key(kind, reads)]
        used.update(keys)
        missing = [k for k in keys if k not in owner]
        if missing:
            what = f"a read of a constant computed from {', '.join(missing)}" if "via" in rec else "a read"
            bad.append(f"{site} (line {rec['line']}): {what} no mocks.json entry covers; name the replacement or "
                       "condition behind it in the `covers` of the entry that accounts for it (FID-1)")
    for item, sym in sorted(owner.items()):
        if item not in used:
            bad.append(f"{sym}: covers {item}, which no reachable site reads (remove it)")
    return bad


def listing_problems(found: dict, listed: dict) -> list:
    """The differences between the sites at the checkout and the committed listing."""
    bad = []
    for site in sorted(set(found) - set(listed)):
        bad.append(f"{site} (line {found[site]['line']}): reachable, and not in oracle/platform_reads.json "
                   "(regenerate it with --write, and name the read in mocks.json)")
    for site in sorted(set(listed) - set(found)):
        bad.append(f"{site}: listed in oracle/platform_reads.json, and no longer at the checkout (regenerate it with --write)")
    for site in sorted(set(found) & set(listed)):
        if found[site] != listed[site]:
            bad.append(f"{site}: listed as {listed[site]}, found as {found[site]} (regenerate it with --write)")
    return bad


def commit_of(root: str) -> str:
    r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{root}: not a git checkout ({r.stderr.strip()})")
    return r.stdout.strip()


def dump(listing: dict) -> str:
    """The listing as JSON, one site to a line, so a regenerated file diffs by site."""
    head = {k: v for k, v in listing.items() if k != "sites"}
    body = ",\n".join(f"    {json.dumps(k)}: {json.dumps(v, sort_keys=True)}" for k, v in sorted(listing["sites"].items()))
    text = json.dumps(head, indent=2)[:-2] + ',\n  "sites": {\n' + body + "\n  }\n}\n"
    json.loads(text)
    return text


def main(argv: list) -> int:
    write = "--write" in argv
    args = [a for a in argv if a != "--write"]
    if not 1 <= len(args) <= 4 or any(a.startswith("--") for a in args):
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    root = args[0]
    harness, mocks_path, listing_path = (args[1:] + [os.path.join(HERE, n) for n in
                                         ("harness.json", "mocks.json", "platform_reads.json")][len(args) - 1:])[:3]
    with open(harness) as f:
        prefixes = json.load(f)["decision_prefixes"]
    commit = commit_of(root)
    mods, found = generate(root, prefixes)
    if write:
        listing = {
            "description": "Every platform read, clock read and thread start reachable from the decision modules the "
                           "oracle drives, at the SGLang commit below; generated by oracle/gen_platform_reads.py "
                           "--write, do not edit. Each is covered by the oracle/mocks.json entry that accounts for it.",
            "sglang_commit": commit, "decision_modules": len(mods), "sites": found}
        with open(listing_path, "w") as f:
            f.write(dump(listing))
        print(f"gen_platform_reads: wrote {len(found)} sites of {len(mods)} decision modules to {listing_path}")
        return 0
    with open(listing_path) as f:
        listing = json.load(f)
    with open(mocks_path) as f:
        mocks = json.load(f)["mocks"]
    bad = []
    if listing.get("sglang_commit") != commit:
        bad.append(f"oracle/platform_reads.json lists the sites of {listing.get('sglang_commit')}, and the checkout "
                   f"is {commit} (regenerate it with --write at the pinned commit)")
    bad += listing_problems(found, listing.get("sites", {}))
    bad += coverage_problems(found, mocks)
    for b in bad:
        print(b)
    kinds = {k: sum(1 for s in found if s.startswith(k + ":")) for k in KINDS}
    print(f"gen_platform_reads: {len(mods)} decision modules, " +
          ", ".join(f"{n} {k}" for k, n in kinds.items()) + f" sites, {len(bad)} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", SyntaxWarning)  # invalid escapes in SGLang's own sources
    sys.exit(main(sys.argv[1:]))
