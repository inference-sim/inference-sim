# Original to blis-vllm.
"""Write fields/value_classes.json: the value classes of each modelled or proven-inert field (VLM-3).

VLM-3: "a field marked modelled records the value classes its evidence covers: for a boolean or
enumerated field, the values its evidence sets; for an ordered numeric field, the range its
evidence sets, split at every threshold the upstream compares the field with, generated from the
upstream's comparisons. The probe values of a proven-inert field follow the same classes."

This reads vLLM's source at the pinned commit with the ast module (vLLM is not imported) and, for
each field of fields.json whose status is modelled or proven_inert:

  1. finds the field's type: bool, a Literal enumeration, a number, a free string, or a structure;
  2. finds every comparison the source makes with the field: `field <op> literal`, `field in
     (literals)`, `field is None`, and a bare truth test (`if field`, `not field`, `a or field`);
  3. cuts the field's domain at those comparisons and writes one class per piece, each with the
     file:line of every comparison that forms its edge ("compared_at") and the text of the
     comparison ("expr"). A class no comparison backs is not written;
  4. lists the traces of the field's evidence whose scenario sets a value in the class
     ("covered_by"), and the other expressions the field is compared with ("relations"), which
     cut the domain at a value that depends on another field and so cannot be a class edge.

What counts as the field. An expression is the field when it is
  - `<base>.<leaf>` (or getattr(<base>, "<leaf>")) where <base> is an owner of the field's
    configuration class: `self` inside that class, a name or attribute whose last part is the
    name VllmConfig (or another config class) gives the class (scheduler_config), a parameter or
    variable annotated with the class, or a name the same function assigned from an owner;
    for a field of EngineArgs, `self` inside EngineArgs and AsyncEngineArgs and the names args,
    engine_args and parsed_args;
  - for an environment variable, `envs.NAME`, os.environ.get("NAME"), os.getenv("NAME") and
    os.environ["NAME"];
  - a name the same function assigned from one of those (x = scheduler_config.max_num_seqs).
Not followed: a value passed into another function, stored in a container, or computed with
arithmetic before the comparison (min(x, 8), x // 2 > 3). A comparison made in such a way is not
found, and a threshold it carries is not a class edge. The limit is stated in the file's
"limits" so a reader of fields.json knows what "generated from the upstream's comparisons" means.

Usage:
    python3 -I fields/gen_classes.py <vllm-source> fields.json                   > fields/value_classes.json
    python3 -I fields/gen_classes.py <vllm-source> fields.json --check fields/value_classes.json
    python3 -I fields/gen_classes.py <vllm-source> fields.json --field NAME ...  (any field, to stdout)
"""

import argparse
import ast
import importlib.util
import json
import os
import re
import sys


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_json(path):
    return json.loads(read_text(path))


def _load_sibling(name):
    """gen_fields.py beside this file, loaded by path (the generators are run as scripts, not as a package)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit("gen_classes: cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gf = _load_sibling("gen_fields")

LIMITS = (
    "Comparisons are found where the field, or a name the same function assigned from it, is compared "
    "with a literal: ==, !=, <, <=, >, >=, in, not in, is None, is not None, and a bare truth test. A "
    "value passed to another function, stored in a container or computed before the comparison is not "
    "followed. A comparison with another field is listed under relations and is not a class edge. "
    "The classes under-approximate what the engine does with the field: a field read by name through a "
    "helper that takes the name as a string (for example _scheduler_config_int(vllm_config, "
    "\"max_num_batched_tokens\", 0)), and a threshold produced by min, range or arithmetic, are not found, so "
    "a field with one class is one that no scanned comparison singles out, not one the engine treats "
    "alike at every value."
)

ARGS_BASES = {"args", "engine_args", "parsed_args"}
ENGINE_ARG_CLASSES = ("EngineArgs", "AsyncEngineArgs")
INF = float("inf")


# ---------------------------------------------------------------- field types and owners

def snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class Source:
    """The parsed vLLM tree: config classes, EngineArgs, Literal aliases."""

    def __init__(self, root):
        self.root = root
        self.classes = gf.config_classes(root)
        self.engine_args = {}
        self.aliases = {}
        for rel in ("vllm/engine/arg_utils.py",):
            tree = gf.parse(os.path.join(root, rel))
            for n in tree.body:
                if isinstance(n, ast.ClassDef) and n.name in ENGINE_ARG_CLASSES:
                    for s in n.body:
                        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                            self.engine_args[s.target.id] = s
        # `scheduler_kwargs = get_kwargs(SchedulerConfig)` ... `**scheduler_kwargs["max_num_seqs"]`:
        # the class a flag's field is taken from.
        self.flag_class = {}
        au = gf.parse(os.path.join(root, "vllm", "engine", "arg_utils.py"))
        var_class = {}
        for n in ast.walk(au):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                    and isinstance(n.value, ast.Call) and ast.unparse(n.value.func).endswith("get_kwargs") and n.value.args:
                var_class[n.targets[0].id] = ast.unparse(n.value.args[0])
        for n in ast.walk(au):
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in var_class \
                    and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                self.flag_class[n.slice.value] = var_class[n.value.id]
        self.aliases = {}
        d = os.path.join(root, "vllm", "config")
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                for n in gf.parse(os.path.join(d, f)).body:
                    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                        v = n.value
                        if isinstance(v, ast.Subscript) and ast.unparse(v.value).endswith("Literal"):
                            self.aliases[n.targets[0].id] = v
        # An annotation that names an enum.Enum class of the config package (CUDAGraphMode) is an
        # enumeration of its member names, the strings the field is stated with.
        self.enums = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                for n in gf.parse(os.path.join(d, f)).body:
                    if isinstance(n, ast.ClassDef) and any(
                            ast.unparse(b).split(".")[-1] in ("Enum", "IntEnum", "StrEnum") for b in n.bases):
                        members = [t.id for st in n.body if isinstance(st, ast.Assign)
                                   for t in st.targets if isinstance(t, ast.Name) and not t.id.startswith("_")]
                        if members:
                            self.enums[n.name] = members
        self.bases = {}  # class name -> attribute names VllmConfig-like classes hold it under
        for node in self.classes.values():
            for fname, ann in gf.class_fields(node):
                sub = gf.named_class(ann, self.classes)
                if sub is not None:
                    self.bases.setdefault(sub, set()).add(fname)

    def frontend_flag(self, name):
        """The AnnAssign of a FrontendArgs field, or a synthetic one built from its add_argument call."""
        tree = gf.parse(os.path.join(self.root, "vllm", "entrypoints", "launchers", "cli_args.py"))
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and n.name in ("BaseFrontendArgs", "FrontendArgs"):
                for st in n.body:
                    if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == name:
                        return st
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument":
                lits = [a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                if lits and lits[0].lstrip("-").replace("-", "_") == name:
                    kw = {k.arg: ast.unparse(k.value) for k in n.keywords}
                    ann = kw.get("type") or ("bool" if kw.get("action", "").strip("'\"").startswith("store_") else "str")
                    return ast.parse("%s: %s" % (name, ann)).body[0]
        return None

    def class_field(self, cls, leaf):
        for s in self.classes[cls].body:
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id == leaf:
                return s
        return None

    def resolve(self, name):
        """(owner class or None, leaf in that class, EngineArgs name or None, AnnAssign node)."""
        parts = name.split(".")
        top = parts[0]
        ea = self.engine_args.get(top)
        cls = None
        if ea is None:
            # A front-end flag (FrontendArgs, or one make_arg_parser adds itself): read from the parsed
            # arguments, as args.<name>; it has no configuration class.
            fe = self.frontend_flag(name)
            if fe is None:
                raise SystemExit("gen_classes: %s is neither an EngineArgs field nor a front-end flag" % name)
            return None, name, name, fe
        if len(parts) == 1:
            val = ast.unparse(ea.value) if ea.value is not None else ""
            m = re.match(r"^([A-Z]\w+)\.(\w+)$", val)
            if m and m.group(1) in self.classes and self.class_field(m.group(1), m.group(2)) is not None:
                return m.group(1), m.group(2), top, self.class_field(m.group(1), m.group(2))
            c = self.flag_class.get(top)
            if c in self.classes and self.class_field(c, top) is not None:
                return c, top, top, self.class_field(c, top)
            return None, top, top, ea
        cls = gf.CONFIG_OF_DICT.get(top) or gf.named_class(ast.unparse(ea.annotation), self.classes)
        for p in parts[1:-1]:
            node = self.class_field(cls, p)
            cls = gf.named_class(ast.unparse(node.annotation), self.classes)
        node = self.class_field(cls, parts[-1])
        if node is None:
            raise SystemExit("gen_classes: %s: class %s has no field %s" % (name, cls, parts[-1]))
        return cls, parts[-1], None, node

    def literal_values(self, node, seen=()):
        """The values of a Literal[...] subscript, with a Literal alias in it expanded."""
        elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        out = []
        for e in elts:
            if isinstance(e, ast.Constant):
                out.append(e.value)
            elif isinstance(e, ast.Name) and e.id in self.aliases and e.id not in seen:
                out += self.literal_values(self.aliases[e.id], seen + (e.id,))
            else:
                raise SystemExit("gen_classes: cannot read the Literal element %s" % ast.unparse(e))
        return list(dict.fromkeys(out))

    def kind_of(self, ann):
        """(kind, enumeration or None, optional) from an annotation's text."""
        text = ast.unparse(ann)
        optional = bool(re.search(r"\bNone\b", text))
        for alias, node in self.aliases.items():
            if re.search(r"\b%s\b" % alias, text):
                return "enumerated", self.literal_values(node), optional
        m = re.search(r"Literal\[(.*)\]", text)
        if m:
            return "enumerated", list(ast.literal_eval("[" + m.group(1) + "]")), optional
        for enum, members in self.enums.items():
            if re.search(r"\b%s\b" % enum, text):
                return "enumerated", list(members), optional
        if re.search(r"\bbool\b", text):
            return "boolean", None, optional
        if re.search(r"\bint\b", text):
            return "numeric", "int", optional
        if re.search(r"\bfloat\b", text):
            return "numeric", "float", optional
        if re.search(r"\bstr\b", text):
            return "open", None, optional
        return "structured", None, optional


def domain_of(node, is_int=False):
    """Bounds a pydantic Field(ge=, gt=, le=, lt=) states for the field, as (lo, lo_open, hi, hi_open).
    On an integer field an open bound moves to the next integer, so the domain has closed bounds."""
    lo, lo_open, hi, hi_open = None, False, None, False
    if node is not None and isinstance(node.value, ast.Call) and ast.unparse(node.value.func).endswith("Field"):
        for k in node.value.keywords:
            if k.arg in ("ge", "gt", "le", "lt") and isinstance(k.value, ast.Constant):
                if k.arg == "ge":
                    lo, lo_open = k.value.value, False
                elif k.arg == "gt":
                    lo, lo_open = k.value.value, True
                elif k.arg == "le":
                    hi, hi_open = k.value.value, False
                else:
                    hi, hi_open = k.value.value, True
    if is_int:
        if lo is not None and lo_open:
            lo, lo_open = lo + 1, False
        if hi is not None and hi_open:
            hi, hi_open = hi - 1, False
    return lo, lo_open, hi, hi_open


# ---------------------------------------------------------------- finding comparisons

class Matcher:
    def __init__(self, src, name, owner, leaf, ea_name, env_name):
        self.src, self.name, self.owner, self.leaf, self.ea_name, self.env_name = src, name, owner, leaf, ea_name, env_name
        self.owner_bases = set()
        if owner is not None:
            self.owner_bases = set(src.bases.get(owner, set())) | {snake(owner)}

    def attr_ref(self, n, cls_name, ctx):
        """Is n `<owner>.<leaf>` (or the EngineArgs name on an args base)?"""
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr" and len(n.args) >= 2 \
                and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
            return self.attr_base(n.args[0], n.args[1].value, cls_name, ctx)
        if isinstance(n, ast.Attribute):
            return self.attr_base(n.value, n.attr, cls_name, ctx)
        return False

    def attr_base(self, base, attr, cls_name, ctx):
        if self.owner is not None and attr == self.leaf and self.is_owner(base, cls_name, ctx):
            return True
        if self.ea_name is not None and attr == self.ea_name and self.is_args(base, cls_name, ctx):
            return True
        return False

    def is_owner(self, base, cls_name, ctx):
        if isinstance(base, ast.Name):
            if base.id == "self":
                return cls_name == self.owner
            if base.id in ctx["owner_vars"]:
                return True
            return base.id in self.owner_bases
        if isinstance(base, ast.Attribute):
            return base.attr in self.owner_bases
        return False

    def is_args(self, base, cls_name, ctx):
        if isinstance(base, ast.Name):
            if base.id == "self":
                return cls_name in ENGINE_ARG_CLASSES
            return base.id in ARGS_BASES or base.id in ctx["args_vars"]
        return isinstance(base, ast.Attribute) and base.attr in ARGS_BASES

    def env_ref(self, n):
        v = self.env_name
        if v is None:
            return False
        if isinstance(n, ast.Attribute) and n.attr == v and isinstance(n.value, (ast.Name, ast.Attribute)) \
                and ast.unparse(n.value).split(".")[-1] == "envs":
            return True
        if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == v:
            f = ast.unparse(n.func)
            return f in ("os.environ.get", "os.getenv", "environ.get", "getenv", "envs.is_set")
        if isinstance(n, ast.Subscript) and ast.unparse(n.value).endswith("environ") \
                and isinstance(n.slice, ast.Constant) and n.slice.value == v:
            return True
        return False

    def is_ref(self, n, cls_name, ctx):
        if isinstance(n, ast.Name) and n.id in ctx["aliases"]:
            return True
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" \
                and n.attr in ctx["self_aliases"]:
            return True
        return self.attr_ref(n, cls_name, ctx) or self.env_ref(n)


ENUM_MEMBERS = {}  # (class, member) -> value, for every class body of constant assignments


def enum_members(root):
    if root not in ENUM_MEMBERS:
        table = {}
        for rel, text, consts, scope_list in parsed_tree(root):
            for scope, cls, nodes in scope_list:
                if isinstance(scope, ast.ClassDef):
                    bases = [ast.unparse(b) for b in scope.bases]
                    if any(re.search(r"Enum", b) for b in bases):
                        for st in scope.body:
                            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                                    and isinstance(st.value, ast.Constant):
                                table[(scope.name, st.targets[0].id)] = st.value.value
        ENUM_MEMBERS[root] = table
    return ENUM_MEMBERS[root]


def literal_values(n, consts):
    """The literal(s) an expression stands for, or None. A tuple, list or set of literals gives all."""
    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and (n.value.id, n.attr) in CURRENT_ENUMS:
        return [CURRENT_ENUMS[(n.value.id, n.attr)]]
    if isinstance(n, ast.Constant):
        return [n.value]
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant) \
            and isinstance(n.operand.value, (int, float)):
        return [-n.operand.value]
    if isinstance(n, (ast.Tuple, ast.List, ast.Set)):
        out = []
        for e in n.elts:
            v = literal_values(e, consts)
            if v is None or len(v) != 1:
                return None
            out += v
        return out
    if isinstance(n, ast.Name) and n.id in consts:
        return consts[n.id]
    return None


def module_consts(tree):
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            v = literal_values(n.value, {})
            if v is not None:
                out[n.targets[0].id] = v
    return out


def scopes(tree):
    """(scope node, enclosing class name or None) for the module, every class body and function."""
    out: list[tuple[ast.AST, str | None]] = [(tree, None)]

    def visit(node, cls):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                out.append((ch, ch.name))
                visit(ch, ch.name)
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                out.append((ch, cls))
                visit(ch, cls)
            else:
                visit(ch, cls)

    visit(tree, None)
    return out


def own_nodes(scope):
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        n = todo.pop()
        yield n
        if not isinstance(n, gf.SCOPES):
            todo.extend(ast.iter_child_nodes(n))


_PARSED = {}
CURRENT_ENUMS = {}


def parsed_tree(root):
    """(path, text, module constants, [(scope, class name, its own nodes)]) for every module, once."""
    if root not in _PARSED:
        out = []
        for p, rel in gf.walk(root, "vllm", (".py",)):
            try:
                text = read_text(p)
                tree = ast.parse(text, filename=p)
            except (SyntaxError, UnicodeDecodeError):
                continue
            out.append((rel, text, module_consts(tree), [(sc, cl, list(own_nodes(sc))) for sc, cl in scopes(tree)]))
        _PARSED[root] = out
    return _PARSED[root]


def scope_ctx(m, scope, cls, nodes, self_aliases):
    ctx = {"owner_vars": set(), "args_vars": set(), "aliases": set(), "self_aliases": self_aliases}
    # Parameters and variables annotated with the owner class.
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for a in scope.args.args + scope.args.kwonlyargs:
            if a.annotation is not None:
                t = ast.unparse(a.annotation)
                if m.owner and re.search(r"\b%s\b" % m.owner, t):
                    ctx["owner_vars"].add(a.arg)
                if re.search(r"\b(EngineArgs|AsyncEngineArgs)\b", t):
                    ctx["args_vars"].add(a.arg)
    for n in nodes:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            t = ast.unparse(n.annotation)
            if m.owner and re.search(r"\b%s\b" % m.owner, t):
                ctx["owner_vars"].add(n.target.id)
    # Names assigned from the field, or from an owner of it (twice, so a chain resolves).
    for _ in range(2):
        for n in nodes:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                v = n.value
                if m.is_ref(v, cls, ctx):
                    ctx["aliases"].add(n.targets[0].id)
                elif m.owner and isinstance(v, (ast.Attribute, ast.Name)) and m.is_owner(v, cls, ctx):
                    ctx["owner_vars"].add(n.targets[0].id)
            elif isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name) and m.is_ref(n.value, cls, ctx):
                ctx["aliases"].add(n.target.id)
    return ctx


def scan_scope(m, scope, cls, nodes, ctx, consts, rel, text, sites, relations):
    for n in nodes:
        if isinstance(n, ast.Compare):
            ops = [n.left] + n.comparators
            for i, op in enumerate(n.ops):
                a, b = ops[i], ops[i + 1]
                for x, y, flipped in ((a, b, False), (b, a, True)):
                    if not m.is_ref(x, cls, ctx):
                        continue
                    opname = type(op).__name__
                    if flipped:
                        opname = {"Lt": "Gt", "Gt": "Lt", "LtE": "GtE", "GtE": "LtE"}.get(opname, opname)
                    vals = literal_values(y, consts)
                    expr = ast.get_source_segment(text, n) or ast.unparse(n)
                    site = {"path": rel, "line": n.lineno, "expr": " ".join(expr.split()), "op": opname}
                    if vals is not None:
                        site["values"] = vals
                        sites.append(site)
                    elif not m.is_ref(y, cls, ctx):
                        site["other"] = " ".join((ast.get_source_segment(text, y) or ast.unparse(y)).split())
                        relations.append(site)
        elif isinstance(n, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            sites.extend(truth_sites(n.test, m, cls, ctx, rel, text))
        elif isinstance(n, ast.BoolOp):
            for v in n.values:
                sites.extend(truth_sites(v, m, cls, ctx, rel, text))


EXTRA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes_extra.json")


def reviewed_sites(root, name):
    """Comparisons a reviewer added because the scan cannot follow the value to them (it crosses a
    function call), from fields/classes_extra.json. Each states the line and the text found there;
    the generator stops when the pinned source no longer has that text on that line."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes_extra.json")
    if not os.path.exists(path):
        return []
    out = []
    for x in read_json(path).get(name, {}).get("sites", []):
        rel, line = x["at"].rsplit(":", 1)
        lines = read_text(os.path.join(root, rel)).splitlines()
        if x["expr"] not in lines[int(line) - 1]:
            raise SystemExit("gen_classes: %s: %s no longer has %r on line %s" % (name, rel, x["expr"], line))
        out.append({"path": rel, "line": int(line), "expr": x["expr"], "op": x["op"], "values": x["values"], "reviewed": True})
    return out


def each_scope(m, root, visit):
    """Call visit(scope, cls, nodes, ctx, consts, rel, text) for every scope of every module that can name
    the field (the file holds one of its names), with the aliases of that scope resolved."""
    CURRENT_ENUMS.clear()
    CURRENT_ENUMS.update(enum_members(root))
    tokens = [t for t in (m.leaf if m.owner else None, m.ea_name, m.env_name) if t]
    for rel, text, consts, scope_list in parsed_tree(root):
        if not any(t in text for t in tokens):
            continue
        # A class that stores the field in an attribute (self.x = scheduler_config.x) reads it as self.x.
        self_aliases = {}
        for _ in range(2):
            for scope, cls, nodes in scope_list:
                ctx = scope_ctx(m, scope, cls, nodes, self_aliases.setdefault(cls, set()))
                for n in nodes:
                    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute) \
                            and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "self" \
                            and m.is_ref(n.value, cls, ctx):
                        self_aliases[cls].add(n.targets[0].attr)
        for scope, cls, nodes in scope_list:
            ctx = scope_ctx(m, scope, cls, nodes, self_aliases.setdefault(cls, set()))
            visit(scope, cls, nodes, ctx, consts, rel, text)


def find_reads(m, root):
    """Every place the field is read: an attribute load of it (or of the EngineArgs name on an args
    object), getattr of it, or an environment read of the variable, as {"at", "to", "expr"} with the
    first and last line of the expression. A read through a name or attribute the function stored
    it in is the read that stored it. Reviewed sites (classes_extra.json) are not reads."""
    reads = []

    def visit(scope, cls, nodes, ctx, consts, rel, text):
        for n in nodes:
            if isinstance(n, (ast.Attribute, ast.Call, ast.Subscript)) and isinstance(getattr(n, "ctx", ast.Load()), ast.Load) \
                    and (m.attr_ref(n, cls, ctx) or m.env_ref(n)):
                reads.append({"at": "%s:%d" % (rel, n.lineno), "to": n.end_lineno or n.lineno,
                              "expr": " ".join((ast.get_source_segment(text, n) or ast.unparse(n)).split())})

    each_scope(m, root, visit)
    seen, out = set(), []
    for r in reads:
        k = (r["at"], r["to"], r["expr"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return sorted(out, key=lambda r: (site_key(r["at"]), r["expr"]))


def site_key(at):
    path, _, line = at.rpartition(":")
    return path, int(line)


def find_sites(m, root):
    """Every comparison with the field. A site is a dict: path, line, expr, op, values (or None)."""
    sites, relations = reviewed_sites(root, m.name), []

    def visit(scope, cls, nodes, ctx, consts, rel, text):
        scan_scope(m, scope, cls, nodes, ctx, consts, rel, text, sites, relations)

    each_scope(m, root, visit)
    seen, out = set(), []
    for s in sites:
        k = (s["path"], s["line"], s["expr"], s["op"], json.dumps(s.get("values")))
        if k not in seen:
            seen.add(k)
            out.append(s)
    seen, rel_out = set(), []
    for s in relations:
        k = (s["path"], s["line"], s["expr"])
        if k not in seen:
            seen.add(k)
            rel_out.append(s)
    key = lambda s: (s["path"], s["line"], s["expr"])  # noqa: E731
    return sorted(out, key=key), sorted(rel_out, key=key)


def truth_sites(test, m, cls, ctx, rel, text):
    out = []
    node = test
    negated = False
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        node, negated = node.operand, not negated
    if m.is_ref(node, cls, ctx):
        expr = ast.get_source_segment(text, test) or ast.unparse(test)
        out.append({"path": rel, "line": test.lineno, "expr": " ".join(expr.split()), "op": "Truth", "values": [not negated]})
    elif isinstance(node, ast.BoolOp):
        for v in node.values:
            out += truth_sites(v, m, cls, ctx, rel, text)
    return out


# ---------------------------------------------------------------- classes from sites

def cuts_from(sites, is_int):
    """Cut points of a numeric domain: (value, after) where after is 0 when the value belongs to the
    upper piece (x < v | x >= v) and 1 when it belongs to the lower (x <= v | x > v)."""
    cuts = {}  # (v, after) -> [site]

    def add(v, after, s):
        if is_int and after == 1 and float(v).is_integer():
            v, after = v + 1, 0
        cuts.setdefault((v, after), []).append(s)

    for s in sites:
        vals = [v for v in s.get("values", []) if isinstance(v, (int, float)) and not isinstance(v, bool)]
        for v in vals:
            op = s["op"]
            if op in ("Lt", "GtE"):
                add(v, 0, s)
            elif op in ("LtE", "Gt"):
                add(v, 1, s)
            elif op in ("Eq", "NotEq", "In", "NotIn"):
                add(v, 0, s)
                add(v, 1, s)
    return cuts


def numeric_classes(sites, is_int, dom):
    lo, lo_open, hi, hi_open = dom
    cuts = cuts_from(sites, is_int)
    keys = sorted(cuts)
    # A cut at or beyond the domain's own edge removes nothing, so it makes no class.
    def inside(k):
        v, after = k
        if lo is not None and (v < lo or (v == lo and (after == 0 or lo_open))):
            return False
        if hi is not None and (v > hi or (v == hi and (after == 1 or (after == 0 and hi_open)))):
            return False
        return True

    keys = [k for k in keys if inside(k)]
    classes = []
    bounds = [None] + keys + [None]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        c = {"lo": lo if a is None else a[0], "hi": hi if b is None else b[0]}
        if a is None:
            c["lo_closed"] = lo is not None and not lo_open
            if lo is None:
                c["lo_closed"] = False
        else:
            c["lo_closed"] = a[1] == 0
        if b is None:
            c["hi_closed"] = hi is not None and not hi_open
        else:
            c["hi_closed"] = b[1] == 1
        backed = []
        for k in (a, b):
            if k is not None:
                backed += cuts[k]
        # A point class (x == v): both edges are the same value.
        if c["lo"] is not None and c["hi"] is not None and c["lo"] == c["hi"] and not (c["lo_closed"] and c["hi_closed"]):
            continue
        c["compared_at"] = sorted({"%s:%d" % (s["path"], s["line"]) for s in backed})
        if not keys:
            # No comparison cuts the domain: one class, which no comparison backs.
            c["uncut"] = True
        classes.append(c)
    return classes


def at_of(sites):
    return sorted({"%s:%d" % (s["path"], s["line"]) for s in sites})


def unset_class(sites):
    none = [s for s in sites if s["op"] in ("Is", "IsNot", "Eq", "NotEq") and None in s.get("values", [])]
    return [{"unset": True, "compared_at": at_of(none)}] if none else []


def value_classes(sites, kind, enum, optional):
    """Classes of a boolean, enumerated, open-string or structured field. A value some comparison
    singles out is a class of its own, backed by that comparison. The values no comparison singles
    out are one class, which no comparison backs, because none tells them apart."""
    classes = []
    if kind == "boolean":
        enum = [False, True]
    singled = []
    for s in sites:
        if s["op"] in ("Eq", "NotEq", "In", "NotIn", "Is", "IsNot"):
            for x in s.get("values", []):
                if x is not None and x not in singled and (kind != "boolean" or isinstance(x, bool)):
                    singled.append(x)
    if kind == "boolean" and any(s["op"] == "Truth" for s in sites):
        singled = [False, True]
    singled.sort(key=repr)
    for x in singled:
        hit = [s for s in sites if s["op"] == "Truth" and kind == "boolean"
               or s["op"] in ("Eq", "NotEq", "In", "NotIn", "Is", "IsNot") and x in s.get("values", [])]
        classes.append({"values": [x], "compared_at": at_of(hit)})
    if kind in ("boolean", "enumerated"):
        rest = [x for x in enum if x not in singled]
        if rest:
            classes.append({"values": rest, "uncompared": True, "compared_at": []})
    else:
        classes.append({"other": True, "compared_at": []})
    if optional:
        classes += unset_class(sites)
    return classes


def class_of_value(classes, v):
    for i, c in enumerate(classes):
        if "values" in c and v in c["values"]:
            return i
        if "unset" in c and v is None:
            return i
    return None


def numeric_index(classes, v):
    for i, c in enumerate(classes):
        if "lo" not in c:
            continue
        lo, hi = c["lo"], c["hi"]
        ok_lo = lo is None or v > lo or (v == lo and c["lo_closed"])
        ok_hi = hi is None or v < hi or (v == hi and c["hi_closed"])
        if ok_lo and ok_hi:
            return i
    return None


def place(entry, v):
    """The index of the class value v falls in, or None; used by the evidence mapping and tests."""
    cl = entry["classes"]
    if entry["kind"] == "numeric":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        return numeric_index(cl, v)
    i = class_of_value(cl, v)
    if i is None and entry["kind"] in ("open", "enumerated"):
        for j, c in enumerate(cl):
            if c.get("other"):
                return j
    return i


# ---------------------------------------------------------------- evidence values

SCENARIO_ALIASES = {
    "kv_events": "kv_events_config.enable_kv_cache_events",
    "use_v2_model_runner": "VLLM_USE_V2_MODEL_RUNNER",
    "speculative.method": "speculative_config.method",
    "speculative.num_speculative_tokens": "speculative_config.num_speculative_tokens",
    "speculative.model": "speculative_config.model",
    "speculative.enable_adaptive_verification": "speculative_config.enable_adaptive_verification",
}


def scenario_values(repo, trace):
    """{field name: [values]} the scenario of a golden trace states, from oracle/scenarios*/<name>.json
    when it can be found; {} otherwise (a Go test or a YAML scenario states no machine-readable value).
    A key the scenario leaves out runs the runner's default and is not counted: covered_by lists the
    traces that state a value in the class, and so under-claims."""
    base = os.path.basename(trace)
    if not trace.endswith(".json") or "/testdata/" not in trace:
        return {}
    stem = base[:-5]
    hits = []
    for root, _, files in os.walk(os.path.join(repo, "oracle")):
        if stem + ".json" in files and "/scenarios" in root:
            hits.append(os.path.join(root, stem + ".json"))
    out = {}
    for h in sorted(hits):
        d = read_json(h)
        cfg = d.get("config", {}) if isinstance(d, dict) else {}
        flat = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[k + "." + k2] = v2
            flat[k] = v
        for k, v in flat.items():
            name = SCENARIO_ALIASES.get(k, k)
            if not isinstance(v, (dict, list)):
                out.setdefault(name, []).append(v)
        # kv transfer scenarios state the connector and role under "connector".
        conn = cfg.get("connector")
        if isinstance(conn, dict):
            for k, name in (("name", "kv_transfer_config.kv_connector"), ("kv_role", "kv_transfer_config.kv_role")):
                if k in conn:
                    out.setdefault(name, []).append(conn[k])
    return out


def reviewed_stated(name):
    """Values a trace's scenario states that no machine-readable scenario shows (a Go test, a YAML
    scenario, a fixture): [{"trace", "values", "why"}] from fields/classes_extra.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes_extra.json")
    if not os.path.exists(path):
        return []
    return read_json(path).get(name, {}).get("stated", [])


def project(e):
    """The value_classes object of the substrate's field list (contracts/spec/fields) for entry e:
    type, and for a boolean or enumerated field the values the evidence states, for a numeric field
    the thresholds and the classes the evidence states a value in."""
    stated = [x["value"] for x in e["stated"]]
    out: dict = {}
    if e["kind"] == "boolean":
        out = {"type": "boolean", "covered": sorted({"true" if v else "false" for v in stated if isinstance(v, bool)})}
    elif e["kind"] in ("enumerated", "open"):
        out = {"type": "enumerated"}
        if e["kind"] == "enumerated":
            # Required for a proven-inert field, whose probes must cover every value; optional for a modelled one.
            out["values"] = [str(v) for v in e["enumeration"]]
        out["covered"] = sorted({str(v) for v in stated if isinstance(v, str)})
    elif e["kind"] == "numeric":
        th = thresholds_of(e)
        out = {"type": "numeric"}
        if th:
            out["thresholds"] = th
        nums = [v for v in stated if isinstance(v, (int, float)) and not isinstance(v, bool)]
        out["covered_classes"] = sorted({class_of(th, v) for v in nums})
    else:
        raise SystemExit("gen_classes: %s: a %s field has no value classes in the substrate's schema" % (e.get("name"), e["kind"]))
    return out


def thresholds_of(e):
    """The thresholds between the numeric classes of e: where one class ends and the next begins, and
    which side takes the value on the line. Two thresholds at one value cannot be written (they must
    increase), which only an equality on a float field makes."""
    cl = [c for c in e["classes"] if "lo" in c]
    out = []
    for a, b in zip(cl, cl[1:]):
        out.append({"value": a["hi"], "closed": "below" if a["hi_closed"] else "above"})
    for x, y in zip(out, out[1:]):
        if y["value"] <= x["value"]:
            raise SystemExit("gen_classes: classes meet at %s twice (an equality on a float field); the schema cannot say it" % x["value"])
    return out


def class_of(thresholds, x):
    """ClassOf of contracts/spec/fields: the thresholds x is above, a value on the line counting as
    above when the threshold is closed above."""
    return sum(1 for t in thresholds if x > t["value"] or (x == t["value"] and t["closed"] == "above"))


# ---------------------------------------------------------------- the entry

def entry_for(src, root, name, field, repo):
    source = field["source"]
    is_env = source == "env"
    override = read_json(EXTRA_PATH).get(name, {}).get("kind") if os.path.exists(EXTRA_PATH) else None
    if is_env:
        owner, leaf, ea_name, node = None, name, None, None
        kind, extra, optional = env_kind(root, name)
        if override == "numeric":
            # A variable no code compares (or reads): one class, all its values (reviewed, with the reason in the file).
            kind, extra, optional = "numeric", "int", False
    else:
        owner, leaf, ea_name, node = src.resolve(name)
        kind, extra, optional = src.kind_of(node.annotation)
    m = Matcher(src, name, owner, leaf, ea_name, name if is_env else None)
    sites, relations = find_sites(m, root)
    e = {"kind": kind}
    if kind == "numeric":
        is_int = extra == "int"
        dom = domain_of(node, is_int)
        e["numeric_type"] = extra
        if dom[0] is not None or dom[2] is not None:
            e["domain"] = {"lo": dom[0], "lo_closed": dom[0] is not None and not dom[1],
                           "hi": dom[2], "hi_closed": dom[2] is not None and not dom[3]}
        classes = numeric_classes(sites, is_int, dom)
        if optional:
            classes += unset_class(sites)
    else:
        classes = value_classes(sites, kind, extra, optional)
    e["classes"] = classes
    if kind == "enumerated":
        e["enumeration"] = extra
    e["relations"] = [{"at": "%s:%d" % (s["path"], s["line"]), "expr": s["expr"], "with": s["other"]} for s in relations][:200]
    e["sites"] = [{"at": "%s:%d" % (s["path"], s["line"]), "expr": s["expr"], "op": s["op"],
                   "values": s.get("values", [])} for s in sites]
    # Which traces state which value (the scenario's own key), and so which class they cover.
    traces = field.get("evidence", {}).get("traces", [])
    stated, cov = [], {}
    if field["status"] == "proven_inert":
        # The probes of the proof are the values the evidence states; the Go test holds them to cover every class.
        proof = field["evidence"]["proof"].split()[0]
        for pr in read_json(os.path.join(repo, proof))["probes"]:
            stated.append({"value": pr["value"], "trace": proof})
    for t in traces:
        for v in scenario_values(repo, t).get(name, []):
            stated.append({"value": v, "trace": t})
    for x in reviewed_stated(name):
        if not os.path.exists(os.path.join(repo, x["trace"])):
            raise SystemExit("gen_classes: %s: reviewed trace %s does not exist" % (name, x["trace"]))
        for v in x["values"]:
            stated.append({"value": v, "trace": x["trace"], "reviewed": True})
    seen = set()
    e["stated"] = [x for x in stated if not ((json.dumps(x["value"]), x["trace"]) in seen or seen.add((json.dumps(x["value"]), x["trace"])))]
    for x in e["stated"]:
        i = place(e, x["value"])
        if i is not None:
            cov.setdefault(i, set()).add(x["trace"])
    for i, c in enumerate(e["classes"]):
        c["covered_by"] = sorted(cov.get(i, []))
    e["name"] = name
    return e


def env_kind(root, name):
    """The type of an environment variable from its parser in vllm/envs.py."""
    tree = gf.parse(os.path.join(root, "vllm", "envs.py"))
    text = read_text(os.path.join(root, "vllm", "envs.py"))
    for n in tree.body:
        target = n.target if isinstance(n, ast.AnnAssign) else (n.targets[0] if isinstance(n, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "environment_variables" and isinstance(n.value, ast.Dict):
            for k, v in zip(n.value.keys, n.value.values):
                if isinstance(k, ast.Constant) and k.value == name:
                    seg = ast.get_source_segment(text, v) or ""
                    if "bool(" in seg or "== \"1\"" in seg or ".lower()" in seg and "true" in seg.lower():
                        return "boolean", None, False
                    if "int(" in seg:
                        return "numeric", "int", False
                    if "float(" in seg:
                        return "numeric", "float", False
                    return "open", None, "None" in seg
    return "open", None, True


# ---------------------------------------------------------------- the document check

def edge_values(c, is_int):
    """The values a comparison can name to put an edge at c's lo or hi. On an integer field an edge
    at v+1 is made by x > v, so v and v-1 name it as well."""
    out = []
    for v in (c.get("lo"), c.get("hi")):
        if v is not None:
            out += [v, v - 1] if is_int else [v]
    return out


def check_document(doc, fl, root=None):
    """The problems of a classes document against the field list and, when root is given, the
    pinned source. The rules (VLM-3):
      - every modelled or proven-inert field has an entry with at least one class, and no other
        field has one;
      - the classes of a numeric field are in order, touch and do not overlap; those of a boolean
        or enumerated field hold each declared value once;
      - a class is backed by a comparison: each place it cites is a site of the entry, the value
        or edge it names is among that site's values, and (with the source) the cited line holds
        the site's text. A class that names no comparison is only one of three kinds: the values
        no comparison singles out (uncompared, other) or a domain no comparison cuts (uncut)."""
    problems = []
    want = {f["name"]: f for f in fl["fields"] if f["status"] in ("modelled", "proven_inert")}
    got = doc.get("fields", {})
    for n in sorted(want):
        if n not in got:
            problems.append("%s: a %s field has no value classes" % (n, want[n]["status"]))
    for n in sorted(got):
        if n not in want:
            problems.append("%s: has value classes but is neither modelled nor proven inert" % n)
    enums = enum_members(root) if root else {}
    cache = {}
    for n, rec in sorted(got.items()):
        e = rec.get("basis", {})
        cl = e.get("classes", [])
        if not cl:
            problems.append("%s: no classes" % n)
            continue
        if n in want:
            try:
                if project(e) != rec.get("value_classes"):
                    problems.append("%s: value_classes is not what its basis gives" % n)
            except SystemExit as x:
                problems.append("%s: %s" % (n, x))
            vc = rec.get("value_classes", {})
            if vc.get("type") == "numeric" and not vc.get("covered_classes"):
                problems.append("%s: the evidence states no value in any class (covered_classes is empty)" % n)
            if vc.get("type") != "numeric" and not vc.get("covered"):
                problems.append("%s: the evidence states no value (covered is empty)" % n)
        sites = {}
        for s in e.get("sites", []):
            sites.setdefault(s["at"], []).append(s)
        if e["kind"] == "numeric":
            num = [c for c in cl if "lo" in c]
            for a, b in zip(num, num[1:]):
                if a["hi"] is None or b["lo"] is None or a["hi"] != b["lo"] or a["hi_closed"] == b["lo_closed"]:
                    problems.append("%s: numeric classes %s and %s do not touch exactly" % (n, a, b))
            if num and num[0]["lo"] is not None and e.get("domain", {}).get("lo") != num[0]["lo"] and not e.get("domain"):
                problems.append("%s: the first class starts at %s but the field states no lower bound" % (n, num[0]["lo"]))
        else:
            seen = []
            for c in cl:
                for v in c.get("values", []):
                    if v in seen:
                        problems.append("%s: value %r is in two classes" % (n, v))
                    seen.append(v)
        for c in cl:
            flags = [k for k in ("other", "uncompared", "uncut") if c.get(k)]
            at = c.get("compared_at", [])
            if not at and not flags:
                problems.append("%s: class %s is backed by no comparison" % (n, {k: v for k, v in c.items() if k not in ("covered_by",)}))
            if flags and at and not c.get("uncut"):
                problems.append("%s: class %s is marked %s and cites a comparison" % (n, c, flags[0]))
            if c.get("uncut") and len(cl) != 1 + (1 if any("unset" in x for x in cl) else 0):
                problems.append("%s: an uncut class beside other classes" % n)
            for a in at:
                here = sites.get(a)
                if here is None:
                    problems.append("%s: class cites %s, which is not a comparison of the entry" % (n, a))
                    continue
                if "values" in c:
                    ok = any(s["op"] == "Truth" or set(map(repr, c["values"])) & set(map(repr, s["values"])) for s in here)
                elif "lo" in c:
                    ok = any(v in s["values"] for s in here for v in edge_values(c, e.get("numeric_type") == "int"))
                elif "unset" in c:
                    ok = any(None in s["values"] for s in here)
                else:
                    ok = True
                if not ok:
                    problems.append("%s: class %s cites %s, which compares %s" % (
                        n, c.get("values", [c.get("lo"), c.get("hi")]), a, [s["values"] for s in here]))
                if root:
                    rel, line = a.rsplit(":", 1)
                    if rel not in cache:
                        cache[rel] = read_text(os.path.join(root, rel))
                    window = " ".join(" ".join(cache[rel].splitlines()[int(line) - 1:int(line) + 8]).split())
                    for s in here:
                        if " ".join(s["expr"].split()) not in window:
                            problems.append("%s: %s no longer holds %r" % (n, a, s["expr"]))
                            continue
                        if s.get("reviewed") or s["op"] == "Truth":
                            continue
                        consts = module_consts(ast.parse(cache[rel]))
                        for v in s["values"]:
                            text = s["expr"]
                            member = any(enums.get((cn, mn)) == v and ("%s.%s" % (cn, mn)) in text for cn, mn in enums)
                            named = any(v in vals and re.search(r"\b%s\b" % re.escape(nm), text) for nm, vals in consts.items())
                            if not (repr(v) in text or str(v) in text or member or named):
                                problems.append("%s: %s does not mention the value %r it is said to compare" % (n, a, v))
    return problems


def reads_of(root, fields_path, name):
    """Every read of one field of the list, whatever its status (see find_reads)."""
    src = Source(root)
    fl = read_json(fields_path)
    for f in fl["fields"]:
        if f["name"] == name:
            if f["source"] == "env":
                m = Matcher(src, name, None, name, None, name)
            else:
                owner, leaf, ea_name, _ = src.resolve(name)
                m = Matcher(src, name, owner, leaf, ea_name, None)
            return find_reads(m, root)
    raise SystemExit("gen_classes: %s is not in the field list" % name)


def build(root, fields_path, repo, only=None):
    src = Source(root)
    fl = read_json(fields_path)
    out = {}
    for f in fl["fields"]:
        if only is not None:
            if f["name"] not in only:
                continue
        elif f["status"] not in ("modelled", "proven_inert"):
            continue
        basis = entry_for(src, root, f["name"], f, repo)
        out[f["name"]] = {"value_classes": project(basis), "basis": basis}
    return {"kind": "FieldValueClasses", "engine": fl["engine"], "limits": LIMITS, "fields": dict(sorted(out.items()))}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("fields")
    ap.add_argument("--check")
    ap.add_argument("--field", nargs="+")
    ap.add_argument("--reads", metavar="NAME", help="print every read of the field NAME (any status) as JSON")
    a = ap.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if a.reads:
        sys.stdout.write(json.dumps(reads_of(a.source, a.fields, a.reads), indent=1) + "\n")
        return 0
    got = build(a.source, a.fields, repo, set(a.field) if a.field else None)
    text = json.dumps(got, indent=1, sort_keys=False, ensure_ascii=False) + "\n"
    if a.check:
        if read_text(a.check) != text:
            sys.stderr.write("gen_classes: %s is not what the pinned source gives; regenerate it\n" % a.check)
            return 1
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
