# Original to blis-sglang.
"""Write fields/value_classes.json: the value classes of each modelled or proven-inert field (VLM-3).

VLM-3, which SGL-3 applies: "a field marked modelled records the value classes its evidence
covers: for a boolean or enumerated field, the values its evidence sets; for an ordered numeric
field, the range its evidence sets, split at every threshold the upstream compares the field with,
generated from the upstream's comparisons. The probe values of a proven-inert field follow the
same classes." The method is blis-vllm's fields/gen_classes.py, applied to SGLang's sources.

This reads SGLang's source at the pinned commit with the ast module (SGLang is not imported) and,
for each field of fields.json whose status is modelled or proven_inert:

  1. takes the field's type from fields/declarations.json (the ServerArgs declaration, or the
     Env* class of an Envs field): bool, an enumeration (its choices), a number, or a free string;
  2. finds every comparison the scanned Python source (gen_fields.SCANNED_PY) makes with the
     field: `field <op> literal`, `field in (literals)`, `field is None`, and a bare truth test
     (`if field`, `not field`, `a or field`, `assert field`);
  3. cuts the field's domain at those comparisons and writes one class per piece, each with the
     file:line of every comparison that forms its edge ("compared_at"). A class no comparison
     backs is not written, except the one class of a field no comparison cuts ("uncut");
  4. lists the evidence traces whose content sets a value in the class ("covered_by").

What counts as the field. For a ServerArgs field, `<base>.<name>` or getattr(<base>, "<name>"),
where <base> is a name or attribute whose last part is one of CONFIG_BASES (server_args, args,
cfg, view, ...: the names SGLang's hooks, managers and workers give a ServerArgs or a resolving
view of one), or `self` inside server_args.py and arg_groups/. For an environment variable,
envs.NAME.get(), os.environ.get("NAME"), os.getenv("NAME"), os.environ["NAME"],
get_bool_env_var("NAME") and get_int_env_var("NAME"). Also a name the same function assigned from
one of those. Not followed: a value passed into another function, stored in a container, or
computed with arithmetic before the comparison. The limit is stated in the file's "limits".

Usage:
    python3 -I fields/gen_classes.py <sglang-source> fields.json fields/declarations.json > fields/value_classes.json
    python3 -I fields/gen_classes.py <sglang-source> fields.json fields/declarations.json --check fields/value_classes.json
    python3 -I fields/gen_classes.py <sglang-source> fields.json fields/declarations.json --field NAME ...
"""

import argparse
import ast
import difflib
import importlib.util
import json
import os
import sys


def _load_sibling(name):
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
    "followed, and a ServerArgs field is recognised only through the base names listed in CONFIG_BASES. "
    "The classes under-approximate what SGLang does with the field: a field with one class is one no "
    "scanned comparison singles out, not one SGLang treats alike at every value."
)

CONFIG_BASES = {"server_args", "args", "cfg", "view", "sa", "server_cfg", "resolved", "srv_args"}
SELF_SCOPES = (gf.SERVER_ARGS, gf.SRT + "/arg_groups/")
ENV_GETS = {"os.getenv", "os.environ.get", "getenv", "environ.get", "get_bool_env_var", "get_int_env_var"}
INF = float("inf")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Field:
    def __init__(self, name, source, decl):
        self.name, self.source, self.decl = name, source, decl or {}

    def kind(self):
        """boolean, enumerated, numeric or open."""
        if self.source == "env":
            t = self.decl.get("type", "")
            return {"EnvBool": "boolean", "EnvBoolWithAlias": "boolean", "EnvInt": "numeric", "EnvIntWithAlias": "numeric",
                    "EnvFloat": "numeric"}.get(t, "open")
        if self.decl.get("choices"):
            return "enumerated"
        t = self.decl.get("type", "").replace("Optional[", "").rstrip("]")
        if t == "bool":
            return "boolean"
        if t in ("int", "float"):
            return "numeric"
        return "open"

    def numeric_type(self):
        t = self.decl.get("type", "")
        return "float" if ("float" in t or self.decl.get("type") == "EnvFloat") else "int"


def is_field_expr(node, field, rel, aliases):
    """Whether node reads the field (see the module docstring)."""
    if isinstance(node, ast.Name):
        return node.id in aliases
    if field.source == "config":
        if isinstance(node, ast.Attribute) and node.attr == field.name:
            base = node.value
            last = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else None
            if last in CONFIG_BASES:
                return True
            return last == "self" and any(rel == s or (s.endswith("/") and rel.startswith(s)) for s in SELF_SCOPES)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2:
            a = node.args[1]
            if isinstance(a, ast.Constant) and a.value == field.name:
                b = node.args[0]
                last = b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else None
                return last in CONFIG_BASES
        return False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and not node.args:
        v = node.func.value
        if isinstance(v, ast.Attribute) and v.attr == field.name and ast.unparse(v.value) in ("envs", "environ.envs"):
            return True
    fn = ast.unparse(node.func) if isinstance(node, ast.Call) else ""
    if fn and (fn in ENV_GETS or fn.endswith((".get_bool_env_var", ".get_int_env_var"))) and node.args:
        a = node.args[0]
        return isinstance(a, ast.Constant) and a.value == field.name
    if isinstance(node, ast.Subscript) and ast.unparse(node.value) in ("os.environ", "environ"):
        return isinstance(node.slice, ast.Constant) and node.slice.value == field.name
    return False


def const_value(node):
    ok, v = gf.literal(node)
    if ok and (v is None or isinstance(v, (bool, int, float, str))):
        return True, v
    if ok and isinstance(v, (list, tuple, set, frozenset)) and all(isinstance(x, (bool, int, float, str)) or x is None for x in v):
        return True, list(v)
    return False, None


def comparisons(src, field):
    """[(site, op, value, expr)] for every comparison of the field in the scanned Python source."""
    out = []
    for path, rel in gf.walk(src, gf.SCANNED_PY, (".py",)):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if field.name not in text:
            continue
        tree = ast.parse(text, filename=rel)
        parent = {}
        for n in ast.walk(tree):
            for c in ast.iter_child_nodes(n):
                parent[c] = n
        fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] + [tree]
        for fn in fns:
            body_nodes = list(gf.scope_nodes(fn))
            aliases = set()
            for n in body_nodes:
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                        and is_field_expr(n.value, field, rel, set()):
                    aliases.add(n.targets[0].id)
            for n in body_nodes:
                site = "%s:%d" % (rel, getattr(n, "lineno", 0))
                if isinstance(n, ast.Compare) and len(n.ops) == 1:
                    left, right, op = n.left, n.comparators[0], type(n.ops[0]).__name__
                    if is_field_expr(left, field, rel, aliases):
                        ok, v = const_value(right)
                    elif is_field_expr(right, field, rel, aliases):
                        ok, v = const_value(left)
                        op = {"Lt": "Gt", "Gt": "Lt", "LtE": "GtE", "GtE": "LtE"}.get(op, op)
                        if op in ("In", "NotIn"):
                            ok = False
                    else:
                        continue
                    if ok:
                        out.append((site, op, tuple(v) if isinstance(v, list) else v, ast.unparse(n)))
                    continue
                tests = []
                if isinstance(n, (ast.If, ast.While, ast.IfExp, ast.Assert)):
                    tests = [n.test]
                elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
                    tests = [n.operand]
                elif isinstance(n, ast.BoolOp):
                    tests = n.values
                for t in tests:
                    if is_field_expr(t, field, rel, aliases):
                        out.append((site, "Truth", None, ast.unparse(t)))
    return [(s, op, list(v) if isinstance(v, tuple) else v, e)
            for s, op, v, e in sorted(set(out), key=lambda s: (gf.site_order(s[0]), s[1], repr(s[2]), s[3]))]


def stated_values(repo, field, evidence):
    """[(value, trace)] for each evidence trace (JSON) that sets the field by its name."""
    out = []
    for tr in evidence.get("traces", []):
        try:
            doc = read_json(os.path.join(repo, tr))
        except (OSError, ValueError):
            continue
        todo = [doc]
        while todo:
            d = todo.pop()
            if isinstance(d, dict):
                for k, v in d.items():
                    if k == field.name and (v is None or isinstance(v, (bool, int, float, str))):
                        out.append((v, tr))
                    todo.append(v)
            elif isinstance(d, list):
                todo.extend(d)
    return out


def numeric_classes(sites, ntype):
    cuts = {}
    for site, op, v, _ in sites:
        vals = v if isinstance(v, list) else [v]
        for x in vals:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                continue
            if op in ("Lt", "GtE"):
                cuts.setdefault((x, "above"), set()).add(site)
            elif op in ("LtE", "Gt"):
                cuts.setdefault((x, "below"), set()).add(site)
            elif op in ("Eq", "NotEq", "In", "NotIn"):
                cuts.setdefault((x, "above"), set()).add(site)
                if ntype == "int":
                    cuts.setdefault((x + 1, "above"), set()).add(site)
                else:
                    cuts.setdefault((x, "below"), set()).add(site)
    # A cut is a threshold: (v, "above") puts v in the class above it, (v, "below") in the class
    # below. For an int, v "below" is v + 1 "above". For a float, a value cut on both sides keeps
    # one threshold ("above"), so the thresholds stay strictly increasing as the substrate asks.
    norm = {}
    for (x, side), at in cuts.items():
        if side == "below" and (ntype == "int" or (x, "above") in cuts):
            x, side = (x + 1, "above") if ntype == "int" else (x, "above")
        norm.setdefault((x, side), set()).update(at)
    cuts = norm
    order = sorted(cuts, key=lambda c: (c[0], 0 if c[1] == "below" else 1))
    # Two cuts at one value (below and above) leave the single point between them as its own class.
    classes, lo, lo_closed, lo_sites = [], None, False, set()
    for value, side in order:
        hi_closed = side == "below"
        classes.append({"lo": lo, "hi": value, "lo_closed": lo_closed, "hi_closed": hi_closed,
                        "compared_at": sorted(lo_sites | cuts[(value, side)], key=gf.site_order)})
        lo, lo_closed, lo_sites = value, side == "above", cuts[(value, side)]
    classes.append({"lo": lo, "hi": None, "lo_closed": lo_closed, "hi_closed": False,
                    "compared_at": sorted(lo_sites, key=gf.site_order)})
    if not cuts:
        classes[0]["uncut"] = True
    thresholds = [{"value": v, "closed": s} for v, s in order]
    return classes, thresholds


def in_class(c, x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    if c["lo"] is not None and (x < c["lo"] or (x == c["lo"] and not c["lo_closed"])):
        return False
    if c["hi"] is not None and (x > c["hi"] or (x == c["hi"] and not c["hi_closed"])):
        return False
    return True


def classes_for(src, repo, field, evidence):
    """(value_classes, basis) of one field."""
    sites = comparisons(src, field)
    stated = stated_values(repo, field, evidence)
    kind = field.kind()
    basis = {"name": field.name, "kind": kind, "sites": [{"at": s, "op": op, "values": v if isinstance(v, list) else [v], "expr": e}
                                                        for s, op, v, e in sites], "stated": [{"value": v, "trace": t} for v, t in stated]}
    truth = sorted({s for s, op, _, _ in sites if op == "Truth"}, key=gf.site_order)
    if kind == "boolean":
        classes = []
        for b in (False, True):
            at = sorted(set(truth) | {s for s, op, v, _ in sites if v == b or (isinstance(v, list) and b in v)}, key=gf.site_order)
            classes.append({"values": [b], "compared_at": at, "covered_by": sorted({t for v, t in stated if v is b})})
        basis["classes"] = classes
        vc = {"type": "boolean", "covered": sorted({"true" if v else "false" for v, _ in stated if isinstance(v, bool)})}
    elif kind == "numeric":
        classes, thresholds = numeric_classes(sites, field.numeric_type())
        for c in classes:
            c["covered_by"] = sorted({t for v, t in stated if in_class(c, v)})
        basis["numeric_type"] = field.numeric_type()
        basis["classes"] = classes
        vc = {"type": "numeric", "covered_classes": [i for i, c in enumerate(classes) if c["covered_by"]]}
        if thresholds:
            vc["thresholds"] = thresholds
    else:
        choices = [str(c) for c in field.decl.get("choices", [])]
        compared = {}
        for s, op, v, _ in sites:
            for x in (v if isinstance(v, list) else [v]):
                compared.setdefault(x, set()).add(s)
        values = choices or sorted({str(x) for x in compared if x is not None})
        classes = [{"values": [v], "compared_at": sorted(compared.get(v, set()), key=gf.site_order),
                    "covered_by": sorted({t for x, t in stated if str(x) == v})} for v in values]
        if None in compared or truth:
            classes.append({"unset": True, "compared_at": sorted(compared.get(None, set()) | set(truth), key=gf.site_order),
                            "covered_by": sorted({t for x, t in stated if x is None})})
        if not choices:
            classes.append({"other": True, "compared_at": [], "covered_by": sorted({t for x, t in stated if x is not None and str(x) not in values})})
        basis["classes"] = classes
        vc = {"type": "enumerated", "covered": sorted({str(x) for x, _ in stated if x is not None})}
        if choices:
            vc["values"] = choices
    return vc, basis


def document(src, repo, fields_json, decls):
    fl = read_json(fields_json)
    out = {}
    for f in fl["fields"]:
        if f["status"] not in ("modelled", "proven_inert"):
            continue
        decl = (decls["server_args"].get(f["name"]) if f["source"] == "config" else decls["env"].get(f["name"]))
        vc, basis = classes_for(src, repo, Field(f["name"], f["source"], decl), f["evidence"])
        out[f["name"]] = {"value_classes": vc, "basis": basis}
    return {"kind": "ValueClasses", "engine": fl["engine"], "limits": LIMITS, "fields": dict(sorted(out.items()))}


def check_document(doc, fields_json):
    """The rules a value_classes.json obeys against fields.json, as problems."""
    fl = read_json(fields_json)
    want = {f["name"] for f in fl["fields"] if f["status"] in ("modelled", "proven_inert")}
    probs = ["%s: a modelled or proven-inert field has no value classes" % n for n in sorted(want - set(doc["fields"]))]
    probs += ["%s: has value classes but is neither modelled nor proven inert" % n for n in sorted(set(doc["fields"]) - want)]
    for n, rec in sorted(doc["fields"].items()):
        b = rec["basis"]
        sites = {s["at"] for s in b["sites"]}
        for i, c in enumerate(b["classes"]):
            if not c["compared_at"] and not (c.get("uncut") or c.get("other")):
                probs.append("%s: class %d is backed by no comparison" % (n, i))
            for a in c["compared_at"]:
                if a not in sites:
                    probs.append("%s: class %d cites %s, which is not a comparison of the field" % (n, i, a))
    return probs


def render(doc):
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sglang_source")
    ap.add_argument("fields_json")
    ap.add_argument("declarations_json")
    ap.add_argument("--check", metavar="VALUE_CLASSES_JSON")
    ap.add_argument("--field", action="append", help="print the classes of any field, whatever its status")
    ap.add_argument("--repo", default=".", help="the blis-sglang checkout the evidence paths are in")
    a = ap.parse_args(argv)
    decls = read_json(a.declarations_json)
    if a.field:
        fl = {f["name"]: f for f in read_json(a.fields_json)["fields"]}
        out = {}
        for n in a.field:
            if n not in fl:
                raise SystemExit("gen_classes: %s is not in %s" % (n, a.fields_json))
            f = fl[n]
            decl = decls["server_args"].get(n) if f["source"] == "config" else decls["env"].get(n)
            vc, basis = classes_for(a.sglang_source, a.repo, Field(n, f["source"], decl), f["evidence"])
            out[n] = {"value_classes": vc, "basis": basis}
        sys.stdout.write(render(out))
        return 0
    doc = document(a.sglang_source, a.repo, a.fields_json, decls)
    probs = check_document(doc, a.fields_json)
    if probs:
        raise SystemExit("gen_classes: " + "\ngen_classes: ".join(probs))
    out = render(doc)
    if a.check:
        with open(a.check, encoding="utf-8") as f:
            have = f.read()
        if have != out:
            sys.stdout.writelines(difflib.unified_diff(have.splitlines(True), out.splitlines(True), a.check, "generated"))
            print("gen_classes: %s is not what the generator writes; regenerate it" % a.check, file=sys.stderr)
            return 1
        return 0
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
