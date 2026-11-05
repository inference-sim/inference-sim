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
An environment variable is named as it is set: every key of vllm/envs.py's
environment_variables, and every NCCL_*, NVSHMEM_* or UCX_* variable vLLM reads or sets by a
literal name anywhere under vllm/ (VLM-3 names these libraries). status.json's "extra_env" adds
other literal reads a reviewer decides belong in the list.
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

COMM_ENV = re.compile(r"^(NCCL|NVSHMEM|UCX)_[A-Z0-9_]+$")


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


def env_names(src, extra):
    tree = parse(os.path.join(src, "vllm", "envs.py"))
    keys = []
    for n in tree.body:
        target = n.target if isinstance(n, ast.AnnAssign) else (n.targets[0] if isinstance(n, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "environment_variables" and isinstance(n.value, ast.Dict):
            keys = [k.value for k in n.value.keys if isinstance(k, ast.Constant)]
    if not keys:
        raise SystemExit("gen_fields: vllm/envs.py has no environment_variables dict")
    names = set(keys) | set(extra)
    for name in sorted(literal_env_reads(src)):
        if COMM_ENV.match(name):
            names.add(name)
    return sorted(names)


ENV_CALLS = {"os.environ.get", "os.getenv", "environ.get", "os.environ.setdefault", "os.environ.pop", "getenv"}


def literal_env_reads(src):
    """Every environment variable read or set by a literal name under vllm/, with its files."""
    out = {}
    for root, dirs, files in os.walk(os.path.join(src, "vllm")):
        dirs.sort()
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            for node in ast.walk(parse(p)):
                name = None
                if isinstance(node, ast.Call) and ast.unparse(node.func) in ENV_CALLS and node.args:
                    a = node.args[0]
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        name = a.value
                elif isinstance(node, ast.Subscript) and ast.unparse(node.value) in ("os.environ", "environ"):
                    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                        name = node.slice.value
                elif (isinstance(node, ast.Compare) and len(node.comparators) == 1
                      and isinstance(node.ops[0], (ast.In, ast.NotIn))
                      and ast.unparse(node.comparators[0]) in ("os.environ", "environ")
                      and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)):
                    name = node.left.value
                if name:
                    out.setdefault(name, set()).add(os.path.relpath(p, src))
    return out


def key(source, name):
    return source + ":" + name


def build(src, status):
    derived = [("config", n) for n in config_names(src)] + [("env", n) for n in env_names(src, status.get("extra_env", []))]
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
    ap.add_argument("--env-reads", action="store_true", help="print every literal env read under vllm/ and exit")
    a = ap.parse_args(argv)
    if a.names:
        for n in config_names(a.vllm_source):
            print("config:" + n)
        for n in env_names(a.vllm_source, []):
            print("env:" + n)
        return 0
    if a.env_reads:
        for n, files in sorted(literal_env_reads(a.vllm_source).items()):
            print(n, " ".join(sorted(files)))
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
