"""Fail if an oracle harness file replaces upstream code that oracle/mocks.json does not list, if
mocks.json lists decision code, or if a harness file copies an upstream decision function that
the manifest does not declare as a transcription (FID-1, FID-8).

The oracle may replace the GPU, the clock, transports, the environment, the platform and outside
services, never a scheduling, KV-cache or connector decision. The checks:

1. Every entry's kind is one of KINDS, and no entry's symbol lies under one of the engine's
   decision modules (the `decision_prefixes` of oracle/harness.json). A clock inside one is
   allowed: the clock is the environment even where a decision reads it. A transcription is
   allowed too, because declaring the copy is what a transcription entry is for.

2. The harness files are read, not trusted. Each file of the oracle directory that matches one
   of the `harness_files` patterns is parsed, and every place it writes into an object it did not
   define is found (Sites below): attribute and item stores, setattr, delattr and del, and objects
   built with object.__new__ instead of their constructor. Each such site must be named, as
   "<file>:<target>", in the `sites` of a mocks.json entry, so a patch nobody listed fails CI, and
   an entry whose sites no harness file makes fails too. A setting changed after an upstream
   config was constructed is a site as well: the constructor's own checks do not see it.

3. The version names the list. mocks.json's version is "gpu-mock/<n>+<digest>", where the
   digest is the first 12 hex digits of the sha256 of the canonical JSON of its `mocks` list
   (mocks_digest). A trace records the version, and the engine's diff tests refuse a trace whose
   version is not the current one, so a mock list that changes without its version changing
   fails here, and one whose version changes makes every golden be refreshed.

4. A copy is declared (FID-8). A harness file whose leading comment block has a "Derived from"
   line, or that defines a function named like an upstream function (the `upstream_names` of
   oracle/harness.json), is refused unless an entry of kind "transcription" names it in its
   `transcribed_in` list, as "<file>:<function>" (or "<file>" for the header). A transcription
   entry names the upstream code it copies in `upstream`, and each place it names must exist.

oracle/harness.json is the engine's configuration of these checks; it is read from the directory
of this script unless --config names another file.

Usage: python3 -I oracle/check_mocks.py [--config <harness.json>] [oracle/mocks.json]
"""
import ast
import fnmatch
import hashlib
import json
import os
import sys

KINDS = {"gpu", "clock", "transport", "environment", "platform", "external-service", "transcription"}

# The keys of oracle/harness.json and what each must hold. Every key is required and no other is
# allowed. A string must not be empty, and a list must hold non-empty strings. golden_files may be
# an empty list (an engine with no golden file outside its golden directories); every other list
# must name something, because a check given nothing to compare passes on nothing (X16).
CONFIG_SCHEMA = {
    "commit_field": "string",
    "ci_platform": "string",
    "golden_dirs": "list",
    "golden_files": "list, may be empty",
    "harness_files": "list",
    "decision_prefixes": "list",
    "upstream_names": "list",
}


def config_problems(cfg) -> list[str]:
    """What is wrong with a parsed oracle/harness.json, each naming its key."""
    if not isinstance(cfg, dict):
        return [f"the configuration is a {type(cfg).__name__}, not an object"]
    bad = [f"{k}: missing" for k in sorted(set(CONFIG_SCHEMA) - set(cfg))]
    bad += [f"{k}: not a key of the configuration" for k in sorted(set(cfg) - set(CONFIG_SCHEMA))]
    for key, kind in sorted(CONFIG_SCHEMA.items()):
        if key not in cfg:
            continue
        v = cfg[key]
        if kind == "string":
            if not isinstance(v, str) or not v.strip():
                bad.append(f"{key}: must be a non-empty string, not {v!r}")
            continue
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
            bad.append(f"{key}: must be a list of non-empty strings, not {v!r}")
        elif not v and kind == "list":
            bad.append(f"{key}: is empty; the check it configures would pass on nothing")
    if isinstance(cfg.get("golden_dirs"), list):
        slashless = [d for d in cfg["golden_dirs"] if isinstance(d, str) and not d.endswith("/")]
        if slashless:
            bad.append(f"golden_dirs: each entry must end with '/': {slashless}")
    return bad


def load_config(path: str) -> dict:
    """Read oracle/harness.json, or exit naming every key that is wrong."""
    with open(path) as f:
        cfg = json.load(f)
    bad = config_problems(cfg)
    if bad:
        raise SystemExit(f"{path}: " + "; ".join(bad))
    return cfg


def _root(node):
    """The name at the root of an attribute chain (a.b.c -> a), or None."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _chain(node) -> str:
    return ast.unparse(node)


class Sites(ast.NodeVisitor):
    """Every place one harness file writes into an object it did not define: an attribute or item
    store on anything but `self` (a harness file's own helper class), a setattr or delattr, a del of
    an attribute, and an object built with object.__new__. The harness writes only into upstream
    modules, classes and objects, so this is default-deny: a write the list does not name fails,
    whatever the object."""

    def __init__(self):
        self.sites: dict[str, int] = {}
        # The name each site binds, when the value written is a plain name (a function the
        # harness defines, for example); None otherwise.
        self.values: dict[str, str | None] = {}

    def add(self, target: str, line: int, value=None) -> None:
        self.sites.setdefault(target, line)
        self.values.setdefault(target, value.id if isinstance(value, ast.Name) else None)

    def _store(self, t, line, value=None):
        if isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._store(e, line)
        elif isinstance(t, ast.Attribute) and _root(t) != "self":
            self.add(_chain(t), line, value)
        elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) and _root(t.value) != "self":
            self.add(_chain(t.value) + "[]", line)

    def visit_Assign(self, node):
        for t in node.targets:
            self._store(t, node.lineno, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._store(node.target, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._store(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Delete(self, node):
        for t in node.targets:
            self._store(t, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node):
        fn = _chain(node.func)
        if fn in ("setattr", "delattr") and node.args and _root(node.args[0]) != "self":
            name = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "?"
            self.add(f"{_chain(node.args[0])}.{name}", node.lineno, node.args[2] if len(node.args) > 2 else None)
        if fn == "object.__new__" and node.args:
            self.add(f"object.__new__({_chain(node.args[0])})", node.lineno)
        self.generic_visit(node)


def harness_files(oracle_dir: str, patterns: list) -> list[str]:
    """The files of the oracle directory (not its subdirectories) that the patterns name."""
    return sorted(n for n in os.listdir(oracle_dir)
                  if n.endswith(".py") and any(fnmatch.fnmatchcase(n, p) for p in patterns))


def parse(oracle_dir: str, name: str):
    with open(os.path.join(oracle_dir, name)) as f:
        src = f.read()
    return src, ast.parse(src, name)


def harness_sites(oracle_dir: str, names: list[str]) -> tuple[dict[str, int], dict[str, str | None]]:
    """Every site the harness files make, as {"<file>:<target>": line}, and the name each binds."""
    out, values = {}, {}
    for name in names:
        _, tree = parse(oracle_dir, name)
        v = Sites()
        v.visit(tree)
        for target, line in v.sites.items():
            out[f"{name}:{target}"] = line
            values[f"{name}:{target}"] = v.values.get(target)
    return out, values


def derived_header(src: str) -> bool:
    """Whether the file's leading comment block has a "Derived from" line."""
    for line in src.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            return False
        if s.lstrip("#").strip().startswith("Derived from"):
            return True
    return False


def functions(tree) -> dict[str, int]:
    """Every function the file defines, at any depth, with its first line."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, node.lineno)
    return out


def copies(oracle_dir: str, names: list[str], upstream_names: set) -> tuple[dict[str, str], dict[str, set]]:
    """The copies FID-8 asks to be declared, as {"<file>[:<function>]": why}, and every function
    each harness file defines."""
    found, defined = {}, {}
    for name in names:
        src, tree = parse(oracle_dir, name)
        fns = functions(tree)
        defined[name] = set(fns)
        if derived_header(src):
            found[name] = "has a Derived from header"
        for fn, line in sorted(fns.items()):
            if fn in upstream_names:
                found[f"{name}:{fn}"] = f"defines {fn} (line {line}), the name of an upstream function"
    return found, defined


def mocks_digest(mocks: list) -> str:
    """The first 12 hex digits of the sha256 of the mock list's canonical JSON."""
    canonical = json.dumps(mocks, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def main(path: str, config: str) -> int:
    cfg = load_config(config)
    with open(path) as f:
        spec = json.load(f)
    oracle_dir = os.path.dirname(os.path.abspath(path))
    names = harness_files(oracle_dir, cfg["harness_files"])
    bad = []
    if not names:
        bad.append(f"no file of {oracle_dir} matches harness_files {cfg['harness_files']}; the check would read nothing")
    version = spec.get("version", "")
    digest = mocks_digest(spec["mocks"])
    if not version.startswith("gpu-mock/") or not version.endswith("+" + digest):
        bad.append(f"version {version!r} does not name this mock list: it must be gpu-mock/<n>+{digest} "
                   "(bump <n> when the list changes; the digest is mocks_digest of the list)")
    prefixes = tuple(cfg["decision_prefixes"])
    found_copies, defined = copies(oracle_dir, names, set(cfg["upstream_names"]))
    listed: dict[str, str] = {}
    declared: set[str] = set()
    for m in spec["mocks"]:
        sym, kind = m["symbol"], m["kind"]
        if kind not in KINDS:
            bad.append(f"{sym}: kind {kind!r} is not one of {sorted(KINDS)}")
        if kind not in ("clock", "transcription") and sym.startswith(prefixes):
            bad.append(f"{sym}: replaces decision code ({kind})")
        if kind == "transcription":
            if not str(m.get("upstream", "")).strip():
                bad.append(f"{sym}: a transcription names the upstream code it copies in `upstream`")
            if not m.get("transcribed_in"):
                bad.append(f"{sym}: a transcription names where the harness holds the copy in `transcribed_in`")
            for t in m.get("transcribed_in", []):
                file, _, fn = t.partition(":")
                if file not in defined or (fn and fn not in defined[file]):
                    bad.append(f"{sym}: transcribed_in names {t}, which no harness file defines")
                declared.add(t)
        elif m.get("transcribed_in"):
            bad.append(f"{sym}: only a transcription has transcribed_in ({kind})")
        for s in m.get("sites", []):
            listed[s] = sym
    found, bound = harness_sites(oracle_dir, names)
    # A transcription of decision code may write only its declared copy into the upstream: each
    # of its sites binds a function its transcribed_in lists.
    for m in spec["mocks"]:
        if m["kind"] != "transcription" or not m["symbol"].startswith(prefixes):
            continue
        for site in m.get("sites", []):
            file = site.split(":", 1)[0]
            value = bound.get(site)
            if value is None or f"{file}:{value}" not in m.get("transcribed_in", []):
                bad.append(f"{m['symbol']}: transcription site {site} binds {value or 'something other than a name'}, "
                           "not a function its transcribed_in lists (a transcription installs its copy, nothing else)")
    for site, line in sorted(found.items()):
        if site not in listed:
            file = site.split(":", 1)[0]
            bad.append(f"{file}:{line}: writes {site.split(':', 1)[1]}, which is not listed in mocks.json "
                       "(add it to the sites of the entry that justifies it, or remove the patch)")
    for site, sym in sorted(listed.items()):
        if site not in found:
            bad.append(f"{sym}: lists site {site}, which no harness file makes (remove it)")
    for where in sorted(declared - set(found_copies)):
        bad.append(f"{where}: declared in transcribed_in, but check_mocks finds no copy there (no Derived from "
                   "header, or a function not named in upstream_names); remove the declaration or name the copy")
    for where, why in sorted(found_copies.items()):
        if where not in declared:
            bad.append(f"{where}: {why}, and no transcription entry declares it in transcribed_in (FID-8)")
    for b in bad:
        print("mocks.json:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness.json")
    if args[:1] == ["--config"]:
        if len(args) < 2:
            sys.exit("--config needs a path")
        config, args = args[1], args[2:]
    sys.exit(main(args[0] if args else "oracle/mocks.json", config))
