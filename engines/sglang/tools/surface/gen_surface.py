# Original to blis-sglang.
"""Write surface.json: every feature SGLang can select at the pinned commit, each with one status.

The ITEMS come from SGLang's own registration tables and selection code at the pinned commit, read
with the ast module (SGLang is not imported), so a feature a new release adds appears in the list
without anyone adding it (design X9, §7.5, SGL-4, SGL-6). The STATUS of each item, with its
evidence, issue or reason, comes from tools/surface/status.json, which people edit. The generator
refuses to write a list in which an item has no status, a status names an item SGLang does not
offer, or a status lacks what X9 asks of it. So the list is complete by construction and cannot
carry a stale entry.

Usage:
    python3 -I tools/surface/gen_surface.py <sglang-source> tools/surface/status.json > surface.json
    python3 -I tools/surface/gen_surface.py <sglang-source> tools/surface/status.json --check surface.json
    python3 -I tools/surface/gen_surface.py <sglang-source> --items     (the derived item ids, one per line)

<sglang-source> is a checkout of the SGLang repository. The working directory is the blis-sglang
checkout, whose provenance.go names the release (--repo names another). A checkout at another
commit is refused.

SGLang has no single registry, so each category is read from the code that selects it. Paths are
under python/sglang/srt/; each item's "site" is the file and line it was read at.

  transfer_backend     the members of TransferBackend (disaggregation/utils.py), each with the
                       manager class get_kv_class maps it to. A member get_kv_class has no branch for
                       fails the generator. Each --disaggregation-transfer-backend choice is a member
                       or an alias the resolution rewrites (ALIASES, checked at its site); any other
                       choice fails the generator.
  hicache_storage      every StorageBackendFactory.register_backend call
                       (mem_cache/storage/backend_factory.py), the class each names (which must
                       derive from HiCacheStorage), and the path-loaded "dynamic" backend. The
                       --hicache-storage-backend choices must be exactly these names.
  tree_cache           every class deriving from BasePrefixCache. Each records where the selection
                       chain (mem_cache/registry.py default_radix_cache_factory) or a registration
                       selects it. A return in the chain that builds anything else fails the
                       generator: that is a branch this list does not hold.
  tree_cache_backend   every register_radix_cache_backend call (--radix-cache-backend).
  tree_attachment      what create_unified_radix_cache attaches to the unified tree: HiCache
                       (init_hicache) and an external linker (init_cache_linker).
  tree_component       the members of ComponentType (mem_cache/unified_cache/component_type.py).
  tree_core            every register_tree_core_backend call (mem_cache/unified_cache/tree_core_registry.py).
  unified_cache_linker each backend create_unified_cache_linker branches on; the
                       --unified-cache-external-linker-backend choices must be exactly these.
  eviction_policy      the keys of _EVICTION_POLICY_FACTORIES (mem_cache/utils.py). A
                       --radix-eviction-policy choice that is not a key fails the generator; a key
                       that is not a choice is recorded with "cli": false.
  allocator, kv_pool,  every class deriving from BaseTokenToKVPoolAllocator, from KVCache or
  req_pool, host_pool  MambaPool, from ReqToTokenPool, and from HostKVCache, anywhere under srt/
                       (test files aside). Each records the first line of the pool configurator
                       (mem_cache/kv_cache_configurator.py, mem_cache/unified_memory_pool.py) that
                       builds it, when one does. A pool or allocator class the configurator builds
                       that derives from none of these bases fails the generator. Each records its
                       HiCache level: L1 device, L2 host.
  hicache_option       each choice of the HiCache and retraction settings (HICACHE_FIELDS).
  schedule_policy      the members of CacheAwarePolicy and CacheAgnosticPolicy
                       (managers/schedule_policy.py). A --schedule-policy choice that is neither is
                       recorded only when SGLang itself rejects it at a site REJECTED_CHOICES names;
                       any other fails the generator.
  spec_algo            the members of SpeculativeAlgorithm but NONE, and each reserved alias.
  spec_worker          each worker class SpeculativeAlgorithm.create_worker returns, with the
                       condition that selects it.
  moe_a2a_backend      the members of MoeA2ABackend (layers/moe/utils.py); a --moe-a2a-backend
                       choice that is not a member fails the generator.
  kv_event_type        every subclass of KVCacheEvent in disaggregation/kv_events.py.
  kv_event_medium      the members of StorageMedium there: L1 to L4.
  kv_event_publisher   the keys of EventPublisherFactory._registry.
  metric_family        every Prometheus metric constructor under srt/ with a literal name.
  framework            the base classes the KV frameworks define (§7.6), FRAMEWORKS.
  extension_point      every place a plugin extends SGLang: each add_*_choices list in
                       arg_groups/choices.py, each register function, the dynamic backends, the
                       resolution-step overrides, and each platform class hook
                       (platforms/interface.py get_*_cls).
  server_args_group    each namespace of arg_groups/fields/*.py.

Statuses (X9). Each status.json entry is keyed by item id ("<category>:<name>") and holds:
  modelled       "evidence": repository paths that exist, at least one an oracle scenario or a
                 golden trace. A framework item also needs "recorded" and "direct": the recorded
                 and direct traces taken at its interface (FID-5).
  refused        "issue": the blis-sglang issue that models it, which stays open while cited (X28).
  out_of_scope   "reason": hardware, no_priced_work or deprecated (§7.5), and "citation": the
                 upstream code that shows it, as python/sglang/<path>:<lines>@<commit8>.
An alias carries its own entry, so its status is stated, never inherited.
"""

import argparse
import ast
import difflib
import json
import os
import re
import sys
import warnings

UPSTREAM = {"tag": None, "commit": None}
ISSUE = re.compile(r"^https://github\.com/inference-sim/blis-sglang/issues/[0-9]+$")
REASONS = ("hardware", "no_priced_work", "deprecated")
STATUSES = ("modelled", "refused", "out_of_scope")
SCENARIO = re.compile(r"^(oracle/scenarios[^/]*/|.*/testdata/golden[^/]*/)")
SRT = "python/sglang/srt/"


def citation_rule():
    return re.compile(r"^python/sglang/[A-Za-z0-9_./-]+\.py:[0-9]+(-[0-9]+)?(, ?[0-9]+(-[0-9]+)?)*@"
                      + UPSTREAM["commit"][:8] + "$")


DISAGG_UTILS = "disaggregation/utils.py"
CHOICES = "arg_groups/choices.py"
PD_HOOK = "arg_groups/pd_disaggregation_hook.py"
BACKEND_FACTORY = "mem_cache/storage/backend_factory.py"
REGISTRY = "mem_cache/registry.py"
TREE_CORES = "mem_cache/unified_cache/tree_core_registry.py"
COMPONENTS = "mem_cache/unified_cache/component_type.py"
LINKERS = "mem_cache/unified_cache/linker_factory.py"
EVICTION = "mem_cache/utils.py"
SCHED_POLICY = "managers/schedule_policy.py"
CONFIGURATOR = "mem_cache/kv_cache_configurator.py"
UNIFIED_POOL = "mem_cache/unified_memory_pool.py"
SPEC_INFO = "speculative/spec_info.py"
SPEC_REGISTRY = "speculative/spec_registry.py"
MOE_UTILS = "layers/moe/utils.py"
KV_EVENTS = "disaggregation/kv_events.py"
PLATFORM = "platforms/interface.py"
RESOLUTION_HOOKS = "arg_groups/resolution_hooks.py"
FIELDS_DIR = "arg_groups/fields"

# A --disaggregation-transfer-backend choice that is not a TransferBackend member, the member the
# resolution rewrites it to, and the line of the rewrite.
ALIASES = {
    "mooncake_tcp": ("mooncake", PD_HOOK, r'disaggregation_transfer_backend == "mooncake_tcp"'),
}

# A CLI choice SGLang accepts at parse time and then rejects, with the line that rejects it.
REJECTED_CHOICES = {
    ("schedule_policy", "priority"): (SCHED_POLICY, r'raise ValueError\(f"Unknown schedule_policy'),
}

# The HiCache and retraction settings whose choices are surface items: (field file, field).
HICACHE_FIELDS = [
    ("memory.py", "hicache_write_policy"),
    ("memory.py", "hicache_io_backend"),
    ("memory.py", "hicache_mem_layout"),
    ("memory.py", "hicache_storage_prefetch_policy"),
    ("memory.py", "hicache_host_memory_mode"),
    ("disagg.py", "disaggregation_decode_retraction_backup"),
]

# The class bases whose subclasses are pool and allocator types, and the category of each.
POOL_ROOTS = {
    "BaseTokenToKVPoolAllocator": "allocator",
    "KVCache": "kv_pool",
    "MambaPool": "kv_pool",
    "ReqToTokenPool": "req_pool",
    "HostKVCache": "host_pool",
}
# Pool and allocator classes that derive from none of POOL_ROOTS (they are duck-typed), each with
# its category. The configurator naming any other such class fails the generator.
STANDALONE = {
    "DecodeReqToTokenPool": "req_pool",
    "UnifiedKVPool": "kv_pool",
    "MambaSlotAllocator": "allocator",
    "UnifiedMambaSlotAllocator": "allocator",
}
LEVEL = {"kv_pool": "L1", "host_pool": "L2", "hicache_storage": "L3"}
# The pool configurator: the class and module functions that choose and build pools and allocators.
CONFIGURATOR_SCOPES = [(CONFIGURATOR, "class", "KVCacheConfigurator"),
                       (UNIFIED_POOL, "prefix", "init_unified_")]
POOL_NAME = re.compile(r"^[A-Z][A-Za-z0-9]*(Pool|Allocator)$")

# The base classes of the KV frameworks (§7.6): (file, class).
FRAMEWORKS = [
    ("mem_cache/base_prefix_cache.py", "BasePrefixCache"),
    ("mem_cache/allocator/base.py", "BaseTokenToKVPoolAllocator"),
    ("mem_cache/memory_pool.py", "KVCache"),
    ("mem_cache/memory_pool.py", "ReqToTokenPool"),
    ("managers/cache_controller.py", "HiCacheController"),
    ("managers/cache_controller.py", "LayerDoneCounter"),
    ("mem_cache/hybrid_cache/hybrid_cache_controller.py", "HybridCacheController"),
    ("mem_cache/pool_host/base.py", "HostKVCache"),
    ("mem_cache/hicache_storage.py", "HiCacheStorage"),
    ("mem_cache/l2_transfer.py", "L2TransferEngine"),
    ("mem_cache/unified_cache/unified_cache_linker.py", "UnifiedCacheLinker"),
    ("disaggregation/base/conn.py", "BaseKVManager"),
    ("disaggregation/base/conn.py", "BaseKVSender"),
    ("disaggregation/base/conn.py", "BaseKVReceiver"),
    ("disaggregation/base/conn.py", "BaseKVBootstrapServer"),
]

# The register functions a plugin calls: (file, function, what the registered name is selected by).
REGISTER_FUNCTIONS = [
    (REGISTRY, "register_radix_cache_backend", "--radix-cache-backend"),
    (TREE_CORES, "register_tree_core_backend", "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"),
    (SPEC_REGISTRY, "register_algorithm", "--speculative-algorithm (SpeculativeAlgorithm.register)"),
    (KV_EVENTS, "register_publisher", "--kv-events-config publisher (EventPublisherFactory.register_publisher)"),
    (RESOLUTION_HOOKS, "register_resolution_hook", "a resolution step of arg_groups/pipeline.py"),
]

# Calls to these build a Prometheus metric; a call whose name is not a literal is a pass-through
# wrapper, allowed only at the sites listed here.
METRIC_CTORS = ("Counter", "Gauge", "Histogram", "Summary", "GaugeHistogram")
METRIC_WRAPPERS = ("utils/gauge_histogram.py", "observability/ray_wrappers.py")


class GenError(Exception):
    pass


def read(src, rel):
    try:
        with open(os.path.join(src, SRT, rel), encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise GenError("cannot read %s%s: %s" % (SRT, rel, e))


_TREES = {}


def parse(src, rel):
    key = (src, rel)
    if key not in _TREES:
        with warnings.catch_warnings():
            # SGLang's own invalid escape sequences are its business, not this list's.
            warnings.simplefilter("ignore", SyntaxWarning)
            _TREES[key] = ast.parse(read(src, rel), filename=rel)
    return _TREES[key]


def site(rel, line):
    return "%s:%d" % (rel, line)


def const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def item(category, name, rel, line, **extra):
    if not name or re.search(r"\s", name):
        raise GenError("%s:%d: a %s item name %r is empty or holds a space" % (rel, line, category, name))
    d = {"id": "%s:%s" % (category, name), "category": category, "name": name, "site": site(rel, line)}
    d.update(extra)
    return d


def line_of(src, rel, pattern):
    rx = re.compile(pattern)
    for i, l in enumerate(read(src, rel).splitlines(), 1):
        if rx.search(l):
            return i
    raise GenError("%s: no line matches %r" % (rel, pattern))


def top_class(src, rel, name):
    for s in parse(src, rel).body:
        if isinstance(s, ast.ClassDef) and s.name == name:
            return s
    raise GenError("%s: no class %s" % (rel, name))


def top_function(src, rel, name):
    """The module-level function name; of several (typing.overload stubs first), the last."""
    found = [s for s in parse(src, rel).body if isinstance(s, ast.FunctionDef) and s.name == name]
    if not found:
        raise GenError("%s: no function %s" % (rel, name))
    return found[-1]


def method(cls, name, rel):
    for s in cls.body:
        if isinstance(s, ast.FunctionDef) and s.name == name:
            return s
    raise GenError("%s: %s has no method %s" % (rel, cls.name, name))


def enum_members(src, rel, cls):
    """(member name, its value or None, line) for each member of an enum class."""
    out = []
    for s in top_class(src, rel, cls).body:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            v = s.value
            out.append((s.targets[0].id, v.value if isinstance(v, ast.Constant) else None, s.lineno))
    if not out:
        raise GenError("%s: %s has no members" % (rel, cls))
    return out


def base_names(c):
    out = []
    for b in c.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
        elif isinstance(b, ast.Subscript) and isinstance(b.value, ast.Name):
            out.append(b.value.id)
    return out


def module_constant(src, rel, name):
    """The value node of a module-level `name = ...`."""
    for s in parse(src, rel).body:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) \
                and s.targets[0].id == name:
            return s.value, s.lineno
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id == name and s.value:
            return s.value, s.lineno
    raise GenError("%s: no module constant %s" % (rel, name))


def string_list(src, rel, node, what):
    """The strings of a list, tuple or set literal, a frozenset/set call over one, a Name bound to one
    in arg_groups/choices.py or in rel, or a sum of these."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out = []
        for e in node.elts:
            s = const_str(e)
            if s is None:
                raise GenError("%s: %s holds a value that is not a string literal" % (rel, what))
            out.append(s)
        return out
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("frozenset", "set", "list") \
            and len(node.args) == 1:
        return string_list(src, rel, node.args[0], what)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return string_list(src, rel, node.left, what) + string_list(src, rel, node.right, what)
    if isinstance(node, ast.Name):
        for where in (rel, CHOICES):
            try:
                v, _ = module_constant(src, where, node.id)
            except GenError:
                continue
            return string_list(src, where, v, node.id)
    raise GenError("%s: %s is not a list of string literals the generator can read" % (rel, what))


# -- ServerArgs field choices ---------------------------------------------------------------------

def field_decl(src, fname, field):
    rel = FIELDS_DIR + "/" + fname
    for c in parse(src, rel).body:
        if not isinstance(c, ast.ClassDef):
            continue
        for s in c.body:
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id == field:
                return rel, s
    raise GenError("%s: no field %s" % (rel, field))


def field_choices(src, fname, field):
    """The choices of a ServerArgs field: its Arg(choices=...), else the Literal[...] of its type."""
    rel, s = field_decl(src, fname, field)
    ann = s.annotation
    if not (isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) and ann.value.id == "A"):
        raise GenError("%s:%d: %s is not declared as A[type, ...]" % (rel, s.lineno, field))
    parts = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
    for p in parts[1:]:
        if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "Arg":
            for k in p.keywords:
                if k.arg == "choices":
                    return rel, s.lineno, string_list(src, rel, k.value, field + " choices")
    t = parts[0]
    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "Optional":
        t = t.slice
    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "Literal":
        return rel, s.lineno, string_list(src, rel, t.slice if isinstance(t.slice, ast.Tuple) else ast.Tuple(elts=[t.slice]),
                                          field + " Literal")
    raise GenError("%s:%d: %s has no choices" % (rel, s.lineno, field))


# -- the categories -------------------------------------------------------------------------------

def transfer_backends(src):
    members = enum_members(src, DISAGG_UTILS, "TransferBackend")
    fn = top_function(src, DISAGG_UTILS, "get_kv_class")
    managers = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "transfer_backend" \
                and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Attribute) \
                and isinstance(t.comparators[0].value, ast.Name) and t.comparators[0].value.id == "TransferBackend":
            member = t.comparators[0].attr
            mgr = None
            for inner in node.body:
                if isinstance(inner, ast.Assign) and isinstance(inner.value, ast.Dict):
                    for k, v in zip(inner.value.keys, inner.value.values):
                        if isinstance(k, ast.Attribute) and k.attr == "MANAGER" and isinstance(v, ast.Name):
                            mgr = v.id
            if mgr is None:
                raise GenError("%s:%d: get_kv_class's %s branch maps no manager class" % (DISAGG_UTILS, node.lineno, member))
            managers[member] = (mgr, node.lineno)
    out, values = [], {}
    for name, value, line in members:
        if name not in managers:
            raise GenError("%s:%d: TransferBackend.%s has no get_kv_class branch" % (DISAGG_UTILS, line, name))
        values[value] = name
        out.append(item("transfer_backend", value, DISAGG_UTILS, line, manager=managers[name][0],
                        selected=site(DISAGG_UTILS, managers[name][1])))
    extra = sorted(set(managers) - {m[0] for m in members})
    if extra:
        raise GenError("%s: get_kv_class branches on %s, which TransferBackend does not have" % (DISAGG_UTILS, extra))
    rel, fline, choices = field_choices(src, "disagg.py", "disaggregation_transfer_backend")
    for c in choices:
        if c in values:
            continue
        if c not in ALIASES:
            raise GenError("%s:%d: --disaggregation-transfer-backend choice %r is not a TransferBackend member "
                           "and no alias rewrites it" % (rel, fline, c))
        target, arel, pattern = ALIASES[c]
        if target not in values:
            raise GenError("alias %s names %s, which TransferBackend does not have" % (c, target))
        out.append(item("transfer_backend", c, arel, line_of(src, arel, pattern), alias_of=target))
    for v in values:
        if v not in choices:
            for it in out:
                if it["name"] == v:
                    it["cli"] = False
    return out


def classes_index(src):
    """Every class under srt/, test files aside: name -> [(file, line, bases)]."""
    if src in _INDEX:
        return _INDEX[src]
    idx = {}
    base = os.path.join(src, SRT)
    for root, dirs, files in os.walk(base):
        dirs.sort()
        for fn in sorted(files):
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            rel = os.path.relpath(os.path.join(root, fn), base).replace(os.sep, "/")
            try:
                tree = parse(src, rel)
            except SyntaxError as e:
                raise GenError("%s%s does not parse: %s" % (SRT, rel, e))
            for n in ast.walk(tree):
                if isinstance(n, ast.ClassDef):
                    idx.setdefault(n.name, []).append((rel, n.lineno, base_names(n)))
    _INDEX[src] = idx
    return idx


_INDEX = {}


def root_of(idx, name, roots, seen=()):
    if name in roots:
        return roots[name]
    if name in seen or name not in idx:
        return None
    for _, _, bases in idx[name]:
        for b in bases:
            r = root_of(idx, b, roots, seen + (name,))
            if r:
                return r
    return None


def derived(src, roots):
    """(category, class, file, line) for each class whose bases reach one of roots."""
    idx = classes_index(src)
    out = []
    for name in sorted(idx):
        cat = root_of(idx, name, roots)
        if cat is None:
            continue
        if len(idx[name]) > 1:
            raise GenError("class %s is defined at %s; the generator cannot tell them apart"
                           % (name, ", ".join(site(r, l) for r, l, _ in idx[name])))
        rel, line, _ = idx[name][0]
        out.append((cat, name, rel, line))
    return out


def hicache_storage(src):
    idx = classes_index(src)
    out, names = [], []
    for stmt in parse(src, BACKEND_FACTORY).body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            continue
        f = stmt.value.func
        if not (isinstance(f, ast.Attribute) and f.attr == "register_backend" and isinstance(f.value, ast.Name)
                and f.value.id == "StorageBackendFactory"):
            continue
        args = [const_str(a) for a in stmt.value.args]
        if len(args) != 3 or None in args:
            raise GenError("%s:%d: a register_backend argument is not a literal" % (BACKEND_FACTORY, stmt.lineno))
        name, module, cls = args
        if root_of(idx, cls, {"HiCacheStorage": "hicache_storage"}) is None:
            raise GenError("%s:%d: %s registers %s, which does not derive from HiCacheStorage"
                           % (BACKEND_FACTORY, stmt.lineno, name, cls))
        names.append(name)
        out.append(item("hicache_storage", name, BACKEND_FACTORY, stmt.lineno,
                        **{"class": "%s.%s" % (module, cls), "level": LEVEL["hicache_storage"]}))
    if not out:
        raise GenError("%s: no StorageBackendFactory.register_backend call" % BACKEND_FACTORY)
    out.append(item("hicache_storage", "dynamic", BACKEND_FACTORY,
                    line_of(src, BACKEND_FACTORY, r'backend_name == "dynamic"'), level=LEVEL["hicache_storage"],
                    path_loaded=True))
    names.append("dynamic")
    rel, fline, choices = field_choices(src, "memory.py", "hicache_storage_backend")
    if sorted(choices) != sorted(names):
        raise GenError("%s:%d: --hicache-storage-backend choices %s are not the registered backends %s"
                       % (rel, fline, sorted(set(choices) ^ set(names)), "(symmetric difference)"))
    return out


def called_name(node):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.BoolOp) or isinstance(node.func, ast.Attribute):
            return None
    return None


def tree_caches(src):
    idx = classes_index(src)
    classes = {name: (rel, line) for cat, name, rel, line in derived(src, {"BasePrefixCache": "tree_cache"})}
    selected = {}
    fn = top_function(src, REGISTRY, "default_radix_cache_factory")
    factories = {}  # a function the chain returns the value of -> the class it builds
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return):
            continue
        v = node.value
        name = called_name(v)
        if name is None:
            raise GenError("%s:%d: default_radix_cache_factory returns %s, which the generator cannot read"
                           % (REGISTRY, node.lineno, ast.unparse(v) if v else "nothing"))
        if name in classes:
            built = name
        else:
            built = factories.get(name) or factory_class(src, name, classes)
            factories[name] = built
        selected.setdefault(built, site(REGISTRY, node.lineno))
    for user in simulated_users(src):
        selected.setdefault(user[0], user[1])
    out = []
    for name in sorted(classes, key=lambda n: (classes[n][0], classes[n][1])):
        rel, line = classes[name]
        extra = {"selected": selected[name]} if name in selected else {}
        out.append(item("tree_cache", name, rel, line, **extra))
    regs = []
    for rel, line, args in calls_anywhere(src, "register_radix_cache_backend"):
        name = const_str(args[0]) if args else None
        if name is None:
            raise GenError("%s:%d: register_radix_cache_backend's name is not a literal" % (rel, line))
        fac = args[1].id if len(args) > 1 and isinstance(args[1], ast.Name) else None
        built = factory_class(src, fac, classes, rel) if fac else None
        regs.append(item("tree_cache_backend", name, rel, line, **({"builds": built} if built else {})))
    _ = idx
    return out + regs


def simulated_users(src):
    """(class, site) for each `X.create_simulated()` call: a tree cache the scheduler builds itself."""
    out = []
    tree = parse(src, SCHED_POLICY)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "create_simulated" \
                and isinstance(n.func.value, ast.Name):
            out.append((n.func.value.id, site(SCHED_POLICY, n.lineno)))
    return out


def calls_anywhere(src, fname):
    """(file, line, args) for each call of fname(...) at module level or in a try body, under srt/."""
    out = []
    base = os.path.join(src, SRT)
    for root, dirs, files in os.walk(base):
        dirs.sort()
        for fn in sorted(files):
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            rel = os.path.relpath(os.path.join(root, fn), base).replace(os.sep, "/")
            if fname not in read(src, rel):
                continue
            stmts = list(parse(src, rel).body)
            while stmts:
                s = stmts.pop(0)
                if isinstance(s, ast.Try):
                    stmts = list(s.body) + stmts
                    continue
                if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name) \
                        and s.value.func.id == fname:
                    out.append((rel, s.lineno, s.value.args))
    return out


def factory_class(src, name, classes, rel_hint=None):
    """The tree-cache class a factory function returns, found by its name under srt/.
    create_unified_radix_cache builds UnifiedRadixCache (or a subclass a caller passes, which that
    caller then names)."""
    if name == "create_unified_radix_cache":
        return "UnifiedRadixCache"
    base = os.path.join(src, SRT)
    hits = []
    for root, dirs, files in os.walk(base):
        dirs.sort()
        for fn in sorted(files):
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            rel = os.path.relpath(os.path.join(root, fn), base).replace(os.sep, "/")
            if ("def " + name + "(") not in read(src, rel):
                continue
            for s in ast.walk(parse(src, rel)):
                if isinstance(s, ast.FunctionDef) and s.name == name:
                    built = set()
                    for r in ast.walk(s):
                        if isinstance(r, ast.Return):
                            n = called_name(r.value)
                            if n in classes:
                                built.add(n)
                            elif n == "create_unified_radix_cache":
                                built.add("UnifiedRadixCache")
                            else:
                                raise GenError("%s:%d: %s returns %s, which is not a tree cache class"
                                               % (rel, r.lineno, name, ast.unparse(r.value) if r.value else "nothing"))
                    hits.append(built)
    if len(hits) != 1 or len(hits[0]) != 1:
        raise GenError("factory %s: cannot tell which one tree cache class it builds" % name)
    return hits[0].pop()


def tree_attachments(src):
    fn = top_function(src, REGISTRY, "create_unified_radix_cache")
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.value.id == "cache" and n.func.attr.startswith("init_"):
            out.append(item("tree_attachment", n.func.attr, REGISTRY, n.lineno))
    if not out:
        raise GenError("%s: create_unified_radix_cache attaches nothing" % REGISTRY)
    return out


def tree_components(src):
    return [item("tree_component", n, COMPONENTS, line) for n, _, line in enum_members(src, COMPONENTS, "ComponentType")]


def tree_cores(src):
    out = []
    for s in parse(src, TREE_CORES).body:
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name) \
                and s.value.func.id == "register_tree_core_backend":
            name = const_str(s.value.args[0])
            if name is None:
                raise GenError("%s:%d: register_tree_core_backend's name is not a literal" % (TREE_CORES, s.lineno))
            out.append(item("tree_core", name, TREE_CORES, s.lineno))
    if not out:
        raise GenError("%s: no register_tree_core_backend call" % TREE_CORES)
    return out


def linkers(src):
    fn = top_function(src, LINKERS, "create_unified_cache_linker")
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) \
                and n.test.left.id == "backend" and len(n.test.comparators) == 1:
            v = const_str(n.test.comparators[0])
            if v is None:
                raise GenError("%s:%d: create_unified_cache_linker compares backend with a non-literal" % (LINKERS, n.lineno))
            extra = {"path_loaded": True} if v == "dynamic" else {}
            out.append(item("unified_cache_linker", v, LINKERS, n.lineno, **extra))
    rel, fline, choices = field_choices(src, "memory.py", "unified_cache_external_linker_backend")
    names = [i["name"] for i in out]
    if sorted(choices) != sorted(names):
        raise GenError("%s:%d: --unified-cache-external-linker-backend choices %s are not the linker branches %s"
                       % (rel, fline, sorted(choices), sorted(names)))
    return out


def eviction_policies(src):
    v, line = module_constant(src, EVICTION, "_EVICTION_POLICY_FACTORIES")
    if not isinstance(v, ast.Dict):
        raise GenError("%s: _EVICTION_POLICY_FACTORIES is not a dict literal" % EVICTION)
    rel, fline, choices = field_choices(src, "memory.py", "radix_eviction_policy")
    keys = [const_str(k) for k in v.keys]
    for c in choices:
        if c not in keys:
            raise GenError("%s:%d: --radix-eviction-policy choice %r has no eviction factory" % (rel, fline, c))
    out = []
    for k, val in zip(v.keys, v.values):
        extra = {"class": val.id} if isinstance(val, ast.Name) else {}
        if const_str(k) not in choices:
            extra["cli"] = False
        out.append(item("eviction_policy", const_str(k), EVICTION, k.lineno, **extra))
    return out


def schedule_policies(src):
    out, values = [], set()
    for cls in ("CacheAwarePolicy", "CacheAgnosticPolicy"):
        for name, value, line in enum_members(src, SCHED_POLICY, cls):
            values.add(value)
            out.append(item("schedule_policy", value, SCHED_POLICY, line, **{"class": cls + "." + name}))
    rel, fline, choices = field_choices(src, "schedule.py", "schedule_policy")
    for c in choices:
        if c in values:
            continue
        if ("schedule_policy", c) not in REJECTED_CHOICES:
            raise GenError("%s:%d: --schedule-policy choice %r is not a policy, and no rejection of it is listed"
                           % (rel, fline, c))
        rrel, pattern = REJECTED_CHOICES[("schedule_policy", c)]
        out.append(item("schedule_policy", c, rel, fline, rejected=site(rrel, line_of(src, rrel, pattern))))
    for it in out:
        if it["name"] not in choices:
            it["cli"] = False
    return out


def configurator_refs(src):
    """{class name: first site} for every pool- or allocator-like name the configurator code uses."""
    refs = {}
    for rel, kind, name in CONFIGURATOR_SCOPES:
        tree = parse(src, rel)
        if kind == "class":
            scopes = [top_class(src, rel, name)]
        else:
            scopes = [s for s in tree.body if isinstance(s, ast.FunctionDef) and s.name.startswith(name)]
            if not scopes:
                raise GenError("%s: no function %s*" % (rel, name))
        for scope in scopes:
            for n in ast.walk(scope):
                if isinstance(n, ast.Name) and POOL_NAME.match(n.id) and isinstance(n.ctx, ast.Load):
                    # The first file of CONFIGURATOR_SCOPES that names a class wins, then its first line.
                    have = refs.get(n.id)
                    if have is None or (have[0] == rel and n.lineno < have[1]):
                        refs[n.id] = (rel, n.lineno)
    return {k: site(*v) for k, v in refs.items()}


def pools(src):
    found = derived(src, POOL_ROOTS)
    idx = classes_index(src)
    for name, cat in sorted(STANDALONE.items()):
        if name not in idx or len(idx[name]) != 1:
            raise GenError("STANDALONE names %s, which is not one class under srt/" % name)
        rel, line, _ = idx[name][0]
        found.append((cat, name, rel, line))
    found.sort(key=lambda f: (f[0], f[1]))
    refs = configurator_refs(src)
    known = {name for _, name, _, _ in found}
    # Type names the configurator names only in annotations and isinstance checks are bases too.
    unknown = sorted(n for n in refs if n not in known)
    if unknown:
        raise GenError("the pool configurator uses %s, which derive from none of %s"
                       % (", ".join("%s (%s)" % (n, refs[n]) for n in unknown), ", ".join(sorted(POOL_ROOTS))))
    out = []
    for cat, name, rel, line in found:
        extra = {}
        if name in refs:
            extra["built"] = refs[name]
        if cat in LEVEL:
            extra["level"] = LEVEL[cat]
        out.append(item(cat, name, rel, line, **extra))
    return out


def hicache_options(src):
    out = []
    for fname, field in HICACHE_FIELDS:
        rel, line, choices = field_choices(src, fname, field)
        for c in choices:
            out.append(item("hicache_option", "%s=%s" % (field, c), rel, line))
    return out


def spec_algos(src):
    out = []
    for name, _, line in enum_members(src, SPEC_INFO, "SpeculativeAlgorithm"):
        if name != "NONE":
            out.append(item("spec_algo", name, SPEC_INFO, line))
    v, line = module_constant(src, SPEC_REGISTRY, "_RESERVED_ALIASES")
    for a in string_list(src, SPEC_REGISTRY, v, "_RESERVED_ALIASES"):
        out.append(item("spec_algo", a, SPEC_REGISTRY, line, alias_of="EAGLE"))
    fn = method(top_class(src, SPEC_INFO, "SpeculativeAlgorithm"), "create_worker", SPEC_INFO)
    workers = 0
    for n in ast.walk(fn):
        if not isinstance(n, ast.If):
            continue
        cond = ast.unparse(n.test)
        if "self.is_" not in cond:
            continue
        ret = [s for s in n.body if isinstance(s, ast.Return)]
        if len(ret) != 1 or not isinstance(ret[0].value, ast.Name):
            raise GenError("%s:%d: create_worker's branch %r returns no worker class" % (SPEC_INFO, n.lineno, cond))
        out.append(item("spec_worker", ret[0].value.id, SPEC_INFO, n.lineno, condition=cond))
        workers += 1
    if not workers:
        raise GenError("%s: create_worker selects no worker" % SPEC_INFO)
    return out


def moe_a2a(src):
    members = enum_members(src, MOE_UTILS, "MoeA2ABackend")
    rel, fline, choices = field_choices(src, "exec_.py", "moe_a2a_backend")
    values = {v for _, v, _ in members}
    for c in choices:
        if c not in values:
            raise GenError("%s:%d: --moe-a2a-backend choice %r is not a MoeA2ABackend member" % (rel, fline, c))
    out = []
    for name, value, line in members:
        extra = {} if value in choices else {"cli": False}
        out.append(item("moe_a2a_backend", value, MOE_UTILS, line, **extra))
    return out


def kv_events(src):
    tree = parse(src, KV_EVENTS)
    out = []
    for c in tree.body:
        if isinstance(c, ast.ClassDef) and "KVCacheEvent" in base_names(c):
            out.append(item("kv_event_type", c.name, KV_EVENTS, c.lineno))
    if not out:
        raise GenError("%s: no KVCacheEvent subclass" % KV_EVENTS)
    for name, value, line in enum_members(src, KV_EVENTS, "StorageMedium"):
        out.append(item("kv_event_medium", value, KV_EVENTS, line))
    for s in top_class(src, KV_EVENTS, "EventPublisherFactory").body:
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id == "_registry":
            for k in s.value.keys:
                out.append(item("kv_event_publisher", const_str(k), KV_EVENTS, k.lineno))
            break
    else:
        raise GenError("%s: no EventPublisherFactory._registry" % KV_EVENTS)
    return out


def metric_families(src):
    base = os.path.join(src, SRT)
    seen = {}
    for root, dirs, files in os.walk(base):
        dirs.sort()
        for fn in sorted(files):
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            rel = os.path.relpath(os.path.join(root, fn), base).replace(os.sep, "/")
            text = read(src, rel)
            if not any(c + "(" in text for c in METRIC_CTORS):
                continue
            for n in ast.walk(parse(src, rel)):
                if not isinstance(n, ast.Call):
                    continue
                f = n.func
                callee = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
                if callee is None or callee.lstrip("_").replace("Prom", "") not in METRIC_CTORS:
                    continue
                name = None
                for k in n.keywords:
                    if k.arg == "name":
                        name = k.value
                if name is None and n.args:
                    name = n.args[0]
                    # collections.Counter takes an iterable, never a metric name.
                    if const_str(name) is None:
                        continue
                if name is None:
                    continue
                s = const_str(name)
                if s is None:
                    if rel in METRIC_WRAPPERS:
                        continue
                    raise GenError("%s:%d: %s's metric name is not a literal" % (rel, n.lineno, callee))
                seen.setdefault(s, item("metric_family", s, rel, n.lineno, kind=callee.lstrip("_").replace("Prom", "")))
    if not seen:
        raise GenError("no Prometheus metric constructor under srt/")
    return [seen[k] for k in sorted(seen)]


def frameworks(src):
    out = []
    for rel, name in FRAMEWORKS:
        c = None
        for n in ast.walk(parse(src, rel)):
            if isinstance(n, ast.ClassDef) and n.name == name:
                c = n
                break
        if c is None:
            raise GenError("%s: no class %s (a framework base, FRAMEWORKS)" % (rel, name))
        out.append(item("framework", name, rel, c.lineno))
    return out


def extension_points(src):
    out = []
    for s in parse(src, CHOICES).body:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) \
                and s.targets[0].id.startswith("add_"):
            v = s.value
            if not (isinstance(v, ast.Attribute) and v.attr in ("extend", "append") and isinstance(v.value, ast.Name)):
                raise GenError("%s:%d: %s is not LIST.extend or LIST.append" % (CHOICES, s.lineno, s.targets[0].id))
            out.append(item("extension_point", s.targets[0].id, CHOICES, s.lineno, extends=v.value.id))
    if not out:
        raise GenError("%s: no add_* extension point" % CHOICES)
    for rel, fname, selects in REGISTER_FUNCTIONS:
        line = None
        for n in ast.walk(parse(src, rel)):
            if isinstance(n, ast.FunctionDef) and n.name == fname:
                line = n.lineno
                break
        if line is None:
            raise GenError("%s: no function %s" % (rel, fname))
        out.append(item("extension_point", fname, rel, line, selected_by=selects))
    pc = platform_class(src)
    for s in pc.body:
        if isinstance(s, ast.FunctionDef) and re.fullmatch(r"get_[a-z0-9_]+_cls", s.name):
            out.append(item("extension_point", pc.name + "." + s.name, PLATFORM, s.lineno))
    return out


def platform_class(src):
    """The class of platforms/interface.py that declares the get_*_cls hooks."""
    for s in parse(src, PLATFORM).body:
        if isinstance(s, ast.ClassDef) and any(isinstance(m, ast.FunctionDef) and m.name == "get_kv_pool_cls" for m in s.body):
            return s
    raise GenError("%s: no class declares get_kv_pool_cls" % PLATFORM)


def server_args_groups(src):
    out = []
    d = os.path.join(src, SRT, FIELDS_DIR)
    try:
        files = sorted(f for f in os.listdir(d) if f.endswith(".py") and f != "__init__.py")
    except OSError as e:
        raise GenError("%s%s: %s" % (SRT, FIELDS_DIR, e))
    for fn in files:
        rel = FIELDS_DIR + "/" + fn
        for c in parse(src, rel).body:
            if isinstance(c, ast.ClassDef):
                ns = None
                for s in c.body:
                    if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "_NS_PATH":
                        ns = const_str(s.value)
                if ns is not None:
                    out.append(item("server_args_group", ns, rel, c.lineno, **{"class": c.name}))
    if not out:
        raise GenError("%s%s: no namespace" % (SRT, FIELDS_DIR))
    return out


def derive(src):
    _TREES.clear()
    _INDEX.clear()
    its = []
    its += transfer_backends(src)
    its += hicache_storage(src)
    its += tree_caches(src)
    its += tree_attachments(src)
    its += tree_components(src)
    its += tree_cores(src)
    its += linkers(src)
    its += eviction_policies(src)
    its += pools(src)
    its += hicache_options(src)
    its += schedule_policies(src)
    its += spec_algos(src)
    its += moe_a2a(src)
    its += kv_events(src)
    its += metric_families(src)
    its += frameworks(src)
    its += extension_points(src)
    its += server_args_groups(src)
    ids = [i["id"] for i in its]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise GenError("item ids derived twice: %s" % dup)
    return its


# -- statuses -------------------------------------------------------------------------------------

def paths_exist(repo, paths, what, errs):
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p for p in paths):
        errs.append("%s: want a non-empty list of repository paths" % what)
        return False
    for p in paths:
        if p.startswith("/") or ".." in p.split("/") or not os.path.exists(os.path.join(repo, p)):
            errs.append("%s: %s is not a path in this repository" % (what, p))
    return True


def check_entry(repo, it, st, errs):
    """X9's demands on one item's status entry, and FID-5's on a framework's."""
    iid = it["id"]
    status = st.get("status")
    if status not in STATUSES:
        errs.append("%s: status %r is not one of %s" % (iid, status, ", ".join(STATUSES)))
        return
    allowed = {"status", "why"}
    if status == "modelled":
        allowed |= {"evidence", "recorded", "direct"}
        if paths_exist(repo, st.get("evidence"), iid + ": evidence", errs) and \
                not any(SCENARIO.match(p) for p in st["evidence"]):
            errs.append("%s: modelled, but its evidence names no oracle scenario or golden trace" % iid)
        if it["category"] == "framework":
            for k in ("recorded", "direct"):
                if k not in st:
                    errs.append("%s: a framework is modelled only with %s traces at its interface (FID-5)" % (iid, k))
                else:
                    paths_exist(repo, st[k], "%s: %s" % (iid, k), errs)
        else:
            for k in ("recorded", "direct"):
                if k in st:
                    errs.append("%s: %s traces are a framework's (FID-5); this is a %s" % (iid, k, it["category"]))
    elif status == "refused":
        allowed |= {"issue"}
        if not ISSUE.match(str(st.get("issue", ""))):
            errs.append("%s: refused, but issue %r is not a blis-sglang issue URL" % (iid, st.get("issue")))
    else:
        allowed |= {"reason", "citation"}
        if st.get("reason") not in REASONS:
            errs.append("%s: out_of_scope reason %r is not one of %s (§7.5)" % (iid, st.get("reason"), ", ".join(REASONS)))
        if not citation_rule().match(str(st.get("citation", ""))):
            errs.append("%s: out_of_scope citation %r is not python/sglang/<path>:<lines>@%s"
                        % (iid, st.get("citation"), UPSTREAM["commit"][:8]))
    extra = sorted(set(st) - allowed)
    if extra:
        errs.append("%s: keys %s do not belong with status %s" % (iid, extra, status))


def build(src, status, repo):
    its = derive(src)
    entries = status.get("items", {})
    errs = []
    known = {i["id"] for i in its}
    for k in sorted(set(entries) - known):
        errs.append("%s: has a status but SGLang %s offers no such item" % (k, UPSTREAM["tag"]))
    out = []
    for it in its:
        st = entries.get(it["id"])
        if st is None:
            errs.append("%s: no status (%s%s); add it to tools/surface/status.json" % (it["id"], SRT, it["site"]))
            continue
        check_entry(repo, it, st, errs)
        row = dict(it)
        row.update(st)
        out.append(row)
    if errs:
        raise GenError("\n".join(errs))
    counts = {}
    for r in out:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "generator": "tools/surface/gen_surface.py",
        "upstream": {"name": "sglang", "tag": UPSTREAM["tag"], "commit": UPSTREAM["commit"]},
        "sites_under": SRT,
        "counts": dict(sorted(counts.items())),
        "items": out,
    }


def render(doc):
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def provenance_upstream(repo):
    """The Tag and Commit of provenance.go's Upstream record."""
    try:
        with open(os.path.join(repo, "provenance.go"), encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise GenError("provenance.go: %s" % e)
    m = re.search(r"var Upstream = harness\.Upstream\{(.*?)\n\}", text, re.S)
    tag = m and re.search(r"\bTag:\s*\"([^\"]+)\"", m.group(1))
    commit = m and re.search(r"\bCommit:\s*\"([0-9a-f]{40})\"", m.group(1))
    if not (tag and commit):
        raise GenError("provenance.go: no Upstream record with a Tag and a 40-hex Commit")
    return tag.group(1), commit.group(1)


def checkout_commit(src):
    """The commit the SGLang checkout is at, or None when it is not a git checkout of its own."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        top = subprocess.run(["git", "-C", src, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    if r.returncode != 0 or top.returncode != 0 or os.path.realpath(top.stdout.strip()) != os.path.realpath(src):
        return None
    return r.stdout.strip()


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sglang_source")
    ap.add_argument("status", nargs="?")
    ap.add_argument("--check", metavar="SURFACE_JSON")
    ap.add_argument("--repo", default=".", help="the blis-sglang checkout (default: the working directory)")
    ap.add_argument("--items", action="store_true", help="print the derived item ids and exit")
    a = ap.parse_args(argv)
    try:
        tag, commit = provenance_upstream(a.repo)
        head = checkout_commit(a.sglang_source)
        if head is not None and head != commit:
            raise GenError("the SGLang checkout is at %s, not %s (%s)" % (head, commit, tag))
        UPSTREAM["tag"], UPSTREAM["commit"] = tag, commit
        if a.items:
            for it in derive(a.sglang_source):
                print(it["id"], SRT + it["site"])
            return 0
        if a.status is None:
            ap.error("status.json is required")
        with open(a.status, encoding="utf-8") as f:
            status = json.load(f)
        out = render(build(a.sglang_source, status, a.repo))
    except GenError as e:
        print("gen_surface: " + str(e).replace("\n", "\ngen_surface: "), file=sys.stderr)
        return 1
    if a.check:
        try:
            with open(a.check, encoding="utf-8") as f:
                have = f.read()
        except OSError as e:
            print("gen_surface: %s" % e, file=sys.stderr)
            return 1
        if have != out:
            sys.stdout.writelines(difflib.unified_diff(
                have.splitlines(True), out.splitlines(True), a.check, "generated"))
            print("gen_surface: %s is not what the generator writes; regenerate it" % a.check, file=sys.stderr)
            return 1
        return 0
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
