# Original to blis-vllm.
"""Write kernel_impls.json: every attention kernel implementation vLLM can emit, with its code hash.

The table is the one design section 6.2 asks each engine release to publish: for each lowered
primitive it covers, every kernel the release can run, by library, the library's own kernel name,
the device architecture and the code hash the coefficient registry keys its entries by (D5,
VAL-8). Its type, validator and hash rule are blis-substrate's contracts/spec/kernelimpls; a Go
test (kernel_impls_test.go) loads the committed file with that loader, which recomputes every
hash, so this script and the rule cannot drift apart.

WHAT COMES FROM vLLM, AND WHAT FROM kernels/<tag>/attention.json.
  From the vLLM release's own source, at the tag --upstream names, read with the ast module (vLLM is
  not imported):
    - the attention backends, every member of AttentionBackendEnum
      (vllm/v1/attention/backends/registry.py), and the module of each;
    - the backends the CUDA platform selects on its own, every AttentionBackendEnum member named
      in _get_backend_priorities (vllm/platforms/cuda.py);
    - for each backend the table lists, every name its module imports from a kernel library's
      modules, and the line of the first call of each kernel;
    - the line of each architecture gate the classification names (the def of the function);
    - the pinned version of each library: the requirement line of a pip library in
      requirements/cuda.txt, the GIT_TAG of a library vLLM's build fetches;
    - the CUDA toolkit of the release's build, ARG CUDA_VERSION in docker/Dockerfile.
  From each library's own source at that pin (a checkout given with --source, whose commit and
  version the script checks against the pin):
    - the git object id of every source path a kernel's hash covers;
    - the SHA-256 the library records for a precompiled kernel package (FlashInfer's
      flashinfer/artifacts.py, CheckSumHash), for a kernel that ships only as a binary.
  From kernels/<tag>/attention.json, which people edit, also the library versions runs of the release
  were seen with (a library's "seen" list, each citing the run record that shows the version); each
  is listed unpinned, with its kernels hashed from that version's own source (design 6.2).
  From kernels/<tag>/attention.json, which people edit: the status of each backend, and for each kernel
  of a listed backend its library, variant, primitives, architectures, gate and hash inputs.

The script refuses to write a table when the classification and the source disagree: an
AttentionBackendEnum member with no status or a status for one that does not exist; a backend the
CUDA platform selects marked as not running on CUDA; a name a listed backend imports from a kernel
library that the classification neither lists as a kernel nor as not one, or one it lists that the
module no longer imports; a kernel never called in its backend's module; a gate function that does
not exist; a checkout at another commit or version than the pin. So the table is complete for the
backends it lists by construction, and a backend it does not list is a gap with an issue.

ONE CLASSIFICATION PER RELEASE. kernels/<tag>/attention.json classifies the vLLM release <tag>
(v0.32.0, v0.33.0); --upstream names the tag, and the generator refuses a classification of another
tag or a source tree at another commit. A new release gets its own classification: copy the
nearest one, set its tag and commit, and run the generator against the new source, which names
every backend and import that has to be classified.

Usage (kernels/regen.py does all of it, fetching the sources, in one command):
    python3 -I kernels/gen_kernel_impls.py <vllm-source> --upstream v0.33.0 \\
        --source flashinfer=<dir> --source vllm-flash-attn=<dir> > kernel_impls.json
    python3 -I kernels/gen_kernel_impls.py ... --check kernel_impls.json
"""

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys

HASH_VERSION = "blis kernel hash v1"
PRIMITIVES = {"attention.prefill", "attention.decode"}
STATUSES = {"listed", "gap", "not_cuda", "not_lowered"}


class Refused(Exception):
    """The classification and the source disagree; the message says where."""


def code_hash(basis, library, variant, arch, inputs):
    """blis-substrate's kernelimpls.CodeHash, restated: the Go test reparses every row with it."""
    lines = [HASH_VERSION, "basis " + basis, "library " + library, "variant " + variant, "arch " + arch]
    lines += ["input " + s for s in sorted(inputs)]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def read(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return f.read()


def cite(rel, line, commit):
    return f"{rel}:{line}@{commit[:8]}"


def git(root, *args):
    out = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise Refused(f"git -C {root} {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


# ---------------------------------------------------------------------------------------------
# vLLM's selection code
# ---------------------------------------------------------------------------------------------


def backend_enum(vllm):
    """Every AttentionBackendEnum member and the module of its class ('' when it names none)."""
    rel = "vllm/v1/attention/backends/registry.py"
    tree = ast.parse(read(vllm, rel))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AttentionBackendEnum":
            out = {}
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    v = stmt.value
                    if not isinstance(v, ast.Constant):
                        raise Refused(f"{rel}:{stmt.lineno}: {stmt.targets[0].id} is not a literal class path")
                    path = v.value or ""
                    out[stmt.targets[0].id] = path.rsplit(".", 1)[0] if path else ""
            return out
    raise Refused(f"{rel}: no class AttentionBackendEnum")


def cuda_priorities(vllm, commit):
    """The members _get_backend_priorities names, each with the line it is first named on."""
    rel = "vllm/platforms/cuda.py"
    tree = ast.parse(read(vllm, rel))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_backend_priorities":
            out = {}
            for n in ast.walk(node):
                if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                        and n.value.id == "AttentionBackendEnum"):
                    out.setdefault(n.attr, cite(rel, n.lineno, commit))
            return out
    raise Refused(f"{rel}: no function _get_backend_priorities")


def module_path(module):
    return module.replace(".", "/") + ".py"


def imports_from(tree, prefixes):
    """Names a module imports from a module under one of prefixes: {name: (module, line)}, and
    the local names it binds to such a module itself ({alias: module})."""
    names, aliases = {}, {}

    def under(mod):
        return any(mod == p or mod.startswith(p + ".") for p in prefixes)

    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                full = n.module + "." + a.name
                if under(n.module):
                    names.setdefault(a.asname or a.name, (n.module, n.lineno))
                elif under(full):
                    # from vllm import _custom_ops as ops: a library module bound to a name.
                    aliases[a.asname or a.name] = full
        elif isinstance(n, ast.Import):
            for a in n.names:
                if under(a.name):
                    aliases[a.asname or a.name.split(".")[0]] = a.name
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
            names.setdefault(n.attr, (aliases[n.value.id], n.lineno))
    return names


def jit_defs(tree):
    """Functions the module decorates with a JIT compiler (@triton.jit, @cute.jit, ...): kernels,
    or device code compiled into one, that an import scan cannot see. {name: line}."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.decorator_list:
                f = d.func if isinstance(d, ast.Call) else d
                if isinstance(f, ast.Attribute) and f.attr == "jit":
                    out.setdefault(n.name, n.lineno)
    return out


def torch_op_calls(tree):
    """Each torch.ops.<namespace>.<op> the module calls: a registered kernel. {name: line}."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f, chain = n.func, []
            while isinstance(f, ast.Attribute):
                chain.append(f.attr)
                f = f.value
            if isinstance(f, ast.Name):
                chain.append(f.id)
            chain.reverse()
            if chain[:2] == ["torch", "ops"] and len(chain) > 2:
                out.setdefault(".".join(chain), n.lineno)
    return out


INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
C_LIKE = (".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".inl", ".jinja", ".inc")


def include_closure(root, entries, lib):
    """The files a C++ or CUDA build compiles from entries, following #include through the
    library's own tree (lib["include_search"]), and the submodules an include reaches
    (lib["submodules"], prefix to path). An include the tree does not hold is refused unless
    lib["unhashed_includes"] names its prefix with the reason it is not part of the hash."""
    files, subs, todo = set(), set(), list(entries)
    search = lib.get("include_search", [])
    while todo:
        rel = todo.pop()
        if rel in files:
            continue
        files.add(rel)
        if not rel.endswith(C_LIKE):
            continue
        for kind, inc in INCLUDE.findall(read(root, rel)):
            cands = ([os.path.normpath(os.path.join(os.path.dirname(rel), inc))] if kind == '"' else [])
            cands += [os.path.normpath(os.path.join(d, inc)) for d in search]
            hit = next((c for c in cands if os.path.isfile(os.path.join(root, c))), None)
            if hit:
                todo.append(hit)
                continue
            sub = next((v for k, v in lib.get("submodules", {}).items() if inc.startswith(k)), None)
            if sub:
                subs.add(sub)
            elif kind == '"' and not any(inc.startswith(k) for k in lib.get("unhashed_includes", {})):
                raise Refused(f"{rel} includes {inc!r}, which {root} does not hold and the classification does not "
                              f"name as unhashed (libraries.<name>.unhashed_includes)")
    return files, subs


def python_closure(root, entries, prefixes):
    """The Python files a kernel module runs, following imports to modules under prefixes
    (vLLM's attention ops, where its Triton helpers live)."""
    files, todo = set(), list(entries)
    while todo:
        rel = todo.pop()
        if rel in files:
            continue
        files.add(rel)
        if not rel.endswith(".py"):
            continue
        for n in ast.walk(ast.parse(read(root, rel))):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods = [n.module] + [n.module + "." + a.name for a in n.names]
            elif isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            for m in mods:
                path = m.replace(".", "/")
                for cand in (path + ".py", path + "/__init__.py"):
                    if any(cand.startswith(p) for p in prefixes) and os.path.isfile(os.path.join(root, cand)):
                        todo.append(cand)
    return files


def first_call(tree, name):
    """The line of the first call of name, as a bare name or an attribute, or None."""
    lines = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id == name) or (isinstance(f, ast.Attribute) and f.attr == name):
                lines.append(n.lineno)
    return min(lines) if lines else None


def def_line(vllm, rel, func):
    tree = ast.parse(read(vllm, rel))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func:
            return n.lineno
    raise Refused(f"{rel}: no function {func}, which a gate names")


# ---------------------------------------------------------------------------------------------
# Library pins and sources
# ---------------------------------------------------------------------------------------------


def pip_pin(vllm, rel, dist, commit):
    for i, line in enumerate(read(vllm, rel).splitlines(), 1):
        m = re.match(r"^\s*" + re.escape(dist) + r"(\[[^\]]*\])?\s*==\s*([^\s;#]+)", line)
        if m:
            return m.group(2), cite(rel, i, commit)
    raise Refused(f"{rel}: no pin of {dist}")


def cmake_pin(vllm, rel, commit):
    for i, line in enumerate(read(vllm, rel).splitlines(), 1):
        m = re.match(r"^\s*GIT_TAG\s+([0-9a-f]{40})\s*$", line)
        if m:
            return m.group(1), cite(rel, i, commit)
    raise Refused(f"{rel}: no GIT_TAG naming a commit")


def cuda_version(vllm):
    rel = "docker/Dockerfile"
    for line in read(vllm, rel).splitlines():
        m = re.match(r"^ARG CUDA_VERSION=(\S+)$", line)
        if m:
            return m.group(1)
    raise Refused(f"{rel}: no ARG CUDA_VERSION with a default")


COMPILERS = {"nvcc", "nvcc+prebuilt", "triton", "cutedsl"}


def toolkit(vllm, commit, compiler):
    """The compiler that builds a kernel, at the version the release pins: nvcc from the
    release image's CUDA toolkit; Triton from the torch the release pins, which pins Triton;
    CuTe DSL from nvidia-cutlass-dsl. A precompiled package is loaded by a launcher nvcc builds."""
    if compiler == "nvcc":
        return f"cuda {cuda_version(vllm)}"
    if compiler == "nvcc+prebuilt":
        return f"cuda {cuda_version(vllm)} launcher, precompiled kernels"
    if compiler == "triton":
        return f"triton (torch=={pip_pin(vllm, 'requirements/cuda.txt', 'torch', commit)[0]})"
    if compiler == "cutedsl":
        return f"nvidia-cutlass-dsl=={pip_pin(vllm, 'requirements/cuda.txt', 'nvidia-cutlass-dsl', commit)[0]}"
    raise Refused(f"compiler {compiler!r} is not one of {sorted(COMPILERS)}")


def at_pin(root, what, commit=None, tag=None):
    """Refuse a checkout that is not exactly the pin: a dirty tree, or HEAD at another commit
    than the pin's (a commit pin, or the commit a release tag names)."""
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise Refused(f"{root} ({what}) has changes; the hash is of the pinned tree")
    head = git(root, "rev-parse", "HEAD")
    if commit is not None and head != commit:
        raise Refused(f"{root} is at {head}; {what} is pinned at {commit}")
    if tag is not None:
        r = subprocess.run(["git", "-C", root, "rev-parse", "--verify", "-q", f"refs/tags/{tag}^{{commit}}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise Refused(f"{root} has no tag {tag}; fetch {what} by its tag (kernels/regen.py does)")
        if r.stdout.strip() != head:
            raise Refused(f"{root} is at {head}, not tag {tag} ({r.stdout.strip()})")
    return head


def entry_kind(root, path):
    """The kind of git object path names at HEAD (blob, tree, or commit for a submodule), or None."""
    out = git(root, "ls-tree", "HEAD", "--", path.rstrip("/"))
    lines = [ln for ln in out.splitlines() if ln.split("\t", 1)[-1] == path.rstrip("/")]
    return lines[0].split("\t", 1)[0].split()[1] if len(lines) == 1 else None


def object_id(root, path, absent_ok=False):
    """The path with the git object it names at HEAD: a blob, a tree, or the commit a submodule
    is pinned to (read from the tree, so the submodule need not be checked out). A path the
    tree does not hold is "<path> absent" when absent_ok: a build entry one version of a library
    does not have is part of that version's build identity."""
    path = path.rstrip("/")
    out = git(root, "ls-tree", "HEAD", "--", path)
    lines = [ln for ln in out.splitlines() if ln.split("\t", 1)[-1] == path]
    if len(lines) != 1:
        if absent_ok:
            return f"{path} absent"
        raise Refused(f"{root}: no single entry {path} at HEAD")
    _, kind, oid = lines[0].split("\t", 1)[0].split()
    return f"{path} {kind} {oid}"


def artifact_checksum(root, rel, artifact):
    """CheckSumHash.<artifact> of the library's artifact list (FlashInfer's flashinfer/artifacts.py):
    the SHA-256 of the checksums.txt that lists the SHA-256 of every file of a precompiled kernel
    package."""
    tree = ast.parse(read(root, rel))
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == "CheckSumHash":
            for stmt in n.body:
                target = stmt.target if isinstance(stmt, ast.AnnAssign) else (
                    stmt.targets[0] if isinstance(stmt, ast.Assign) else None)
                if isinstance(target, ast.Name) and target.id == artifact:
                    v = stmt.value
                    if isinstance(v, ast.Constant) and re.fullmatch(r"[0-9a-f]{64}", v.value or ""):
                        return f"{artifact} checksums.txt sha256:{v.value}"
    raise Refused(f"{rel}: CheckSumHash has no SHA-256 {artifact}")


# ---------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------


def classification_path(tag):
    """The classification of the vLLM release tag: kernels/<tag>/attention.json."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), tag, "attention.json")


def library_pins(vllm, cls):
    """Each library's pinned version, where vLLM states it, and the git ref of its source at that
    pin: {name: (version, pin, repo, ref)}. A pip library's ref is its release tag, the
    classification's tag_format applied to the version; a library vLLM's build fetches is fetched
    at the commit its GIT_TAG names; the engine's own kernels are vLLM itself (repo None)."""
    commit = git(vllm, "rev-parse", "HEAD")
    out = {}
    for name, lib in sorted(cls["libraries"].items()):
        if "pip" in lib:
            version, pin = pip_pin(vllm, lib["requirements"], lib["pip"], commit)
            ref = lib.get("tag_format", "v{version}").format(version=version)
        elif "cmake" in lib:
            version, pin = cmake_pin(vllm, lib["cmake"], commit)
            ref = version
        elif lib.get("engine"):
            version, pin, ref = commit, "the engine release itself", None
        else:
            raise Refused(f"library {name}: no pip, cmake or engine pin")
        out[name] = (version, pin, None if lib.get("engine") else lib["repo"], ref)
        for v in lib.get("seen", []):
            out[f"{name}@{v['version']}"] = (v["version"], v.get("seen_in", ""), lib["repo"],
                                             lib.get("tag_format", "v{version}").format(version=v["version"]))
    return out


def library_of(module, libs):
    """The library whose module prefixes hold module, by the longest prefix, or None."""
    best = None
    for name, lib in libs.items():
        for p in lib["modules"]:
            if (module == p or module.startswith(p + ".")) and (best is None or len(p) > best[1]):
                best = (name, len(p))
    return best[0] if best else None


def hash_inputs(vllm, cls, k, lib_name, root, engine_lib):
    """The basis and content identities a kernel's hash covers: its build closure. For a source
    kernel, every file its entries compile (through #include, or through Python imports for
    vLLM's own Triton kernels), each directory it names as a whole, and each submodule an
    include reaches; for a precompiled one, the package's checksum as well, beside its
    launcher's closure. Files of vLLM's own tree that dispatch or parameterise the kernel
    (engine_source) are prefixed "vllm:" when the kernel is another library's."""
    lib = cls["libraries"][lib_name]
    spec = k["hash"]
    inputs, files, trees, subs = [], set(), [], set()
    for path in spec.get("source", []):
        kind = entry_kind(root, path)
        if kind is None:
            inputs.append(object_id(root, path, absent_ok=True))
        elif kind in ("tree", "commit"):
            trees.append(path.rstrip("/"))
        else:
            files.add(path)
    if files:
        if lib.get("engine"):
            files = python_closure(root, sorted(files), lib.get("python_closure", []))
        else:
            cfiles, subs = include_closure(root, sorted(f for f in files if f.endswith(C_LIKE)), lib)
            files = cfiles | {f for f in files if not f.endswith(C_LIKE)}
    inputs += [object_id(root, f) for f in sorted(files)] + [object_id(root, t) for t in trees]
    inputs += [object_id(root, sub) for sub in sorted(subs)]
    eng = python_closure(vllm, spec.get("engine_source", []), cls["libraries"][engine_lib].get("python_closure", []))
    inputs += [("" if lib.get("engine") else "vllm:") + object_id(vllm, f) for f in sorted(eng)]
    basis = "source"
    if "artifact" in spec:
        if "checksums" not in lib:
            raise Refused(f"kernel {k['variant']} is a precompiled package, and library {lib_name} names no checksum list")
        basis = "binary"
        inputs.append(artifact_checksum(root, lib["checksums"], spec["artifact"]))
    if not inputs:
        raise Refused(f"kernel {k['variant']}: the hash covers nothing")
    return basis, sorted(set(inputs))


def generate(vllm, cls, sources):
    up = cls["upstream"]
    commit = at_pin(vllm, f"vLLM {up['tag']}", commit=up["commit"])

    archs = cls["archs"]
    libs = {}
    pins = library_pins(vllm, cls)
    engine_lib = next((n for n, lib in cls["libraries"].items() if lib.get("engine")), None)
    if engine_lib is None:
        raise Refused("the classification names no library for the engine's own kernels (engine: true)")
    for name, lib in sorted(cls["libraries"].items()):
        version, pin, _, ref = pins[name]
        root = vllm if lib.get("engine") else sources.get(name)
        if root is None:
            raise Refused(f"library {name}: no checkout given (--source {name}=<dir>)")
        if "cmake" in lib:
            head = at_pin(root, f"{name} ({pin})", commit=version)
        elif "pip" in lib:
            head = at_pin(root, f"{name} {version} ({pin})", tag=ref)
        else:
            head = commit
        if "version_file" in lib:
            have = read(root, lib["version_file"]).strip()
            if have != version:
                raise Refused(f"{root} is version {have}; vLLM pins {name} {version} ({pin})")
        libs[name] = {"version": version, "pin": pin, "root": root, "origin": f"{lib['repo']} {head}",
                      "modules": lib["modules"]}
        seen = []
        for v in lib.get("seen", []):
            if not v.get("seen_in"):
                raise Refused(f"library {name} {v.get('version')}: a version seen in runs cites the run record (seen_in)")
            if v["version"] == version:
                raise Refused(f"library {name} {version} is the pin; it is not listed again as seen in runs")
            sroot = sources.get(f"{name}@{v['version']}")
            if sroot is None:
                raise Refused(f"library {name} {v['version']}: no checkout given (--source {name}@{v['version']}=<dir>)")
            if "version_file" in lib and read(sroot, lib["version_file"]).strip() != v["version"]:
                raise Refused(f"{sroot} is not {name} {v['version']}")
            shead = at_pin(sroot, f"{name} {v['version']}", tag=pins[f"{name}@{v['version']}"][3])
            seen.append({"version": v["version"], "root": sroot, "pin": v["seen_in"], "origin": f"{lib['repo']} {shead}"})
        libs[name]["seen"] = seen

    enum = backend_enum(vllm)
    prio = cuda_priorities(vllm, commit)
    backends = cls["backends"]
    for b in sorted(set(enum) - set(backends)):
        raise Refused(f"AttentionBackendEnum.{b} has no status in the classification")
    for b in sorted(set(backends) - set(enum)):
        raise Refused(f"the classification gives a status to {b}, which AttentionBackendEnum does not have")
    for b, st in sorted(backends.items()):
        if st["status"] not in STATUSES:
            raise Refused(f"{b}: status {st['status']!r} is not one of {sorted(STATUSES)}")
        if st["status"] in ("not_cuda", "not_lowered") and not st.get("reason"):
            raise Refused(f"{b}: a {st['status']} backend states its reason")
        if st["status"] == "gap" and not re.fullmatch(r"[\w.-]+/[\w.-]+#[1-9]\d*", st.get("issue", "")):
            raise Refused(f"{b}: a gap names its issue as <owner>/<repo>#<n>")
        if b in prio and st["status"] in ("not_cuda", "not_lowered"):
            raise Refused(f"{b} is {st['status']}, but the CUDA platform selects it ({prio[b]})")

    kernels_by_backend = {}
    for k in cls["kernels"]:
        kernels_by_backend.setdefault(k["backend"], []).append(k)
    rows, gaps, used = [], [], set()
    for b, st in sorted(backends.items()):
        selection = f"attention backend {b}"
        if st["status"] == "gap":
            for prim in st.get("primitives", sorted(PRIMITIVES)):
                gaps.append({"primitive": prim, "selection": selection, "issue": st["issue"]})
        if st["status"] != "listed":
            if b in kernels_by_backend or st.get("gaps"):
                raise Refused(f"{b} is {st['status']} but the classification lists kernels or kernel gaps for it")
            continue
        rel = module_path(enum[b])
        tree = ast.parse(read(vllm, rel))
        prefixes = sorted({m for lib in libs.values() for m in lib["modules"]})
        imported = imports_from(tree, prefixes)
        jits, ops = jit_defs(tree), torch_op_calls(tree)
        kernels = kernels_by_backend.get(b, [])
        part_of = {n: j["part_of"] for n, j in st.get("jit", {}).items() if "part_of" in j}
        gap_calls = {g["call"]: g for g in st.get("gaps", [])}
        # Every name the module imports from a kernel library, every JIT-compiled function it
        # defines and every registered op it calls is a kernel, part of one, a gap or not a kernel.
        found = {**{n: f"{rel}:{ln} imports {n} from {imported[n][0]}" for n, (_, ln) in imported.items()},
                 **{n: f"{rel}:{ln} defines JIT function {n}" for n, ln in jits.items()},
                 **{n: f"{rel}:{ln} calls {n}" for n, ln in ops.items()}}
        classified = {k["call"] for k in kernels} | set(st.get("not_kernels", {})) | set(gap_calls) | set(st.get("jit", {}))
        for name in sorted(set(found) - classified):
            raise Refused(f"{found[name]} ({b}); the classification neither lists it as a kernel, part of one or a gap, "
                          f"nor as not a kernel")
        for name in sorted(classified - set(found)):
            raise Refused(f"{b}: the classification names {name}, which {rel} neither imports from a library module, "
                          f"defines as a JIT function nor calls as a registered op")
        for g in st.get("gaps", []):
            if not re.fullmatch(r"[\w.-]+/[\w.-]+#[1-9]\d*", g.get("issue", "")):
                raise Refused(f"{b}: gap {g['call']} names its issue as <owner>/<repo>#<n>")
            glib = library_of(imported[g["call"]][0], libs) if g["call"] in imported else engine_lib
            for prim in g["primitives"]:
                gaps.append({"primitive": prim, "selection": selection, "library": glib, "variant": g["call"],
                             "issue": g["issue"]})
        for k in kernels:
            if k["call"] not in imported:
                raise Refused(f"{b}: kernel {k['variant']} is called as {k['call']}, which {rel} does not import "
                              f"from a library module")
            mlib = library_of(imported[k["call"]][0], libs)
            if mlib != k["library"]:
                raise Refused(f"{b}: kernel {k['variant']} is classified under {k['library']}, but {rel} imports "
                              f"{k['call']} from {imported[k['call']][0]}, a module of {mlib}")
            line = first_call(tree, k["call"])
            if line is None:
                raise Refused(f"{rel}: {b} never calls {k['call']}")
            for n, v in part_of.items():
                if v == k["variant"] and rel not in k["hash"].get("engine_source", []):
                    raise Refused(f"{b}: {n} is compiled into {v}, so {v}'s hash covers {rel} (hash.engine_source)")
            lib = libs[k["library"]]
            used.add(k["library"])
            selected = {cite(rel, line, commit), prio.get(b, cite("vllm/v1/attention/backends/registry.py",
                                                                  enum_line(vllm, b), commit))}
            for g in k.get("gates", []):
                grel, func = g.split(":")
                selected.add(cite(grel, def_line(vllm, grel, func), commit))
            tk = toolkit(vllm, commit, k.get("compiler", ""))
            # The pinned build, and the build of each version runs were seen with, from that
            # version's own source.
            for version, root in [(lib["version"], lib["root"])] + [(v["version"], v["root"]) for v in lib["seen"]]:
                basis, inputs = hash_inputs(vllm, cls, k, k["library"], root, engine_lib)
                for prim in k["primitives"]:
                    if prim not in PRIMITIVES:
                        raise Refused(f"{b}: kernel {k['variant']} names primitive {prim!r}, which the table does not cover")
                    for arch in k["archs"]:
                        if arch not in archs:
                            raise Refused(f"{b}: kernel {k['variant']} names arch {arch}, which the table does not cover")
                        rows.append({
                            "primitive": prim, "selection": selection, "library": k["library"],
                            "library_version": version, "variant": k["variant"], "target_arch": arch,
                            "hash_basis": basis, "hash_inputs": inputs,
                            "code_hash": code_hash(basis, k["library"], k["variant"], arch, inputs),
                            "toolkit": tk, "selected_by": sorted(selected)})
        for n, v in part_of.items():
            if v not in {k["variant"] for k in kernels}:
                raise Refused(f"{b}: {n} is classified as part of {v}, which the backend does not list")
    for b in sorted(set(kernels_by_backend) - set(backends)):
        raise Refused(f"the classification lists kernels for {b}, which has no status")

    key = lambda r: (r["primitive"], r["selection"], r["library"], r["library_version"], r["variant"], r["target_arch"])
    rows.sort(key=key)
    for a, b in zip(rows, rows[1:]):
        if key(a) == key(b):
            raise Refused(f"two kernels of the classification are {b['library']} {b['variant']} for {b['primitive']} "
                          f"under {b['selection']} on {b['target_arch']}")
    gkey = lambda g: (g["primitive"], g["selection"], g.get("library", ""), g.get("variant", ""))
    by = {}
    for g in gaps:
        if gkey(g) in by and by[gkey(g)]["issue"] != g["issue"]:
            raise Refused(f"gap {gkey(g)} is named twice, with issues {by[gkey(g)]['issue']} and {g['issue']}")
        by[gkey(g)] = g
    gaps = sorted(by.values(), key=gkey)
    return {
        "kind": "KernelImplTable",
        "engine": {"kind": "vllm", "version": up["tag"].removeprefix("v"), "commit": commit},
        "primitives": sorted(PRIMITIVES),
        "archs": sorted(archs),
        "gaps": [{"primitive": g["primitive"], "selection": g["selection"],
                  **({"library": g["library"], "variant": g["variant"]} if g.get("variant") else {}),
                  "issue": g["issue"]} for g in gaps],
        "libraries": sorted([{"name": n, "version": libs[n]["version"], "pinned": True, "pin": libs[n]["pin"],
                              "origin": libs[n]["origin"]} for n in used] +
                            [{"name": n, "version": v["version"], "pinned": False, "pin": v["pin"], "origin": v["origin"]}
                             for n in used for v in libs[n]["seen"]],
                            key=lambda l: (l["name"], l["version"])),
        "implementations": rows,
    }


def enum_line(vllm, member):
    rel = "vllm/v1/attention/backends/registry.py"
    for n in ast.walk(ast.parse(read(vllm, rel))):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == member:
            return n.lineno
    raise Refused(f"{rel}: no member {member}")


def render(table):
    if not table["gaps"]:
        del table["gaps"]
    return json.dumps(table, indent=2, ensure_ascii=False) + "\n"


def load_classification(tag, path=None):
    """The classification for the vLLM release tag (kernels/<tag>/attention.json unless path is
    given). It must name the same tag: a table is generated for one release."""
    path = path or classification_path(tag)
    if not os.path.exists(path):
        raise Refused(f"no classification for vLLM {tag}: {path} does not exist; copy the nearest release's and "
                      f"run the generator against {tag}'s source, which names every backend and import to classify")
    with open(path, encoding="utf-8") as f:
        cls = json.load(f)
    if cls["upstream"].get("tag") != tag:
        raise Refused(f"{path} classifies vLLM {cls['upstream'].get('tag')}, not {tag}")
    return cls


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vllm_source")
    ap.add_argument("--upstream", required=True, metavar="TAG",
                    help="the vLLM release tag the source is checked out at, such as v0.33.0")
    ap.add_argument("--classification", metavar="FILE",
                    help="the classification to use instead of kernels/<TAG>/attention.json")
    ap.add_argument("--source", action="append", default=[], metavar="LIBRARY=DIR",
                    help="a library's source checkout at vLLM's pin")
    ap.add_argument("--check", metavar="FILE", help="compare with FILE instead of writing")
    args = ap.parse_args(argv)
    sources = {}
    for s in args.source:
        name, _, d = s.partition("=")
        if not d:
            ap.error(f"--source {s}: want LIBRARY=DIR")
        sources[name] = d
    try:
        cls = load_classification(args.upstream, args.classification)
        out = render(generate(args.vllm_source, cls, sources))
    except Refused as e:
        print(f"gen_kernel_impls: {e}", file=sys.stderr)
        return 1
    if args.check:
        with open(args.check, encoding="utf-8") as f:
            have = f.read()
        if have != out:
            print(f"gen_kernel_impls: {args.check} is not current; regenerate it", file=sys.stderr)
            return 1
        return 0
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
