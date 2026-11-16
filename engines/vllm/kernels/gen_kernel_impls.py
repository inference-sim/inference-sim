# Original to blis-vllm.
"""Write kernel_impls.json: every attention kernel implementation vLLM can emit, with its code hash.

The table is the one design section 6.2 asks each engine release to publish: for each lowered
primitive it covers, every kernel the release can run, by library, the library's own kernel name,
the device architecture and the code hash the coefficient registry keys its entries by (D5,
VAL-8). Its type, validator and hash rule are blis-substrate's contracts/spec/kernelimpls; a Go
test (kernel_impls_test.go) loads the committed file with that loader, which recomputes every
hash, so this script and the rule cannot drift apart.

WHAT COMES FROM vLLM, AND WHAT FROM kernels/attention.json.
  From vLLM v0.32.0's own source, read with the ast module (vLLM is not imported):
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
  From kernels/attention.json, which people edit: the status of each backend, and for each kernel
  of a listed backend its library, variant, primitives, architectures, gate and hash inputs.

The script refuses to write a table when the classification and the source disagree: an
AttentionBackendEnum member with no status or a status for one that does not exist; a backend the
CUDA platform selects marked as not running on CUDA; a name a listed backend imports from a kernel
library that the classification neither lists as a kernel nor as not one, or one it lists that the
module no longer imports; a kernel never called in its backend's module; a gate function that does
not exist; a checkout at another commit or version than the pin. So the table is complete for the
backends it lists by construction, and a backend it does not list is a gap with an issue.

Usage:
    python3 -I kernels/gen_kernel_impls.py <vllm-source> kernels/attention.json \\
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
        m = re.match(r"^\s*" + re.escape(dist) + r"\s*==\s*([^\s;#]+)", line)
        if m:
            return m.group(1), cite(rel, i, commit)
    raise Refused(f"{rel}: no pin of {dist}")


def cmake_pin(vllm, rel, commit):
    for i, line in enumerate(read(vllm, rel).splitlines(), 1):
        m = re.match(r"^\s*GIT_TAG\s+([0-9a-f]{40})\s*$", line)
        if m:
            return m.group(1), cite(rel, i, commit)
    raise Refused(f"{rel}: no GIT_TAG naming a commit")


def toolkit(vllm, commit):
    rel = "docker/Dockerfile"
    for i, line in enumerate(read(vllm, rel).splitlines(), 1):
        m = re.match(r"^ARG CUDA_VERSION=(\S+)$", line)
        if m:
            return m.group(1)
    raise Refused(f"{rel}: no ARG CUDA_VERSION with a default")


def object_id(root, path):
    """The path with the git object it names at HEAD: a blob, a tree, or the commit a submodule
    is pinned to (read from the tree, so the submodule need not be checked out)."""
    out = git(root, "ls-tree", "HEAD", "--", path)
    lines = [ln for ln in out.splitlines() if ln.split("\t", 1)[-1] == path]
    if len(lines) != 1:
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


def generate(vllm, cls, sources):
    up = cls["upstream"]
    commit = git(vllm, "rev-parse", "HEAD")
    if commit != up["commit"]:
        raise Refused(f"{vllm} is at {commit}; the classification is for vLLM {up['version']} at {up['commit']}")

    archs = cls["archs"]
    libs = {}
    for name, lib in sorted(cls["libraries"].items()):
        if "pip" in lib:
            version, pin = pip_pin(vllm, lib["requirements"], lib["pip"], commit)
        elif "cmake" in lib:
            version, pin = cmake_pin(vllm, lib["cmake"], commit)
        elif lib.get("engine"):
            version, pin = commit, "the engine release itself"
        else:
            raise Refused(f"library {name}: no pip, cmake or engine pin")
        root = vllm if lib.get("engine") else sources.get(name)
        if root is None:
            raise Refused(f"library {name}: no checkout given (--source {name}=<dir>)")
        head = git(root, "rev-parse", "HEAD")
        if "cmake" in lib and head != version:
            raise Refused(f"{root} is at {head}; vLLM builds {name} from {version} ({pin})")
        if "version_file" in lib:
            have = read(root, lib["version_file"]).strip()
            if have != version:
                raise Refused(f"{root} is version {have}; vLLM pins {name} {version} ({pin})")
        libs[name] = {"version": version, "pin": pin, "root": root,
                      "origin": f"{lib['repo']} {head}" if "repo" in lib else f"vLLM {head}",
                      "modules": lib["modules"]}

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
        if st["status"] == "gap":
            for prim in st.get("primitives", sorted(PRIMITIVES)):
                gaps.append({"primitive": prim, "selection": f"attention backend {b}", "issue": st["issue"]})
        for g in st.get("gaps", []):
            gaps.append({"primitive": g["primitive"], "selection": f"attention backend {b}: {g['selection']}",
                         "issue": g["issue"]})
        if st["status"] != "listed":
            if b in kernels_by_backend:
                raise Refused(f"{b} is {st['status']} but the classification lists kernels for it")
            continue
        rel = module_path(enum[b])
        tree = ast.parse(read(vllm, rel))
        prefixes = sorted({m for lib in libs.values() for m in lib["modules"]})
        imported = imports_from(tree, prefixes)
        kernels = kernels_by_backend.get(b, [])
        gap_calls = {g["call"] for g in st.get("gaps", []) if g.get("call")}
        classified = {k["call"] for k in kernels} | set(st.get("not_kernels", {})) | gap_calls
        for name in sorted(set(imported) - classified):
            mod, line = imported[name]
            raise Refused(f"{rel}:{line}: {b} imports {name} from {mod}, which the classification neither lists "
                          f"as a kernel nor as not one")
        for name in sorted((set(st.get("not_kernels", {})) | gap_calls) - set(imported)):
            raise Refused(f"{b}: the classification names {name}, but {rel} does not import it from a library module")
        for k in kernels:
            if k["call"] not in imported:
                raise Refused(f"{b}: kernel {k['variant']} is called as {k['call']}, which {rel} does not import "
                              f"from a library module")
            line = first_call(tree, k["call"])
            if line is None:
                raise Refused(f"{rel}: {b} never calls {k['call']}")
            lib = libs[k["library"]]
            used.add(k["library"])
            selected = {cite(rel, line, commit), prio.get(b, cite("vllm/v1/attention/backends/registry.py",
                                                                  enum_line(vllm, b), commit))}
            for g in k.get("gates", []):
                grel, func = g.split(":")
                selected.add(cite(grel, def_line(vllm, grel, func), commit))
            if "artifact" in k["hash"]:
                if "checksums" not in cls["libraries"][k["library"]]:
                    raise Refused(f"{b}: kernel {k['variant']} is a binary artifact, and library {k['library']} "
                                  f"names no checksum list")
                basis, inputs = "binary", [artifact_checksum(lib["root"], cls["libraries"][k["library"]]["checksums"],
                                                             k["hash"]["artifact"])]
            else:
                basis, inputs = "source", [object_id(lib["root"], p) for p in k["hash"]["source"]]
            for prim in k["primitives"]:
                if prim not in PRIMITIVES:
                    raise Refused(f"{b}: kernel {k['variant']} names primitive {prim!r}, which the table does not cover")
                for arch in k["archs"]:
                    if arch not in archs:
                        raise Refused(f"{b}: kernel {k['variant']} names arch {arch}, which the table does not cover")
                    rows.append({
                        "primitive": prim, "library": k["library"], "library_version": lib["version"],
                        "variant": k["variant"], "target_arch": arch, "hash_basis": basis,
                        "hash_inputs": sorted(inputs),
                        "code_hash": code_hash(basis, k["library"], k["variant"], arch, inputs),
                        "toolkit": toolkit(vllm, commit), "selected_by": sorted(selected)})
    for b in sorted(set(kernels_by_backend) - set(backends)):
        raise Refused(f"the classification lists kernels for {b}, which has no status")

    rows.sort(key=lambda r: (r["primitive"], r["library"], r["library_version"], r["variant"], r["target_arch"]))
    for a, b in zip(rows, rows[1:]):
        if (a["primitive"], a["library"], a["variant"], a["target_arch"]) == (
                b["primitive"], b["library"], b["variant"], b["target_arch"]):
            raise Refused(f"two kernels of the classification are {b['library']} {b['variant']} for {b['primitive']} "
                          f"on {b['target_arch']}")
    gaps = sorted({(g["primitive"], g["selection"]): g for g in gaps}.values(),
                  key=lambda g: (g["primitive"], g["selection"]))
    return {
        "kind": "KernelImplTable",
        "engine": {"kind": "vllm", "version": up["version"], "commit": commit},
        "primitives": sorted(PRIMITIVES),
        "archs": sorted(archs),
        "gaps": gaps,
        "libraries": [{"name": n, "version": libs[n]["version"], "pinned": True, "pin": libs[n]["pin"],
                       "origin": libs[n]["origin"]} for n in sorted(used)],
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


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vllm_source")
    ap.add_argument("classification")
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
    with open(args.classification, encoding="utf-8") as f:
        cls = json.load(f)
    try:
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
