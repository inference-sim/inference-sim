# Original to inference-sim.
"""The upgrade report of design 12.4: what moving an engine to another commit changes, and what it blocks.

The functions are the library: the dashboard imports them to grade a run with the same diff the
upgrade preflight uses (design 11.3), so the grade and the report cannot disagree. main() is a thin
command line over them: it reads an engine's committed lists, derives the new commit's from a
source checkout with the engine's own generators, and prints the report as JSON.

The report has five parts.
  surface       the items the new commit adds or removes, each added item with the status the
                engine's status.json gives it, or "unstatused" (an item no one has decided on); the
                items present at both whose "value" (the fact an item records from source, such as a
                default or a priority list) differs, each with its status; the transformers_fallback
                items are the Transformers-backed fallback paths
  fields        the fields added, removed or changed in status, and for a demand (field -> value ->
                points the dashboard could not run) which of it the engine models
  source        the ported files (every Go file's "Derived from" header, file level) whose upstream
                file changed, the host and graph code paths whose function hashes changed, and the
                kernel implementations whose code hash changed
  models        for each catalog model: its entry, how the engine supports it before and after, the
                models that run only through a Transformers fallback, and the missing coefficients
  coefficients  the CodeIDs and kernel code hashes of the new commit that no registry coefficient names

A coefficient names the code it was fitted for (code_ids, kernel_impls), so one the new commit does
not name is missing for every model on the engine; the report does not attribute it to one model.

Usage:
    python3 -I upgrade_report.py --engine vllm|sglang --repo <inference-sim checkout> \\
        --source <engine git checkout at the new commit> [--old-source <checkout at the pin>] \\
        [--demand demand.json] [--new-fields fields.json] [--new-kernel-impls kernel_impls.json]
"""

import argparse
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys

FALLBACK = "transformers_fallback"
DERIVED = re.compile(r"^// Derived from (\S+) at ", re.M)
CODE_IDS = re.compile(r"\bcode_ids:\s*\[([^\]]*)\]")
IMPLS = re.compile(r"\bkernel_impls:\s*\[(.*)\]\s*$", re.M)
HASH = re.compile(r"\bcode_hash:\s*([0-9a-f]{64})")
ARCH = re.compile(r"^\s+architecture:\s*(\S+)\s*$", re.M)


def _same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def surface_diff(old_items, new_items, statuses):
    """The surface items a commit adds, removes or changes. old_items are the committed surface.json
    items, new_items the items derived at the commit ({id, category, site, ...}), statuses the
    engine's status.json "items". Every added item carries its status, or "unstatused". An item at
    both whose "value" differs is changed: a status given for the old value does not carry over."""
    old = {i["id"]: i for i in old_items}
    new = {i["id"]: i for i in new_items}
    added = []
    for k in sorted(set(new) - set(old)):
        st = statuses.get(k) or {}
        row = {"id": k, "category": new[k]["category"], "site": new[k]["site"], "status": st.get("status", "unstatused")}
        for extra in ("issue", "reason"):
            if extra in st:
                row[extra] = st[extra]
        added.append(row)
    removed = [{"id": k, "category": old[k]["category"], "status": old[k]["status"]} for k in sorted(set(old) - set(new))]
    changed = [{"id": k, "category": new[k]["category"], "status": (statuses.get(k) or {}).get("status", "unstatused"),
                "from": old[k].get("value"), "to": new[k].get("value")}
               for k in sorted(set(old) & set(new)) if not _same(old[k].get("value"), new[k].get("value"))]
    by_status = {}
    for r in added:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "distance": {"added": len(added), "removed": len(removed), "changed": len(changed), "unstatused": by_status.get("unstatused", 0),
                     "added_by_status": dict(sorted(by_status.items()))},
        "transformers_fallback": [r["id"] for r in added if r["category"] == FALLBACK],
    }


def _covers(fld, value):
    vc = fld.get("value_classes") or {}
    if vc.get("type") in ("enumerated", "boolean") and "covered" in vc:
        return value in vc["covered"]
    return None


def field_demand(old_fields, new_fields, demand):
    """The field changes of a commit and what a demand needs of it. old_fields and new_fields are
    the "fields" of fields.json; demand is {field name: {value: points}}, the points the dashboard
    could not run for lack of that field's value ("" is the field with no value in view). A row is
    blocked when the field is absent or not modelled, or its value is outside the covered classes
    (numeric classes are not evaluated here)."""
    old = {(f["source"], f["name"]): f for f in old_fields}
    new = {(f["source"], f["name"]): f for f in new_fields}
    changed = [{"name": k[1], "source": k[0], "from": old[k]["status"], "to": new[k]["status"]}
               for k in sorted(set(old) & set(new)) if old[k]["status"] != new[k]["status"]]
    by_name = {}
    for f in new_fields:
        by_name.setdefault(f["name"], []).append(f)
    rows = []
    for name, values in demand.items():
        fields = by_name.get(name)
        for value, points in values.items():
            if not fields:
                rows.append({"field": name, "value": value, "points": points, "status": "absent", "blocked": True})
                continue
            modelled = [f for f in fields if f["status"] == "modelled"]
            status = "modelled" if modelled else fields[0]["status"]
            seen = [c for c in (_covers(f, value) for f in modelled) if c is not None] if value else []
            value_covered = any(seen) if seen else None
            rows.append({"field": name, "value": value, "points": points, "status": status,
                         "value_covered": value_covered, "blocked": status != "modelled" or value_covered is False})
    rows.sort(key=lambda r: (-r["points"], r["field"], r["value"]))
    return {
        "added": [{"name": k[1], "source": k[0], "status": new[k]["status"]} for k in sorted(set(new) - set(old))],
        "removed": [{"name": k[1], "source": k[0]} for k in sorted(set(old) - set(new))],
        "status_changed": changed,
        "demand": rows,
        "blocked_points": sum(r["points"] for r in rows if r["blocked"]),
    }


def code_path_diff(old_table, new_table):
    """Per path (host, graphs): the CodeIDs and the functions whose hash changed, appeared or went."""
    old = {p["path"]: p for p in old_table["paths"]}
    out = []
    for p in new_table["paths"]:
        o = old.get(p["path"], {"code_id": None, "functions": []})
        of = {f["id"]: f["hash"] for f in o["functions"]}
        nf = {f["id"]: f["hash"] for f in p["functions"]}
        out.append({"path": p["path"], "from": o["code_id"], "to": p["code_id"],
                    "changed": sorted(k for k in nf if k in of and of[k] != nf[k]),
                    "added": sorted(set(nf) - set(of)), "removed": sorted(set(of) - set(nf))})
    return out


def kernel_impls_diff(old, new):
    """The kernel implementations whose selection (primitive, selection, library, variant, target
    arch) is new, gone or has another code hash; hashes are the new code hashes of the added and
    changed ones, which a coefficient fitted for the build must be refitted or moved for."""
    def key(i):
        return (i["primitive"], i["selection"], i["library"], i["variant"], i["target_arch"])
    o = {key(i): i["code_hash"] for i in old["implementations"]}
    n = {key(i): i["code_hash"] for i in new["implementations"]}
    added = sorted(k for k in n if k not in o)
    changed = sorted(k for k in n if k in o and o[k] != n[k])
    return {"added": added, "removed": sorted(k for k in o if k not in n), "changed": changed,
            "hashes": sorted({n[k] for k in added + changed})}


def derived_files(texts):
    """{upstream path: [file]} from the "Derived from <path> at ..." header of each file; texts is
    {file: text}. The unit is a file: a ported Go file is not mapped to a function."""
    out = {}
    for name, text in texts.items():
        m = DERIVED.search(text)
        if m:
            out.setdefault(m.group(1), []).append(name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def source_diff(derived, changed_files, code_paths, kernel_impls):
    """The ported files whose upstream file changed in the commit range, with code_paths (the
    code_path_diff) and kernel_impls (the kernel_impls_diff, or None when the engine has none)."""
    changed = set(changed_files)
    return {
        "ported_files_changed": [{"upstream": u, "files": f} for u, f in derived.items() if u in changed],
        "code_paths": code_paths,
        "kernel_impls": kernel_impls,
    }


def uncovered(code_ids, impl_hashes, covered_code_ids, covered_hashes):
    """The new commit's CodeIDs ({path: id}) and the new code hashes of its added or changed kernel
    implementations that no coefficient names."""
    return {"code_ids": {p: c for p, c in sorted(code_ids.items()) if c not in covered_code_ids},
            "kernel_code_hashes": sorted(h for h in set(impl_hashes) if h not in covered_hashes)}


def model_report(models, old_arch, new_arch, missing):
    """models is {name: architecture} (the catalog); old_arch and new_arch are {architecture:
    "native" | "transformers"} before and after (old_arch None when no old tree is given); an
    architecture in neither table is "unregistered", which the engine runs through its Transformers
    backend or refuses. missing is the uncovered coefficients."""
    rows = []
    for name, arch in sorted(models.items()):
        after = new_arch.get(arch, "unregistered")
        before = None if old_arch is None else old_arch.get(arch, "unregistered")
        rows.append({"model": name, "architecture": arch, "catalog_entry": True, "support": {"from": before, "to": after},
                     "changed": None if before is None else before != after,
                     "transformers_only": after != "native", "missing_coefficients": missing})
    return {"models": rows, "transformers_only": [r["model"] for r in rows if r["transformers_only"]]}


# ----------------------------------------------------------------------------------------------
# The command line: reading the repository and the checkouts.


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def catalog_models(catalog):
    """{model: architecture} from each catalog entry's derived graph."""
    out = {}
    models = os.path.join(catalog, "models")
    for name in sorted(os.listdir(models)):
        g = os.path.join(models, name, "graph.yaml")
        if os.path.isfile(g):
            with open(g, encoding="utf-8") as f:
                m = ARCH.search(f.read())
            if m:
                out[name] = m.group(1)
    return out


def coverage(coefficients):
    """(CodeIDs, kernel code hashes) that some registry coefficient names."""
    ids, hashes = set(), set()
    for fn in sorted(os.listdir(coefficients)):
        if not fn.endswith(".yaml"):
            continue
        with open(os.path.join(coefficients, fn), encoding="utf-8") as f:
            text = f.read()
        for m in CODE_IDS.finditer(text):
            ids.update(x.strip() for x in m.group(1).split(",") if x.strip())
        for m in IMPLS.finditer(text):
            hashes.update(HASH.findall(m.group(1)))
    return ids, hashes


def go_headers(root):
    out = {}
    for d, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".go"):
                p = os.path.join(d, fn)
                with open(p, encoding="utf-8") as f:
                    out[os.path.relpath(p, root)] = f.read(400)
    return out


def git(src, *args):
    r = subprocess.run(["git", "-C", src] + list(args), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("upgrade_report: git %s: %s" % (" ".join(args), r.stderr.strip()))
    return r.stdout.strip()


def derive_items(gen, src, repo, manifest):
    """The generators differ in signature: vLLM's reads the metric manifest from the engine directory (repo)."""
    return gen.derive(src, repo, manifest) if len(inspect.signature(gen.derive).parameters) == 3 else gen.derive(src)


def build_report(a):
    base = os.path.join(a.repo, "engines", a.engine)
    surf = os.path.join(base, "tools", "surface")
    gen = load_module(os.path.join(surf, "gen_surface.py"), "gen_surface_" + a.engine)
    cp = load_module(os.path.join(a.repo, "substrate", "tools", "codepaths", "gen_code_paths.py"), "gen_code_paths")
    old_table = load_json(os.path.join(base, "code_paths.json"))
    commit, pin = git(a.source, "rev-parse", "HEAD"), old_table["engine"]["commit"]
    gen.UPSTREAM["tag"], gen.UPSTREAM["commit"] = gen.resolve_upstream(base, a.source, None, None)

    report = {"engine": a.engine, "from": {"version": old_table["engine"]["version"], "commit": pin}, "to": {"commit": commit}}
    statuses = load_json(os.path.join(surf, "status.json")).get("items", {})
    report["surface"] = surface_diff(load_json(os.path.join(base, "surface.json"))["items"],
                                     derive_items(gen, a.source, base, a.manifest), statuses)

    old_fields = load_json(os.path.join(base, "fields.json"))["fields"]
    new_fields = load_json(a.new_fields)["fields"] if a.new_fields else old_fields
    report["fields"] = field_demand(old_fields, new_fields, load_json(a.demand) if a.demand else {})

    spec = load_json(os.path.join(base, "code_paths", "spec.json"))
    spec["engine"]["commit"] = commit
    try:
        new_table = cp.generate(a.source, spec)
        paths, new_ids = code_path_diff(old_table, new_table), {p["path"]: p["code_id"] for p in new_table["paths"]}
    except cp.Refused as e:
        paths, new_ids = {"error": str(e)}, {}
    impls, new_hashes = None, []
    impls_path = os.path.join(base, "kernel_impls.json")
    if os.path.isfile(impls_path):
        old_impls = load_json(impls_path)
        new_impls = load_json(a.new_kernel_impls) if a.new_kernel_impls else old_impls
        impls = kernel_impls_diff(old_impls, new_impls)
        new_hashes = impls["hashes"]
    changed = git(a.source, "diff", "--name-only", pin, commit).splitlines() if pin != commit else []
    report["source"] = source_diff(derived_files(go_headers(base)), changed, paths, impls)

    ids, hashes = coverage(a.coefficients or os.path.join(a.repo, "substrate", "registry", "coefficients"))
    report["coefficients"] = uncovered(new_ids, new_hashes, ids, hashes)
    old_arch = gen.model_architectures(a.old_source) if a.old_source else None
    report["models"] = model_report(catalog_models(a.catalog or os.path.join(a.repo, "substrate", "catalog")),
                                    old_arch, gen.model_architectures(a.source), report["coefficients"])
    return report


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", required=True, choices=("vllm", "sglang"))
    ap.add_argument("--repo", default=".", help="the inference-sim checkout (default: the working directory)")
    ap.add_argument("--source", required=True, help="a git checkout of the engine at the new commit; it must hold the pin too")
    ap.add_argument("--old-source", help="a checkout at the pin, for the models' old support (omitted: not reported)")
    ap.add_argument("--demand", help='JSON {"field": {"value": points}}')
    ap.add_argument("--new-fields", help="the fields.json gen_fields wrote for the new commit")
    ap.add_argument("--new-kernel-impls", help="the kernel_impls.json gen_kernel_impls wrote for the new commit")
    ap.add_argument("--manifest", help="vLLM only: the metric manifest for the new commit")
    ap.add_argument("--catalog", help="default: <repo>/substrate/catalog")
    ap.add_argument("--coefficients", help="default: <repo>/substrate/registry/coefficients")
    json.dump(build_report(ap.parse_args(argv)), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
