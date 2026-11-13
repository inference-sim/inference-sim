#!/usr/bin/env python3
"""Fail if a job reads a file that changed_jobs.py does not tie to it.

changed_jobs.py skips a job when no changed file is one the job reads. That is only safe while its
lists are complete, so this audit derives what the jobs read from the repository itself and checks that
changing each such file selects the job:

  go tests   a *_test.go that opens a path outside its own package's directory (../../oracle/mocks.json,
             filepath.Join("..", "..", "testdata", ...)) must be selected by the job that runs that package;
  ci.yml     every repository path a job's steps name (python3 -I oracle/compare_goldens.py core/...) must
             select that job;
  oracle     every repository path a Python file under oracle/ names must select the oracle job.

Usage: audit_triggers.py            (from anywhere in the checkout; exit 1 and a list on a gap)
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import changed_jobs as cj  # noqa: E402

ROOT = cj.ROOT
CI = os.path.join(ROOT, ".github", "workflows", "ci.yml")
failures = []
checked = 0
_decisions = {}


def selects(path, key):
    """Does changing PATH select the job KEY? PATH is a file, or a directory: then a file inside it."""
    full = os.path.join(ROOT, path)
    if os.path.isdir(full):
        inside = [os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(full) for f in fs]
        probes = sorted(inside)[:1] or [path + "/x"]
    else:
        probes = [path]
    for p in probes:
        if p not in _decisions:
            _decisions[p] = cj.decide([p])
        if _decisions[p][key]:
            return True
    return False


def need(path, keys, why):
    global checked
    checked += 1
    if not any(selects(path, k) for k in keys):
        failures.append("%s: changing it selects none of %s (%s)" % (path, "/".join(keys), why))


def strip_wild(p):
    for i, ch in enumerate(p):
        if ch in "*?[":
            p = p[:i]
            break
    return p.rstrip("/") if "/" in p else p


def go_tests():
    legs = cj.json.load(open(cj.LEGS_FILE))
    raced = set()
    for l in legs:
        for p in (l.get("pkgs") or l["race_pkgs"]).split():
            raced.add(os.path.normpath(p))
    lit = re.compile(r'"((?:\.\./)+[^"\n]*)"')
    join = re.compile(r'filepath\.Join\(((?:\s*"[^"\n]*"\s*,?)+)')
    for d, dirs, files in os.walk(ROOT):
        rel = os.path.relpath(d, ROOT)
        dirs[:] = [x for x in dirs if x not in (".git", "testdata", "node_modules")]
        for f in files:
            if not f.endswith("_test.go"):
                continue
            src = open(os.path.join(d, f), encoding="utf-8").read()
            found = set()
            for m in lit.finditer(src):
                found.add(m.group(1))
            for m in join.finditer(src):
                parts = re.findall(r'"([^"\n]*)"', m.group(1))
                if parts and parts[0] == "..":
                    found.add("/".join(parts))
            targets = set()
            for lt in found:
                targets.add(os.path.normpath(os.path.join(rel, strip_wild(lt))))
            if found:
                # A test that climbs out of its directory often joins a list of repository-relative paths
                # onto the root ("testdata/scenarios", "vllmengine/testdata"): count those literals too.
                for m in re.finditer(r'"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.*-]+)+)"', src):
                    cand = os.path.normpath(strip_wild(m.group(1)))
                    if os.path.exists(os.path.join(ROOT, cand)) and not cand.startswith(rel + os.sep):
                        targets.add(cand)
            for target in sorted(targets):
                if target in (".", "..") or target.startswith("..") or not os.path.exists(os.path.join(ROOT, target)):
                    continue
                if target == rel or target.startswith(rel + os.sep):
                    continue  # inside its own package
                if rel.startswith("kerneltests/_inpackage"):
                    keys = ["inpackage"]
                elif rel == "kerneltests" or rel.startswith("kerneltests/"):
                    keys = ["kerneltests"]
                else:
                    keys = ["rest"]
                need(target, keys, "read by %s/%s" % (rel, f))
                if os.path.normpath("./" + rel) in raced:
                    need(target, ["race"], "read by %s/%s, which a race leg runs" % (rel, f))


# Jobs of ci.yml and the output that selects each.
JOB_KEYS = {
    "go-build": ["go"],
    "go-race-guard": ["race_guard"],
    "lint": ["lint"],
    "metrics-manifest": ["metrics"],
    "goldens-reviewed": ["goldens"],
}
PATH_TOKEN = re.compile(r"(?<![\w$/.{}-])((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.*-]*)")


def jobs_of(text):
    """job id -> its text, from the top-level 'jobs:' mapping."""
    out, cur, name = {}, [], None
    in_jobs = False
    for line in text.split("\n"):
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        m = re.match(r"^  ([a-z0-9][a-z0-9-]*):\s*$", line)
        if m:
            if name:
                out[name] = "\n".join(cur)
            name, cur = m.group(1), []
        elif name:
            cur.append(line)
    if name:
        out[name] = "\n".join(cur)
    return out


def ci_yml():
    text = open(CI, encoding="utf-8").read()
    jobs = jobs_of(text)
    keys = dict(JOB_KEYS)
    keys.update({j: ["oracle"] for j in jobs if j.startswith("oracle")})
    keys.update({"packs-against-vllm": ["packs_v0_29", "packs_v0_31", "packs_v0_32"]})
    for job, ks in keys.items():
        if job not in jobs:
            continue
        body = "\n".join(l for l in jobs[job].split("\n") if not l.lstrip().startswith("#"))
        for m in PATH_TOKEN.finditer(body):
            tok = strip_wild(m.group(1))
            if tok.startswith(("./", "http", "oracle/out", "vllm-source", "actions/", "manifest-out", "tier1-out", "oracle-")):
                continue
            if tok in ("", ".") or not os.path.exists(os.path.join(ROOT, tok)):
                continue
            if tok.startswith(".github/workflows/"):
                continue
            # A job that reads the path in any of its legs is selected by any of KS.
            need(tok, ks, "named in ci.yml job %s" % job)


def oracle_python():
    top = r"(?:core|compat|rules|docs|fields|registry|kerneltests|testdata|cmd|vllmengine|internal)"
    rx = re.compile(r"[\"'](%s/[A-Za-z0-9_./*-]+|fields\.json|go\.mod)[\"']" % top)
    for py in glob.glob(os.path.join(ROOT, "oracle", "**", "*.py"), recursive=True):
        if os.path.basename(py).startswith("mutate"):
            continue  # mutation.yml's sweeps, run by hand, not by ci.yml
        src = open(py, encoding="utf-8").read()
        for m in rx.finditer(src):
            tok = strip_wild(m.group(1))
            if tok and os.path.exists(os.path.join(ROOT, tok)):
                need(tok, ["oracle"], "named by %s" % os.path.relpath(py, ROOT))


def main():
    go_tests()
    ci_yml()
    oracle_python()
    if failures:
        print("\n".join(sorted(set(failures))))
        print("audit: %d of %d reads are not tied to their job: add the path to changed_jobs.py" % (len(set(failures)), checked))
        sys.exit(1)
    print("audit: %d file reads, each tied to the job that reads it" % checked)


if __name__ == "__main__":
    main()
