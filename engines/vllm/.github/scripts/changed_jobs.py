#!/usr/bin/env python3
"""Say which ci.yml jobs a change needs, from the files it changes.

Reads the changed files (one repository-relative path per line) on standard input and prints
"job=true|false" lines, the form a workflow appends to $GITHUB_OUTPUT, and, for the race
detector's matrix, "race_matrix=<json>". With --all every job is true (a push to main, a run by
hand).

The rule, for each job: it runs if a changed file is one the job reads, and it is skipped when none is.
What a job reads is stated here and nowhere else:
  - the paths its own steps name (the lists below, one per job), and
  - for a job that runs Go tests, the directory of every package of this module its test
    binaries link (go list -deps -test), with everything under that directory, testdata
    included, plus the data its tests open from elsewhere (the *_EXTRA lists).
A path nobody has classified is read by no job and runs none, so the audit (test-changed-jobs.sh,
`audit`) fails the build when a Go test opens a file the lists do not tie to a job, or when a step
of ci.yml names a path its job's list does not cover. Files that no job reads (a README, the
documents under docs/, the other workflows) run only the jobs that always run.

Usage: changed_jobs.py [--all] [--legs FILE] < changed-files
"""
import functools
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LEGS_FILE = os.path.join(ROOT, ".github", "shards", "race-legs.json")

# Changing any of these changes every job of ci.yml: the workflow, the selection itself (a wrong selection skips
# jobs), and the Go setup every job uses. Other files under .github/ are data or code that one kind of job reads
# (the shard tool and weights, the race legs, the oracle's composite actions) or none (the other workflows, the
# pull-request template, scripts that only people run): they are listed with the jobs that read them below, so a
# new weight or a line in a pull-request template does not run the cost of the whole workflow.
SHARED = (r"^\.github/(workflows/ci\.yml|actions/go-setup/|scripts/(changed_jobs|audit_triggers)\.py|scripts/test-changed-jobs\.sh)")
SHARD_TOOL = (r"^\.github/scripts/(shard\.py|run-go-shard\.sh|go-test-items\.sh|go-test-json\.sh|test_shard\.py)$|"
              r"^\.github/shards/(weights|split)\.txt$")
RACE_LEGS = r"^\.github/shards/race-legs\.json$"
GOMOD = r"^go\.(mod|sum)$"
GOFILE = r"\.go$"

# Data the Go tests of the packages outside kerneltests open from outside their own directories.
REST_EXTRA = [
    r"^kerneltests/testdata/",
    r"^testdata/",
    r"^oracle/(scenarios[^/]*/|mocks\.json$|drafts/)",
    r"^docs/metrics-(manifest|buckets)\.json$",
    r"^fields\.json$",
    r"^vllmengine/testdata/",
    r"^core/(kv|lower)/testdata/",
    r"^oracle/engineargs-cases\.json$",
]
# The census reads kerneltests/testdata (REST_EXTRA); the race legs of other packages read what REST_EXTRA names.
RACE_EXTRA = REST_EXTRA

# kerneltests/docs is read by the examples package (its documents' code blocks); the other directories under
# kerneltests/ are packages or testdata inside packages.
KERNELTESTS_PATHS = [r"^kerneltests/docs/"]
# run-inpackage.sh copies the blis-substrate this module pins, the tests under _inpackage and the fixtures under testdata.
INPACKAGE_PATHS = [r"^kerneltests/(testdata|scripts)/"]

ORACLE_GO = ["./core/sched", "./core/metrics", "./core/engine", "./core/connector/...", "./cmd/provenance", "./core/kv", "./core/vllm"]
ORACLE_PATHS = [
    r"^oracle/",
    r"^\.github/actions/oracle-(fuzz|env)/",
    r"^\.github/scripts/shard\.py$",
    r"^\.github/shards/weights-inertness\.txt$",
    r"^docs/metrics-(manifest|buckets)\.json$",
    r"^core/kv/testdata/",
    r"^compat/quantization/resolved\.json$",
    r"^core/lower/testdata/",
    r"^rules/engineconfig/testdata/",
    r"^rules/v0_32/testdata/",
]
METRICS_GO = ["./cmd/metricsmanifest", "./registry"]
METRICS_PATHS = [r"^fields/", r"^fields\.json$", r"^docs/metrics-manifest\.json$"]
GOLDEN_PATHS = [r"^oracle/", r"^core/.*/testdata/", r"^tools/migrateengineblock/"]
BENCH_GO = ["./core/sched", "./core/vllm"]
PACKS = {"v0_29": "./rules/v0_29", "v0_31": "./rules/v0_31", "v0_32": "./rules/v0_32"}


@functools.lru_cache(maxsize=None)
def _run(args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def run(*args):
    return _run(tuple(args))


def rel(path):
    """PATH (an absolute directory or file go list printed) relative to the checkout. A path outside the checkout
    means go list ran somewhere else than we think, and every comparison after it would be wrong: stop."""
    r = os.path.relpath(os.path.realpath(path), ROOT)
    if r == ".." or r.startswith("../"):
        raise SystemExit("changed_jobs: %s is outside the checkout %s" % (path, ROOT))
    return r


def checked_list(patterns):
    """go list -e hides a package it cannot load; an empty or short closure would then skip jobs that must run.
    Fail instead."""
    bad = run("go", "list", "-e", "-f", "{{if .Error}}{{.ImportPath}}: {{.Error}}{{end}}", *patterns).strip()
    if bad:
        raise SystemExit("changed_jobs: go list cannot load packages, so the selection cannot be trusted:\n" + bad)


MODULE = None


def module():
    global MODULE
    if MODULE is None:
        MODULE = run("go", "list", "-m").strip()
    return MODULE


def packages(patterns, exclude=None):
    pk = run("go", "list", "-e", *patterns).split()
    if exclude:
        pk = [p for p in pk if not re.search(exclude, p)]
    return pk


_ALL_DIRS = None


def all_package_dirs():
    """The directory of every package of the module (kerneltests/_inpackage included, which wildcards skip)."""
    global _ALL_DIRS
    if _ALL_DIRS is None:
        fmt = "{{.Dir}}"
        checked_list(["./...", "./kerneltests/_inpackage"])
        ds = run("go", "list", "-e", "-f", fmt, "./...", "./kerneltests/_inpackage").split()
        _ALL_DIRS = {rel(d) for d in ds}
        if len(_ALL_DIRS) < 10:
            raise SystemExit("changed_jobs: go list found only %d packages; the selection cannot be trusted" % len(_ALL_DIRS))
    return _ALL_DIRS


def owner(path):
    """The package a file belongs to: the deepest package directory above it. None for a file that
    sits in no package directory (docs/, oracle/, fields/, the root's data files)."""
    d = os.path.dirname(path)
    dirs = all_package_dirs()
    while d and d != ".":
        if d in dirs:
            return d
        d = os.path.dirname(d)
    return None


# The files of a package that its importers compile: not its tests, not its testdata.
_BUILD_FILES = ("GoFiles", "CgoFiles", "CFiles", "CXXFiles", "MFiles", "HFiles", "SFiles", "SwigFiles", "SwigCXXFiles",
                "SysoFiles", "EmbedFiles")
_TEST_FILES = ("TestGoFiles", "XTestGoFiles", "TestEmbedFiles", "XTestEmbedFiles")


def closure(patterns, exclude=None):
    return _closure(tuple(patterns), exclude)


@functools.lru_cache(maxsize=None)
def _closure(patterns, exclude):
    """What running the tests of PATTERNS reads from this module's packages:
      dirs   the packages themselves: everything that belongs to them, testdata and tests included;
      files  the files other linked packages of the module compile into the binaries (their sources and
             embedded files, not their tests or testdata), and the root package's own files.
    A package's tests are its own input only; an importer's tests are not."""
    pk = packages(patterns, exclude)
    if not pk:
        raise SystemExit("changed_jobs: go list %s names no package" % " ".join(patterns))
    checked_list(pk)
    dirs = {rel(d) for d in run("go", "list", "-e", "-f", "{{.Dir}}", *pk).split()}
    def tmpl(fields):
        return "".join("{{range $.%s}}{{$.Dir}}/{{.}}\n{{end}}" % f for f in fields)
    mod = module()
    fmt = "{{with .Module}}{{if eq .Path \"%s\"}}%s{{end}}{{end}}" % (mod, tmpl(_BUILD_FILES))
    files = {rel(f) for f in run("go", "list", "-e", "-deps", "-test", "-f", fmt, *pk).split()}
    if "." in dirs:
        dirs.discard(".")
        fmt2 = tmpl(_BUILD_FILES + _TEST_FILES)
        files |= {rel(f) for f in run("go", "list", "-e", "-f", fmt2, ".").split()}
    return dirs, files


def explicit(patterns):
    return closure(patterns)


@functools.lru_cache(maxsize=None)
def embedded():
    """Every file a package of the module embeds: a change to one changes what compiles."""
    fmt = "{{$d := .Dir}}{{range .EmbedFiles}}{{$d}}/{{.}}\n{{end}}{{range .TestEmbedFiles}}{{$d}}/{{.}}\n{{end}}{{range .XTestEmbedFiles}}{{$d}}/{{.}}\n{{end}}"
    return {rel(p) for p in run("go", "list", "-e", "-f", fmt, "./...").split("\n") if p.strip()}


def any_match(files, patterns):
    rx = [re.compile(p) for p in patterns]
    return any(r.search(f) for f in files for r in rx)


def in_closure(files, dirs, exact):
    for f in files:
        if f in exact:
            return True
        o = owner(f)
        if o is not None and o in dirs:
            return True
    return False


def is_build_go(f):
    """A .go file `go build ./...` and the race guard read: not a test, not testdata, not an underscore directory."""
    return f.endswith(".go") and not f.endswith("_test.go") and "/testdata/" not in "/" + f and not any(
        seg.startswith(("_", ".")) for seg in f.split("/")[:-1])


def is_lint_go(f):
    """golangci-lint ./... lints tests too, but not testdata or underscore directories."""
    return f.endswith(".go") and "/testdata/" not in "/" + f and not any(seg.startswith(("_", ".")) for seg in f.split("/")[:-1])


def decide(files, everything=False, legs_file=LEGS_FILE):
    out = {}
    jobs = ["go", "bench", "lint", "rest", "kerneltests", "inpackage", "race", "race_guard", "metrics", "goldens", "oracle"] + ["packs_" + k for k in PACKS]
    if everything:
        for j in jobs:
            out[j] = True
        legs = json.load(open(legs_file))
        out["race_matrix"] = json.dumps({"include": legs}, separators=(",", ":"))
        return out
    shared = [SHARED]
    modfiles = shared + [GOMOD]

    emb = embedded()
    out["go"] = any_match(files, modfiles + [GOFILE]) or any(f in emb for f in files)
    out["lint"] = any_match(files, modfiles + [r"^\.golangci\.yml$"]) or any(is_lint_go(f) for f in files)

    d, rf = closure(["./..."], exclude=r"/kerneltests(/|$)")
    out["rest"] = any_match(files, modfiles + [SHARD_TOOL] + REST_EXTRA) or in_closure(files, d, rf)

    kd, krf = closure(["./kerneltests/..."])
    out["kerneltests"] = any_match(files, modfiles + [SHARD_TOOL] + KERNELTESTS_PATHS) or in_closure(files, kd, krf)

    idir, irf = explicit(["./kerneltests/_inpackage"])
    out["inpackage"] = any_match(files, modfiles + [SHARD_TOOL] + INPACKAGE_PATHS) or in_closure(files, idir, irf)

    out["race_guard"] = any_match(files, shared + [RACE_LEGS, r"^\.github/scripts/check_race_legs\.py$"]) or any(is_build_go(f) for f in files)
    legs = json.load(open(legs_file))
    chosen = []
    cache = {}
    for leg in legs:
        pk = leg.get("pkgs") or leg["race_pkgs"]
        if pk not in cache:
            cache[pk] = closure(pk.split())
        dd, rr = cache[pk]
        if any_match(files, modfiles + [SHARD_TOOL, RACE_LEGS] + RACE_EXTRA) or in_closure(files, dd, rr):
            chosen.append(leg)
    out["race"] = bool(chosen)
    out["race_matrix"] = json.dumps({"include": chosen}, separators=(",", ":"))

    bd, brf = closure(BENCH_GO)
    out["bench"] = any_match(files, modfiles) or in_closure(files, bd, brf)

    md, mrf = closure(METRICS_GO)
    out["metrics"] = any_match(files, modfiles + METRICS_PATHS) or in_closure(files, md, mrf)

    out["goldens"] = any_match(files, shared + GOLDEN_PATHS)

    od, orf = closure(ORACLE_GO)
    out["oracle"] = any_match(files, modfiles + ORACLE_PATHS) or in_closure(files, od, orf)

    for k, pkg in PACKS.items():
        pd, prf = closure([pkg])
        out["packs_" + k] = any_match(files, modfiles) or in_closure(files, pd, prf)
    return out


def batch(text):
    """--batch: change lists separated by lines of '--'; one line each: the jobs selected, a '|', the race legs."""
    for block in text.split("\n--\n"):
        files = sorted({l.strip() for l in block.split("\n") if l.strip() and l.strip() != "--"})
        out = decide(files)
        jobs = " ".join(sorted(k for k, v in out.items() if v is True)) or "-"
        legs = " ".join(l["name"] for l in json.loads(out["race_matrix"])["include"]) or "-"
        print("%s|%s" % (jobs, legs))


def main():
    args = sys.argv[1:]
    if "--batch" in args:
        batch(sys.stdin.read())
        return
    everything = "--all" in args
    legs_file = LEGS_FILE
    if "--legs" in args:
        legs_file = args[args.index("--legs") + 1]
    files = sorted({l.strip() for l in sys.stdin if l.strip()})
    out = decide(files, everything, legs_file)
    for k in sorted(out):
        v = out[k]
        print("%s=%s" % (k, ("true" if v else "false") if isinstance(v, bool) else v))


if __name__ == "__main__":
    main()
