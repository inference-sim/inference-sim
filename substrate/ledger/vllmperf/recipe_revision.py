#!/usr/bin/env python3
"""Count how many vLLM CI perf rows have a different recipe at their own date than at the pinned one.

    recipe_revision.py --api perf.json --perf-eval DIR

perf.json is the body of https://ci.vllm.ai/api/perf; DIR is a full checkout of
vllm-project/perf-eval at the commit sources.yaml pins (git history needed). A row joins a recipe
as export.py and the ingest join them: model, GPU, input and output lengths and concurrency. The
row's date is the time its result was posted (lib/ingest_perf.py takes the UTC time of the post),
so the recipe at the date is the last commit of the workload file at or before it. Prints counts
only; nothing is written. This backs Table 2.3 in evidence/SETTINGS-PROVENANCE.md.
"""
import argparse
import collections
import datetime as dt
import glob
import json
import os
import subprocess

import yaml


def norm(s):
    return " ".join(str(s or "").split())


def recipes(text):
    w = yaml.safe_load(text) or {}
    v = w.get("vllm") or {}
    for c in (w.get("vllm_bench") or {}).get("configs") or []:
        conc = c["max_concurrency"]
        for n in conc if isinstance(conc, list) else [conc]:
            yield ((str(v.get("model")), str(w.get("gpu")).lower(), str(c["input_len"]), str(c["output_len"]), str(n)), norm(v.get("serve_args")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--perf-eval", required=True)
    a = ap.parse_args()

    def git(*args):
        return subprocess.run(["git", "-C", a.perf_eval, *args], capture_output=True, text=True, check=True).stdout

    rows = json.load(open(a.api))["rows"]
    head, history = {}, {}
    for path in sorted(glob.glob(os.path.join(a.perf_eval, "workloads", "*.yaml"))):
        rel = "workloads/" + os.path.basename(path)
        for key, args in recipes(open(path).read()):
            head.setdefault(key, []).append((rel, args))
        history[rel] = [(l.split()[0], dt.datetime.fromisoformat(l.split()[1])) for l in git("log", "--format=%H %cI", "--", rel).splitlines()]
    at = {}

    def recipe_at(rel, commit):
        if (rel, commit) not in at:
            at[rel, commit] = dict(recipes(git("show", f"{commit}:{rel}")))
        return at[rel, commit]

    n = collections.Counter()
    for r in rows:
        key = (r["model"], r["device"].lower(), r["isl"], r["osl"], r["conc"])
        found = head.get(key, [])
        if len(set(found)) != 1:
            n["no_single_recipe_at_pin"] += 1
            continue
        rel, pinned = found[0]
        n["joined_at_pin"] += 1
        d = dt.datetime.fromisoformat(r["date"].replace(" ", "T") + "+00:00")
        before = [c for c, t in history[rel] if t <= d]
        if any(t.date() == d.date() and t <= d for _, t in history[rel]):
            n["file_changed_earlier_that_utc_day"] += 1
        if not before or key not in recipe_at(rel, before[0]):
            n["recipe_absent_at_date"] += 1
        elif recipe_at(rel, before[0])[key] != pinned:
            n["serve_args_differ_at_date"] += 1
            if any(t.date() == d.date() and t <= d for _, t in history[rel]):
                n["serve_args_differ_and_file_changed_that_day"] += 1
    print(json.dumps({"rows": len(rows), **n}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
