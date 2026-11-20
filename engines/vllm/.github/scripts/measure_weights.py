#!/usr/bin/env python3
"""Write .github/shards/weights.txt from a `go test -json` run.

  go test -count=1 -json ./... > results.json
  python3 -I .github/scripts/measure_weights.py results.json > .github/shards/weights.txt

A package named in .github/shards/split.txt weighs test by test (top-level tests only); every other
package weighs once. The unit is seconds on the machine that ran it; only the ratios matter.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(path):
    split = {l.strip() for l in open(os.path.join(ROOT, ".github/shards/split.txt")) if l.strip() and not l.startswith("#")}
    module = None
    for l in open(os.path.join(ROOT, "go.mod")):
        if l.startswith("module "):
            module = l.split()[1]
    pk, tests = {}, {}
    for line in open(path, encoding="utf-8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("Action") not in ("pass", "fail", "skip") or "Elapsed" not in e:
            continue
        rel = e["Package"][len(module):].lstrip("/") or "."
        if e.get("Test"):
            if "/" not in e["Test"]:
                tests.setdefault(rel, {})[e["Test"]] = e["Elapsed"]
        elif e["Action"] != "skip":  # a package with no test files is skipped, and is not an item
            pk[rel] = e["Elapsed"]
    print("# Seconds a test or package took on one machine (go test -json, -count=1): the weights the shards are balanced by.")
    print("# Only the ratios matter. An item missing here weighs 1 second, so a new test is placed in a shard and never dropped.")
    print("# Regenerate with .github/scripts/measure_weights.py after a large change in what the slow tests do.")
    for rel, t in sorted(tests.items()):
        if rel in split:
            for n, s in sorted(t.items()):
                print("%s %s\t%.2f" % (rel, n, max(s, 0.05)))
    for rel, s in sorted(pk.items()):
        if rel not in split:
            print("%s\t%.2f" % (rel, max(s, 0.05)))


if __name__ == "__main__":
    main(sys.argv[1])
