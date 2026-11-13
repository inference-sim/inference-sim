#!/usr/bin/env python3
"""The race detector's legs (.github/shards/race-legs.json) run exactly the packages ci.yml's go-race-guard guards.

A package in RACE_PKGS and in no leg would pass the guard and never be raced; a leg whose package is not in
RACE_PKGS (the census, which names its package as race_pkgs) is allowed only as a leg of its own kind. The shard
legs of one package (vllmengine-0..2) must cover shard 0..shards-1 exactly once each, and the slices of the census
(k/n) must cover 0..n-1 exactly once each. Exit 1 with a message on any gap.
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def golist(pkgs):
    return set(subprocess.run(["go", "list", *pkgs], cwd=ROOT, check=True, capture_output=True, text=True).stdout.split())


def main(legs_file=None, ci_file=None):
    legs = json.load(open(legs_file or os.path.join(ROOT, ".github/shards/race-legs.json")))
    text = open(ci_file or os.path.join(ROOT, ".github/workflows/ci.yml"), encoding="utf-8").read()
    m = re.search(r"RACE_PKGS:\s*>-\n((?:\s+\S.*\n)+)", text)
    if not m:
        sys.exit("check_race_legs: no RACE_PKGS in ci.yml: the guard was restructured, and this check must be updated with it")
    pkgs = [t for line in m.group(1).split("\n") if line.strip().startswith("./") for t in line.split()]
    guarded = golist(pkgs)
    if not legs:
        sys.exit("check_race_legs: race-legs.json has no legs")
    names = [l["name"] for l in legs]
    if len(set(names)) != len(names):
        sys.exit("check_race_legs: duplicate leg names")
    raced = set()
    shards, slices = {}, {}
    for l in legs:
        if bool(l.get("pkgs")) == bool(l.get("race_pkgs")):
            sys.exit("check_race_legs: leg %s names exactly one of pkgs, race_pkgs" % l["name"])
        if l.get("pkgs"):
            raced |= golist(l["pkgs"].split())
        if "shards" in l:
            shards.setdefault(l["pkgs"], []).append((l["shards"], l["shard"]))
        if "slice" in l:
            k, n = map(int, l["slice"].split("/"))
            slices.setdefault((l["race_pkgs"], l.get("run", "")), []).append((n, k))
    if raced != guarded:
        sys.exit("check_race_legs: RACE_PKGS and the legs differ\n  guarded only: %s\n  legs only: %s" % (sorted(guarded - raced), sorted(raced - guarded)))
    for pk, ent in shards.items():
        n = {e[0] for e in ent}
        got = sorted(e[1] for e in ent)
        if len(n) != 1 or got != list(range(n.pop())):
            sys.exit("check_race_legs: the shard legs of %s are %s, want 0..shards-1 once each" % (pk, got))
    for key, ent in slices.items():
        n = {e[0] for e in ent}
        got = sorted(e[1] for e in ent)
        if len(n) != 1 or got != list(range(n.pop())):
            sys.exit("check_race_legs: the slices of %s are %s, want 0..n-1 once each" % (key[0], got))
    print("check_race_legs: %d legs run the %d guarded packages; shard and slice legs are complete" % (len(legs), len(guarded)))


if __name__ == "__main__":
    main()
