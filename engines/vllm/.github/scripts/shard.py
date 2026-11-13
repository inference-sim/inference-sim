#!/usr/bin/env python3
"""Partition a list of test items into shards, and check that a shard ran exactly its share.

An item is one line: a Go package ("compat/build"), or a package and one of its top-level
tests ("compat/kernelmodel TestCensus"). Every shard lists the whole set itself, takes its share by
a rule that depends on the list and the weights alone, and then proves it ran exactly that share.
A package or test that is in no shard is therefore impossible by construction (the shards are
the classes of one function over one list), and the checks below fail the job if the list is
too short to be the real one, a shard is empty, or the shard's results do not hold the
items it was given.

  shard.py select -of N -index K -weights FILE      < items   > the items of shard K
  shard.py plan   -of N -weights FILE               < items   > "K<TAB>item" for every item
  shard.py check  -of N -min-items M -weights FILE  < items   (exit 1 unless the shards partition the items)
  shard.py verify -results go-test.json -mode package|test    < items   (exit 1 unless every item has a result)
  shard.py names  -results go-test.json                       > "package<TAB>test<TAB>action" for every test event

Weights: lines "item<TAB>seconds" (blank lines and # lines ignored). An item with no weight
weighs the default (-default, 1.0), so a new item is placed, never dropped.
"""
import argparse
import json
import sys


def read_items(stream):
    seen, items = set(), []
    for line in stream:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            sys.exit("shard: item listed twice: %r" % line)
        seen.add(line)
        items.append(line)
    return items


def read_weights(path):
    w = {}
    if not path:
        return w
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            item, sep, sec = line.rpartition("\t")
            if not sep:
                sys.exit("%s:%d: want 'item<TAB>seconds'" % (path, n))
            try:
                w[item] = float(sec)
            except ValueError:
                sys.exit("%s:%d: %r is not a number of seconds" % (path, n, sec))
    return w


def assign(items, of, weights, default):
    """Longest processing time first: heaviest item to the lightest shard. Ties break by name and
    shard index, so the result depends on the item set and the weights only."""
    if of < 1:
        sys.exit("shard: -of must be at least 1")
    load = [0.0] * of
    out = {}
    for it in sorted(items, key=lambda i: (-weights.get(i, default), i)):
        k = min(range(of), key=lambda j: (load[j], j))
        out[it] = k
        load[k] += weights.get(it, default)
    return out, load


def cmd_select(a):
    items = read_items(sys.stdin)
    if not 0 <= a.index < a.of:
        sys.exit("shard: -index %d is outside 0..%d" % (a.index, a.of - 1))
    got, _ = assign(items, a.of, read_weights(a.weights), a.default)
    for it in items:
        if got[it] == a.index:
            print(it)


def cmd_plan(a):
    items = read_items(sys.stdin)
    got, _ = assign(items, a.of, read_weights(a.weights), a.default)
    for it in items:
        print("%d\t%s" % (got[it], it))


def cmd_check(a):
    items = read_items(sys.stdin)
    if len(items) < a.min_items:
        sys.exit("shard: only %d items listed, want at least %d: the listing is broken, and a "
                 "short list would hide a test from every shard" % (len(items), a.min_items))
    w = read_weights(a.weights)
    got, load = assign(items, a.of, w, a.default)
    per = [0] * a.of
    for k in got.values():
        per[k] += 1
    if any(c == 0 for c in per):
        sys.exit("shard: a shard has no item (%s of %d items in %d shards): use fewer shards" % (per, len(items), a.of))
    # The partition: every item is in exactly one shard of 0..of-1. The union over all shards
    # recomputed one shard at a time must be the item list.
    union = []
    for k in range(a.of):
        union += [it for it in items if got[it] == k]
    if sorted(union) != sorted(items):
        sys.exit("shard: the shards do not partition the items")
    stale = sorted(set(w) - set(items))
    print("shard: %d items in %d shards, items per shard %s, seconds per shard %s" % (
        len(items), a.of, per, [round(x) for x in load]))
    if stale:
        print("shard: note: %d weight lines name items that no longer exist (harmless): %s" % (len(stale), ", ".join(stale[:5])))


def events(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def cmd_verify(a):
    items = read_items(sys.stdin)
    if not items:
        sys.exit("shard: the shard has no items to verify")
    result = {}
    for e in events(a.results):
        act = e.get("Action")
        if act not in ("pass", "fail", "skip"):
            continue
        pkg, test = e.get("Package", ""), e.get("Test", "")
        if a.mode == "package" and not test:
            result[pkg] = act
        if a.mode == "test" and test and "/" not in test:
            result[(pkg, test)] = act
    bad = []
    for it in items:
        if a.mode == "package":
            key = a.module + "/" + it if a.module and it != "." else (a.module or it)
        else:
            pkg, _, test = it.partition(" ")
            key = ((a.module + "/" + pkg) if a.module and pkg != "." else (a.module or pkg), test)
        r = result.get(key)
        if r is None:
            bad.append("no result: %s" % it)
        elif r == "fail":
            bad.append("failed: %s" % it)
    if bad:
        print("\n".join(bad))
        sys.exit("shard: %d of %d items did not run to a pass" % (len(bad), len(items)))
    print("shard: all %d items ran to a result" % len(items))


def cmd_names(a):
    seen = {}
    for e in events(a.results):
        if e.get("Test") and e.get("Action") in ("pass", "fail", "skip"):
            seen[(e["Package"], e["Test"])] = e["Action"]
    for (p, t), act in sorted(seen.items()):
        print("%s\t%s\t%s" % (p, t, act))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("select", cmd_select), ("plan", cmd_plan), ("check", cmd_check)):
        s = sub.add_parser(name)
        s.add_argument("-of", type=int, required=True)
        s.add_argument("-weights", default="")
        s.add_argument("-default", type=float, default=1.0)
        if name == "select":
            s.add_argument("-index", type=int, required=True)
        if name == "check":
            s.add_argument("-min-items", type=int, default=1)
        s.set_defaults(fn=fn)
    s = sub.add_parser("verify")
    s.add_argument("-results", required=True)
    s.add_argument("-mode", choices=("package", "test"), required=True)
    s.add_argument("-module", default="", help="module path: items are relative to it")
    s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("names")
    s.add_argument("-results", required=True)
    s.set_defaults(fn=cmd_names)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
