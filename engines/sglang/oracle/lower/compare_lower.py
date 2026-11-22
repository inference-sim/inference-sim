# Original to blis-sglang.
"""Compare the lowering answers a sharded run wrote (extract.py --shard I/N) with the committed ones.

  shard    <committed> <fresh shard>
           The shard's answers must equal the committed answers of the cases the shard ran
           ("ran" in the shard's file), in order. A case SGLang refuses has no answer, in either.
  complete <committed> <serverargs cases.json> <extra cases.json> <shard>...
           The shards are every shard 0..N-1 of one N, shard I ran exactly the cases at the
           positions I mod N of the case list (the file's, then the extra ones), so each case ran
           once, and every committed answer is for a case in that list.

Together the two checks are the one the unsharded run made: the committed file equals what SGLang
lowers for every case.

Usage: python3 -I oracle/lower/compare_lower.py shard <committed> <fresh>
       python3 -I oracle/lower/compare_lower.py complete <committed> <cases.json> <extra cases.json> <shard>...
"""
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def check_shard(committed, fresh):
    """Problems with one shard's answers against the committed ones."""
    if "shard" not in fresh or "ran" not in fresh:
        return ["the answers are not a shard's (no \"shard\" and \"ran\")"]
    ran = set(fresh["ran"])
    problems = []
    for c in fresh["cases"]:
        if c["name"] not in ran:
            problems.append("%s: answered but not in the cases the shard ran" % c["name"])
    want = [c for c in committed["cases"] if c["name"] in ran]
    got = fresh["cases"]
    if [c["name"] for c in want] != [c["name"] for c in got]:
        problems.append("committed answers for the shard's cases %s, fresh %s" % ([c["name"] for c in want], [c["name"] for c in got]))
        return problems
    for x, y in zip(want, got):
        if x != y:
            problems.append("%s: %s differ" % (x["name"], sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))))
    return problems


def check_complete(committed, names, shards):
    """Problems with the shards together: every case ran in exactly the shard that owns it."""
    if not shards:
        return ["no shard"]
    problems = []
    ns = {s["shard"]["of"] for s in shards if "shard" in s}
    if len(ns) != 1 or any("shard" not in s for s in shards):
        return ["the shards do not share one N: %s" % sorted(ns)]
    n = ns.pop()
    seen = sorted(s["shard"]["index"] for s in shards)
    if seen != list(range(n)):
        problems.append("shards %s of %d, want every one of 0..%d once" % (seen, n, n - 1))
    for s in shards:
        i = s["shard"]["index"]
        want = [name for k, name in enumerate(names) if k % n == i]
        if s["ran"] != want:
            problems.append("shard %d/%d ran %d cases, the case list gives it %d (or in another order)" % (i, n, len(s["ran"]), len(want)))
    known = set(names)
    for c in committed["cases"]:
        if c["name"] not in known:
            problems.append("%s: committed answer for a case that is in no case file" % c["name"])
    if len(known) != len(names):
        problems.append("two cases share a name")
    return problems


def main(argv):
    if len(argv) == 3 and argv[0] == "shard":
        problems = check_shard(load(argv[1]), load(argv[2]))
    elif len(argv) >= 5 and argv[0] == "complete":
        names = [c["name"] for c in load(argv[2])["cases"]] + [c["name"] for c in load(argv[3])["cases"]]
        problems = check_complete(load(argv[1]), names, [load(p) for p in argv[4:]])
    else:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    for p in problems:
        print(p)
    if not problems:
        print("lowering answers: ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
