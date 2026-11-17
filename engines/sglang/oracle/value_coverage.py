# Original to blis-sglang.
"""The value-coverage report (FID-10): fail unless every compared field of the golden traces takes
its neutral value and at least one other value somewhere in the scenario set, so that a comparison
of it can fail.

The compared fields are every key of a golden outside its `provenance` block (what
compare_goldens.py and the Go diff tests compare). A field is named by its path, with `[]` for a
list element and `*` for a key of a map that oracle/value_coverage.json lists in `maps` (a map keyed
by request id, say). A list is a field of its own, whose values are empty and not empty.

A path is covered when it takes two distinct values in the scenario set, which is the check FID-10
names ("fails on any field or series seen only at one value"). For a boolean and for a list, two
values are the neutral one (false, empty) and another. A number that never takes 0, and a string
(an enumeration such as a forward mode has no neutral member), need two values only; --report
marks a covered number that never takes 0, since its neutral value is the invariant's aim.

A path not covered must be listed in `held` in oracle/value_coverage.json, with its `kind` and a
`reason`:

- `fixed`: the oracle holds it at one value by construction (the CPU device, the Python tree core
  SGL-1 requires), so no scenario can move it;
- `no_neutral`: its neutral value cannot occur (a forward pass has at least one request);
- `gap`: a scenario could move it and none does yet. The reason cites the open issue that adds the
  scenario (`inference-sim/blis-sglang#N` or `#N`, X28).

A held key ending in `**` holds a whole subtree: the path before it and every path under it. It is for
a record whose fields mean different things by call (the recorded calls of the cache and HiCache
interface traces, FID-5), where one path per call argument would restate the same gap hundreds of times.
It is a `gap` or a `fixed`, with the same reason and issue rules, and it must match some path.

The report fails on a path neither covered nor held, on a held path that is now covered (so the list
stays true), on a held path no golden has, and on an entry with no reason. No golden at all fails
too (X16).

Usage: python3 -I oracle/value_coverage.py [<repository root>] [--report]
       --report prints every path with the values it took.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("fixed", "no_neutral", "gap")
ISSUE = re.compile(r"(inference-sim/blis-sglang)?#\d+")


def kind(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "list"
    return "object"


def neutral(v) -> bool:
    return v is None or v is False or (kind(v) == "number" and v == 0) or v == [] or v == ""


def walk(v, path: str, maps: set, seen: dict) -> None:
    """Record every leaf value of v under its path in seen (path -> set of JSON texts)."""
    if isinstance(v, dict):
        for k, x in v.items():
            child = f"{path}.*" if path in maps else (f"{path}.{k}" if path else k)
            walk(x, child, maps, seen)
        return
    if isinstance(v, list):
        # A list is a field of its own (empty or not), and its elements are another.
        seen.setdefault(path, set()).add("[]" if not v else "[...]")
        for x in v:
            walk(x, path + "[]", maps, seen)
        return
    seen.setdefault(path, set()).add(json.dumps(v, sort_keys=True))


def covered(values: set) -> bool:
    return len(values) >= 2


def no_zero(values: set) -> bool:
    """A number field that never takes its neutral value."""
    if "[]" in values or "[...]" in values:
        return False
    parsed = [json.loads(t) for t in values]
    return all(kind(v) == "number" for v in parsed) and not any(neutral(v) for v in parsed)


def held_by(held: dict, path: str):
    """The held entry that covers path: its own, or the subtree (`prefix**`) that contains it."""
    if path in held:
        return path
    for key in held:
        if key.endswith("**"):
            base = key[:-2]
            if path == base or path.startswith(base + ".") or path.startswith(base + "[]"):
                return key
    return None


def golden_traces(root: str) -> list:
    with open(os.path.join(root, "oracle", "harness.json")) as f:
        cfg = json.load(f)
    out = []
    for d in cfg["golden_dirs"]:
        for dirpath, _, names in sorted(os.walk(os.path.join(root, d))):
            for n in sorted(names):
                if n.endswith(".json"):
                    with open(os.path.join(dirpath, n)) as f:
                        out.append(json.load(f))
    for rel in cfg["golden_files"]:
        with open(os.path.join(root, rel)) as f:
            out += json.load(f).get("cases", [])
    return out


def config_problems(cfg) -> list:
    bad = []
    if not isinstance(cfg, dict) or set(cfg) - {"description", "maps", "held"}:
        return ["oracle/value_coverage.json: keys must be description, maps and held"]
    for path, entry in sorted(cfg.get("held", {}).items()):
        if not isinstance(entry, dict) or entry.get("kind") not in KINDS:
            bad.append(f"held {path}: kind must be one of {', '.join(KINDS)}")
            continue
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            bad.append(f"held {path}: no reason")
        elif entry["kind"] == "gap" and not ISSUE.search(reason):
            bad.append(f"held {path}: a gap cites the open issue that closes it (X28)")
    return bad


def main(root: str, report: bool = False) -> int:
    with open(os.path.join(root, "oracle", "value_coverage.json")) as f:
        cfg = json.load(f)
    bad = config_problems(cfg)
    goldens = golden_traces(root)
    if not goldens:
        print("value_coverage: no golden trace in the golden directories or files of oracle/harness.json")
        return 1
    seen: dict = {}
    maps = set(cfg.get("maps", []))
    for g in goldens:
        walk({k: v for k, v in g.items() if k != "provenance"}, "", maps, seen)
    held = cfg.get("held", {})
    for path in sorted(seen):
        ok = covered(seen[path])
        if report:
            label = ("covered, never 0" if no_zero(seen[path]) else "covered") if ok else \
                ("held " + held[held_by(held, path)].get("kind", "?") if held_by(held, path) else "ONE VALUE")
            print(f"{label:16} {path}: "
                  + ", ".join(sorted(seen[path]))[:200])
        if not ok and held_by(held, path) is None:
            bad.append(f"{path}: takes only {', '.join(sorted(seen[path]))} in the scenario set, so a comparison "
                       "of it cannot fail; add a scenario that moves it, or hold it in oracle/value_coverage.json")
        elif ok and path in held:
            bad.append(f"{path}: is held as {held[path].get('kind')} but is now covered; remove it from held")
    for path in sorted(held):
        if path.endswith("**"):
            if not any(held_by({path: held[path]}, p) for p in seen):
                bad.append(f"{path}: is held, but no golden has it")
        elif path not in seen:
            bad.append(f"{path}: is held, but no golden has it")
    for b in bad:
        print(b)
    print(f"value_coverage: {len(goldens)} golden(s), {len(seen)} field(s), "
          f"{sum(covered(v) for v in seen.values())} covered, {len(held)} held, {len(bad)} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--report"]
    if len(args) > 1:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        sys.exit(2)
    sys.exit(main(os.path.abspath(args[0]) if args else os.path.dirname(HERE), "--report" in sys.argv))
