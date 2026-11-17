# Original to blis-sglang.
"""SGLang's two tree cores compared on the cache sequences (S45; design §6.4, SGL-1, SGL-5).

Production SGLang on Linux builds the unified radix cache over the Rust tree core
(SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND defaults to "rust", python/sglang/srt/environ.py:741
@297e54fd), and the oracle pins the Python one (SGL-1), which the Go port follows. This runner
drives every sequence of oracle/scenarios-cache-iface through oracle/run_cache_iface.py twice,
once over each core, and compares what the cache did at the interfaces the two cores share: each
answer of the radix cache to the driver's calls (with node ids paired one to one, since the cores
number nodes differently), the cache's calls at the allocator and the request pool, in order, and
the KV events it queued (SGL-5: both cores must merge them the same way).

A difference is a known divergence when oracle/treecore-divergences.json lists it, with the
sequence, the call's position among the driver's calls, what differs and the open issue that
records it (X28); that the issue is still open is the nightly X28 job's check, not this run's. The
run fails on an unlisted difference, on a listed one that no longer occurs,
and when SGLang builds the Python core where the Rust one was asked for (a fallback,
tree_core_registry.py:44-120, would compare the Python core with itself).

Usage: python -I oracle/run_treecore_diff.py <sglang checkout> <report.json>
Environment as for oracle/run_cache_iface.py direct.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SEQUENCES = os.path.join(HERE, "scenarios-cache-iface")
DIVERGENCES = os.path.join(HERE, "treecore-divergences.json")
# Interfaces both cores answer at: the cache's own calls, and those it makes at the pools.
COMPARED = ("prefix_cache", "allocator", "req_to_token")
NODE_FIELDS = ("last_device_node", "last_host_node", "best_match_node", "last_visited_node_id", "node", "node_id")


def run(sglang_root: str, seq: str, tree_core: str, out: str) -> dict:
    cmd = [sys.executable, "-I", os.path.join(HERE, "run_cache_iface.py"), "direct", sglang_root, seq, out,
           f"--tree-core={tree_core}"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode:
        raise SystemExit(f"{os.path.basename(seq)} over the {tree_core} core failed:\n{p.stderr[-3000:]}")
    with open(out) as f:
        return json.load(f)


class Pairing:
    """Node ids of the Python core paired one to one with the Rust core's."""

    def __init__(self):
        self.a, self.b = {}, {}

    def same(self, x, y) -> bool:
        if x is None or y is None:
            return x is None and y is None
        if self.a.setdefault(x, y) != y or self.b.setdefault(y, x) != x:
            return False
        return True


def differs(a, b, pairing: Pairing, key=None) -> bool:
    """Whether two encoded values differ, node ids compared through the pairing."""
    if key in NODE_FIELDS and (isinstance(a, int) or a is None) and (isinstance(b, int) or b is None):
        return not pairing.same(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        ks = set(a) | set(b)
        return any(k not in a or k not in b or differs(a[k], b[k], pairing, k) for k in ks)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) != len(b) or any(differs(x, y, pairing, key) for x, y in zip(a, b))
    return a != b


def driver_calls(trace: dict) -> list:
    """Each call the driver made at the cache, with the calls at the pools it made in turn."""
    calls = trace["calls"]
    by_seq = {c["seq"]: c for c in calls}

    def top(c):
        while c.get("in") is not None:
            c = by_seq[c["in"]]
        return c["seq"]

    out, index = [], {}
    for c in calls:
        if c["i"] not in COMPARED:
            continue
        t = top(c)
        if c.get("in") is None:
            index[t] = len(out)
            out.append({"call": c, "made": []})
        elif t in index and c["i"] != "prefix_cache":
            out[index[t]]["made"].append(c)
    return out


def compare(name: str, py: dict, rs: dict) -> list:
    a, b = driver_calls(py), driver_calls(rs)
    found = []
    if len(a) != len(b):
        return [{"sequence": name, "call": None, "what": f"{len(a)} driver calls over Python, {len(b)} over Rust"}]
    pairing = Pairing()
    for i, (x, y) in enumerate(zip(a, b)):
        cx, cy = x["call"], y["call"]
        label = f"{cx['i']}.{cx['m']}"
        if cx["m"] == "__init__" and cx["i"] == "prefix_cache":
            continue  # the construction differs in its backend by design
        if differs(cx.get("ret"), cy.get("ret"), pairing) or cx.get("raise") != cy.get("raise"):
            found.append({"sequence": name, "call": i, "what": f"{label} answers differently",
                          "python": cx.get("ret", cx.get("raise")), "rust": cy.get("ret", cy.get("raise"))})
        mx = [(m["i"], m["m"], m.get("args"), m.get("ret")) for m in x["made"]]
        my = [(m["i"], m["m"], m.get("args"), m.get("ret")) for m in y["made"]]
        if len(mx) != len(my) or any(p[:2] != q[:2] or differs(list(p[2:]), list(q[2:]), pairing)
                                     for p, q in zip(mx, my)):
            found.append({"sequence": name, "call": i, "what": f"{label} makes different calls at the pools",
                          "python": mx, "rust": my})
    return found


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    sglang_root, report_path = argv
    with open(DIVERGENCES) as f:
        known = json.load(f)["divergences"]
    found = []
    names = sorted(n for n in os.listdir(SEQUENCES) if n.endswith(".json"))
    with tempfile.TemporaryDirectory() as tmp:
        for n in names:
            seq = os.path.join(SEQUENCES, n)
            py = run(sglang_root, seq, "python", os.path.join(tmp, "py.json"))
            rs = run(sglang_root, seq, "rust", os.path.join(tmp, "rs.json"))
            found += compare(n[:-5], py, rs)
    key = lambda d: (d["sequence"], d["call"], d["what"])  # noqa: E731
    listed = {key(d): d for d in known}
    bad = [f"unlisted divergence: {d}" for d in found if key(d) not in listed]
    bad += [f"listed divergence that no longer occurs: {d}" for k, d in listed.items() if k not in {key(f) for f in found}]
    bad += [f"divergence without an open issue: {d}" for d in known if not str(d.get("issue", "")).startswith(
        "https://github.com/inference-sim/blis-sglang/issues/")]
    with open(report_path, "w") as f:
        json.dump({"sequences": names, "divergences": found, "problems": bad}, f, indent=1, sort_keys=True)
    for b in bad:
        print(b)
    print(f"run_treecore_diff: {len(names)} sequences, {len(found)} divergence(s), {len(bad)} problem(s)")
    return 1 if bad or not names else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
