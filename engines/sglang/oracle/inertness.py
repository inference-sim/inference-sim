# Original to blis-sglang.
"""Tier-1 inertness proofs (design 7.4, issue #2350): vary a field over its probe values, run SGLang's own
scheduler on oracle scenarios, compare, and record which of the field's reads each run executed.

A field is proven inert when the runs at its default and at each probe value produce the same result and
the runs executed every line of SGLang's source that reads the field (found by fields/gen_classes.py
--reads, which follows the resolved-config bags) and the lines that use what a read stored. This driver
does both for one field and writes the outcome into the field's proof file, oracle/inertness/<field>.json,
which engines/sglang/inertness_test.go holds the field list to.

    python3 -I oracle/inertness.py <sglang checkout> oracle/inertness/<field>.json           (print the observed runs)
    python3 -I oracle/inertness.py <sglang checkout> oracle/inertness/<field>.json --write   (update the file)
    python3 -I oracle/inertness.py <sglang checkout> oracle/inertness/<field>.json --check   (fail if the file differs)

Environment: as oracle/run_scheduler.py (it runs oracle/run_inertness.py as a child process).

A proof file of kind "observed_reads" holds:
    field, release, claim     what the field does, as read at the pinned release
    probes                    [{"value": V}, ...], one per value class of fields/value_classes.json
    scenarios                 [{"name", "path", "golden"}, ...]: oracle scenarios and their goldens; the default
                              non-streaming run must reproduce the golden
    consumers                 [{"at": "path:line", "expr", "why"}, ...]: the reviewed lines that use a value a
                              read stored; a run must execute each
    reads                     the reads gen_classes.py finds; the check fails when the source has others
    observed                  written by --write: the runs, and the verdict

Every scenario runs at the default and at each probe, twice: as the harness runs it (it sends no streaming
request) and with every request a streaming one, so the code that sends a streamed token runs.
A non-streaming run's whole trace is compared with the scenario's golden; a streaming run's trace is compared
by the projection `frames`: every step and every state equal, the stream of output frames replaced by what
each request was handed in total and the step it finished (the frame boundaries are what the field sets).
The verdict is "inert" when no problem is found (problems()).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fields"))
import goldengz  # noqa: E402


def drop_run_identity(trace: dict) -> dict:
    """The trace without what names the run (its provenance) and the coverage record."""
    t = {k: v for k, v in trace.items() if k != "provenance"}
    if "config" in t:
        t["config"] = {k: v for k, v in t["config"].items() if k != "inertness"}
    return t


def projection(trace: dict, frames: bool) -> dict:
    t = drop_run_identity(trace)
    if not frames:
        return t
    tokens, finish = {}, {}
    steps = []
    for i, step in enumerate(t["steps"]):
        for out in step.get("outputs", []):
            for rid, n in zip(out.get("rids", []), out.get("output_lens", [])):
                tokens[rid] = tokens.get(rid, 0) + n
            for rid in out.get("finished", []):
                finish[rid] = i
        steps.append({k: v for k, v in step.items() if k != "outputs"})
    t["steps"] = steps
    t["delivered"] = {rid: {"tokens": tokens[rid], "finish_step": finish.get(rid)} for rid in sorted(tokens)}
    return t


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def problems(reads: list, scanned: list, consumers: list, probes: list, observed: dict) -> list:
    """Why the observed runs do not prove the field inert (empty when they do)."""
    out = []
    if reads != scanned:
        out.append("the reads in the proof file are not the reads the scan finds in the source")
    runs = observed["runs"]
    wanted = [None] + list(probes)
    groups = {}
    for r in runs:
        groups.setdefault((r["scenario"], r["stream"]), []).append(r)
    for (sc, stream), rs in sorted(groups.items()):
        seen = [r["value"] for r in rs]
        if sorted(seen, key=repr) != sorted(wanted, key=repr):
            out.append("%s (stream %s) ran at %s, not at the default and each probe %s" % (sc, stream, seen, probes))
        if len({r["digest"] for r in rs}) != 1:
            out.append("%s (stream %s): the result differs between values: %s" % (
                sc, stream, {repr(r["value"]): r["digest"][:12] for r in rs}))
        if not stream and rs[0]["digest"] != observed["golden"].get(sc):
            out.append("%s: the non-streaming run is not the scenario's golden" % sc)
    executed = {w for r in runs for w in r["executed"]}
    for rd in reads:
        if rd["at"] not in executed:
            out.append("no run executed the read at %s" % rd["at"])
    for v in probes:
        if not any(r["value"] == v and r["executed"] for r in runs):
            out.append("no run at the probe value %r executed a read" % (v,))
    consumed = {w for r in runs if r["stream"] for w in r["consumers"]}
    for c in consumers:
        if c["at"] not in consumed:
            out.append("no streaming run executed the consumer at %s" % c["at"])
    return out


def run_scheduler(root: str, scenario: dict, tmp: str, name: str) -> dict:
    spath, opath = os.path.join(tmp, name + ".scenario.json"), os.path.join(tmp, name + ".trace.json")
    with open(spath, "w") as f:
        json.dump(scenario, f)
    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_inertness.py"), root, spath, opath],
                       cwd=os.path.dirname(HERE), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise SystemExit("inertness: %s failed (exit %d)\n%s" % (name, p.returncode, p.stderr[-2000:]))
    with open(opath) as f:
        return json.load(f)


def observe(root: str, proof: dict, reads: list) -> dict:
    field = proof["field"]
    golden, runs = {}, []
    base = os.path.dirname(HERE)
    with tempfile.TemporaryDirectory() as tmp:
        for sc in proof["scenarios"]:
            with open(os.path.join(base, sc["path"])) as f:
                scenario = json.load(f)
            gold = projection(goldengz.load(os.path.join(base, sc["golden"])), False)
            golden[sc["name"]] = digest(gold)
            for stream in (False, True):
                for value in [None] + [p["value"] for p in proof["probes"]]:
                    s = json.loads(json.dumps(scenario))
                    if value is not None:
                        s.setdefault("server_args", {})[field] = value
                    s["inertness"] = {"field": field, "stream": stream, "sites": [{"at": r["at"], "to": r["to"]} for r in reads],
                                      "consumers": [{"at": c["at"], "expr": c["expr"]} for c in proof["consumers"]]}
                    name = "%s-%s-%s" % (sc["name"], "stream" if stream else "whole", "default" if value is None else value)
                    trace = run_scheduler(root, s, tmp, name)
                    cov = trace["config"]["inertness"]
                    runs.append({"scenario": sc["name"], "stream": stream, "value": value,
                                 "digest": digest(projection(trace, stream)), **cov})
    observed = {"golden": golden, "runs": runs}
    found = problems(proof["reads"], reads, proof["consumers"], [p["value"] for p in proof["probes"]], observed)
    observed["verdict"] = "inert" if not found else "not proven: " + "; ".join(found)
    return observed


def main(argv: list) -> int:
    flags = [a for a in argv if a.startswith("--")]
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2 or set(flags) - {"--write", "--check"} or len(flags) > 1:
        print("usage: inertness.py <sglang checkout> <proof.json> [--write | --check]", file=sys.stderr)
        return 2
    root, path = args
    with open(path) as f:
        proof = json.load(f)
    if proof.get("kind") != "observed_reads":
        print("inertness: %s is of kind %r, not observed_reads" % (path, proof.get("kind")), file=sys.stderr)
        return 2
    import gen_classes

    base = os.path.dirname(HERE)
    with open(os.path.join(base, "fields.json")) as f:
        entry = next(x for x in json.load(f)["fields"] if x["name"] == proof["field"])
    reads = gen_classes.reads_of(root, entry)
    proof["reads"] = proof.get("reads") or reads
    observed = observe(root, proof, reads)
    new = dict(proof)
    new["reads"] = reads
    new["observed"] = observed
    text = json.dumps(new, indent=1) + "\n"
    if "--write" in flags:
        with open(path, "w") as f:
            f.write(text)
        print("%s: %s" % (path, observed["verdict"]))
    elif "--check" in flags:
        with open(path) as f:
            have = f.read()
        if have != text:
            print("inertness: %s is not what the runs give; run with --write" % path, file=sys.stderr)
            return 1
        print("%s: %s" % (path, observed["verdict"]))
    else:
        sys.stdout.write(text)
    return 0 if observed["verdict"] == "inert" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
