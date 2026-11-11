# Original to blis-vllm.
"""Tier-1 inertness proofs (design 7.4): vary a field over its value classes, run vLLM's own code, compare.

A field is proven inert when two things hold at its default and at each probe value: the scenario set
produces identical traces and identical kernel choices, and code coverage shows every upstream function
that reads the field ran. This harness does both for one field and writes the result into the field's
proof file, oracle/inertness/<field>.json, which the Go test TestEveryInertFieldHasAPassingProof reads.

    python3 -I oracle/inertness.py <vllm checkout> oracle/inertness/<field>.json          (print the result)
    python3 -I oracle/inertness.py <vllm checkout> oracle/inertness/<field>.json --write  (update the file)
    python3 -I oracle/inertness.py <vllm checkout> oracle/inertness/<field>.json --check  (fail if the file differs)

The proof file holds the claim, the probes (one value per value class of fields/value_classes.json,
chosen by the file, checked by the Go test), the runs, a disposition for every read of the field, and,
after --write, the result. Runs:

  A proof file may instead be of kind "unread": {"kind": "unread", "names": [...]}, for a variable no file of
  the serving process (vllm/, csrc/, rust/) names; the result lists the files scanned.

  scheduler   vLLM's real Scheduler on an oracle scenario (oracle/run_scheduler.py) with the probe applied
              through the scenario's scheduler_config; the trace without provenance is the decisions.
  config      EngineArgs(...).create_engine_config(): the value the field resolves to and what the
              resolved scheduler/cache/parallel config holds; this is the tier's "defaults resolved".
  outputproc  vLLM's OutputProcessor fed a scripted EngineCoreOutput stream, for a front-end field: what it
              hands back to the engine core (requests to abort) is compared; what it delivers to the
              client is recorded as frontend_effect and may differ.

A read of the field is found by fields/gen_classes.py (--reads) and must have a disposition:
  run                    a line of the read executed in a named run (line coverage, sys.settrace; the hits
                         are those of that run alone, not of the runs before it);
  pass_through           the read is a keyword argument of a call to a named callee and does nothing else
                         (checked on the syntax tree), and that callee's own read has a disposition.
A read the file does not list, or a listed read that is not there, fails the check (a new release changes
the reads and the field is classified again). A read the scan cannot follow (the value was passed to another
function) is listed with "reviewed": true, its line and the text found there, which the check verifies.
Design 7.4 also admits a reviewed static argument for reads in a surface item the engine refuses by name
and for reads behind a platform check false on CUDA. This harness does not take them: a disposition of
another kind is an error ("unknown disposition"), until a proof needs one and the harness checks it.

The scan of the reads is of Python (vllm/**/*.py). A file may add "named_in": {"vllm": [...], "rust": [...]},
the files under vllm/ and rust/ (tests excluded for rust) that name the field at all, and the check
compares the list with the tree: a read in a front end the scan does not parse (the Rust front end) is
then listed and reviewed, and a file of the engine that starts to name the field fails the check.

Coverage names a vLLM file by its path inside vLLM and matches the executed file by that path's suffix, so
it counts the code that runs whether vLLM is imported from the checkout or installed from a wheel (CI).

Verdict. inert when every run's engine-facing decisions are identical across the probes and every read has
a disposition that holds; not_inert, naming the first run and the probes that differ, otherwise; a
field that selects a kernel is not inert for pricing unless the file states a priced-equivalent argument
the design accepts (it never does here).
"""

import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def sibling(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit("inertness: cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class Coverage:
    """Which lines of a set of files ran (sys.settrace, Python 3.12).

    A file is named by its path inside vLLM ("vllm/v1/engine/output_processor.py") and matched by the
    suffix "/" + that path on the file a frame runs. It is not matched by the checkout's own path: the
    oracle's environment installs vLLM from a wheel (site-packages/vllm/...), and the runner appends the
    checkout to sys.path after it, so the code that runs is the installed copy, not the checkout's. Both
    carry the same text at the pinned release, which the version pin in requirements.lock holds.

    The hits are kept for one run: reset() clears them, and the caller reads ran() after each run."""

    def __init__(self, vllm_root, files):
        self.suffixes = {"/" + f: f for f in files}
        self.hit = {f: set() for f in files}
        self.cache = {}

    def rel_of(self, filename):
        """The vLLM-relative name of an executed file, or None when the file is not one we count."""
        if filename in self.cache:
            return self.cache[filename]
        path = filename.replace(os.sep, "/")
        rel = None
        for suffix, f in self.suffixes.items():
            if path.endswith(suffix):
                rel = f
                break
        self.cache[filename] = rel
        return rel

    def reset(self):
        for lines in self.hit.values():
            lines.clear()

    def tracer(self, frame, event, arg):
        rel = self.rel_of(frame.f_code.co_filename)
        if rel is None:
            return None

        def local(fr, ev, ar):
            if ev == "line":
                self.hit[rel].add(fr.f_lineno)
            return local

        self.hit[rel].add(frame.f_lineno)
        return local

    def __enter__(self):
        sys.settrace(self.tracer)
        return self

    def __exit__(self, *exc):
        sys.settrace(None)

    def ran(self, at, to):
        rel, line = at.rsplit(":", 1)
        return any(int(line) <= n <= to for n in self.hit.get(rel, ()))


def patched_scenario(path, apply_to, value, tmp):
    """The scenario with the probe applied, written to tmp under its own name (the runner reads a path)."""
    with open(path, encoding="utf-8") as f:
        sc = json.load(f)
    if "scheduler_config" in apply_to:
        sc["config"].setdefault("scheduler_config", {})[apply_to["scheduler_config"]] = value
    out = os.path.join(tmp, os.path.basename(path))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sc, f)
    return out


def run_scheduler(vllm_root, run, value):
    rs = sibling(os.path.join(HERE, "run_scheduler.py"), "run_scheduler")
    with tempfile.TemporaryDirectory() as tmp:
        decisions = {}
        for sc in run["scenarios"]:
            p = patched_scenario(os.path.join(REPO, sc), run["apply"], value, tmp)
            trace = rs.run_one(vllm_root, p)
            trace.pop("provenance", None)
            decisions[sc] = trace
    return decisions


def dump(obj):
    """A plain value for a resolved config: dataclasses by field, containers by element, anything else by
    repr with memory addresses removed (two probes must not differ by where an object was allocated)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: dump(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [dump(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(repr(dump(x)) for x in obj)
    if isinstance(obj, dict):
        return {str(k): dump(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(obj))


def without(tree, path):
    """tree without the value at the dotted path (the probed field itself, which differs by construction)."""
    parts = path.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def differing(a, b, path=""):
    """The dotted paths at which two dumped configs differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += differing(a.get(k), b.get(k), path + "." + k if path else k)
        return out
    return [] if a == b else [path]


def run_config(vllm_root, run, value):
    """EngineArgs(...).create_engine_config() with the probe stated. "observe" lists the resolved values the
    file wants to see in the result; "compare" lists the config sections (scheduler_config, cache_config, ...)
    whose every field is compared across the probes, except the field the probe states ("echo"), which holds
    the stated value by construction."""
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    from vllm.engine.arg_utils import EngineArgs  # noqa: E402

    model = os.environ.get("ORACLE_MODEL", "facebook/opt-125m")
    args = EngineArgs(model=model, **{run["apply"]["engine_args"]: value}, **run.get("base", {}))
    cfg = args.create_engine_config()

    def dig(path):
        obj = cfg
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj if isinstance(obj, (bool, int, float, str, type(None))) else repr(obj)

    whole = {section: dump(getattr(cfg, section)) for section in run.get("compare", [])}
    without(whole, run["echo"])
    # echo is where the field itself resolves: it must be the stated value, and is not compared across probes.
    return {"resolved": {path: dig(path) for path in run["observe"]}, "echo": dig(run["echo"]), "stated": value,
            "whole": whole}


def run_outputproc(vllm_root, run, value):
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    from vllm.sampling_params import RequestOutputKind, SamplingParams  # noqa: E402
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason  # noqa: E402
    from vllm.v1.engine.output_processor import OutputProcessor  # noqa: E402

    # The probe is the processor's stream_interval, or, for a control that shows the run sees an
    # engine-facing change, the step at which the script aborts the requests apply["abort_step"] names.
    apply = run.get("apply", {})
    script = dict(run["script"])
    if "abort_step" in apply:
        script["abort"] = {"step": value, "ids": apply["abort_step"]}
        value = 1
    proc = OutputProcessor(None, log_stats=False, stream_interval=value)
    # script: {"requests": {id: {"prompt": n, "tokens": [[token ids of step 0], ...]}}}
    for rid in sorted(script["requests"]):
        params = SamplingParams(max_tokens=1 << 20, detokenize=False, output_kind=RequestOutputKind.DELTA,
                                stream_interval=script["requests"][rid].get("stream_interval"))
        req = EngineCoreRequest(request_id=rid, prompt_token_ids=[1] * script["requests"][rid]["prompt"],
                                mm_features=None, sampling_params=params, pooling_params=None, arrival_time=0.0,
                                lora_request=None, cache_salt=None, data_parallel_rank=None, external_req_id=rid)
        proc.add_request(req, None)
    aborted, delivered = [], []
    steps = max(len(r["tokens"]) for r in script["requests"].values())
    for step in range(steps):
        outs = []
        aborts = []
        if script.get("abort", {}).get("step") == step:
            aborts = sorted(proc.abort_requests(script["abort"]["ids"], internal=True))
        for rid in sorted(script["requests"]):
            toks = script["requests"][rid]["tokens"]
            if step < len(toks):
                last = step == len(toks) - 1
                outs.append(EngineCoreOutput(request_id=rid, new_token_ids=list(toks[step]),
                                             finish_reason=FinishReason.STOP if last else None))
        result = proc.process_outputs(outs)
        aborted.append(aborts + sorted(result.reqs_to_abort))
        for o in result.request_outputs:
            delivered.append([step, o.request_id, [len(c.token_ids) for c in o.outputs], bool(o.finished)])
    return {"to_engine": {"aborted_per_step": aborted, "open_requests": sorted(proc.request_states)},
            "frontend_effect": delivered}


RUNNERS = {"scheduler": run_scheduler, "config": run_config, "outputproc": run_outputproc}


def reader_ok(vllm_root, r, source_lines):
    """Check a pass_through reader's syntax: the read is the value of a keyword of a call to the callee."""
    d = r["disposition"]
    rel, line = r["at"].rsplit(":", 1)
    tree = ast.parse("\n".join(source_lines))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            callee = ast.unparse(n.func).split(".")[-1]
            for k in n.keywords:
                if k.arg == d["keyword"] and callee == d["callee"] and k.value.lineno <= int(line) <= (k.value.end_lineno or k.value.lineno):
                    return True
    return False


def files_naming(root, top, name, suffixes, skip=lambda rel: False):
    """The files under root/top with one of the suffixes whose text contains name, relative to root."""
    found = []
    for dirpath, _, files in os.walk(os.path.join(root, top)):
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            if not fn.endswith(suffixes) or skip(rel):
                continue
            try:
                with open(os.path.join(root, rel), encoding="utf-8") as f:
                    if name in f.read():
                        found.append(rel)
            except (UnicodeDecodeError, OSError):
                continue
    return sorted(found)


def rust_test_file(rel):
    return "/tests/" in rel or rel.endswith("/tests.rs")


def evaluate(vllm_root, spec):
    gc = sibling(os.path.join(REPO, "fields", "gen_classes.py"), "gen_classes")
    root = os.path.realpath(vllm_root)
    fl = os.path.join(REPO, "fields.json")
    reads = gc.reads_of(root, fl, spec["field"])
    listed = {r["at"]: r for r in spec["readers"] if not r.get("reviewed")}
    found = {r["at"] for r in reads}
    result = {"vllm_commit": spec["release"], "problems": []}
    for at in sorted(found - set(listed)):
        result["problems"].append("a read of the field has no disposition: %s" % at)
    for at in sorted(set(listed) - found):
        result["problems"].append("a listed read is not in the source: %s" % at)
    for r in spec["readers"]:
        if r.get("reviewed"):
            rel, line = r["at"].rsplit(":", 1)
            text = read_lines(os.path.join(root, rel))[int(line) - 1]
            if r["expr"] not in text:
                result["problems"].append("the reviewed read %s no longer holds %r" % (r["at"], r["expr"]))
    if "named_in" in spec:
        name = spec["field"]
        tree = {"vllm": files_naming(root, "vllm", name, (".py",)),
                "rust": files_naming(root, "rust", name, (".rs",), rust_test_file)}
        for top, have in sorted(tree.items()):
            want = sorted(spec["named_in"].get(top, []))
            for rel in sorted(set(have) - set(want)):
                result["problems"].append("%s names the field and the proof does not list it (named_in)" % rel)
            for rel in sorted(set(want) - set(have)):
                result["problems"].append("%s is listed in named_in and does not name the field" % rel)
        result["named_in"] = {top: len(have) for top, have in sorted(tree.items())}
    want_files = sorted({r["at"].rsplit(":", 1)[0] for r in spec["readers"]})
    cov = Coverage(root, want_files)
    probes = spec["probes"]
    runs_out = {}
    ran_in = {}
    with cov:
        for run in spec["runs"]:
            cov.reset()  # a line counts for the run that executed it, not for every later run
            outs = {}
            for p in probes:
                outs[p["label"]] = RUNNERS[run["kind"]](root, run, p["value"])
            engine_facing = {k: (v.get("to_engine", v) if run["kind"] == "outputproc" else v) for k, v in outs.items()}
            whole_same = True
            if run["kind"] == "config":
                for k, v in outs.items():
                    if v["echo"] != v["stated"]:
                        result["problems"].append("config run: %s resolved to %r, not the stated %r" % (run["echo"], v["echo"], v["stated"]))
                engine_facing = {k: v["resolved"] for k, v in outs.items()}
                # The whole of the named config sections must agree too, not only the observed values.
                labels = sorted(outs)
                for other in labels[1:]:
                    diff = differing(outs[labels[0]]["whole"], outs[other]["whole"])
                    if diff:
                        whole_same = False
                        result["problems"].append("config run %s: %s resolves differently at %r and %r" % (run["id"], ", ".join(diff[:5]), labels[0], other))
            hashes = {k: digest(v) for k, v in engine_facing.items()}
            entry = {"decisions": hashes, "identical": len(set(hashes.values())) == 1 and whole_same}
            if run["kind"] == "config":
                entry["compared_sections"] = run.get("compare", [])
            if run["kind"] == "outputproc":
                entry["frontend_effect"] = {k: digest(v["frontend_effect"]) for k, v in outs.items()}
                entry["frontend_effect_differs"] = len(set(entry["frontend_effect"].values())) > 1
            if run["kind"] == "config":
                entry["resolved"] = {k: v["resolved"] for k, v in outs.items()}
                entry["echo"] = {k: v["echo"] for k, v in outs.items()}
            runs_out[run["id"]] = entry
            for r in spec["readers"]:
                rel = r["at"].rsplit(":", 1)[0]
                to = r.get("to", int(r["at"].rsplit(":", 1)[1]))
                if cov.ran(r["at"], to):
                    ran_in.setdefault(r["at"], []).append(run["id"])
    result["runs"] = runs_out
    result["readers"] = []
    for r in spec["readers"]:
        d = r["disposition"]
        rec = {"at": r["at"], "disposition": d["kind"]}
        if d["kind"] == "run":
            rec["ran_in"] = ran_in.get(r["at"], [])
            if d["run"] not in rec["ran_in"]:
                result["problems"].append("%s is marked run in %s and that run did not execute it" % (r["at"], d["run"]))
        elif d["kind"] == "pass_through":
            rel = r["at"].rsplit(":", 1)[0]
            lines = read_lines(os.path.join(root, rel))
            if not reader_ok(root, r, lines):
                result["problems"].append("%s is not a keyword %s of a call to %s" % (r["at"], d["keyword"], d["callee"]))
        else:
            result["problems"].append("%s: unknown disposition %r" % (r["at"], d["kind"]))
        result["readers"].append(rec)
    for rid, e in runs_out.items():
        if not e["identical"]:
            result["problems"].append("the decisions of run %s differ across the probes" % rid)
    result["verdict"] = "inert" if not result["problems"] else "not_inert"
    return result


def evaluate_unread(vllm_root, spec):
    """A proof of kind "unread": no file of the serving process names the variable. The scan reads every text
    file under vllm/, csrc/ and rust/ (the paths gen_fields.py does not exclude are all of them) for each
    name, as a whole word. A name found there is a read or a write the generator's rule would list, so the
    verdict is not_inert and the field is classified again."""
    import re

    root = os.path.realpath(vllm_root)
    found = {n: [] for n in spec["names"]}
    pats = {n: re.compile(r"\b%s\b" % re.escape(n)) for n in spec["names"]}
    scanned = 0
    for top in ("vllm", "csrc", "rust"):
        for dirpath, _, files in os.walk(os.path.join(root, top)):
            for fn in sorted(files):
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, encoding="utf-8") as f:
                        text = f.read()
                except (UnicodeDecodeError, OSError):
                    continue
                scanned += 1
                for n, pat in pats.items():
                    if pat.search(text):
                        found[n].append(os.path.relpath(path, root))
    result = {"vllm_commit": spec["release"], "files_scanned": scanned, "occurrences": {n: sorted(v) for n, v in found.items()},
              "problems": ["%s is named in %s" % (n, ", ".join(v)) for n, v in sorted(found.items()) if v]}
    if scanned < 1000:
        result["problems"].append("only %d files were scanned: the checkout is not a vLLM tree" % scanned)
    result["verdict"] = "inert" if not result["problems"] else "not_inert"
    return result


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    vllm_root, path = argv[0], argv[1]
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    result = evaluate_unread(vllm_root, spec) if spec.get("kind") == "unread" else evaluate(vllm_root, spec)
    if "--check" in argv:
        if spec.get("result") != result:
            sys.stderr.write("inertness: %s differs from what the harness computes now\n" % path)
            sys.stderr.write(json.dumps(result, indent=1) + "\n")
            return 1
        return 0
    if "--write" in argv:
        spec["result"] = result
        with open(path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=1)
            f.write("\n")
    else:
        print(json.dumps(result, indent=1))
    return 0 if result["verdict"] == "inert" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
