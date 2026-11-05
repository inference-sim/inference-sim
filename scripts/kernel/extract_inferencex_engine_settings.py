#!/usr/bin/env python3
"""Extract the ENGINE SETTINGS each measured run was launched with, from InferenceX's dump.

# Why a lookup and not a rule

The accuracy snapshot publishes a deployment's parallelism and precision and nothing about
how its engine was configured, so scripts/gen_aisimulate_scenarios.py wrote a default and
said so: "max_num_seqs, block_size, gpu_memory_utilization and cudagraph_mode are engine
defaults: the comparison scores a ratio across concurrency at a fixed deployment, so a
constant cancels."

A constant does cancel in a step-time ratio. It does not cancel when it decides WHETHER A
REQUEST WAITS, which is what the sequence cap does, and time to first token is mostly made of
waiting. Measured against the 46 vLLM sweeps this corpus scores, the assumed 256 was wrong on
every chip.

No rule replaces it either. Reading what the runs actually passed shows three regimes in
roughly equal measure across 254 points: max_num_seqs equal to the client concurrency (92),
a fixed 512 (69), and not passed at all (93). Any single rule is wrong on two thirds of the
corpus. So this reads the value per point.

# Where the settings come from

Every vLLM server log opens with a `non-default args` line listing exactly what was passed on
the command line, and later reports what the engine RESOLVED. Both are captured:

    non-default args: {'model': ..., 'max_model_len': 9416, 'max_num_seqs': 32,
                       'max_num_batched_tokens': 8192, 'enable_prefix_caching': False, ...}
    GPU KV cache size: 6,161,456 tokens
    Maximum concurrency for 9,416 tokens per request: 693.86x

A setting absent from the args line was not passed, and the engine resolved it from its own
defaults -- for max_num_seqs and max_num_batched_tokens, from device memory
(vllm/engine/arg_utils.py get_batch_defaults); the others have their own config defaults. That absence is recorded as null rather
than filled in here, so a consumer can apply the engine's own resolution and a reader can tell
a measurement from a default.

# The join, and the check that validates it

The run a sweep is scored against is already settled: testdata/measurements/
inferencex_absolutes.json records the (date, config_id) whose relatives reproduce the
artifact's own, for all 83 sweeps at 0.000%. This script reuses that decision rather than
re-deriving it, so the settings and the latencies cannot come from different runs -- which is
the failure that would make the comparison quietly stop being apples to apples.

Requires the InferenceX dump restored locally (benchmark_results, configs, server_logs).

Usage:
    python scripts/extract_inferencex_engine_settings.py \
        -c testdata/measurements/aisimulate_e2e.json \
        -a testdata/measurements/inferencex_absolutes.json \
        -o testdata/measurements/inferencex_engine_settings.json
"""

import argparse
import json
import re
import subprocess
import sys

PSQL = "psql"
DB = "inferencex"

# Settings read from the args line. Each is a vLLM CLI flag; the value is whatever the run
# passed, or null when it passed nothing.
ARG_FIELDS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "block_size",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "cuda_graph_sizes",
    "max_cudagraph_capture_size",
    # Fields that identify WHICH DEPLOYMENT a run is, not how it was tuned. Without them
    # two rows at the same (model, hardware, tp, isl, osl, concurrency) can be different
    # deployments and get pooled as repeated measurements of one. On MiniMax-M3 h200 tp8
    # at concurrency 4 that pooling spans 4,862 to 10,085 us: two of the five runs serve
    # Inferact/MiniMax-M3-EAGLE3 with eagle3 speculation at 3 draft tokens, and three
    # serve the plain MXFP8 checkpoint. Grouping by these as well takes the median
    # within-cell spread from 1.384x to 1.016x over 630 cells.
    "model",
    "speculative_config",
    "num_speculative_tokens",
    "enable_expert_parallel",
)

# Lines the engine prints AFTER resolving. These are measurements of the running engine, not
# requests, and are recorded separately so the two are never confused.
RESOLVED_PATTERNS = {
    "resolved_gpu_kv_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
    "resolved_max_concurrency": r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x",
    "engine_version": r"V1 LLM engine \(v([0-9.]+)\)",
}


def query(sql: str) -> list[list[str]]:
    out = subprocess.run([PSQL, "-d", DB, "-t", "-A", "-F", "\x1f", "-c", sql],
                         capture_output=True, text=True, check=True)
    return [line.split("\x1f") for line in out.stdout.strip().split("\n") if line]


def query_scalar(sql: str) -> str:
    """One text value, which may contain newlines."""
    out = subprocess.run([PSQL, "-d", DB, "-t", "-A", "-c", sql],
                         capture_output=True, text=True, check=True)
    return out.stdout


def args_body(log: str) -> str | None:
    r"""The text inside the `non-default args` braces, matched by balance rather than by
    the first closing brace.

    A non-greedy `\{(.*?)\}` stops at the first `}`, which on this corpus is inside the
    nested speculative_config dict rather than at the end of the args: 2,628 of the
    9,192 args lines in the dump carry a nested dict, and every field after it was
    silently dropped -- max_cudagraph_capture_size on 757, block_size and four
    others on 83.
    """
    i = log.find("non-default args: {")
    if i < 0:
        return None
    start = log.index("{", i)
    depth = 0
    for k in range(start, len(log)):
        if log[k] == "{":
            depth += 1
        elif log[k] == "}":
            depth -= 1
            if depth == 0:
                return log[start + 1:k]
    return None


def top_level_fields(body: str) -> dict[str, str]:
    """Split the args body into its TOP-LEVEL `key: value` pairs.

    Split by brace depth rather than by comma, because `speculative_config` holds a
    nested dict whose own keys include `model`. A depth-blind search finds the DRAFT
    checkpoint there and reports it as the served model: on a MiniMax-M3 MTP run the
    served MiniMax-M3-MXFP8 came back as Inferact/MiniMax-M3-EAGLE3, which is a
    different model with a different graph. Nested values are returned whole, so a
    caller that wants a field inside one reads it from the returned text.
    """
    out: dict[str, str] = {}
    depth = 0
    part = ""
    parts = []
    for ch in body:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(part)
            part = ""
        else:
            part += ch
    parts.append(part)
    for piece in parts:
        m = re.match(r"\s*'([A-Za-z_0-9]+)':\s*(.+)$", piece, re.S)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_args_line(log: str) -> dict:
    """The `non-default args` dict, as a mapping of the fields this project reads."""
    body = args_body(log)
    if body is None:
        return {}
    fields = top_level_fields(body)
    found = {}
    for field in ARG_FIELDS:
        if field not in fields:
            continue
        raw = fields[field].strip().strip("'\"")
        if raw == "None":
            # The run passed the flag with a literal None, which is the absence of a
            # setting. Recording the string makes it look like a value: a reader asking
            # whether a capture size was set would get "None" and read it as one.
            continue
        if raw in ("True", "False"):
            found[field] = raw == "True"
        elif re.fullmatch(r"-?\d+", raw):
            found[field] = int(raw)
        elif re.fullmatch(r"-?\d*\.\d+", raw):
            found[field] = float(raw)
        else:
            found[field] = raw
    # num_speculative_tokens and the draft method live INSIDE speculative_config on the
    # runs that carry one, so they are read from there rather than from the top level.
    # Recorded because they change tokens per step: a step that accepts drafts serves
    # more than one token per request, so an ITL read against it is not a step time.
    nested = found.get("speculative_config")
    if isinstance(nested, str):
        n = re.search(r"'num_speculative_tokens':\s*(\d+)", nested)
        if n:
            found["num_speculative_tokens"] = int(n.group(1))
        meth = re.search(r"'method':\s*'([A-Za-z_0-9]+)'", nested)
        if meth:
            found["speculative_method"] = meth.group(1)
        draft = re.search(r"'model':\s*'([^']+)'", nested)
        if draft:
            found["draft_model"] = draft.group(1)
    return found


def parse_resolved(log: str) -> dict:
    out = {}
    for name, pattern in RESOLVED_PATTERNS.items():
        m = re.search(pattern, log)
        if not m:
            continue
        raw = m.group(1).replace(",", "")
        if re.fullmatch(r"\d+", raw):
            out[name] = int(raw)
        elif re.fullmatch(r"\d*\.\d+", raw):
            out[name] = float(raw)
        else:
            out[name] = raw
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--corpus", required=True)
    ap.add_argument("-a", "--absolutes", required=True,
                    help="inferencex_absolutes.json, which settles which run each sweep uses")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    corpus = json.load(open(args.corpus))
    absolutes = json.load(open(args.absolutes))
    chosen = {(a["scenario"], a["label"]): a for a in absolutes["anchors"]}
    sweeps = {(s["scenario"], s["label"]): s for s in corpus["sweeps"]}

    out, missing = [], []
    for key, anchor in chosen.items():
        sweep = sweeps.get(key)
        if sweep is None:
            missing.append({"scenario": key[0], "label": key[1],
                            "reason": "sweep absent from the corpus"})
            continue
        isl, osl = sweep["workload"].split(":")
        concs = [p["concurrency"] for p in sweep["points"]]
        # One query per concurrency. A server log contains newlines, so psql's
        # row-per-line output cannot carry several of them in one result set; asking for
        # one at a time is slower and unambiguous.
        per_conc = {}
        for conc in concs:
            log = query_scalar(f"""
                select sl.server_log
                from benchmark_results b
                join server_logs sl on sl.id = b.server_log_id
                where b.config_id = {anchor['source_config_id']}
                  and b.date = '{anchor['source_run_date']}'
                  and b.benchmark_type = 'single_turn'
                  and b.isl = {isl} and b.osl = {osl}
                  and b.error is null and b.conc = {conc}
                limit 1
            """)
            if not log:
                continue
            passed = parse_args_line(log)
            if not passed:
                continue
            per_conc[str(conc)] = {"passed": passed, "resolved": parse_resolved(log)}
        absent = [c for c in concs if str(c) not in per_conc]
        if absent:
            missing.append({"scenario": key[0], "label": key[1],
                            "reason": f"no engine-args log at concurrency {absent}"})
        if not per_conc:
            continue
        out.append({
            "scenario": key[0],
            "label": key[1],
            "framework": anchor["framework"],
            "gpu": anchor["gpu"],
            "source_run_date": anchor["source_run_date"],
            "source_config_id": anchor["source_config_id"],
            "by_concurrency": per_conc,
        })

    out.sort(key=lambda s: (s["scenario"], s["label"]))
    doc = {
        "source": "SemiAnalysis InferenceX database dump, vLLM server logs",
        "source_url": corpus["snapshot"].get("measurement_source_url"),
        "release_tag": corpus["snapshot"].get("release_tag"),
        "what": "per (scenario, label, concurrency): the settings the run PASSED, and what "
                "the engine RESOLVED. A field absent from `passed` was not passed, and the "
                "engine's own default applies; it is left absent rather than filled in.",
        "run_selection": "the same (date, config_id) each sweep's absolutes resolved to, so "
                         "settings and latencies come from one run",
        "sweeps": len(out),
        "incomplete": missing,
        "settings": out,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    pts = sum(len(s["by_concurrency"]) for s in out)
    print(f"{len(out)} sweeps, {pts} concurrency points -> {args.out}")
    for m in missing[:10]:
        print(f"  incomplete: {m['scenario']} {m['label']}: {m['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
