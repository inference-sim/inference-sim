#!/usr/bin/env python3
"""Collect vLLM's own CUDA-graph capture-memory measurements from its public issue logs.

    mine_vllm_cudagraph_memory.py fetch   CACHE_DIR
    mine_vllm_cudagraph_memory.py extract CACHE_DIR > docs/vllm-cudagraph-capture-samples.csv

WHY THIS SOURCE. No dataset this project reads measures the device memory a CUDA-graph
capture holds. AISimulate's capacity model takes it as a caller-supplied
`cuda_graph_reserved_bytes` defaulting to zero (`aisimulate/capacity.py`), and its
`misc.other_mem` is annotated as EXCLUDING the graph reservation
(`aisimulate/support/config_profile.py`). The FPM dataset records the capture
configuration of each run (`cudagraph_mode`, `cudagraph_capture_sizes`) but no memory
figure, and its SGLang raw evidence was collected with prefill capture disabled.

vLLM itself measures the quantity on every start-up, as a free-memory delta around
capture, and logs it:

  * `Graph capturing finished in S secs, took X GiB`  (gpu_model_runner.capture_model)
  * `CUDA graph pool memory: A GiB (actual), E GiB (estimated)`  (gpu_worker, v0.21+)

Users paste those start-up logs into bug reports, so the issue tracker holds hundreds of
real measurements across parts, models and versions. They are uncontrolled -- each is a
different deployment -- which is why the coefficient they anchor is `method: assumed`
and why this script records every sample rather than a summary: the spread is part of
the finding.

WHAT A SAMPLE IS. One issue body or one comment. A multi-rank log repeats the line once
per rank; the largest is kept (ranks differ only in which shards they hold, and the
capacity question is about the fullest rank). `pool_actual` is preferred over `took`
when both appear, since it is the newer, pool-scoped measurement.

Two fields are INFERRED rather than read, and the CSV says which:

  * `mode` is read from a stated `cudagraph_mode`; failing that, it is vLLM's default
    for the stated version (PIECEWISE for 0.9-0.10, FULL_AND_PIECEWISE from 0.11);
    failing both it is empty. `mode_source` is `stated`, `default`, or empty.
  * `gpu` is the most-mentioned accelerator name in the text, which is right for a
    pasted log (nvidia-smi and the device banner name it repeatedly) and can be wrong
    for a comparison thread.

Stdlib only, so it runs in the validator's environment. `fetch` needs an authenticated
`gh` CLI; `extract` needs only the cache it wrote.
"""

from __future__ import annotations

import collections
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "vllm-project/vllm"
QUERIES = (
    '"CUDA graph pool memory"',
    '"Graph capturing finished in"',
    '"Estimated CUDA graph memory"',
)

CAPTURED = re.compile(r"Graph capturing finished in ([\d.]+) secs?, took (-?[\d.]+) GiB")
POOL = re.compile(r"CUDA graph pool memory: ([\d.]+) GiB \(actual\), ([\d.]+) GiB \(estimated\)")
MODE = re.compile(
    r"cudagraph_mode['\"]?\s*[:=]\s*['\"]?<?(?:CUDAGraphMode\.)?"
    r"(FULL_AND_PIECEWISE|FULL_DECODE_ONLY|PIECEWISE|FULL|NONE)\b")
VERSION = re.compile(
    r"vLLM (?:API server )?version:? ?v?(\d+\.\d+(?:\.\d+)?)"
    r"|engine \(v(\d+\.\d+(?:\.\d+)?)"
    r"|\bvllm\s*==\s*v?(\d+\.\d+(?:\.\d+)?)", re.I)
GPU = re.compile(
    r"\b(GB300|GB200|B300|B200|H200|H100|H800|H20|A100|A800|L40S|L40|L4|A10G?|A6000|"
    r"RTX ?\d{4}\w*|MI3\d\dX?|V100|T4|GH200)\b", re.I)


def fetch(cache: Path) -> int:
    cache.mkdir(parents=True, exist_ok=True)
    numbers: set[int] = set()
    for q in QUERIES:
        for page in range(1, 11):
            out = subprocess.run(
                ["gh", "api", "-X", "GET", "search/issues", "-f", f"q=repo:{REPO} {q}",
                 "-f", "per_page=100", "-f", f"page={page}", "--jq", ".items[].number"],
                capture_output=True, text=True)
            got = [int(n) for n in out.stdout.split()]
            numbers.update(got)
            if len(got) < 100:
                break
    for n in sorted(numbers):
        body, comments = cache / f"{n}.json", cache / f"{n}.c.json"
        if not body.exists():
            body.write_text(subprocess.run(
                ["gh", "api", f"repos/{REPO}/issues/{n}",
                 "--jq", "{n:.number,created:.created_at,body:.body}"],
                capture_output=True, text=True, check=True).stdout)
        if not comments.exists():
            comments.write_text(subprocess.run(
                ["gh", "api", "--paginate", f"repos/{REPO}/issues/{n}/comments?per_page=100",
                 "--jq", "[.[].body]"],
                capture_output=True, text=True, check=True).stdout)
    print(f"{len(numbers)} issues cached in {cache}", file=sys.stderr)
    return 0


def default_mode(version: str) -> str:
    m = re.match(r"(\d+)\.(\d+)", version)
    if not m:
        return ""
    v = (int(m[1]), int(m[2]))
    if v >= (0, 11):
        return "FULL_AND_PIECEWISE"
    if v >= (0, 9):
        return "PIECEWISE"
    return ""


def texts(cache: Path):
    for path in sorted(cache.glob("*.json"), key=lambda p: (len(p.name), p.name)):
        if path.name.endswith(".c.json"):
            continue
        issue = json.loads(path.read_text())
        yield issue["n"], 0, issue["created"][:10], issue.get("body") or ""
        side = path.with_name(f"{issue['n']}.c.json")
        if side.exists():
            for i, body in enumerate(pages(side.read_text()), start=1):
                yield issue["n"], i, issue["created"][:10], body or ""


def pages(raw: str):
    """Yield each element of the JSON arrays `gh --paginate` concatenates, one per page.

    Decoded array by array rather than split on `][`, which also occurs inside a pasted
    log and would cut a comment in half.
    """
    dec = json.JSONDecoder()
    raw = raw.strip()
    i = 0
    while i < len(raw):
        arr, i = dec.raw_decode(raw, i)
        yield from arr
        while i < len(raw) and raw[i].isspace():
            i += 1


def extract(cache: Path) -> int:
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["issue", "part", "created", "gpu", "version", "mode", "mode_source",
                "capture_gib", "measure", "capture_secs"])
    for n, part, created, t in texts(cache):
        took = [float(g) for _, g in CAPTURED.findall(t) if float(g) >= 0]
        pool = [float(a) for a, _ in POOL.findall(t)]
        if not (took or pool):
            continue
        gpus = collections.Counter(g.upper().replace(" ", "") for g in GPU.findall(t))
        versions = [next(x for x in m if x) for m in VERSION.findall(t)]
        version = versions[0] if versions else ""
        modes = MODE.findall(t)
        if modes:
            mode, source = modes[0], "stated"
        else:
            mode = default_mode(version)
            source = "default" if mode else ""
        secs = [float(s) for s, _ in CAPTURED.findall(t)]
        w.writerow([n, part, created, gpus.most_common(1)[0][0] if gpus else "", version,
                    mode, source,
                    f"{max(pool) if pool else max(took):.2f}",
                    "pool_actual" if pool else "took",
                    f"{max(secs):.2f}" if secs else ""])
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("fetch", "extract"):
        print(__doc__, file=sys.stderr)
        return 2
    return (fetch if argv[1] == "fetch" else extract)(Path(argv[2]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
