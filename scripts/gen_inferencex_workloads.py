#!/usr/bin/env python3
"""Generate the InferenceX benchmark's exact per-request target lengths for every scored point.

InferenceX's client (SemiAnalysisAI/InferenceX, utils/bench_serving/benchmark_serving.py at the
dates the measured runs were taken; inferencex-e2e/infx/bench_serving/benchmark_serving.py at
HEAD) seeds NumPy's legacy global generator and then draws every request's target lengths
before sending anything:

    main():                   random.seed(args.seed); np.random.seed(args.seed)   # --seed default 0
    sample_random_requests(): prefix_token_ids = np.random.randint(0, vocab, size=prefix_len)
                              input_lens  = np.random.randint(int(isl*r), isl + 1, size=num_prompts)
                              output_lens = np.random.randint(int(osl*r), osl + 1, size=num_prompts)

with prefix_len = --random-prefix-len (default 0; no launcher passes it), r =
--random-range-ratio (RANDOM_RANGE_RATIO: 0.8 in every benchmark workflow), and num_prompts =
10 * CONC. Each concurrency point is its own workflow job and its own client process, so the
generator is freshly seeded per point. This script reproduces exactly those calls, per point,
and writes the vectors in request order. docs/kernel-exclusive/APPLES-TO-APPLES.md cites every
line this depends on.

The vectors are TARGET lengths. The client then builds a text prompt from token ids, decodes and
re-encodes it up to ten times to land on the target, and records the length it actually got; that
tokenizer residual ("mismatch") is not in the measurements and is not modelled by this generator.
It is instead reconstructed offline by scripts/inferencex_prompt_lengths.py at the point's pinned
tokenizer revisions and verified: for every verified point the residual is zero, so the sent length
equals the target this script emits. See sim/kernelmodel/harness/promptlengths.go and
docs/kernel-exclusive/APPLES-TO-APPLES.md "The sent prompt lengths".

Usage (numpy required; any version -- legacy RandomState streams are frozen by NEP 19):

    python -I scripts/gen_inferencex_workloads.py --corpus $BLIS_MEASUREMENTS/aisimulate_e2e.json
    python -I scripts/gen_inferencex_workloads.py --check            # regenerate and diff
    python -I scripts/gen_inferencex_workloads.py --seed 1 --out /tmp/seed1.json \
        --corpus $BLIS_MEASUREMENTS/aisimulate_e2e.json            # a sensitivity draw

--check regenerates every point already in the fixture and fails on any difference. With
--corpus it also fails if the fixture's point set differs from the corpus's.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "testdata", "inferencex-workloads", "seed0.json")

# The benchmark's constants, from the source cited in the module docstring.
RANGE_RATIO = 0.8          # RANDOM_RANGE_RATIO in .github/workflows/benchmark-tmpl.yml
REQUESTS_PER_CONC = 10     # --num-prompts "$((CONC * 10))" in every single-node recipe
PREFIX_LEN = 0             # --random-prefix-len default; no launcher passes it
# The prefix draw's bound is the tokenizer's vocab size, which differs per model. A size-0 draw
# consumes no generator state for ANY bound (asserted below), so the value cannot matter.
PLACEHOLDER_VOCAB = 200000


def corpus_points(path):
    """Every (isl, osl, concurrency) the corpus scores, enumerated as the harness does:
    each sweep's "<isl>:<osl>" workload identity, at each of its points' concurrency."""
    with open(path) as f:
        corpus = json.load(f)
    points = set()
    for sweep in corpus["sweeps"]:
        isl, osl = (int(v) for v in sweep["workload"].split(":"))
        for p in sweep["points"]:
            points.add((isl, osl, int(p["concurrency"])))
    return sorted(points)


def draw(isl, osl, conc, seed):
    """The client's draws for one point, in the client's call order."""
    np.random.seed(seed)
    before = np.random.get_state()
    prefix = np.random.randint(0, PLACEHOLDER_VOCAB, size=PREFIX_LEN)
    after = np.random.get_state()
    assert len(prefix) == 0 and before[2] == after[2] and (before[1] == after[1]).all(), \
        "a size-0 prefix draw advanced the generator"
    n = REQUESTS_PER_CONC * conc
    inputs = np.random.randint(int(isl * RANGE_RATIO), isl + 1, size=n).tolist()
    outputs = np.random.randint(int(osl * RANGE_RATIO), osl + 1, size=n).tolist()
    return inputs, outputs


def build(points, seed):
    out = []
    for isl, osl, conc in points:
        inputs, outputs = draw(isl, osl, conc, seed)
        out.append({"isl": isl, "osl": osl, "concurrency": conc,
                    "input_lens": inputs, "output_lens": outputs})
    return {
        "what": "InferenceX benchmark_serving.py target input/output lengths per request, in "
                "request order, one entry per (isl, osl, concurrency)",
        "generator": "scripts/gen_inferencex_workloads.py",
        "seed": seed,
        "range_ratio": RANGE_RATIO,
        "requests_per_concurrency": REQUESTS_PER_CONC,
        "prefix_len": PREFIX_LEN,
        "points": out,
    }


def render(doc):
    """One point per line, so a diff names the point that changed."""
    head = {k: v for k, v in doc.items() if k != "points"}
    lines = ["{"]
    for k, v in head.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v)},")
    lines.append('  "points": [')
    pts = [json.dumps(p, separators=(",", ":")) for p in doc["points"]]
    lines.append(",\n".join("    " + p for p in pts))
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="aisimulate_e2e.json; enumerates the points")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0, help="numpy seed (the benchmark's is 0)")
    ap.add_argument("--check", action="store_true", help="regenerate --out and fail on any difference")
    args = ap.parse_args()

    if args.check:
        with open(args.out) as f:
            text = f.read()
        doc = json.loads(text)
        points = [(p["isl"], p["osl"], p["concurrency"]) for p in doc["points"]]
        if args.corpus and sorted(points) != corpus_points(args.corpus):
            print(f"{args.out}: point set differs from {args.corpus}", file=sys.stderr)
            return 1
        if render(build(points, doc["seed"])) != text:
            print(f"{args.out}: stale -- regenerate with this script", file=sys.stderr)
            return 1
        print(f"{args.out}: {len(points)} points match")
        return 0

    if not args.corpus:
        ap.error("--corpus is required to enumerate the points (or pass --check)")
    doc = build(corpus_points(args.corpus), args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(render(doc))
    print(f"wrote {args.out}: {len(doc['points'])} points, seed {args.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
