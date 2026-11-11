"""Run the router sensitivity sweep and write one compact row per run.

    python3 -I sweep.py --blis /path/to/blis --work /tmp/router-sweep --out runs.csv [--jobs 4]
        [--shapes 2P3D] [--workloads prefill-bound] [--concurrency 1,2,4] [--policies round-robin] [--seeds 41,42,43]

For every prefill/decode shape, workload shape, concurrency, router policy and seed it writes
a scenario into --work, runs `blis run`, and appends the figures it needs to --out. A run whose
process fails stops the sweep with its stderr. A run that ends with requests timed out is written
with its `completed` count below `requests`; summarise.py refuses a file that has one.
Reads only the files beside it; writes only under --work and to --out. See
docs/contracts/guides/router-sensitivity.md for what the sweep is for and what it does not show.
"""
import argparse, concurrent.futures as cf, csv, itertools, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# shape: (prefill instances, decode instances). A decode instance is one four-GPU node and the
# prefill pool is always one node of one-GPU instances, so GPUs are prefill + 4 * decode.
SHAPES = {"1P3D": (1, 3), "2P3D": (2, 3), "3P2D": (3, 2), "4P1D": (4, 1)}
# workload kind: (prompt tokens, output tokens). Fixed lengths, closed loop.
WORKLOADS = {"balanced": (1024, 1024), "prefill-bound": (8192, 16), "decode-bound": (256, 2048)}
CONCURRENCY = [16, 64, 256, 1024]
POLICIES = ["round-robin", "random-picker", "least-active-requests", "queue-scorer", "default"]
SEEDS = [41, 42, 43]
FIGURES = ["ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "e2e_p50_ms", "output_tok_s", "request_throughput"]
COLUMNS = ["shape", "workload", "concurrency", "policy", "seed", "requests", "completed"] + FIGURES


def requests_for(conc):
    return min(4000, max(500, 5 * conc))


def scenario_text(shape, pol):
    p, d = SHAPES[shape]
    s = open(os.path.join(HERE, "scenario.yaml")).read()
    return s.replace("@TOPO@", shape.lower()).replace("@ROUTER@", pol).replace("@NODES@", str(1 + d)).replace("@DECODE@", str(d))


def run_one(args, job):
    shape, wl, conc, pol, seed = job
    p, d = SHAPES[shape]
    pt, ot = WORKLOADS[wl]
    name = f"{shape}_{wl}_{conc}_{pol}_{seed}"
    out = os.path.join(args.work, "out", name + ".json")
    n = requests_for(conc)
    cmd = [args.blis, "run", "--scenarios", os.path.join(args.work, "scn"), "--scenario", f"s_{shape}_{pol}.yaml",
           "--num-instances", str(p + d), "--prefill-instances", str(p), "--decode-instances", str(d),
           "--concurrency", str(conc), "--num-requests", str(n), "--seed", str(seed),
           "--prompt-tokens", str(pt), "--prompt-tokens-min", str(pt), "--prompt-tokens-max", str(pt), "--prompt-tokens-stdev", "0",
           "--output-tokens", str(ot), "--output-tokens-min", str(ot), "--output-tokens-max", str(ot), "--output-tokens-stdev", "0",
           "--output", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{name} failed ({r.returncode}):\n{r.stderr[-600:]}")
    res = json.load(open(out))
    row = {"shape": shape, "workload": wl, "concurrency": conc, "policy": pol, "seed": seed,
           "requests": n, "completed": res["requests"]["completed"]}
    for f in FIGURES:
        row[f] = repr(float(res["metrics"][f]["value"]))
    os.remove(out)  # a Result is large; the row keeps what the bound reads
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blis", required=True, help="the blis binary (inference-sim; the guide names the commit)")
    ap.add_argument("--work", required=True, help="scratch directory, created if absent")
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--concurrency", default=",".join(map(str, CONCURRENCY)))
    ap.add_argument("--policies", default=",".join(POLICIES))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.work, "out"), exist_ok=True)
    sdir = os.path.join(args.work, "scn")
    shutil.rmtree(sdir, ignore_errors=True)
    shutil.copytree(os.path.join(HERE, "router"), os.path.join(sdir, "router"))
    shapes = args.shapes.split(",")
    policies = args.policies.split(",")
    for shape, pol in itertools.product(shapes, policies):
        open(os.path.join(sdir, f"s_{shape}_{pol}.yaml"), "w").write(scenario_text(shape, pol))
    jobs = list(itertools.product(shapes, args.workloads.split(","), [int(x) for x in args.concurrency.split(",")], policies,
                                  [int(x) for x in args.seeds.split(",")]))
    rows = []
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for i, row in enumerate(ex.map(lambda j: run_one(args, j), jobs), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", file=sys.stderr)
    rows.sort(key=lambda r: (list(SHAPES).index(r["shape"]), list(WORKLOADS).index(r["workload"]), r["concurrency"],
                             policies.index(r["policy"]), r["seed"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
