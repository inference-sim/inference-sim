"""Print the ratios a router-table bound states, from the sweep's runs.csv.

    python3 -I summarise.py runs.csv                      # the tables in the guide
    python3 -I summarise.py runs.csv --check              # exit 1 unless every cell has three complete seeds
    python3 -I summarise.py runs.csv --emit               # the ratios block of a catalog entry
    python3 -I summarise.py runs.csv --spread             # largest over smallest across all five policies
    python3 -I summarise.py runs.csv --policy round-robin --against random-picker \\
        --metric ttft_p50_ms --workload balanced --concurrency 16-1024 [--shapes 1P3D,2P3D]

The ratio of a cell (shape, workload, concurrency) between policy A and policy B is the larger
of mean_A / mean_B and mean_B / mean_A, each mean taken over the seeds. A bound is the largest
ratio over the cells of a regime and over the alternatives. It is stated rounded up to two
decimals. The Go test in contracts/spec/routertable recomputes the bounds of the committed
catalog entries from this file with its own code, so the two cannot drift.
"""
import argparse, collections, csv, math, sys

FIGURES = ["ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "e2e_p50_ms", "output_tok_s", "request_throughput"]
SEEDS = 3


def load(path):
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["completed"] != r["requests"]:
            raise SystemExit(f"{r['shape']} {r['workload']} {r['concurrency']} {r['policy']} seed {r['seed']}: "
                             f"{r['completed']} of {r['requests']} requests completed; a run that did not finish has no row")
        cells[(r["shape"], r["workload"], int(r["concurrency"]))][r["policy"]].append(r)
    return cells


def mean(rows, metric):
    return sum(float(r[metric]) for r in rows) / len(rows)


def ratio(cells, policy, other, metric, workload, lo, hi, shapes=None):
    best, at = 1.0, None
    for (shape, wl, conc), by in cells.items():
        if wl != workload or not lo <= conc <= hi or (shapes and shape not in shapes):
            continue
        a, b = mean(by[policy], metric), mean(by[other], metric)
        r = max(a / b, b / a)
        if r > best:
            best, at = r, (shape, conc)
    return best, at


def up(x):
    return math.ceil(x * 100 - 1e-9) / 100


def check(cells):
    bad = 0
    for k, by in cells.items():
        for p, rows in by.items():
            if len(rows) != SEEDS or len({r["seed"] for r in rows}) != SEEDS:
                print(f"{k} {p}: {len(rows)} rows, want {SEEDS} distinct seeds", file=sys.stderr)
                bad += 1
    return bad


# The rows a catalog entry carries: four figures, each for the three workload kinds, against the
# other load-blind policy (round-robin and random-picker stand for each other) and against the three
# policies that read load. One regime for every
# row: all the concurrencies swept and all four pool shapes.
BOUND_METRICS = ["ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "output_tok_s"]
BOUND_WORKLOADS = ["balanced", "prefill-bound", "decode-bound"]
LOAD_READING = ["default", "least-active-requests", "queue-scorer"]
OTHER_LOAD_BLIND = {"round-robin": "random-picker", "random-picker": "round-robin"}
MODEL, HARDWARE = "minimax-m2.5", "gb200-nvl72"
# What scenario.yaml runs: the engine release, the fabric between the pools, and the parallelism of
# one prefill instance and one decode instance. A test compares them with the scenario.
ENGINE_VERSION, FABRIC = "0.32.0", "ib-400g"
PREFILL = "{tp: 1, pp: 1, dp: 1, expert_parallel: false}"
DECODE = "{tp: 4, pp: 1, dp: 1, expert_parallel: true}"


def emit(cells, policy):
    concs = sorted({k[2] for k in cells})
    shapes = sorted({k[0] for k in cells})
    print("  ratios:")
    for group in ([OTHER_LOAD_BLIND[policy]], LOAD_READING):
        for w in BOUND_WORKLOADS:
            for m in BOUND_METRICS:
                best = max(ratio(cells, policy, o, m, w, concs[0], concs[-1])[0] for o in group)
                print(f"    - metric: {m}\n      against: [{', '.join(sorted(group))}]\n      regime:\n"
                      f"        workload: {w}\n        concurrencies: [{', '.join(str(c) for c in concs)}]\n"
                      f"        model: {MODEL}\n        hardware: {HARDWARE}\n        engine_version: \"{ENGINE_VERSION}\"\n"
                      f"        fabric: {FABRIC}\n        prefill: {PREFILL}\n        decode: {DECODE}\n        shapes: [{', '.join(shapes)}]\n"
                      f"      max_ratio: {up(best):.2f}")


def spread(cells):
    """Largest to smallest cell mean across every policy, for each figure and concurrency: the
    smallest and the largest over the pool shapes. The measure the first scoping analysis used."""
    concs = sorted({k[2] for k in cells})
    print("| Figure | Workload | " + " | ".join(str(c) for c in concs) + " |")
    print("|---|---|" + "---:|" * len(concs))
    for m in BOUND_METRICS:
        for w in BOUND_WORKLOADS:
            row = []
            for c in concs:
                v = []
                for (shape, wl, conc), by in cells.items():
                    if wl == w and conc == c:
                        means = [mean(rs, m) for rs in by.values()]
                        v.append(max(means) / min(means))
                row.append(f"{min(v):.2f} to {max(v):.2f}")
            print(f"| `{m}` | {w} | " + " | ".join(row) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--policy", default="round-robin")
    ap.add_argument("--against")
    ap.add_argument("--metric")
    ap.add_argument("--workload")
    ap.add_argument("--concurrency")
    ap.add_argument("--shapes")
    ap.add_argument("--emit", action="store_true", help="print the ratios block the catalog entries carry")
    ap.add_argument("--spread", action="store_true", help="print the spread across all policies, per concurrency")
    a = ap.parse_args()
    cells = load(a.runs)
    if a.check:
        sys.exit(1 if check(cells) else 0)
    if check(cells):
        sys.exit("incomplete cells")
    policies = sorted({p for by in cells.values() for p in by})
    if a.spread:
        spread(cells)
        return
    if a.emit:
        emit(cells, a.policy)
        return
    if a.against:
        lo, hi = map(int, a.concurrency.split("-"))
        shapes = a.shapes.split(",") if a.shapes else None
        worst = max((ratio(cells, a.policy, o, a.metric, a.workload, lo, hi, shapes) + (o,) for o in a.against.split(",")),
                    key=lambda t: t[0])
        print(f"{up(worst[0]):.2f}  (raw {worst[0]:.4f}, at {worst[1]} against {worst[2]})")
        return
    concs = sorted({k[2] for k in cells})
    workloads = ["balanced", "prefill-bound", "decode-bound"]
    for pol in policies:
        if pol == a.policy:
            continue
        print(f"\n### {a.policy} against {pol}: largest ratio over the four pool shapes\n")
        print("| Figure | Workload | " + " | ".join(str(c) for c in concs) + " |")
        print("|---|---|" + "---:|" * len(concs))
        for m in FIGURES:
            for w in workloads:
                row = [f"{up(ratio(cells, a.policy, pol, m, w, c, c)[0]):.2f}" for c in concs]
                print(f"| `{m}` | {w} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
