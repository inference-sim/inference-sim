"""Run the router sensitivity sweep and write one compact row per run.

    python3 -I sweep.py --blis /path/to/blis --work /tmp/router-sweep --out runs.csv [--jobs 4]
        [--shapes 2P3D] [--workloads prefill-bound] [--concurrency 1,2,4] [--policies round-robin] [--seeds 41,42,43]
    python3 -I sweep.py --blis ... --work ... --out layouts.csv --layouts layouts.json [--layout ID,ID] [--keep-going] [--resume]

Without --layouts the sweep is the original one: the scenario in scenario.yaml (MiniMax-M2.5 on GB200) at
every pool shape in SHAPES. With --layouts it runs each layout of the file: a layout names the model, the part,
the fabric, the engine release, the parallelism of one prefill and one decode instance, the workload kind and the
pool shapes of a group of real runs, and the CSV records those fields in every row so the ratios can be
recomputed per layout. A layout's pools are written from scenario-layout.yaml.

For every prefill/decode shape, workload shape, concurrency, router policy and seed it writes
a scenario into --work, runs `blis run`, and appends the figures it needs to --out. A run whose
process fails stops the sweep with its stderr. A run that ends with requests timed out is written
with its `completed` count below `requests`; summarise.py refuses a file that has one.
Reads only the files beside it; writes only under --work and to --out.

What the sweep is for: a router-table entry stands an llm-d policy in for a router BLIS does not host,
and the entry's `bound` states how far BLIS's own prediction moves when the policy changes. The sweep
holds the engine, model, hardware and workload fixed, changes only the policy, and records the ratio
max(mean_A / mean_B, mean_B / mean_A) of each figure between two policies, over the three seeds.
What it does not show: it is a sensitivity of BLIS's prediction to the policy, not an error against the
hosted router (no run behind that router was compared), and it says nothing about differences between
BLIS's tie-breaking and the router's that BLIS does not model; the entry's `difference` lists those.
"""
import argparse, concurrent.futures as cf, csv, itertools, json, math, os, re, shutil, subprocess, sys, threading

HERE = os.path.dirname(os.path.abspath(__file__))

# shape: (prefill instances, decode instances). A decode instance is one four-GPU node and the
# prefill pool is always one node of one-GPU instances, so GPUs are prefill + 4 * decode.
SHAPES = {"1P3D": (1, 3), "2P3D": (2, 3), "3P2D": (3, 2), "4P1D": (4, 1)}
# workload kind: (prompt tokens, output tokens). Fixed lengths, closed loop.
WORKLOADS = {"balanced": (1024, 1024), "prefill-bound": (8192, 16), "decode-bound": (256, 2048), "long-prompt": (8192, 1024)}
CONCURRENCY = [16, 64, 256, 1024]
# The client counts a layout run covers: the powers of two the real runs used most.
LAYOUT_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
POLICIES = ["round-robin", "random-picker", "least-active-requests", "queue-scorer", "default"]
SEEDS = [41, 42, 43]
FIGURES = ["ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "e2e_p50_ms", "output_tok_s", "request_throughput"]
COLUMNS = ["shape", "workload", "concurrency", "policy", "seed", "requests", "completed"] + FIGURES
# What a layout row adds in front: the fields of the regime a bound names, so a row says what it ran.
LAYOUT_COLUMNS = ["layout", "model", "hardware", "engine_version", "fabric", "prefill", "decode"]
SHAPE = re.compile(r"^([1-9][0-9]*)P([1-9][0-9]*)D$")


def requests_for(conc):
    return min(4000, max(500, 5 * conc))


def scenario_text(shape, pol):
    p, d = SHAPES[shape]
    s = open(os.path.join(HERE, "scenario.yaml")).read()
    return s.replace("@TOPO@", shape.lower()).replace("@ROUTER@", pol).replace("@NODES@", str(1 + d)).replace("@DECODE@", str(d))


def par_text(p):
    """The parallelism of one instance as a CSV field, for example tp4.pp1.dp1.ep1."""
    return f"tp{p['tp']}.pp{p['pp']}.dp{p['dp']}.ep{int(p['expert_parallel'])}"


def layout_scenario(lay, shape, pol):
    """The scenario text for a layout at a pool shape and router policy. Pools do not share a node, so each pool
    takes the nodes its instances need."""
    p, d = map(int, SHAPE.match(shape).groups())
    gpn = lay["gpus_per_node"]
    pg = lay["prefill"]["tp"] * lay["prefill"]["pp"] * lay["prefill"]["dp"]
    dg = lay["decode"]["tp"] * lay["decode"]["pp"] * lay["decode"]["dp"]
    pn, dn = math.ceil(p * pg / gpn), math.ceil(d * dg / gpn)
    sub = {"@NAME@": f"{lay['id']}-{shape.lower()}-{pol}", "@ENGINE@": lay["engine_version"], "@MODEL@": lay["model"],
           "@HARDWARE@": lay["hardware"], "@FABRIC@": lay["fabric"], "@GPN@": gpn, "@NODES@": pn + dn,
           "@PNODES@": pn, "@DNODES@": dn, "@ROUTER@": pol}
    for role, key in (("prefill", "P"), ("decode", "D")):
        x = lay[role]
        sub.update({f"@{key}TP@": x["tp"], f"@{key}PP@": x["pp"], f"@{key}DP@": x["dp"],
                    f"@{key}EP@": "true" if x["expert_parallel"] else "false"})
    s = open(os.path.join(HERE, "scenario-layout.yaml")).read()
    eng = lay["engine_by_shape"][shape]
    for role, key in (("prefill", "@PENG@"), ("decode", "@DENG@")):
        st = dict(eng[role]["set"])
        s = s.replace(key, "\n".join(f"      {k}: {json.dumps(v)}" for k, v in st.items()))
    for k, v in sub.items():
        s = s.replace(k, str(v))
    return s


def run_one(args, job):
    lay, shape, wl, conc, pol, seed = job
    if lay:
        p, d = map(int, SHAPE.match(shape).groups())
        name = f"{lay['id']}_{shape}_{wl}_{conc}_{pol}_{seed}"
        scn = f"l_{lay['id']}_{shape}_{pol}.yaml"
    else:
        p, d = SHAPES[shape]
        name = f"{shape}_{wl}_{conc}_{pol}_{seed}"
        scn = f"s_{shape}_{pol}.yaml"
    pt, ot = WORKLOADS[wl]
    out = os.path.join(args.work, "out", name + ".json")
    n = requests_for(conc)
    cmd = [args.blis, "run", "--scenarios", os.path.join(args.work, "scn"), "--scenario", scn,
           "--num-instances", str(p + d), "--prefill-instances", str(p), "--decode-instances", str(d),
           "--concurrency", str(conc), "--num-requests", str(n), "--seed", str(seed),
           "--prompt-tokens", str(pt), "--prompt-tokens-min", str(pt), "--prompt-tokens-max", str(pt), "--prompt-tokens-stdev", "0",
           "--output-tokens", str(ot), "--output-tokens-min", str(ot), "--output-tokens-max", str(ot), "--output-tokens-stdev", "0",
           "--output", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RunFailed(name, f"{name} failed ({r.returncode}):\n{r.stderr[-600:]}")
    res = json.load(open(out))
    row = {"shape": shape, "workload": wl, "concurrency": conc, "policy": pol, "seed": seed,
           "requests": n, "completed": res["requests"]["completed"]}
    if lay:
        row.update({"layout": lay["id"], "model": lay["model"], "hardware": lay["hardware"],
                    "engine_version": lay["engine_version"], "fabric": lay["fabric"],
                    "prefill": par_text(lay["prefill"]), "decode": par_text(lay["decode"])})
    for f in FIGURES:
        row[f] = repr(float(res["metrics"][f]["value"]))
    os.remove(out)  # a Result is large; the row keeps what the bound reads
    return row


class RunFailed(Exception):
    def __init__(self, name, text):
        super().__init__(text)
        self.name = name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blis", required=True, help="the blis binary (inference-sim; the guide names the commit)")
    ap.add_argument("--work", required=True, help="scratch directory, created if absent")
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--concurrency", default=None, help="client counts; default %s, or %s with --layouts" % (CONCURRENCY, LAYOUT_CONCURRENCY))
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--layouts", help="a layouts.json: run each layout instead of the original scenario")
    ap.add_argument("--stage", choices=["1", "2"], help="with --layouts: 1 runs the shapes with one prefill instance, 2 those with more")
    ap.add_argument("--layout", help="only these layout ids, comma-separated")
    ap.add_argument("--keep-going", action="store_true", help="with --layouts: record a failing run in <out>.failures and go on; its layout is then incomplete")
    ap.add_argument("--resume", action="store_true", help="with --layouts: skip runs already in <out>.partial")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.work, "out"), exist_ok=True)
    sdir = os.path.join(args.work, "scn")
    if not args.resume:
        shutil.rmtree(sdir, ignore_errors=True)
    os.makedirs(sdir, exist_ok=True)
    shutil.rmtree(os.path.join(sdir, "router"), ignore_errors=True)
    shutil.copytree(os.path.join(HERE, "router"), os.path.join(sdir, "router"))
    policies = args.policies.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.layouts:
        layouts = json.load(open(args.layouts))["layouts"]
        if args.layout:
            want = args.layout.split(",")
            byid = {l["id"]: l for l in layouts}
            layouts = [byid[w] for w in want]
        concs = [int(x) for x in (args.concurrency or ",".join(map(str, LAYOUT_CONCURRENCY))).split(",")]
        jobs = []
        for lay in layouts:
            lay = dict(lay, shapes=[x for x in lay["shapes"] if not args.stage or ((SHAPE.match(x).group(1) == "1") == (args.stage == "1"))])
            for shape in lay["shapes"]:
                for pol in policies:
                    open(os.path.join(sdir, f"l_{lay['id']}_{shape}_{pol}.yaml"), "w").write(layout_scenario(lay, shape, pol))
            jobs += list(itertools.product([lay], lay["shapes"], [lay["workload"]], concs, policies, seeds))
        columns = LAYOUT_COLUMNS + COLUMNS
        order = {l["id"]: i for i, l in enumerate(layouts)}
        shape_order = lambda r: (order[r["layout"]], tuple(map(int, SHAPE.match(r["shape"]).groups())))
    else:
        shapes = args.shapes.split(",")
        for shape, pol in itertools.product(shapes, policies):
            open(os.path.join(sdir, f"s_{shape}_{pol}.yaml"), "w").write(scenario_text(shape, pol))
        concs = [int(x) for x in (args.concurrency or ",".join(map(str, CONCURRENCY))).split(",")]
        jobs = [(None,) + j for j in itertools.product(shapes, args.workloads.split(","), concs, policies, seeds)]
        columns = COLUMNS
        shape_order = lambda r: (list(SHAPES).index(r["shape"]), list(WORKLOADS).index(r["workload"]))
    partial = args.out + ".partial"
    failures = args.out + ".failures"
    rows, done = [], set()
    if args.resume and os.path.exists(partial):
        for r in csv.DictReader(open(partial)):
            rows.append(r)
            done.add((r.get("layout", ""), r["shape"], r["workload"], int(r["concurrency"]), r["policy"], int(r["seed"])))
    todo = [j for j in jobs if ((j[0] or {}).get("id", ""), j[1], j[2], j[3], j[4], j[5]) not in done]
    lock = threading.Lock()
    fh = open(partial, "a" if args.resume else "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
    if not (args.resume and done):
        wr.writeheader()
    failed = []

    def work(j):
        try:
            row = run_one(args, j)
        except RunFailed as e:
            if not args.layouts or not args.keep_going:
                raise SystemExit(str(e))
            with lock:
                failed.append(e)
                open(failures, "a").write(str(e) + "\n")
            return None
        with lock:
            wr.writerow(row)
            fh.flush()
        return row

    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for i, row in enumerate(ex.map(work, todo), 1):
            if row:
                rows.append(row)
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", file=sys.stderr, flush=True)
    fh.close()
    rows.sort(key=lambda r: shape_order(r) + (int(r["concurrency"]), policies.index(r["policy"]), int(r["seed"])))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}; {len(failed)} runs failed", file=sys.stderr)


if __name__ == "__main__":
    main()
