"""Coverage harness: replay InferenceX end-to-end runs in BLIS and score them with InferenceX's metric
definitions over matched windows. See README.md.

  plan       inventory + partition -> work/plan.jsonl (one line per point, FIT first in priority order)
  run        render + simulate planned points through scripts/heavy/run.sh (resumable)
  aggregate  per-run tables and aggregates (FIT -> results/fit, HELD -> results/heldout only)

Usage: uv run --no-project python -I cov.py <cmd> [--limit N] [--width K] [--set FIT|HELD] [--filter k=v]
       [--catalog DIR] [--registry DIR] [--blis BIN]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blockers as BLK  # noqa: E402
import distance as DIST  # noqa: E402
import logfacts as LF  # noqa: E402
import metrics as M  # noqa: E402
import paths as PX  # noqa: E402
import spec as S  # noqa: E402

INV = PX.INV
PART = PX.PART
RECON = PX.RECON
PINS = PX.PINS
RESULTS = PX.ROOT / "results"
WORK = PX.ROOT / "work-r7"
FIT_PRIORITY = ["admission_context_proportional", "host_scheduler_step", "mixed_step", "eager_launch_per_gpu",
                "spec_decode", "collectives_at_scale", "cpu_offload", "kv_transfer_pd"]
GPU_PRIORITY = {"h100": 0, "h200": 1, "b200": 2, "b300": 3, "gb200": 4, "gb300": 5}
RANGE_RATIO = 0.8
HELD_SEED = 20261017
ANCHOR_RUNS = {"ixdb-c114784d44a688eb", "ixdb-88c64d6bf2eb9b77", "ixdb-d35720993a0fb0d8"}  # InferenceX DSV4.1 runs h100-tp8-0918,
# h200-tp8-0921, h200-tp4-0921 (the request-level DSV4.1 anchors)


def stratum(x):
    inv = INVLITE.get(x["run_id"], {})
    par = (inv.get("parallel") or {}).get("decode") or {}
    b = 1
    while b < x["conc"]:
        b *= 2
    return tuple(str(v) for v in (x["model"], x["gpu"], f"tp{par.get('tp')}-ep{par.get('ep')}-dpa{int(bool(par.get('dp_attention')))}",
            "pd" if (inv.get("parallel") or {}).get("disagg") else "colocated", inv.get("offload"), inv.get("spec_method"),
            inv.get("frontend"), x["scenario"], b))


INVLITE: dict = {}  # InferenceX .github/workflows/benchmark-tmpl.yml RANDOM_RANGE_RATIO (every commit read)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def psql(sql: str) -> str:
    return subprocess.run(["psql", "-d", PX.PGDATABASE, "-At", "-c", sql], check=True, capture_output=True,
                          text=True).stdout


def pin_value(key: str):
    return json.loads(PINS.read_text()).get(key) if PINS.exists() else None


def pins(args) -> dict:
    p = json.loads(PINS.read_text())
    blis = Path(args.blis)
    return {"blis_binary": str(blis), "blis_binary_sha256": sha_file(blis),
            "pins_file": str(PINS), "pins_sha256": sha_file(PINS),
            "registry_root": args.registry, "catalog_root": args.catalog,
            "pr_heads": {k: p[k] for k in ("inference_sim", "blis_latency_kernel", "blis_schemas", "blis_catalog",
                                           "blis_registry")},
            "inventory_sha256": sha_file(INV), "partition_sha256": sha_file(PART)}


@__import__("functools").lru_cache(maxsize=None)
def catalog_quant(cat: Path) -> dict:
    """catalog model -> {repo, quant_method} from its model.yaml and config.json (served-checkpoint parity)."""
    out = {}
    for f in sorted((cat / "models").glob("*/model.yaml")):
        repo = next((ln.split(":", 1)[1].strip() for ln in f.read_text().splitlines() if ln.strip().startswith("repo:")), "")
        q = {}
        cj = f.parent / "config.json"
        if cj.exists():
            c = json.loads(cj.read_text())
            q = c.get("quantization_config") or (c.get("text_config") or {}).get("quantization_config") or {}
        out[f.parent.name] = {"repo": repo, "quant_method": q.get("quant_method")}
    return out


def catalog_models(cat: Path) -> dict:
    out = {}
    for f in sorted((cat / "models").glob("*/model.yaml")):
        for ln in f.read_text().splitlines():
            if ln.strip().startswith("repo:"):
                out[ln.split(":", 1)[1].strip().split("/")[-1].lower()] = f.parent.name
                break
    return out


# ---------------------------------------------------------------- plan
def cmd_plan(args):
    part = json.loads(PART.read_text())
    assign = part["assignment"]
    gtargets = {}
    for g in part["groups"]:
        for pid in g["points"]:
            gtargets[pid] = g["fit_targets"]
    rows = []
    for ln in INV.open():
        r = json.loads(ln)
        if r["blocker"] != "ok" or r["gpu"] not in GPU_PRIORITY:
            continue
        INVLITE[r["run_id"]] = {k: r.get(k) for k in ("parallel", "offload", "spec_method", "frontend")}
        for p in r["points"]:
            s = assign.get(p["point_id"])
            if s is None:
                continue
            t = gtargets.get(p["point_id"], [])
            pri = min([FIT_PRIORITY.index(x) for x in t if x in FIT_PRIORITY], default=len(FIT_PRIORITY))
            rows.append({"point_id": p["point_id"], "run_id": r["run_id"], "conc": p["conc"], "set": s,
                         "scenario": r["scenario"], "gpu": r["gpu"], "model": r["catalog_model"],
                         "has_log": r["has_server_log"], "row_ids": r["rows"].get(str(p["conc"]), []),
                         "fit_targets": t, "_k": (s != "FIT", not r["has_server_log"], r["scenario"] != "single_turn",
                                                  GPU_PRIORITY[r["gpu"]], pri, p["conc"], p["point_id"])})
    rows.sort(key=lambda x: x["_k"])
    # HELD order (coordinator, 2026-10-17 ~02:58): the named gates first (the DSV4.1 Hopper anchors, then the
    # other DSV4.1 runs), then a SEEDED STRATIFIED RANDOM order over the partition's factors, so any prefix of the
    # queue covers every stratum in proportion. Stratum = (model, gpu, layout, pd, offload, spec, frontend,
    # scenario, conc_bin). Within each stratum the points are shuffled with HELD_SEED; the queue then takes the
    # stratum whose sampled fraction is lowest next (ties by a seeded key), i.e. proportional allocation.
    held = [x for x in rows if x["set"] == "HELD"]
    anchors = [x for x in held if x["model"] == "deepseek-v4.1-flash"]
    anchors.sort(key=lambda x: (x["run_id"] not in ANCHOR_RUNS, x["gpu"], x["conc"], x["point_id"]))
    rest = [x for x in held if x["model"] != "deepseek-v4.1-flash"]
    rng = random.Random(HELD_SEED)
    strata = {}
    for x in rest:
        strata.setdefault(stratum(x), []).append(x)
    for k in sorted(strata):
        rng.shuffle(strata[k])
    tie = {k: rng.random() for k in sorted(strata)}
    taken = {k: 0 for k in strata}
    order = []
    import heapq
    h = [(0.0, tie[k], k) for k in strata]
    heapq.heapify(h)
    while h:
        _, t, k = heapq.heappop(h)
        order.append(strata[k][taken[k]])
        taken[k] += 1
        if taken[k] < len(strata[k]):
            heapq.heappush(h, (taken[k] / len(strata[k]), t, k))
    for i, x in enumerate(anchors + order):
        x["held_order"] = i
        x["stratum"] = "|".join(map(str, stratum(x)))
    held = anchors + order
    rows = [x for x in rows if x["set"] == "FIT"] + held
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "plan.jsonl").open("w") as f:
        for x in rows:
            x.pop("_k")
            f.write(json.dumps(x) + "\n")
    (WORK / "plan_meta.json").write_text(json.dumps({"held_seed": HELD_SEED, "anchor_runs": sorted(ANCHOR_RUNS),
        "held_strata": len({x.get("stratum") for x in rows if x["set"] == "HELD"}), "inventory_sha256": sha_file(INV),
        "partition_sha256": sha_file(PART)}, indent=1))
    print(f"{len(rows)} points planned: FIT {sum(x['set'] == 'FIT' for x in rows)}, "
          f"HELD {sum(x['set'] == 'HELD' for x in rows)} -> {WORK / 'plan.jsonl'}")


# ---------------------------------------------------------------- measured side
def db_rows(row_ids: list[int]) -> list[dict]:
    if not row_ids:
        return []
    q = ("select json_build_object('row_id', br.id, 'metrics', br.metrics, 'server_log_id', br.server_log_id, "
         "'image', br.image, 'isl', br.isl, 'osl', br.osl, 'conc', br.conc, 'date', br.date, 'error', br.error, "
         "'head_sha', wr.head_sha, 'html_url', wr.html_url, 'head_branch', wr.head_branch) from benchmark_results br "
         f"join workflow_runs wr on wr.id = br.workflow_run_id where br.id in ({','.join(map(str, row_ids))})")
    return [json.loads(x) for x in psql(q).splitlines() if x.strip()]


def server_log(slid) -> str | None:
    if not slid:
        return None
    t = psql(f"select server_log from server_logs where id={int(slid)}")
    return t or None


def bench_log(slid) -> str:
    if not slid:
        return ""
    return psql(f"select log_text from server_log_files where server_log_id={int(slid)} "
                "and file_name in ('results/benchmark.log','benchmark.log') limit 1")


def recon_for(run_id: str) -> dict | None:
    if not hasattr(recon_for, "cache"):
        recon_for.cache = {}
        if RECON.exists():
            for ln in RECON.open():
                r = json.loads(ln)
                recon_for.cache[r["run_id"]] = r
    r = recon_for.cache.get(run_id)
    if not r or r["status"] != "EXACT":
        return None
    ln = next(iter(r["launchers"].values()))
    out = {**ln.get("flags", {}), **ln.get("config_yaml", {})}
    out["_provenance"] = r["provenance"]
    out["_head_sha"] = r["head_sha"]
    return out


def bind_recipe(rec: dict | None, conc: int, isl, osl, tp, ep) -> dict | None:
    """Bind the recipe's shell variables for one point. $CALCULATED_MAX_MODEL_LEN is the launcher's own rule,
    evaluated from its text at the run's commit (e.g. benchmarks/gptoss_fp4_b200_docker.sh:23-29 @4e52ed4c):
    1024/1024 -> ISL+OSL+20; ISL or OSL 8192 -> ISL+OSL+200; else ${MAX_MODEL_LEN:-10240}."""
    if rec is None:
        return None
    env = {"CONC": conc, "TP": tp, "EP_SIZE": ep, "ISL": isl, "OSL": osl}
    out = {}
    for k, v in rec.items():
        if isinstance(v, str) and "$" in v:
            name = v.strip('"').lstrip("$").strip("{}")
            if name in env and env[name] is not None:
                v = str(env[name])
            elif name == "CALCULATED_MAX_MODEL_LEN":
                v = str(calc_max_model_len(rec["_head_sha"], isl, osl))
            else:
                raise S.Unexpressible(f"recipe flag {k}={v} has an unbound variable")
        out[k] = v
    return out


def calc_max_model_len(sha, isl, osl):
    import re as _re
    root = PX.RECIPES / sha / "benchmarks"
    rules = set()
    for f in root.glob("*.sh"):
        t = f.read_text(errors="replace")
        i = t.find("CALCULATED_MAX_MODEL_LEN=")
        if i >= 0:
            rules.add(_re.sub(r"\s+", " ", t[max(0, t.rfind("\nif ", 0, i)):t.find("\nfi", i)]))
    want = ('if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 20)) '
            'elif [ "$ISL" = "8192" ] || [ "$OSL" = "8192" ]; then CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 200)) '
            'else CALCULATED_MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}')
    if not rules or any(r.strip() != want for r in rules):
        raise S.Unexpressible(f"CALCULATED_MAX_MODEL_LEN rule at {sha[:8]} is not the known rule")
    if isl == 1024 and osl == 1024:
        return isl + osl + 20
    if isl == 8192 or osl == 8192:
        return isl + osl + 200
    raise S.Unexpressible("CALCULATED_MAX_MODEL_LEN falls back to ${MAX_MODEL_LEN:-10240}: MAX_MODEL_LEN unrecorded")


# ---------------------------------------------------------------- workloads
def single_turn_workload(isl: int, osl: int, conc: int, seed: int = 0) -> str:
    """vllm bench serving --dataset-name random as InferenceX runs it (benchmarks/benchmark_lib.sh
    run_benchmark_serving): uniform integer lengths in [int(L*0.8), L] (benchmark_serving.py
    sample_random_requests), num_prompts = CONC*10, max_concurrency CONC, request rate inf, ignore_eos."""
    def uni(L):
        lo = int(L * RANGE_RATIO)
        p = 1.0 / (L - lo + 1)
        return "{" + ", ".join(f'"{n}": {p:.12g}' for n in range(lo, L + 1)) + "}"
    return (f'version: "1"\nseed: {seed}\ncategory: language\nnum_requests: {conc * 10}\nclients:\n'
            f"  - id: bench\n    concurrency: {conc}\n    streaming: true\n"
            f"    input_distribution: {{type: empirical, params: {uni(isl)}}}\n"
            f"    output_distribution: {{type: empirical, params: {uni(osl)}}}\n")


def single_turn_mismatches(conc: int) -> list[dict]:
    return [S._mm("request draw", "numpy seed 0 draw of lengths and random token ids",
                  "same distribution (uniform integer bins), BLIS's own seeded draw", False),
            S._mm("warmup", f"{2 * conc} warmup requests of the first prompt (--num-warmups 2*CONC), excluded",
                  "none", False, "affects only prefix-cache state of one prompt"),
            S._mm("client", "vllm bench serving (InferenceX utils/bench_serving)", "BLIS closed-loop concurrency client",
                  False)]


# --public-dataset -> (data dir, HF repo, HF revision). weka-062126: HF sha 23f152f6 (the DSV4.1 replay CORPUS_REV,
# verified). weka-256k: revision not recorded at download (MISMATCH-free: the sha256 pins the bytes).
AIPERF_CORPORA = {"semianalysis_cc_traces_weka_062126": ("weka-062126", "semianalysisai/cc-traces-weka-062126",
                                                         "23f152f6f0f9399a85901b89a6458def0ef16729"),
                  "semianalysis_cc_traces_weka_062126_256k": ("weka-256k", "semianalysisai/cc-traces-weka-062126-256k",
                                                              "unknown")}


def parse_aiperf(bench: str) -> dict:
    i = bench.find("aiperf profile ")
    if i < 0:
        return {}
    body = bench[i:i + 6000].replace("\n", " ")
    toks = body.split()
    flags, j = {}, 0
    while j < len(toks):
        t = toks[j].strip("'")
        if t.startswith("--"):
            if j + 1 < len(toks) and not toks[j + 1].startswith("--"):
                flags[t[2:]] = toks[j + 1].strip("'")
                j += 2
                continue
            flags[t[2:]] = True
        j += 1
    return flags


# ---------------------------------------------------------------- run one point
def point_dir(x: dict) -> Path:
    return WORK / "points" / x["run_id"] / f"c{x['conc']}_{x['point_id']}"


def run_point(x: dict, args, cmods: dict, pin: dict) -> dict:
    pdir = point_dir(x)
    pdir.mkdir(parents=True, exist_ok=True)
    res = {"point_id": x["point_id"], "run_id": x["run_id"], "conc": x["conc"], "set": x["set"],
           "scenario": x["scenario"], "gpu": x["gpu"], "model_slug": x["model"], "fit_targets": x["fit_targets"]}
    try:
        meas = db_rows(x["row_ids"])
        if not meas:
            raise S.Unexpressible("no database row")
        r0 = meas[0]
        inv = INVMAP[x["run_id"]]
        slid = r0.get("server_log_id")
        logt = server_log(slid)
        log = S.parse_log(logt) if logt else None
        par = inv["parallel"]["decode"]
        recon = None if log and log.get("non_default") else bind_recipe(
            recon_for(x["run_id"]), x["conc"], r0.get("isl"), r0.get("osl"), par.get("tp"), par.get("ep"))
        if log is None and recon is None:
            raise S.Unexpressible("no server log and no EXACT recipe reconstruction")
        d = S.resolve(inv, log if (log and log.get("non_default")) else None, recon, x["conc"], cmods)
        res["source"] = {"url": r0["html_url"], "head_sha": r0["head_sha"], "head_branch": r0["head_branch"],
                         "label": "InferenceX" if r0["head_branch"] == "main" else "Unofficial InferenceX",
                         "db_rows": [m["row_id"] for m in meas], "server_log_id": slid,
                         "db_dump": "https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-09-28"}
        res["resolved"] = {k: v for k, v in d.items() if k not in ("mismatches",)}
        res["mismatches"] = list(d["mismatches"])
        name = f"cov-{x['point_id']}"
        (pdir / f"{name}.yaml").write_text(S.scenario_yaml(name, d))
        extra = []
        if d.get("acceptance_length"):
            extra += ["--speculative-acceptance-length", str(d["acceptance_length"])]
        elif d.get("spec_method"):
            extra += ["--speculative-category", "coding", "--speculative-thinking", "off"]
            res["mismatches"].append(S._mm("acceptance category", "random-token prompts (single-turn) or coding traces",
                                           "registry coding profile, thinking off", False))
        if x["scenario"] == "single_turn":
            isl, osl = int(r0["isl"]), int(r0["osl"])
            (pdir / "workload.yaml").write_text(single_turn_workload(isl, osl, x["conc"]))
            res["mismatches"] += single_turn_mismatches(x["conc"])
            cmd = [args.blis, "run", "--workload-spec", str(pdir / "workload.yaml"), "--seed", "0", "--timeout", "-1"]
        else:
            bench = bench_log(slid)
            fl = parse_aiperf(bench)
            ds = fl.get("public-dataset")
            if ds not in AIPERF_CORPORA:
                raise S.Unexpressible(f"aiperf corpus {ds!r} not on disk / not mapped")
            cdir = PX.CORPORA / AIPERF_CORPORA[ds][0]
            corpus = cdir / "traces.jsonl"
            csha = (cdir / "traces.sha256").read_text().split()[0] if (cdir / "traces.sha256").exists() else sha_file(corpus)
            (pdir / "workload.yaml").write_text(agentic_workload(fl, AIPERF_CORPORA[ds][1], csha, AIPERF_CORPORA[ds][2]))
            res["mismatches"] += agentic_mismatches(fl)
            cmd = [args.blis, "replay", "--agentic-scenario", "inferencex-agentx-mvp", "--workload-spec",
                   str(pdir / "workload.yaml"), "--corpus", str(corpus), "--corpus-sha256", csha,
                   "--corpus-cache", str(PX.DATA / "corpus-cache")]
        cmd += ["--scenario", f"{name}.yaml", "--scenarios", str(pdir), "--registry", args.registry,
                "--catalog", args.catalog, "--metrics-path", str(pdir / "metrics.json")] + extra
        stamp = " ".join(cmd) + f"\n#blis_sha256={pin['blis_binary_sha256']}\n#scenario_sha256=" + \
            sha_file(pdir / f"{name}.yaml") + "\n#workload_sha256=" + sha_file(pdir / "workload.yaml") + "\n"
        res["command"] = " ".join(cmd)
        mp = pdir / "metrics.json"
        res["_stamp"] = stamp
        if mp.exists() and (pdir / "command.txt").exists() and (pdir / "command.txt").read_text() == stamp:
            res["reused"] = True
        elif args.phase == "prepare":
            if mp.exists():
                res["rescored_scenario_changed"] = True
            mp.unlink(missing_ok=True)
            res["status"] = "prepared"
            res["_cmd"] = cmd
            res["_ctx"] = (d, meas, log, inv)
            if res.get("rescored_scenario_changed"):
                (pdir / "rescored").write_text("scenario/command changed under the behaviour table\n")
            return res
        else:
            raise S.Unexpressible("BLIS refused/failed: " + _stderr_reason(pdir))
        if (pdir / "wall_s").exists():
            res["wall_s"] = float((pdir / "wall_s").read_text())
        res["metrics_sha256"] = sha_file(mp)
        res["metrics_path"] = str(mp.relative_to(PX.ROOT))
        facts = LF.facts(logt) if logt else None
        res.update(score(x, d, json.loads(mp.read_text()), meas, log, inv, facts))
        res["status"] = "scored"
        pb = BLK.parity_blockers(res, facts, catalog_quant(Path(args.catalog)).get(d["model"]), r0.get("osl"),
                                 int(par.get("tp") or d["tp"]))
        if pb:
            res["parity_blockers"] = pb
            res["status"] = "blocked"
            res["blocked_reason"] = "; ".join(f"{b['name']}: {b['reason']}" for b in pb)
    except S.Unexpressible as e:
        res["status"] = "blocked"
        res["blocked_reason"] = str(e)
    except Exception as e:  # an apparatus error is never silent: recorded and listed
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"[:800]
    res["pins"] = pin
    (pdir / "result.json").write_text(json.dumps({k: v for k, v in res.items() if not k.startswith("_")},
                                                 indent=1, default=str))
    return res


def _stderr_reason(pdir: Path) -> str:
    e = (pdir / "stderr.log").read_text() if (pdir / "stderr.log").exists() else ""
    err = [ln for ln in e.splitlines() if "fatal" in ln or "error" in ln.lower() or "timeout" in ln]
    return (err[-1] if err else e[-400:]).strip()[:600]


def agentic_workload(fl: dict, hf: str, csha: str, rev: str = "unknown") -> str:
    assert fl.get("scenario") == "inferencex-agentx-mvp", fl.get("scenario")
    return f"""# GENERATED by validation/coverage/harness from the run's aiperf command (stateable fields only).
workload:
  replay:
    scenario: inferencex-agentx-mvp
    harness_version: 0.12.0
    corpus:
      loader: {fl['public-dataset']}
      format: weka-trace-v7
      encoding: jsonl
      source:
        huggingface: {hf}
        revision: {rev}
        file: traces.jsonl
        split: train
      sha256: {csha}
    concurrency: {fl['concurrency']}
    random_seed: {fl.get('random-seed', 42)}
    benchmark_duration_s: {fl.get('benchmark-duration', 1800)}
    grace_period_s: 30
    cache_bust: first_turn_prefix
    system_idle_gap_cap_s: 10
    trajectory_start:
      min_ratio: {fl.get('trajectory-start-min-ratio', 0)}
      max_ratio: {fl.get('trajectory-start-max-ratio', 1)}
    warmup:
      mode: synthesized
    split_flattened_agents: true
    category: coding
    thinking: "on"
"""


def agentic_mismatches(fl: dict) -> list[dict]:
    out = []
    if "trace-idle-gap-cap-seconds" in fl:
        out.append(S._mm("per-trace idle-gap cap", f"{fl['trace-idle-gap-cap-seconds']} s",
                         "unset (blis-schemas Replay refuses trace_idle_gap_cap_s under the 0.12.0 mvp lock)", False))
    if "warmup-requests-per-lane" in fl:
        out.append(S._mm("warmup", f"synthesized primers + {fl['warmup-requests-per-lane']} cache-pressure requests per lane",
                         "synthesized primers only", False))
    out.append(S._mm("harness", "aiperf SemiAnalysisAI fork (pyproject 0.12.x)", "aiperf release 0.12.0 transcription", False))
    if "max-context-length" in fl:
        out.append(S._mm("max-context-length", fl["max-context-length"], "not passed", True))
    return out


# ---------------------------------------------------------------- scoring
def host_completion_offset(m: dict, registry: str | None = None) -> float:
    """The completion overhead BLIS's per-request e2e carries after the last chunk (B2): the registry's
    host_completion (cost-model-host-overheads.yaml), in whole simulator ticks (us). Where the rows are the
    aggregate's population, removing it must reproduce the aggregate's e2e_p50/p90 to 1 ns."""
    reg = Path(registry or os.environ.get("IX_REGISTRY") or pin_value("registry_root")) / "coefficients" / "cost-model-host-overheads.yaml"
    txt = reg.read_text()
    i = txt.index("- host_completion:")
    us = float(txt[i:i + 200].split("value:")[1].split()[0])
    off = int(us) / 1e3
    fixed = M.blis_rows(m, off, True)
    if len(fixed) == m["measured_completed_requests"]:
        lat = [r["lat_ms"] for r in fixed]
        assert abs(M.pct(lat, 50) - m["e2e_p50_ms"]) < 1e-6 and abs(M.pct(lat, 90) - m["e2e_p90_ms"]) < 1e-6, \
            "row latency minus host_completion does not reproduce the aggregate"
    return off


_KVU = None


# Models whose logged "GPU KV cache size" (get_kv_cache_capacity = max_concurrency x max_model_len,
# kv_cache_utils.py:2444-2472@v0.31.0) is not pool tokens: DeepSeek-V4-Pro's compressor state caches are
# SlidingWindowMLASpec groups reserving ~190 pool blocks a request (DSV4PRO_KVPOOL.md). The size and concurrency
# lines come from the same function, so their self-consistency cannot detect it: the guard is by architecture.
KV_SIZE_NOT_POOL_TOKENS = {"deepseek-v4-pro"}


def kv_units_differ(slid, model: str | None = None) -> bool:
    """True when the log's "GPU KV cache size" is not token capacity: the model is in KV_SIZE_NOT_POOL_TOKENS, or the
    size line is inconsistent (>5%) with its own "Maximum concurrency" line (work/kv_units.json). Such a pool is not
    BLIS's kv_capacity_tokens_per_replica: a named mismatch, not compared as tokens."""
    global _KVU
    if model in KV_SIZE_NOT_POOL_TOKENS:
        return True
    if _KVU is None:
        f = WORK / "kv_units.json"
        _KVU = json.loads(f.read_text()) if f.exists() else {}
    return (_KVU.get(str(slid)) or {}).get("consistent") is False


def pool_blocks_measured(d: dict, facts: dict | None) -> int | None:
    """DSV4-Pro, single DP rank: vLLM's pool blocks derived from the log (blockers.dsv4_pro_pool_blocks), compared
    with BLIS's blocks, so the pool is scored block for block as vLLM defines it."""
    if d["model"] != "deepseek-v4-pro" or d["dp"] != 1 or not facts:
        return None
    e = d["engine"]
    return BLK.dsv4_pro_pool_blocks(facts, e["max_model_len"], e["max_num_batched_tokens"])


def tag(v, d, w):
    return {"value": v, "def": d, "window": w}


def score(x, d, m, meas_rows, log, inv, facts=None) -> dict:
    hc = host_completion_offset(m)
    # aiperf records a request that finishes inside the grace period (blis-schemas replay.go GracePeriodS);
    # BLIS keeps such rows in phase 'profile', so every completed profile row is in (as compare_ix).
    rows = M.blis_rows(m, hc, True)
    out = {"b2_host_completion_ms": hc, "grace_rows": len(rows) - m.get("measured_completed_requests", len(rows)), "blis_versions": {k: m.get(k) for k in ("blis_version", "kernel_version",
                                                                             "schemas_version")}}
    if x["scenario"] == "single_turn":
        tpdiv = int(inv["parallel"]["decode"].get("tp") or d["tp"])
        pred = M.single_turn_blis(m, rows, tpdiv)
        keymap, win, dname = M.IX_SINGLE, "benchmark (all measured requests; warmups excluded)", "vllm-bench-serving"
        unexp = dict(M.SINGLE_UNEXPRESSED)
        keymap, unexp = M.drop_chunk_itl(keymap, unexp, d.get("stream_interval"))
    else:
        pred = M.request_metrics(rows)
        keymap, win, dname = M.IX_AGENTIC, "aiperf profiling phase", "aiperf-agentic"
        unexp = M.AGENTIC_UNEXPRESSED
    P = {k: tag(v, dname, win) for k, v in pred.items()}
    rowsout = []
    for mr in meas_rows:
        mm = mr["metrics"]
        Mm = {}
        for ixk, (hk, sc) in keymap.items():
            if ixk in mm and mm[ixk] is not None:
                Mm[hk] = tag(mm[ixk] * sc, dname, win)
        if x["scenario"] != "single_turn":
            if mm.get("server_gpu_cache_hit_rate") is not None:
                Mm["gpu_cache_hit_rate_whole_run"] = tag(mm["server_gpu_cache_hit_rate"], "prefix hit/query tokens",
                                                         "whole server run (warmup + profiling)")
            if mm.get("server_cpu_cache_hit_rate") is not None:
                Mm["cpu_cache_hit_rate_whole_run"] = tag(mm["server_cpu_cache_hit_rate"], "external hit/query tokens",
                                                         "whole server run (warmup + profiling)")
        kvm = mm.get("kv_cache_pool_tokens") or ((log or {}).get("kv_tokens") or [None])[0]
        if kvm and d["dp"] == 1 and not kv_units_differ(meas_rows[0].get("server_log_id"), d["model"]):
            Mm["kv_cache_pool_tokens"] = tag(kvm, "engine KV pool tokens", "engine start")
        vb = pool_blocks_measured(d, facts)
        if vb:
            Mm["kv_cache_pool_blocks"] = tag(vb, "engine KV pool blocks (vLLM num_gpu_blocks)", "engine start")
        rowsout.append({"row_id": mr["row_id"], "measured": Mm})
    if x["scenario"] != "single_turn":
        P["gpu_cache_hit_rate_whole_run"] = tag(m["whole_run"]["cache_hit_rate"], "prefix hit/query tokens",
                                                "whole server run (warmup + profiling)")
        P["cpu_cache_hit_rate_whole_run"] = tag(m.get("cpu_tier_hit_rate", 0.0) if not d.get("offload") else None,
                                                "external hit/query tokens", "whole server run (warmup + profiling)")
        if d.get("offload"):
            P.pop("cpu_cache_hit_rate_whole_run")
    if d["dp"] == 1:
        P["kv_cache_pool_tokens"] = tag(m["kv_capacity_tokens_per_replica"], "engine KV pool tokens", "engine start")
        if pool_blocks_measured(d, facts):
            P["kv_cache_pool_blocks"] = tag(m["kv_capacity_tokens_per_replica"] // d["engine"]["block_size"],
                                            "engine KV pool blocks (vLLM num_gpu_blocks)", "engine start")
    for r in rowsout:
        M.assert_same(r["measured"], P)
        r["signed_err_pct"] = {k: M.signed_pct(P[k]["value"], r["measured"][k]["value"])
                               for k in r["measured"] if k in P and P[k]["value"] is not None}
    out["predicted"] = P
    out["measured_rows"] = rowsout
    out["distances"] = point_distances(x, d, m, hc, meas_rows, out)
    out["unexpressed_metrics"] = unexp
    out["acceptance"] = {"blis": m.get("acceptance"), "log_mean_al_last": (log or {}).get("mean_al", [])[-3:],
                         "note": "log value is vLLM's 10 s interval mean; BLIS's is the resolved mean AL: not scored"}
    return out


def point_distances(x, d, m, hc, meas_rows, scored) -> dict:
    """Distribution distances (distance.py) for the point: agentic from the InferenceX request timeline and server
    series over the profiling window; single-turn has aggregates only (quantile errors)."""
    if x["scenario"] == "single_turn":
        return DIST.single_turn_quantiles(scored)
    r0 = meas_rows[0]
    return DIST.point_distances(m, hc, r0["row_id"], r0.get("metrics"), int(d.get("dp") or 1))


# ---------------------------------------------------------------- driver
INVMAP: dict = {}


def load_inv(ids: set):
    for ln in INV.open():
        r = json.loads(ln)
        if r["run_id"] in ids:
            INVMAP[r["run_id"]] = r


def cmd_run(args):
    plan = [json.loads(ln) for ln in (WORK / "plan.jsonl").open()]
    for f in args.filter or []:
        k, v = f.split("=", 1)
        plan = [x for x in plan if str(x.get(k)) == v]
    if args.set:
        plan = [x for x in plan if x["set"] == args.set]
    # Resume: every point is re-resolved and re-rendered; a metrics file is reused only when the command, BLIS
    # sha, scenario and workload are byte-identical to the ones that produced it (command.txt stamp).
    # --skip-done skips points whose last result is final without re-resolving (fast restart, no change).
    done = set()
    if args.skip_done:
        for x in plan:
            rp = point_dir(x) / "result.json"
            if rp.exists() and json.loads(rp.read_text()).get("status") in ("scored", "blocked"):
                done.add(x["point_id"])
    if args.redo:
        for x in plan:
            (point_dir(x) / "command.txt").unlink(missing_ok=True)
    plan = [x for x in plan if x["point_id"] not in done][: args.limit]
    load_inv({x["run_id"] for x in plan})
    cmods = catalog_models(Path(args.catalog))
    pin = pins(args)
    print(f"running {len(plan)} points (slots {args.width}, batch {args.batch}); {len(done)} skipped", flush=True)
    # Pipeline: each of `width` lanes takes the next `batch` points from the shared (ordered) queue, prepares
    # them, runs them in one heavy slot, scores them, and repeats -- no lane waits for another lane's batch.
    import threading
    lock = threading.Lock()
    queue = list(plan)

    def lane():
        while True:
            with lock:
                if not queue:
                    return
                batch_pts = queue[:args.batch]
                del queue[:args.batch]
            a = argparse.Namespace(**vars(args))
            a.phase = "prepare"
            with cf.ThreadPoolExecutor(4) as ex:
                prep = list(ex.map(lambda x: run_point(x, a, cmods, pin), batch_pts))
            todo = [r for r in prep if r.get("status") == "prepared"]
            for r in prep:
                if r.get("status") != "prepared":
                    print(r["set"], r["run_id"], r["conc"], r["status"], (r.get("blocked_reason") or r.get("error") or "")[:160],
                          flush=True)
            if todo:
                exec_batch(todo, a)
            a.phase = "score"
            byid = {x["point_id"]: x for x in batch_pts}
            with cf.ThreadPoolExecutor(4) as ex:
                for r in ex.map(lambda r: run_point(byid[r["point_id"]], a, cmods, pin), todo):
                    print(r["set"], r["run_id"], r["conc"], r["status"], (r.get("blocked_reason") or r.get("error") or "")[:160],
                          flush=True)

    ths = [threading.Thread(target=lane) for _ in range(args.width)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


def exec_batch(batch: list[dict], args):
    """One heavy slot runs the batch's simulations one after another (batch_exec.py)."""
    items = []
    for r in batch:
        pdir = point_dir({"run_id": r["run_id"], "conc": r["conc"], "point_id": r["point_id"]})
        items.append({"cmd": r["_cmd"], "cwd": str(pdir), "timeout": args.timeout, "stderr": str(pdir / "stderr.log"),
                      "stamp": r["_stamp"], "pdir": str(pdir)})
    bf = WORK / "batches" / f"{batch[0]['point_id']}.json"
    bf.parent.mkdir(parents=True, exist_ok=True)
    bf.write_text(json.dumps(items))
    subprocess.run(PX.heavy([sys.executable, "-I", str(HERE / "batch_exec.py"), str(bf)]), capture_output=True)
    done = json.loads(Path(str(bf) + ".done.json").read_text()) if Path(str(bf) + ".done.json").exists() else []
    for it, d in zip(items, done):
        if d["rc"] == 0 and (Path(it["pdir"]) / "metrics.json").exists():
            (Path(it["pdir"]) / "command.txt").write_text(it["stamp"])
            (Path(it["pdir"]) / "wall_s").write_text(str(d["wall_s"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--set")
    ap.add_argument("--filter", action="append")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--catalog", default=pin_value("catalog_root"))
    ap.add_argument("--registry", default=pin_value("registry_root"))
    ap.add_argument("--blis", default=pin_value("blis_binary"))
    args = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
