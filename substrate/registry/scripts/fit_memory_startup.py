#!/usr/bin/env python3
"""Per-term non-KV memory, fitted to vLLM's own start-up decomposition on fit units (inference-sim/blis-substrate#76).

    python scripts/fit_memory_startup.py            # print the fits, rewrite the memory set
    python scripts/fit_memory_startup.py --check    # exit 1 if the set would change

THE DATA. validator/testdata/anchors/memory_startup_fit_units.json: derived figures from the
frozen InferenceX server logs of FIT-role units only (catalog/ledger/published/fit.yaml, D7),
one row per unit, each citing its run and its log's SHA-256. The logs are not redistributed.
Only vLLM >= 0.26 prints the decomposition this reads:

  "Actual usage is X GiB for consumed memory (weights + non-torch), Y GiB for peak activation,
   and Z GiB for CUDAGraph memory"   and   "CUDA graph pool memory: A GiB (actual), E GiB
   (estimated)"  (or "Estimated CUDA graph memory: E GiB total")

WHAT vLLM SUBTRACTS FROM THE KV BUDGET (v0.33.0 vllm/v1/worker/gpu_worker.py:720-756):
available = requested - non_kv_cache_memory - E, where non_kv_cache_memory is the weights, the
non-torch growth and the transient peak. The logged peak activation is
peak_activation_memory = transient_peak_headroom + E (:748-750): it ALREADY HOLDS the CUDA-graph
estimate. On every unit, util x total - (Model loading took) - (Available KV cache memory)
equals (X - load) + Y to within rounding: the logged Z is reported beside the budget, not in it
again. So the three terms are fitted from:

  CUDA graph   E, against the capture ceiling and the uniform-decode FULL ceiling the log states
               ("Profiling CUDA graph memory: PIECEWISE=n (largest=Cmix), FULL=m (largest=Cdec)"),
               in the kernel's form floor + per_mixed_token x Cmix + per_decode_token x Cdec.
  activation   Y - E, the transient, as a multiple of max_num_batched_tokens x h x 2 (the
               kernel's activation_buffer_count, h its activationWidth: the hidden size on a
               latent-attention model, query heads x head width otherwise).
  non-torch    X - load: vLLM's non-torch growth, inside the budget. The kernel holds
               nccl_communicator_bytes and engine_workspace_bytes OUTSIDE it (they precede the
               snapshot), so this is its own term, nontorch_in_budget_bytes_<N>rank.

GROUPS. One entry set per (model, part), and per tensor-parallel width N = min(tp, 8) for the
activation and non-torch terms, wherever at least MIN_UNITS fit units print the decomposition
(the 2026-11-17 pre-registration). Every other model or part keeps the committed values, and the
non-torch term has no entry there, which the kernel records as a stated gap on the point.
Not fitted, and named in each entry's rationale: the activation of deepseek-v4.1-flash, whose
transient the kernel splits into the activation multiple and its own engine-buffer model
(engineBufferBytes), so Y - E is not the multiple's share alone; an activation transient below
the dummy sampler's peak, where the multiple is not identified.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO.parent / "catalog" / "models"
sys.path.insert(0, str(REPO / "scripts"))
import coefficient_blocks as cb  # noqa: E402
from fit_deepseek_v41_capture import capture_sizes, decode_graphs  # noqa: E402

MEMORY = REPO / "coefficients" / "cost-model-memory.yaml"
ANCHOR = REPO / "validator" / "testdata" / "anchors" / "memory_startup_fit_units.json"
GIB = 1 << 30
MODE = "full_and_piecewise"
MIN_UNITS = 5
ACT_BYTES = 2  # bf16, the kernel's price.ActivationBytes
ENGINE_BUFFER_MODELS = {"deepseek-v4.1-flash"}
DEFAULT_MAX_NUM_SEQS = 1024  # vLLM's default on a >= 70 GiB part (engine/arg_utils.py:2311-2320@v0.22.0)
SEED = 20261117
BOOT = 1000
# The log states each figure to 0.01 GiB, so an interval narrower than that rounding would claim
# more precision than the data hold.
QUANTUM = 0.005 * GIB


def ceilings(u: dict) -> tuple[int, int]:
    """(Cmix, Cdec): the capture ceiling and the largest uniform-decode FULL graph. The log's own
    "Profiling CUDA graph memory ... (largest=...)" figures where it prints them; otherwise vLLM's
    capture list for the stated max_cudagraph_capture_size (the v0.31.0 rule the V4.1 writer uses),
    each decode size rounded up to 1 + num_spec_tokens and kept within max_num_seqs requests."""
    if u.get("piecewise_largest") and u.get("full_largest"):
        return u["piecewise_largest"], u["full_largest"]
    seqs = u["max_num_seqs"] or DEFAULT_MAX_NUM_SEQS
    sizes = capture_sizes(u["ceiling"], u["mnbt"])
    dec = decode_graphs(u["ceiling"], u["mnbt"], seqs, 1 + int(u.get("spec_tokens") or 0))
    return max(sizes), (max(dec) if dec else 0)


def widen(lo: float, hi: float, v: float) -> tuple[float, float]:
    return min(lo, v - QUANTUM), max(hi, v + QUANTUM)


def activation_width(graph: dict) -> int:
    """The kernel's activationWidth: per layer kind with a KV-holding attention, the hidden size
    for a latent kind, else query heads x head width; the largest, or the hidden size."""
    hidden = int(graph["global"]["hidden_size"])
    h = 0
    for kind in graph["layer_kinds"]:
        for n in kind["nodes"]:
            if n.get("op") != "Attention" or not n.get("holds_engine_kv"):
                continue
            w = hidden if n.get("kind") in ("mla", "sparse_mla") else int(n["n_q"]) * int(n["d_h"])
            h = max(h, w)
    return h or hidden


def vocab(graph: dict) -> int:
    return int(graph["global"]["vocab_size"])


def boot(values: list[float], stat) -> tuple[float, float]:
    """The 2.5 and 97.5 percentiles of stat over BOOT resamples (fixed seed)."""
    rng = random.Random(SEED)
    xs = sorted(stat([rng.choice(values) for _ in values]) for _ in range(BOOT))
    return xs[int(0.025 * BOOT)], xs[int(0.975 * BOOT) - 1]


def lstsq(rows: list[tuple[float, ...]], y: list[float]) -> list[float] | None:
    """Ordinary least squares by the normal equations, None if singular."""
    k = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * yi for r, yi in zip(rows, y)) for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-9 * max(1.0, max(abs(v) for row in a for v in row)):
            return None
        a[c], a[p], b[c], b[p] = a[p], a[c], b[p], b[c]
        for r in range(k):
            if r != c:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * z for x, z in zip(a[r], a[c])]
                b[r] -= f * b[c]
    return [b[i] / a[i][i] for i in range(k)]


def capture_fit(units: list[dict], separated: bool | None = None):
    """(floor, per_mixed, per_decode) bytes from E against (Cmix, Cdec); per_decode is 0 and
    named when the two ceilings cannot be separated (rank-deficient or a negative slope).
    `separated` forces the form, so a bootstrap resample keeps the one the full data chose."""
    pts = [(1.0, *map(float, ceilings(u))) for u in units]
    e = [u["cudagraph_estimate_gib"] * GIB for u in units]
    if separated is not False:
        full = lstsq(pts, e)
        if full and min(full) >= 0:
            return full, True
        if separated:
            return None, True
    two = lstsq([p[:2] for p in pts], e)
    if two and min(two) >= 0:
        return [two[0], two[1], 0.0], False
    return None, False


def capture_ci(units: list[dict], separated: bool) -> list[tuple[float, float]]:
    """2.5 and 97.5 percentiles of each parameter over BOOT unit resamples that still fit."""
    rng = random.Random(SEED)
    draws = []
    for _ in range(BOOT):
        params, _ = capture_fit([rng.choice(units) for _ in units], separated)
        if params is not None:
            draws.append(params)
    out = []
    for i in range(3):
        xs = sorted(d[i] for d in draws)
        out.append((xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]))
    return out


def groups(units: list[dict], key) -> dict:
    out: dict = {}
    for u in units:
        out.setdefault(key(u), []).append(u)
    return out


def fit(table: dict) -> dict:
    units = [u for u in table["units"] if u.get("cudagraph_estimate_gib") is not None
             and u.get("mode") == MODE.upper()]
    graphs = {m: yaml.safe_load((CATALOG / m / "graph.yaml").read_text())
              for m in sorted({u["model"] for u in units})}
    res = {"capture": [], "activation": [], "nontorch": [], "named": []}
    # Per tensor-parallel width as well: a rank's graph buffers follow its share of the heads and
    # the hidden width, so a TP1 start-up is not a TP4 one (Kimi-K2.5 on B300: 2.58 GiB at ceiling
    # 256 at TP1, 1.36 at 512 at TP4).
    for (m, hw, tp), us in sorted(groups(units, lambda u: (u["model"], u["hardware"], int(u["tp"]))).items()):
        if (m, hw, tp) in UNCHECKED:
            res["named"].append(f"capture {m} {hw} tp{tp}: no anchor unit to check it out of sample")
            continue
        cmixes = {ceilings(u)[0] for u in us}
        if len(us) < MIN_UNITS or len(cmixes) < 3:
            res["named"].append(f"capture {m} {hw} tp{tp}: {len(us)} fit units, {len(cmixes)} distinct ceilings")
            continue
        params, separated = capture_fit(us)
        if params is None:
            res["named"].append(f"capture {m} {hw} tp{tp}: no non-negative fit")
            continue
        resid = [u["cudagraph_estimate_gib"] - (params[0] + params[1] * ceilings(u)[0]
                                                 + params[2] * ceilings(u)[1]) / GIB for u in us]
        res["capture"].append(dict(model=m, hw=hw, tp=tp, n=len(us), floor=params[0], mixed=params[1],
                                   decode=params[2], separated=separated, resid=resid,
                                   cmix=(min(cmixes), max(cmixes)), ci=capture_ci(us, separated),
                                   seqs=(min(u["max_num_seqs"] or DEFAULT_MAX_NUM_SEQS for u in us),
                                         max(u["max_num_seqs"] or DEFAULT_MAX_NUM_SEQS for u in us)),
                                   runs=sorted({u["run"] for u in us})))
    by_width = groups(units, lambda u: (u["model"], u["hardware"], int(u["tp"])))
    for (m, hw, tp), us in sorted(by_width.items()):
        n = min(tp, 8)
        if (m, hw, tp) in UNCHECKED:
            res["named"].append(f"activation and non-torch {m} {hw} {n}rank: no anchor unit to check it out of sample")
            continue
        if len(us) < MIN_UNITS:
            res["named"].append(f"activation and non-torch {m} {hw} {n}rank: {len(us)} fit units")
            continue
        nt = [(u["consumed_gib"] - u["load_gib"]) * GIB for u in us]
        lo, hi = widen(*boot(nt, statistics.median), statistics.median(nt))
        res["nontorch"].append(dict(model=m, hw=hw, tp=tp, rank=n, n=len(us), value=statistics.median(nt),
                                    ci=(lo, hi), mnbt=(min(u["mnbt"] for u in us), max(u["mnbt"] for u in us)),
                                    runs=sorted({u["run"] for u in us})))
        if m in ENGINE_BUFFER_MODELS:
            res["named"].append(f"activation {m} {hw} {n}rank: the kernel's engine-buffer model shares the transient")
            continue
        g = graphs[m]
        h = activation_width(g)
        tr = [(u["peak_activation_gib"] - u["cudagraph_estimate_gib"]) * GIB for u in us]
        sampler = [min(u["mnbt"], u["max_num_seqs"] or DEFAULT_MAX_NUM_SEQS) * vocab(g) * (ACT_BYTES + 4)
                   for u in us]
        if statistics.median(tr) <= statistics.median(sampler):
            res["named"].append(f"activation {m} {hw} {n}rank: transient at or below the sampler peak")
            continue
        counts = [t / (u["mnbt"] * h * ACT_BYTES) for t, u in zip(tr, us)]
        q = 2 * QUANTUM / statistics.median([u["mnbt"] * h * ACT_BYTES for u in us])  # Y and E each rounded
        lo, hi = boot(counts, statistics.median)
        lo, hi = min(lo, statistics.median(counts) - q), max(hi, statistics.median(counts) + q)
        res["activation"].append(dict(model=m, hw=hw, tp=tp, rank=n, n=len(us), value=statistics.median(counts),
                                      ci=(lo, hi), h=h, mnbt=(min(u["mnbt"] for u in us), max(u["mnbt"] for u in us)),
                                      runs=sorted({u["run"] for u in us})))
    return res


# InferenceX's part names that the catalog spells otherwise.
CATALOG_PART = {"gb200": "gb200-nvl72"}


# Why each model's terms are its own (inference-sim/blis-substrate#294 seeks the general form). Every
# one of these models logs the same FlashInfer allreduce-norm fusion workspace (mnnvl backend), so the
# per-model differences lie in what only that model runs.
MECHANISM = {
    "deepseek-v4.1-flash": ("its DSpark drafter and its Engram path (tables offloaded to pinned host memory, 23.6 "
                            "GiB per rank host-side in these logs), which only this model runs; the logs do not "
                            "size their device-side share, so the 5-8 GiB above the other models is not attributed "
                            "further"),
    "kimi-k2.5-nvfp4": ("its latent attention, whose activation width is the 7,168 hidden size (MLA), and its "
                        "routed-expert workspace at 384 experts; no drafter"),
    "minimax-m3-mxfp8": ("its EAGLE3 drafter (method eagle3 in every fitted run), whose forward and graphs share "
                         "the profile, and its block-sparse attention"),
}
GENERAL_FORM = "inference-sim/blis-substrate#294"

# Groups whose entries could not be checked out of sample, because no published anchor unit of the
# same (model, part, tp) prints the decomposition. Each per-model entry lands only once it improves
# its own term on that model's anchor units (the coordinator's guard for the single-model exception;
# the check is recorded on blis-substrate#76); these stay a recorded result until an anchor covers them.
UNCHECKED = {("deepseek-v4.1-flash", "gb300", 2)}


def scope(r: dict) -> str:
    return f"{{hardware: [{CATALOG_PART.get(r['hw'], r['hw'])}], model: [{r['model']}], tp: [{r['tp']}]}}"


def entry(name: str, value, units: str, sc: str, ci: tuple, fit_range: list[tuple], rationale: str) -> list[str]:
    lines = [f"  - {name}:", f"      value: {value}", f"      units: {units}", "      method: measured",
             "      fitted: true", f"      scope: {sc}",
             "      measures: {time: device, execution: eager, harness: \"vLLM's start-up memory profile "
             "(gpu_worker.py determine_available_memory) on InferenceX fit-role units\"}",
             "      fit_range:"]
    lines += [f"        - {{dim: {d}, low: {lo}, high: {hi}}}" for d, lo, hi in fit_range]
    lines += [f"      ci95: [{ci[0]}, {ci[1]}]", "      sources:",
              f'        - {{kind: discussion, cite: "InferenceX server logs of fit-role units, derived figures '
              f'vendored in validator/testdata/anchors/{ANCHOR.name}", role: primary}}',
              '        - {kind: discussion, cite: "vLLM v0.33.0 vllm/v1/worker/gpu_worker.py:720-756 '
              '(determine_available_memory: the budget, and peak activation holding the CUDA-graph estimate)", '
              'role: supporting}',
              "      rationale: >", f"        {rationale}"]
    return lines


def entries(res: dict) -> list[list[str]]:
    out = []
    base_note = ("Fitted by scripts/fit_memory_startup.py on fit-role units only (D7); groups below "
                 f"{MIN_UNITS} fit units keep the committed value. The logged peak activation holds the "
                 "CUDA-graph estimate (gpu_worker.py:748-750), so this split charges the estimate once; the "
                 "estimate is profile_cudagraph_memory's (vllm/v1/worker/gpu_model_runner.py:6268@v0.33.0).")

    def note_for(r: dict) -> str:
        return (f"A single-model entry, a recorded exception to the single-model ratchet: the term is "
                f"{r['model']}'s own because of {MECHANISM[r['model']]}. The general form is "
                f"{GENERAL_FORM}. {base_note}")
    for r in res["capture"]:
        runs = ", ".join(str(x) for x in r["runs"][:6])
        tail = (f"{r['n']} fit units of {r['model']} on {r['hw']} (runs {runs}); estimate residuals "
                f"{min(r['resid']):+.2f} to {max(r['resid']):+.2f} GiB. "
                + ("" if r["separated"] else "The two ceilings move together in these runs, so the decode "
                   "slope is not identified and is 0; the mixed slope carries both families. ") + note_for(r))
        fr = [("tokens", float(r["cmix"][0]), float(r["cmix"][1])), ("requests", float(r["seqs"][0]), float(r["seqs"][1]))]
        if r["seqs"][0] == r["seqs"][1]:
            fr = fr[:1]
        sc = scope(r)
        out.append(entry(f"cudagraph_capture_floor_bytes_{MODE}", int(round(r["floor"])), "bytes_per_rank", sc,
                         tuple(int(round(x)) for x in widen(*r["ci"][0], r["floor"])), fr,
                         f"The floor of the FULL_AND_PIECEWISE capture estimate vLLM subtracts, "
                         f"{r['floor'] / GIB:.3f} GiB per rank. {tail}"))
        out.append(entry(f"cudagraph_capture_bytes_per_mixed_token_{MODE}", int(round(r["mixed"])), "bytes_per_token", sc,
                         tuple(int(round(x)) for x in r["ci"][1]), fr,
                         f"Capture-estimate bytes per rank per token of the capture ceiling, "
                         f"{r['mixed'] / 2**20:.3f} MiB. {tail}"))
        if r["separated"]:
            out.append(entry(f"cudagraph_capture_bytes_per_decode_token_{MODE}", int(round(r["decode"])),
                             "bytes_per_token", sc, tuple(int(round(x)) for x in r["ci"][2]), fr,
                             f"Capture-estimate bytes per rank per token of the uniform-decode FULL ceiling, "
                             f"{r['decode'] / 2**20:.3f} MiB. {tail}"))
        else:
            # The kernel reads the form all or none, so the decode slope is stated, as a term these
            # runs cannot price apart from the mixed one: not_charged, not a measured zero.
            out.append([f"  - cudagraph_capture_bytes_per_decode_token_{MODE}:", "      value: 0",
                        "      units: bytes_per_token", "      method: not_charged", "      fitted: false",
                        f"      scope: {sc}", "      sources:",
                        f'        - {{kind: discussion, cite: "InferenceX server logs of fit-role units, derived '
                        f'figures vendored in validator/testdata/anchors/{ANCHOR.name}", role: primary}}',
                        "      rationale: >",
                        f"        Not charged apart: in these runs the uniform-decode FULL ceiling moves with the "
                        f"capture ceiling, so the decode family's bytes are not separable from the mixed slope, "
                        f"which carries both. {tail}"])
    for r in res["activation"]:
        out.append(entry(f"activation_buffer_count_moe_{r['rank']}rank", round(r["value"], 3), "dimensionless", scope(r),
                         (round(r["ci"][0], 3), round(r["ci"][1], 3)),
                         [("tokens", float(r["mnbt"][0]), float(r["mnbt"][1]))],
                         f"The activation multiple of {r['model']} on {r['hw']} at {r['rank']}-rank tensor "
                         f"parallelism: the transient (logged peak activation minus the logged CUDA-graph "
                         f"estimate) over max_num_batched_tokens x {r['h']} x 2, median of {r['n']} fit units. "
                         f"{note_for(r)}"))
    for r in res["nontorch"]:
        out.append(entry(f"nontorch_in_budget_bytes_{r['rank']}rank", int(round(r["value"])), "bytes_per_rank", scope(r),
                         (int(round(r["ci"][0])), int(round(r["ci"][1])) + 1),
                         [("tokens", float(r["mnbt"][0]), float(r["mnbt"][1]))],
                         f"vLLM's non-torch memory growth inside the KV budget on {r['model']} on {r['hw']} at "
                         f"{r['rank']}-rank tensor parallelism, {r['value'] / GIB:.3f} GiB per rank: consumed "
                         f"memory minus Model loading took, median of {r['n']} fit units. It is part of "
                         f"non_kv_cache_memory (gpu_worker.py:720-756) and is not the communicator or workspace "
                         f"reservation the kernel holds before the snapshot. {note_for(r)}"))
    return out


def render(text: str, new: list[list[str]]) -> str:
    blocks = cb.split(text)
    families = ("activation_buffer_count_", "cudagraph_capture_", "nontorch_in_budget_bytes_")
    for e in new:
        cb.upsert(blocks, e, after=lambda b: (cb.name_of(b) or "").startswith(families))
    return cb.join(blocks)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv[1:])
    res = fit(json.loads(ANCHOR.read_text()))
    for k in ("capture", "activation", "nontorch"):
        for r in res[k]:
            print(k, {x: (round(v, 4) if isinstance(v, float) else v) for x, v in r.items() if x not in ("resid", "runs")})
    for line in res["named"]:
        print("not fitted:", line)
    t0 = MEMORY.read_text(encoding="utf-8")
    t1 = render(t0, entries(res))
    if args.check:
        if t0 != t1:
            print(f"would change: {MEMORY.name}", file=sys.stderr)
            return 1
        print("the memory set already matches")
        return 0
    MEMORY.write_text(t1, encoding="utf-8")
    print("the memory set written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
