"""Draw the FIT / HELD-OUT partition of InferenceX end-to-end NVIDIA points (pre-registered).

Unit: a GROUP = one deployment configuration x one concurrency. Every point (run x concurrency)
of a group goes to one set, so repeated runs (seeds, re-runs, retries) of one configuration at one
level never straddle the split. Group key = model, GPU, engine, image, precision, layout (TP/EP/DP,
P/D and worker counts), spec method and its config, offload mode/backend/connector, frontend,
scenario, ISL/OSL, concurrency. Run ids and dates are NOT in the key.

Algorithm (seed S, target FIT share P = 0.40), "hash + stratification repair":
 1. u(g) = int(sha256(f"{S}|{group_key}"), 16) / 2**256; FIT iff u < P. Independent per group,
    so the draw does not depend on which other points exist.
 2. Repair, in factor order F (model, gpu, engine_image, layout, pd, spec, offload, frontend,
    scenario, conc_bin): for each level with >= 2 groups that has no FIT group, move the HELD
    group with the smallest u to FIT; with no HELD group, move the FIT group with the largest u
    to HELD. Passes repeat until no change (at most 20). Forced-HELD groups never move.
 3. Forced HELD (named subsets): every group containing a DSV4.1 Hopper anchor point
    (validation/runs*/ix_dsv41: config ids 2560/2570/2640, all dates and levels).
 4. Singleton levels (1 group) stay where step 1 put them; they are listed, and pushed to HELD
    as the brief requires.
EXTENSION RULE (no re-draw): assignments in this file are frozen. A point added later joins
its group if the group exists; a new group gets step 1, then step 2 restricted to new groups
(only a new group may be flipped). The seed, P and F never change. Any change to them is a new
pre-registered partition with a stated reason.
Only InferenceX end-to-end serving results on NVIDIA GPUs are partitioned; AMD rows are recorded
in the inventory and excluded. Kernel/operator, FPM, overhead, network and acceptance data are
fit-only and outside this split.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math

SEED = 20261016
P = 0.40
FACTORS = ["model", "gpu", "engine_image", "layout", "pd", "spec", "offload", "frontend", "scenario", "conc_bin"]
ANCHOR_CFG = {2560, 2570, 2640}


def conc_bin(c: int) -> str:
    return "c1" if c <= 1 else f"c{2 ** int(math.floor(math.log2(c)))}-{2 ** (int(math.floor(math.log2(c))) + 1) - 1}"


def factors(r: dict, c: int) -> dict:
    p = r["parallel"]
    pf, dc = p.get("prefill") or {}, p.get("decode") or {}
    if p.get("disagg"):
        lay = (f"P(tp{pf.get('tp')},ep{pf.get('ep')},dpa{int(bool(pf.get('dp_attention')))},w{pf.get('workers')})"
               f"D(tp{dc.get('tp')},ep{dc.get('ep')},dpa{int(bool(dc.get('dp_attention')))},w{dc.get('workers')})")
    else:
        tp = p.get("tp", dc.get("tp"))
        ep = p.get("ep", dc.get("ep"))
        dpa = p.get("dp_attention", dc.get("dp_attention"))
        lay = f"tp{tp},ep{ep},dpa{int(bool(dpa))}"
    spec = r.get("spec_method") or "none"
    sc = r.get("spec_config") or ""
    k = None
    for tok in sc.replace("'", '"').split(","):
        if "num_speculative_tokens" in tok:
            k = tok.split(":")[-1].strip(" }")
    off = "none" if r.get("offload") != "on" else (r.get("kv_connector") or r.get("offload_backend") or "on")
    return {"model": r["model_slug"], "gpu": r["gpu"], "engine_image": f"{r['engine']}|{r.get('engine_image')}",
            "layout": lay, "pd": "disagg" if p.get("disagg") else "colocated",
            "spec": f"{spec}{'/K' + k if k else ''}", "offload": off, "frontend": r.get("frontend") or "unknown",
            "scenario": f"{r['scenario']}|{r.get('isl')}x{r.get('osl')}", "conc_bin": conc_bin(c),
            "precision": r.get("precision")}


TERMS = [  # FIT-point targeting (priority only; never changes assignment)
    ("cpu_offload", lambda f, r, c: f["offload"] != "none"),
    ("kv_transfer_pd", lambda f, r, c: f["pd"] == "disagg"),
    ("collectives_at_scale", lambda f, r, c: ("ep1," not in f["layout"] and "ep1)" not in f["layout"]) or r.get("is_multinode")),
    ("admission_context_proportional", lambda f, r, c: r["scenario"] == "agentic_traces" and c <= 4),
    ("host_scheduler_step", lambda f, r, c: c <= 4 and r["scenario"] == "single_turn"),
    ("mixed_step", lambda f, r, c: c >= 32),
    ("eager_launch_per_gpu", lambda f, r, c: c >= 64),
    ("spec_decode", lambda f, r, c: not f["spec"].startswith("none")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze", help="a previous partition.json: its point assignments are kept (EXTENSION RULE)")
    a = ap.parse_args()
    frozen = json.load(open(a.freeze))["assignment"] if a.freeze else {}
    groups = {}
    for ln in open(a.inventory):
        r = json.loads(ln)
        if r["blocker"] in ("AMD", "gated"):
            continue  # AMD is out of scope; gated data enters (by the extension rule) only once fetched
        for pt in r["points"]:
            c = pt["conc"]
            f = factors(r, c)
            # Group key from source-native fields only (db/artifact columns, never log-parsed ones),
            # so a later, better log parse cannot move a point between groups.
            p = r["parallel"]
            gk = "|".join(str(x) for x in (
                r["model_slug"], r["gpu"], r["engine"], r.get("engine_image"), r.get("precision"),
                json.dumps(p, sort_keys=True), r.get("spec_method"), r.get("offload"), r.get("offload_backend"),
                r["scenario"], r.get("isl"), r.get("osl"), r.get("recipe_fingerprint"), f"c={c}"))
            g = groups.setdefault(gk, {"group_key": gk, "factors": f, "points": [], "blockers": set(),
                                       "anchor": False, "terms": set()})
            g["points"].append(pt["point_id"])
            g["blockers"].add(r["blocker"])
            if r.get("config_id") in ANCHOR_CFG and r["scenario"] == "agentic_traces":
                g["anchor"] = True
            for t, fn in TERMS:
                if fn(f, r, c):
                    g["terms"].add(t)
    for g in groups.values():
        g["u"] = int(hashlib.sha256(f"{SEED}|{g['group_key']}".encode()).hexdigest(), 16) / 2 ** 256
        g["set"] = "HELD" if g["anchor"] else ("FIT" if g["u"] < P else "HELD")
        prev = {frozen[p] for p in g["points"] if p in frozen}
        g["frozen"] = bool(prev)
        if prev:  # a group with any frozen point keeps that set; its new points join it
            g["set"] = prev.pop() if len(prev) == 1 else "HELD"
    log = []
    for _ in range(20):
        changed = False
        for fac in FACTORS:
            lv = collections.defaultdict(list)
            for g in groups.values():
                lv[g["factors"][fac]].append(g)
            for level, gs in lv.items():
                if len(gs) < 2:
                    continue
                fit = [g for g in gs if g["set"] == "FIT"]
                held = [g for g in gs if g["set"] == "HELD"]
                if not fit:
                    cand = sorted((g for g in held if not g["anchor"] and not g["frozen"]), key=lambda g: g["u"])
                    if cand:
                        cand[0]["set"] = "FIT"; changed = True
                        log.append({"factor": fac, "level": level, "moved": cand[0]["group_key"], "to": "FIT"})
                elif not held:
                    cand = sorted((g for g in fit if not g["frozen"]), key=lambda g: -g["u"])
                    if not cand:
                        continue
                    cand[0]["set"] = "HELD"; changed = True
                    log.append({"factor": fac, "level": level, "moved": cand[0]["group_key"], "to": "HELD"})
        if not changed:
            break
    for fac in FACTORS:  # step 4: a group alone in some factor level is HELD
        lv = collections.defaultdict(list)
        for g in groups.values():
            lv[g["factors"][fac]].append(g)
        for level, gs in lv.items():
            if len(gs) == 1 and gs[0]["set"] == "FIT" and not gs[0]["frozen"]:
                gs[0]["set"] = "HELD"
                log.append({"factor": fac, "level": level, "moved": gs[0]["group_key"], "to": "HELD", "why": "singleton"})
    cov, singletons = {}, []
    for fac in FACTORS:
        lv = collections.defaultdict(lambda: {"FIT": 0, "HELD": 0, "groups": 0})
        for g in groups.values():
            x = lv[g["factors"][fac]]
            x[g["set"]] += len(g["points"])
            x["groups"] += 1
        for level, x in lv.items():
            if x["groups"] < 2:
                singletons.append({"factor": fac, "level": level})
        both = sum(1 for x in lv.values() if x["groups"] >= 2 and x["FIT"] and x["HELD"])
        multi = sum(1 for x in lv.values() if x["groups"] >= 2)
        cov[fac] = {"levels": len(lv), "levels_with_2plus_groups": multi, "levels_in_both_sets": both,
                    "per_level": dict(sorted(lv.items()))}
    pts = collections.Counter()
    pts_ok = collections.Counter()
    for g in groups.values():
        pts[g["set"]] += len(g["points"])
        if g["blockers"] == {"ok"}:
            pts_ok[g["set"]] += len(g["points"])
    assign = {}
    for g in groups.values():
        for pid in g["points"]:
            assign[pid] = g["set"]
    doc = {
        "algorithm": __doc__.strip(), "seed": SEED, "frozen_from": a.freeze and {"path": a.freeze,
            "sha256": hashlib.sha256(open(a.freeze, "rb").read()).hexdigest(), "points": len(frozen)}, "fit_share_target": P, "factor_order": FACTORS,
        "inventory_sha256": hashlib.sha256(open(a.inventory, "rb").read()).hexdigest(),
        "summary": {"groups": len(groups), "points": dict(pts), "points_scoreable_today": dict(pts_ok),
                    "fit_share_points": pts["FIT"] / max(1, sum(pts.values())), "repairs": len(log)},
        "named_held_out_subsets": {
            "dsv41_hopper_anchors": {"rule": "InferenceX config ids 2560 (H200 TP8), 2570 (H100 TP8), 2640 (H200 TP4), agentic_traces, every date and level",
                                      "groups": sorted(g["group_key"] for g in groups.values() if g["anchor"])},
            "h200_calibration_runs_anchor_C": {"rule": "the internal deployment H200 calibration runs (anchor C; local only); not InferenceX points, held out in full, never fitted",
                                               "path": "local only (anchor C)"}},
        "per_factor_coverage": cov, "singleton_levels": singletons, "repair_log": log,
        "groups": [{"group_key": g["group_key"], "set": g["set"], "u": g["u"], "points": g["points"],
                    "blockers": sorted(g["blockers"]), "fit_targets": sorted(g["terms"]) if g["set"] == "FIT" else []}
                   for g in sorted(groups.values(), key=lambda g: g["group_key"])],
        "assignment": assign,
    }
    json.dump(doc, open(a.out, "w"), indent=1, sort_keys=False)
    print(json.dumps(doc["summary"]))
    for fac in FACTORS:
        c = cov[fac]
        print(fac, c["levels"], c["levels_with_2plus_groups"], c["levels_in_both_sets"])


if __name__ == "__main__":
    main()
