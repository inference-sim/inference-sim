"""PRE-REGISTERED (2026-10-17 ~06:00, before any r7 HELD point is scored) r6-vs-r7 HELD subsample.

Population: HELD points of work/plan.jsonl that r6 scored (status "scored" in results/heldout/points.jsonl),
excluding the three DSV4.1 anchor runs (cov.ANCHOR_RUNS), which are re-scored IN FULL.
Rule: stratify by the plan's own stratum (model, gpu, layout, pd, offload, spec, frontend, scenario, conc bin);
in each stratum sort by point_id, shuffle with random.Random(f"{SEED}|{stratum}"), take ceil(0.25 n).
SEED = 7017. Output: r7cmp/subsample.json (point ids) + sha256. Both builds are scored on exactly these points.
"""
import hashlib, json, math, random, collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402
C = PX.ROOT
SEED = 7017
ANCHORS = {"ixdb-c114784d44a688eb", "ixdb-88c64d6bf2eb9b77", "ixdb-d35720993a0fb0d8"}
scored = {json.loads(l)["point_id"] for l in (C / "results/heldout/points.jsonl").open()}
plan = [json.loads(l) for l in (C / "work/plan.jsonl").open()]
held = [x for x in plan if x["set"] == "HELD" and x["point_id"] in scored]
anchors = sorted(x["point_id"] for x in held if x["run_id"] in ANCHORS)
st = collections.defaultdict(list)
for x in held:
    if x["run_id"] not in ANCHORS:
        st[x["stratum"]].append(x["point_id"])
pick = []
for s in sorted(st):
    ids = sorted(st[s]); random.Random(f"{SEED}|{s}").shuffle(ids)
    pick += ids[:math.ceil(0.25 * len(ids))]
doc = {"seed": SEED, "fraction": 0.25, "population": len(held) - len(anchors), "strata": len(st),
       "anchors": anchors, "sample": sorted(pick)}
out = C / "r7cmp" / "subsample.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(doc, indent=1) + "\n")
print(len(anchors), "anchor points;", len(pick), "of", doc["population"], "in", len(st), "strata; sha256",
      hashlib.sha256(out.read_bytes()).hexdigest())
