"""Adds the partly answered case to the combined parity fixture, combined/parity.json.

Run from the root of the training pipeline's repository:

  third_party/.venv-l2/bin/python -I <this file> --scripts experiments/scripts --artifact <shipped artifact dir> --fixture <combined/parity.json>

The expected values come from the pipeline's own code (export_stage_artifact.combined, which
applies the simulator model to a pair both of whose points the simulator answered, and the
fallback with the simulator columns missing to any other pair, each pair normalised), run with the
shipped models. The case is the fixture's first case with the simulator's time-to-first-token
removed from two points it answered (X and I given, T null), because the corpus holds no such
point. The first case is regenerated and must equal the one in the file; the new case is
appended, or replaced when a case of that name is there.
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--scripts", required=True)
ap.add_argument("--artifact", required=True)
ap.add_argument("--fixture", required=True)
a = ap.parse_args()
os.environ["RCS_SIM_V19"] = "all"; os.environ["RCS_V19_FILE"] = "blis-v019-blis2-decision.csv"
sys.path.insert(0, a.scripts)
import numpy as np
import xgboost as xgb
import export_stage_artifact as E
import fin_run as FR
import l2_lib as L

fx = json.load(open(a.fixture))
first = fx["cases"][0]["case"]
U = FR.Final(FR.sim_file("clean", "lomco"), "l2nsall", "lomco"); P = U.P
models, fallbacks = {}, {}
for f in L.FRONTS:
    for d, fn in ((models, "model-"), (fallbacks, "fallback-")):
        c = xgb.XGBClassifier(); c.load_model(os.path.join(a.artifact, fn + f + ".json")); d[f] = c

idx = np.where(P.case == first)[0]
def record(name):
    rec = dict(case=name, points=[E.point_json(P, k) for k in idx], fronts={})
    for f in L.FRONTS:
        pi, ans = E.combined(models, fallbacks, P, idx, f); np.fill_diagonal(pi, 0.0)
        q = 1.0 - pi; np.fill_diagonal(q, 1.0)
        rec["fronts"][f] = dict(pi=pi.tolist(), answered=[bool(x) for x in ans], on_front=[bool(x) for x in L.parameter_free(pi)], probability=[float(x) for x in q.prod(0)])
    return rec

# the regenerated first case must equal the stored one
again = record(first)
for f in L.FRONTS:
    assert np.allclose(np.array(again["fronts"][f]["pi"]), np.array(fx["cases"][0]["fronts"][f]["pi"]), atol=1e-9), f
assert again["points"] == fx["cases"][0]["points"]

full = [k for k in idx if np.isfinite(P.S[k]).all()]
assert len(full) >= 4
for k in (full[1], full[4]):                     # X and I stay, T goes
    P.S[k, 2] = np.nan
name = first + "|partial"
rec = record(name)
fx["cases"] = [c for c in fx["cases"] if c["case"] != name] + [rec]
json.dump(fx, open(a.fixture, "w"), separators=(",", ":"))
print("case", name, "points", len(idx), "T removed from", [str(P.ids[k]) for k in (full[1], full[4])])
