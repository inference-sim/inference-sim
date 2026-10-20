"""Regenerate numpy_rng_golden.json (run with numpy, e.g.
uv run --no-project --with numpy python -I gen_numpy_rng_golden.py > numpy_rng_golden.json).

Each case is a default_rng(seed) stream exactly as aiperf's TrajectorySource draws
t* (timing/trajectory_source.py:172-180, 602-637): seed = int.from_bytes(
sha256(f"{base}:{trace_id}:{lane}").digest()[:8], "big"), then rng.uniform(lo, hi).
"""
import hashlib, json, sys
import numpy as np

def lane_seed(base, trace_id, lane):
    return int.from_bytes(hashlib.sha256(f"{base}:{trace_id}:{lane}".encode()).digest()[:8], "big")

cases = []
seeds = [0, 1, 42, 2**32 - 1, 2**32, 2**63 + 12345, 2**64 - 1]
for tid, lane in [("002001296e8a8c38ad9d7cc436d691afc602", 0), ("abc", 7), ("trace-x", 511)]:
    seeds.append(("lane", 42, tid, lane))
for s in seeds:
    label = None
    if isinstance(s, tuple):
        _, base, tid, lane = s
        label = {"base": base, "trace_id": tid, "lane": lane}
        s = lane_seed(base, tid, lane)
    raw = np.random.default_rng(s).bit_generator.random_raw(4)
    rnd = np.random.default_rng(s).random(3)
    lo, hi = 1234.5, 98765.25
    uni = float(np.random.default_rng(s).uniform(lo, hi))
    cases.append({"seed": str(s), "lane": label, "raw": [str(int(x)) for x in raw],
                  "random": [float(x).hex() for x in rnd], "uniform_lo": lo, "uniform_hi": hi,
                  "uniform": uni.hex()})
json.dump({"numpy_version": np.__version__, "cases": cases}, sys.stdout, indent=1)
print()
