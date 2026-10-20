"""Run a list of BLIS commands ONE AT A TIME inside one heavy slot (heavy/run.sh wraps this whole script).

Input: a JSON file [{"cmd": [...], "cwd": dir, "timeout": s, "stderr": path}]; output: <input>.done.json with
each item's returncode and wall time. Sequential by construction (one simulation at a time in this slot).
"""

import json
import os
import subprocess
import sys
import time

items = json.load(open(sys.argv[1]))
out = []
for it in items:
    t0 = time.monotonic()
    try:
        p = subprocess.run(it["cmd"], cwd=it["cwd"], capture_output=True, text=True, timeout=it["timeout"])
        rc, err = p.returncode, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, err = 124, f"timeout after {it['timeout']} s\n{(e.stderr or b'')[-4000:]!r}"
    open(it["stderr"], "w").write(err[-20000:])
    w = time.monotonic() - t0
    if rc == 0 and os.path.exists(os.path.join(it["pdir"], "metrics.json")):
        open(os.path.join(it["pdir"], "command.txt"), "w").write(it["stamp"])
        open(os.path.join(it["pdir"], "wall_s"), "w").write(str(w))
    out.append({"rc": rc, "wall_s": w})
json.dump(out, open(sys.argv[1] + ".done.json", "w"))
