#!/usr/bin/env python3
"""Mutation probes for the scoring harness in cmd/score.

The harness decides which modelling changes ship. A roofline fit that beat every alternative
on the component sweeps was rejected because the harness said it made end-to-end error worse —
so a harness that silently admitted bad points, or derived a batch or context wrongly, would
have let it through and every later decision would rest on it.

Each probe breaks one of the harness's judgements and checks that a test fails. Run from the
repository root:

    python scripts/probe_score_harness.py

Two things this script reports that a plain pass/fail would hide:

  BUILD BROKEN means the mutation did not compile, so the probe tested nothing. A probe that
  reports a survivor because the build failed is worse than no probe, and that exact mistake
  happened twice in this project — once with a sed that silently edited nothing, once with a
  Go mutation that left a variable unused.

  A SURVIVED verdict is not automatically a test gap. Disabling one of two equivalent branches
  — the `running` and `effconc` batch sources, which agree on the corpus — changes no output,
  so that probe is benign and was replaced with one that disables both.
"""

import subprocess, shutil, sys
SRC='cmd/score/main.go'; BK='/tmp/sc.bk'
PROBES=[
 ("preemption check removed","case p.Preempted != nil && *p.Preempted > 0:","case false:"),
 ("queue check removed","case p.Waiting != nil && *p.Waiting > 1:","case false:"),
 ("TTFT check removed","case !r.batchStated && p.TTFTms != nil && *p.TTFTms > 200:","case false:"),
 ("both batch sources ignored","case p.Running != nil && *p.Running > 0:\n\t\tr.batch = int(math.Round(*p.Running / float64(r.replicas)))\n\t\tr.batchStated = true\n\tcase p.EffConc != nil && *p.EffConc > 0:","case false:\n\t\tr.batch = int(math.Round(*p.Running / float64(r.replicas)))\n\t\tr.batchStated = true\n\tcase false:"),
 ("batchStated never set","r.batchStated = true\n\tcase p.EffConc","r.batchStated = false\n\tcase p.EffConc"),
 ("replicas ignored in batch","r.batch = int(math.Round(*p.Running / float64(r.replicas)))","r.batch = int(math.Round(*p.Running))"),
 ("speculation divisor dropped","return step / perStep","return step"),
 ("host per-token cost dropped","step := (e.Overlap + k.OutputTokenOverhead()).Seconds() * 1e3","step := e.Overlap.Seconds() * 1e3"),
 ("context ignores output length","r.context = int(*p.ISL + out/2)","r.context = int(*p.ISL)"),
 ("derived context ignores ratio","inLen := outLen * (*p.InTPS / *p.OutTPS)","inLen := outLen"),
 ("no-context check removed","case r.context <= 0:","case false:"),
]
for name, before, after in PROBES:
    s=open(SRC).read()
    if before not in s:
        print(f"{name:40s} ANCHOR MISSING"); continue
    open(SRC,'w').write(s.replace(before, after, 1))
    r=subprocess.run(['go','test','-count=1','./cmd/score/'],capture_output=True,text=True)
    out=r.stdout+r.stderr
    if 'build failed' in out or 'cannot use' in out or 'undefined' in out or 'declared and not used' in out:
        verdict='BUILD BROKEN (invalid probe)'
    else:
        n=out.count('--- FAIL')
        verdict=f'CAUGHT ({n})' if n else '*** SURVIVED ***'
    print(f"{name:40s} {verdict}")
    shutil.copy(BK, SRC)
