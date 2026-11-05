#!/usr/bin/env python3
"""PROBE, not a fit that ships: why the MoE grouped GEMM is not priced as a roofline.

This script is kept because its negative result is worth more than its numbers. It fits

    latency = floor + max(routed_flops / (peak * eff), local_weight_bytes / (bw * fraction))

to AISimulate's MoE sweeps and reaches a geometric error of 1.59-1.98x against 13.4-34.2x for
the FLOPs-only form. That looks like a clear win and it is not one: wired into the kernel it
takes the published-report corpus MAPE from 47% to 105%, and Nemotron-3-Ultra from 87% to
314%.

WHY. The roofline reads every expert the rank HOLDS. At Granite-5's shape that is 302 MB, so
the read term binds at every token count and the predicted latency is flat at 112 us where the
measurement rises from 36.9 to 190.2. It cannot be right: 302 MB in the measured 36.9 us would
be 8.2 TB/s on a 4.8 TB/s part.

WHAT IS ACTUALLY HAPPENING. The read scales with the experts the step TOUCHES, which is the
coverage term the kernel already carries — expected touched count
`local * (1 - ((E-k)/E)^tokens)`. Against Granite's measured curve that term's read time is a
steady 0.34 to 0.50 of measured across the whole 1-to-512 token range, and it reproduces the
sharp rise from 1 to 32 tokens that the all-experts reading flattens away.

So the sweeps and the corpus do not disagree about the mechanism. The roofline fitted the
sweeps better while modelling the wrong thing, and the corpus is what caught it. Two lessons
worth keeping: a better fit to a component benchmark can be a worse model, and a per-primitive
fit needs an end-to-end check before it ships.

Usage (to reproduce the rejected fit):
    python scripts/probe_moe_roofline.py
"""

import pyarrow.parquet as pq, math, yaml
import os
from pathlib import Path
B='/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data/'
ENV={'h200':(0.788,94),'h100':(0.784,144),'l40s':(0.586,135),'gb200-nvl72':(0.688,172)}
for sku,chip in [('h200_sxm','h200'),('h100_sxm','h100'),('l40s','l40s'),('gb200','gb200-nvl72')]:
    try: d=pq.read_table(f'{B}{sku}/moe/trtllm/1.3.0rc20/moe_perf.parquet').to_pydict()
    except Exception: print(chip,'no data'); continue
    hw=yaml.safe_load(open(Path(os.environ.get('BLIS_CATALOG') or Path(__file__).resolve().parent.parent / 'validator' / 'testdata' / 'catalog') / 'hardware' / f'{chip}.yaml'))
    peak=hw['TFlopsFP8']*1e12; bw=hw['BwPeakTBs']*1e12
    eps,mh=ENV[chip]; eff=lambda M: eps*M/(M+mh)
    pts=[]
    for i in range(len(d['latency'])):
        if d['distribution'][i]!='balanced' or d['moe_dtype'][i]!='fp8': continue
        lat=d['latency'][i]*1e-3
        if lat<=0: continue
        t,k,E=d['num_tokens'][i],d['topk'][i],d['num_experts'][i]
        H,I=d['hidden_size'][i],d['inter_size'][i]
        tp,ep=max(d['moe_tp_size'][i],1),max(d['moe_ep_size'][i],1)
        local=max(E//ep,1)
        flops=2*(t*k*local/E)*3*H*I/tp
        wbytes=local*3*H*I/tp
        pts.append((flops/(peak*eff(t)), wbytes, lat))
    best=None
    for frac in [x*0.02 for x in range(2,101)]:
        for fl in (0.0,10.0,20.0,30.0,50.0,75.0,100.0,150.0,200.0,300.0):
            e=math.exp(sum(abs(math.log((fl*1e-6+max(c, w/(bw*frac)))/lat)) for c,w,lat in pts)/len(pts))
            if best is None or e<best[0]: best=(e,frac,fl)
    e,frac,fl=best
    comp=math.exp(sum(abs(math.log(max(c,1e-12)/lat)) for c,_,lat in pts)/len(pts))
    print(f'{chip:14s} n={len(pts):6d} bw_frac={frac:.2f} floor={fl:5.1f}us geo-err {e:.3f}x  (compute-only {comp:.1f}x)')
