# The Hopper dense MXFP8 ramp and its landing criteria

<p class="lede">Layer 17 prices the dense MXFP8 matrix multiply on h200 with a pair taken from the same chip's
measured grouped kernel, labelled `transferred`. The h100 pair is recorded as a candidate and is not selected. A change that moves a published number needs written criteria and a
held-out verdict (X13). The criteria below were fixed before any held-out evaluation, and the maintainer authorised
one run of them on 2026-11-12, after the final pins.</p>

## The criteria, fixed before any held-out evaluation

What the change does: on h200 it replaces the dense MXFP8 GEMM's compute ramp (asymptote and knee) with that
part's own measured Marlin grouped pair (0.245 and 26 tokens), lowering the GEMM's
efficiency by about 3.5 times at prefill sizes and moving the knee down. It therefore (1) lengthens prefill steps of
models that read the dense MXFP8 ramp on h200, by the whole-step factors the price test states (1.45 to 3.00 times at
512 to 8192 tokens), (2) barely moves decode (under a tenth, because the FP8 weight read dominates), and (3) moves
nothing else. The claim being tested is that the bound underpriced Hopper prefill and the transferred pair corrects it
without overshooting. The criteria follow from that claim, not from the development residuals (which had been seen when
this was written).

Unit: the held-out Hopper points that read the ramp, scored in one harness run with two arms from the same sources
(the registry as committed, and the same registry with the two h200 transferred entries removed, so the bound prices).

- (a) Prefill moves as priced. Every held-out Hopper point that reads the ramp has a simulated TTFT with the pair at
  least as high as with the bound, and at ISL 8192 at least 1.5 times as high. (Mechanical; no measured value is read.)
- (b) The error shrinks and does not flip. Over the held-out Hopper points that read the ramp, TTFT MAPE with the pair
  is at least 5 points lower than with the bound, and the median signed TTFT error with the pair is closer to zero than
  with the bound and not above +10 percent. A median above +10 percent means the pair overshoots, which the bound's
  direction ("optimistic") does not allow.
- (c) Decode is untouched. TPOT MAPE over the same points does not worsen by more than 1 point.
- (d) Nothing else moves. Every held-out point that is Blackwell, or Hopper and does not read the ramp, has identical
  predictions in the two arms. (Mechanical.)
- (e) There is enough to judge. At least 10 held-out points are scored per part; with fewer, there is no verdict for
  that part.

Each part is judged on its own held-out points. A part with none (see the unit note) is not validated: its entries stay
an exploratory candidate, outside the registry the lookup reads
(`registry/candidates/dense-marlin-mxfp8-transferred.yaml`, written by `emit_marlin_dense_transfer.py`), and the bound
keeps pricing it in released output. A test prices h100 and requires the assumed bound. If (a) to (e) hold for a part, its pair
lands. If any fails, or (e) leaves no verdict, the pair for that part is not selected for published pricing: it stays in
the registry as an exploratory candidate and the lookup does not let its rank displace the bound. Report aggregates
only, one row in the dashboard's evaluation log.

## Held-out unit and how the run is made

- The only withheld series that reads the ramp is the h200 MiniMax-M3 FP8 vLLM MTP series (44 rows). h100 has no
  withheld series, so its pair has no held-out unit and stays an exploratory candidate, not in the registry the lookup reads; the bound
  keeps pricing it. Selecting it needs a held-out verdict for h100 first.
- The series is speculative, so it also exercises drafter pricing. The two arms share that, so the difference is the
  ramp's.
- The dashboard's gate job scores both arms in one run, once, and reports aggregates only. Nobody reads or prints a
  withheld point. The result goes in the evaluation log and on pull request #67 before the pull request merges.
