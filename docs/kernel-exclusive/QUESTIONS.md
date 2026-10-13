# Ten questions that could invalidate this experiment

Asked before writing the adapter, answered by investigation rather than assumption.
Each records what was checked and what the answer implies.

1. Does BLIS's scheduler actually produce a resident batch below the client concurrency?
   If it does not, the thesis (scheduling explains the gap) is untestable here.
2. Is BLIS's concurrency knob a closed-loop client level, or an arrival rate?
3. What TPOT does BLIS report, and is it `client_observed` end-to-end as the snapshot is?
4. Does BLIS model the engines the measurements came from (sglang, trtllm, vllm)?
5. Can BLIS represent the 39 deployments' parallelism (tp/dp/moe_ep) faithfully?
6. Does BLIS's own step loop already add costs the kernel also charges (double counting)?
7. Is `StepTime`'s microsecond truncation large enough to bias a 447-point MAPE?
8. Does BLIS need memory/KV-capacity answers the kernel must supply consistently?
9. Are the 39 scenarios' models all present in BLIS's own catalog path?
10. Does the comparison stay honest — same anchor, same metric — end to end?

## Answers, round 1 (before writing the adapter)

**Q1. Does BLIS's scheduler produce a resident batch below the client concurrency?**
Yes, and this is what makes the experiment meaningful. `sim/batch_formation.go:250` gates
admission on three things at once: `MaxNumSeqs`, a token budget, and KV capacity, with
preemption when capacity runs out. So the resident batch is genuinely decoupled from the
client level rather than tracking it. The kernel's 13.67% rests on assuming they are equal;
BLIS can disagree with that assumption, which is the only reason wiring them together
tests anything.

**Q2. Closed loop or arrival rate?**
Closed loop is native. `sim/workload/spec.go:120` has a `concurrency` field, mutually
exclusive with rate (`Validate()` rejects both). `sim/workload/synthesis.go:7` documents it
as "number of concurrent sessions". This matches the snapshot, whose `concurrency` is a
client level, and means the harness does not have to emulate a closed loop by hand.

**Q3. Does BLIS report the same quantity as the snapshot's TPOT?**
Yes. `sim/simulator.go:1099`:
`req.ITL = append(req.ITL, currStepAdvance + NumNewTokens*OutputTokenProcessingTime())`.
That is per-decode-step inter-token latency including host detokenization — the
client-observed inter-token time, which is what TPOT means. Both kernel terms reach it:
`currStepAdvance` from `StepTime` and the host term from `OutputTokenOverhead`.
`sim/metrics.go:321` converts ticks to ms for reporting.

**Q4. Does BLIS model the engines the measurements came from?**
Partly, and this is the experiment's main external-validity limit. BLIS models vLLM
(README, and the config fields mirror vLLM flags). The corpus is:

| Framework | Sweeps | Points | Share |
|---|---|---|---|
| vllm | 46 | 238 | 53.2% |
| sglang | 27 | 155 | 34.7% |
| trt | 10 | 54 | 12.1% |

So 46.8% of points were measured on an engine BLIS does not model. The comparison must
report the vLLM subset separately from the whole, or it claims fidelity it does not have.
This is a property of the corpus, not a fixable defect.

**Q5. Can BLIS represent the 39 deployments' parallelism?**
Yes. `sim/config.go` carries `EnableExpertParallel` (mirroring vLLM
`--enable-expert-parallel`), TP, DP, and an expert-parallel group DP width, with
`EffectiveEPSize` as TP·DP. The scenario files state `tp_size`, `pp_size`,
`attention_dp_size`, `moe_ep_size`, `moe_tp_size`; every one has a BLIS counterpart except
`pp_size`, which is 1 in all 39 selected deployments, so nothing is lost.

**Q7. Is microsecond truncation a material bias?**
Measured, not assumed. The smallest step in the corpus's regime is a 1-request decode at
6849 us and an empty batch at 553 us; truncation is at most 1 us, so under 0.02% on the
smallest real step. It is far below the 9.41% target and it is a truncation of the same
quantity on both sides of a normalised ratio, so it largely cancels. Not a threat.

Still open, to be answered by investigation below: Q6 (double counting), Q8 (memory
consistency), Q9 (catalog coverage), Q10 (comparison honesty).

**Q6. Does BLIS add costs the kernel also charges?**
No, for the deployments in this corpus. `sim/simulator.go:1063-1071` is the whole of it:
`StepTime`, then `+= KVCache.ConsumePendingTransferLatency()`, then `max(1, ...)`. The
transfer term is zero for a single-tier cache, which all 39 scenarios are (no offload
tier). The floor only binds below 1 tick. So the step advance is the kernel's number and
nothing else. Per-token and per-request host costs are added at `simulator.go:1099` and in
`recordRequestCompletion` from the interface's own methods, which the adapter routes to the
kernel -- charged once each, in the right place.

**Q8. Does BLIS need memory answers, and can the kernel give them consistently?**
This is the one finding that changes the design. BLIS derives KV capacity through
`latency.CalculateKVBlocks` (`sim/latency/kv_capacity.go:400`), called from `cmd/root.go`
and `cmd/replay.go` -- entirely OUTSIDE the `sim.LatencyModel` seam. So registering the
adapter at that seam would leave step time priced by the kernel while KV capacity, and
therefore the resident batch, was still decided by the legacy path. Since the resident batch
is precisely the quantity this experiment is about, that would invalidate the result.

The kernel answers the same question directly and was built to:

| Kernel method | Answer on glm-5 / h200 / tp8 |
|---|---|
| `FixedBytes().Total()` | 678 GiB (node aggregate across 8 ranks) |
| `SequenceFixedBytes()` | 0 (pure-attention model, no recurrent state) |
| `SequenceVariableBytes(1024)` | 92,012,544 B = 87.75 MiB |
| `SequenceVariableBytes(8192)` | 736,100,352 B = 702 MiB |

So KV capacity must ALSO come from the kernel for the claim "BLIS relies exclusively on the
kernel" to be true. The harness derives `TotalKVBlocks` from the kernel's own memory
methods rather than from `CalculateKVBlocks`, and a behavioural test asserts the legacy
function is not consulted on a kernel run. The per-rank versus per-node convention of
`FixedBytes` must be pinned by a test before it is divided by anything.

**Q10. Does the comparison stay honest?**
The rule adopted: reproduce AISimulate's own published whole-snapshot shape error (10.05%)
from its own per-point data, as `blis-latency-kernel/cmd/shape` does, and refuse to report
any figure from a harness that cannot. This is the check that caught the metric-identity
error already made once in this project, where a shape score was compared against an
absolute MAPE and flattered the kernel by 2.4x. Both sides are normalised to their OWN
anchor at the sweep's lowest concurrency.

## Round-1 conclusion

Two things changed as a result of asking:

1. Q8 moved KV capacity into scope. Without it, "exclusive" would have been false in the
   one place that matters most for this experiment.
2. Q4 bounds the claim. 46.8% of points come from sglang or trtllm, which BLIS does not
   model. The vLLM subset -- 46 sweeps, 238 points -- is the defensible comparison, and
   the full 447 is reported alongside it as context rather than as the headline.

## Round 2: a kernel defect found by integrating

Deriving the KV budget from `FixedBytes()` (Q8's requirement) made five of the 39
deployments report that they cannot serve a single request:

```
glm-5-h200-fp8-sglang-tp8: a 141.0 GiB device at utilization 0.90 gives a 126.9 GiB
budget, and the kernel reports 678.7 GiB of fixed occupancy per rank.
```

The interface documents `FixedBytes` as "per rank" (`blis-schemas/kernel/kernel.go`), and
678.7 GiB per rank on a 141 GiB part is impossible. The breakdown shows it is almost all
weights: 677.5 of 678.7 GiB.

**The defect.** `blis-latency-kernel/new.go:493` computes MoE expert weights as

```go
weights += c * l.ExpertWeightBytesPerExpert * k.expertsPerRank
```

with no division by `expertTensorShards`. The STEP-TIME path, `kernel.go:331`, gets this
right:

```go
weightBytes += l.ExpertWeightBytesPerExpert / k.expertTensorShards * ...
```

`expertTensorShards` is 1 when expert parallelism is on (each rank holds whole experts) and
`tp` when it is off (experts sliced across the tensor-parallel group) -- `new.go:274-277`,
matching vLLM `fused_moe/config.py:1225`. So with EP off, `FixedBytes` overstates MoE
weights by the tensor-parallel width.

**Arithmetic, independent of the kernel.** GLM-5 has 75 MoE layers, 256 experts, n=2048,
k=6144, three matrices per expert, fp8 at 1 byte:

| Quantity | Value |
|---|---|
| MoE expert weights, whole model | 675.0 GiB |
| Divided by TP=8 (sliced experts) | 84.4 GiB |
| Kernel's reported weights at TP=8 | 677.5 GiB |
| H200 budget at utilization 0.90 | 126.9 GiB |

675.0 against a reported 677.5 (the remainder being dense layers, embeddings and the head)
confirms the expert term is undivided. The correct 84.4 GiB fits the budget and leaves
42.5 GiB for KV.

**Which deployments are affected.** All five with EP off and an MoE model where the
overstatement pushes fixed occupancy past the device: the three glm-5 TP-8 and TP-4
scenarios plus the two b300 ones. The other 34 either have small enough expert weights
(gpt-oss at 53.5 GiB) or enough device memory to stay positive -- so their KV budgets are
WRONG BUT NON-NEGATIVE, which is worse, because they would have produced a plausible number.

**Scope of the error.** It is confined to `FixedBytes`, so it does not affect the 13.67%
shape result already reported: `cmd/shape` scores step time, and the step-time path divides
correctly. It affects any memory or capacity answer, which is exactly what this experiment
needs.

**What this means for the goal.** The fix belongs in `blis-latency-kernel/new.go`, which is
outside this worktree. Per the instruction to keep all work inside the worktree and not to
modify other repos, the fix is NOT applied there. The options are recorded here for
decision rather than chosen unilaterally.

## Round 2, measured: the defect's blast radius

Every one of the 39 evaluation deployments was classified by deriving its KV budget:

| Class | Count | Meaning |
|---|---|---|
| Refused | 17 | Per-rank occupancy exceeds device memory; no budget derivable |
| Suspect | 20 | Expert parallelism off, so expert bytes undivided; budget positive but too small |
| Unaffected | 2 | Expert parallelism on, so `expertTensorShards` is legitimately 1 |

The two unaffected deployments are exactly the two scenarios in the corpus carrying
`enable_expert_parallel: true`:

```
minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml   epw=4  fixed=28.3 GiB
minimax-m2.5-b300-fp4-vllm-tp2-ep4-dp2.yaml   epw=4  fixed=28.3 GiB
```

That correspondence is exact, with no exceptions in either direction, which confirms the
diagnosis rather than merely being consistent with it: the error appears if and only if
expert parallelism is off, which is precisely when `expertTensorShards` should be `tp` and
the memory path uses 1.

**Consequence for the goal.** KV-driven admission cannot be exercised on 37 of 39
deployments without the upstream fix. The resident batch would be bounded by `max_num_seqs`
and the token budget alone. Since the thesis under test is that the resident batch explains
the 13.67%-vs-9.41% gap, running the comparison with KV pressure disabled on 95% of the
corpus would not test the thesis -- it would produce a number whose provenance nobody could
defend to a reviewer.

The goal's constraints say to keep all work in this worktree and not to modify any other
repo, and the fix belongs in `blis-latency-kernel/new.go`. A one-line change there, plus a
behavioural test, is the correct remedy. That decision is the user's, not this document's,
and it is recorded here rather than taken.

What is NOT blocked, and is being built regardless: the adapter, its tests, the closed-loop
harness, and the comparison on the 2 unaffected deployments plus every deployment under
`max_num_seqs`-only admission. That establishes the pipeline end to end and quantifies what
KV pressure would add, so applying the upstream fix later is a one-line change followed by a
re-run rather than new work.

## Round 3: ten questions on correctness and on reaching SOTA

Asked after the first scores, prompted by one flaw whose shape is worth naming: a
justification that was TRUE in one measurement regime was carried unexamined into another
where it is false. "max_num_seqs is a default, and a constant cancels in a ratio" holds for a
ratio of two StepTime calls. It fails the moment a scheduler is in the loop, because
max_num_seqs bounds the resident batch and the resident batch sets the curve's shape.

Every question below is therefore of the form "what else did I carry across a regime change,
and does it still hold?"

### Correctness

1. Which values does this comparison supply that the snapshot does not state, and how much
   does each move the score? (Answered by measurement, not judgement -- cmd/sensitivity.)
2. Is `cache_dtype: fp8` an assumption that changes KV bytes per token, and therefore the
   KV-bound resident batch, rather than a cosmetic label?
3. Does `engine_version: "0.29.0"` select engine rules that differ from what the measured
   runs used, and would a different pack change the collective backends or async scheduling?
4. Does the adapter's `CachedTokens: 0` still hold? It assumed BLIS subtracts prefix-cache
   hits before setting NumNewTokens. With prefix caching off this is vacuous; if a default
   turned it on, the kernel would be told a hit it has already been charged for.
5. Is `Overlap` the right band edge for a SIMULATED step, or does a real engine's step sit
   between Overlap and NoOverlap in a way that a step-time ratio hid?
6. Does the harness's zero think time match how the snapshot's client behaved, or does a
   saturating loop overstate the offered load?

### Reaching SOTA

7. Where is the remaining error concentrated after the dp fix -- is it a few arms, a
   concurrency band, or spread evenly? A spread residual is a modelling gap; a concentrated
   one is usually a specific wrong assumption.
8. Is the residual the same SIGN everywhere? Systematic over-prediction points at one term;
   mixed signs point at the scheduler or at the data.
9. Can any part of the gap be closed by something CHECKABLE against vLLM's source or
   NVIDIA's data, rather than by adjusting a number until the score improves?
10. What is the floor this comparison can reach at all, given that AISimulate replays the
    engine that produced the measurements and therefore knows the settings BLIS must guess?

## Round 3 answers

**Q1. How much does each unstated setting move the score?** Measured, one at a time, on the
213 monotone vLLM points:

| Setting | n | MAPE | delta |
|---|---|---|---|
| scenario values (baseline) | 213 | 11.20% | — |
| max_num_seqs x2 | 213 | 11.05% | -0.16 |
| max_num_seqs x4 | 213 | 11.03% | -0.17 |
| max_num_seqs /2 | 213 | 11.56% | +0.36 |
| token budget x2 | 213 | 11.24% | +0.03 |
| token budget /2 | 213 | 11.22% | +0.02 |
| admission: seqs only, no KV bound | 213 | 11.21% | +0.00 |

**This overturns the conclusion recorded in RESULTS.md.** I had written that part of the
residual is attributable to the unstated engine settings and that the comparison therefore
cannot be driven to AISimulate's figure by correct modelling. The sensitivity sweep says the
opposite: quadrupling max_num_seqs moves the score by 0.17 points and halving it by 0.36,
against a 4.5-point gap. Even removing the KV bound on admission entirely changes nothing to
two decimal places.

So the unstated settings are a footnote, not a ceiling. The resident batch in this corpus is
set by the offered concurrency and the work per request, not by the caps -- which is why
scaling the caps barely registers. The 4.5-point gap is a modelling gap, and it is mine to
close. That correction matters more than the original claim: it would have excused the
residual on a cause that measurement does not support.

The dp fix remains a genuine bug fix for a different reason: those 3 points sat AT the cap,
where the cap binds absolutely rather than marginally.

**Q4. Does `CachedTokens: 0` still hold?** Yes, but the justification was wrong and is
corrected. BLIS implements prefix caching UNCONDITIONALLY (`sim/kv/cache.go`), not behind a
flag, so the assumption cannot rest on caching being off. The real mechanism:
`batch_formation` sets `numNewTokens = InputLen() - ProgressIndex`, and the cache-aware
allocation path advances `ProgressIndex`, so a hit is excluded from `Scheduled` before the
adapter sees it. Passing a cached count again would subtract the same tokens twice. A
behavioural test pins that a half-computed prompt costs less than a cold one; it SKIPS on the
double-subtraction half with a stated reason, because this kernel does not consume
`CachedTokens` for that shape and the zero is harmless either way.

**Q7/Q8. Where is the residual, and what sign?** Concentrated and signed. After the dp fix
the error is still worst on minimax-m2.5 (7 of the 10 worst sweeps) and grows monotonically
with concurrency within an arm -- on `minimax-m2.5-b200-fp8-vllm-tp4` at 1k1k it runs 6.34%,
15.23%, 22.11%, 32.41% across c=8..64, every point an OVER-prediction. A systematic,
batch-growing over-prediction on one model points at a term that scales with batch on that
model specifically, not at the scheduler and not at the data.

**Q9. Is `ExpertsTouched` that term?** No, rejected by arithmetic:
`local * (1 - ((E-k)/E)^tokens)` is identical for minimax (E=256, k=8) and gpt-oss (E=128,
k=4) at every batch size, because the two E/k ratios coincide. A minimax-specific error
cannot come from a function that treats them identically.

**Q10. What floor can this reach?** Higher than I claimed. With the settings shown to be
worth under half a point, there is no measured ceiling from the unstated-configuration
argument. AISimulate's advantage on the vLLM subset is 6.68% against 10.72%, and nothing
measured so far explains it away.

Still open: Q2 (cache_dtype), Q3 (engine_version rules pack), Q5 (Overlap vs NoOverlap for a
simulated step), Q6 (think time).
