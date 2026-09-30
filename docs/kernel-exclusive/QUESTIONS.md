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
