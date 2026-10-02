# Is this an apples-to-apples comparison?

Two things must match for the comparison to mean anything, and they were checked separately:
the WORKLOAD (three of them, not two) and the ERROR DEFINITION (two of them).

Both had a mismatch. Neither was visible from the corpus alone; both required reading
AISimulate's and vLLM's source.

## The three workloads

| # | Who | What it runs |
|---|---|---|
| 1 | The real benchmark | vLLM `benchmark_serving` against real GPUs; produced `mean_tpot` |
| 2 | AISimulate's simulation | `randomized_synthetic_engine_replay` |
| 3 | BLIS's simulation | this worktree's harness |

### What each actually does at the `1024:1024` label

**AISimulate (2)** is fully specified in source. `scripts/run_e2e_accuracy.py:230-236` sets
`random_range_ratio: 0.8`, `random_seed: 0`, `request_count = concurrency * 10`. And
`python/aisimulate/src/aisimulate/runner.py:1838-1852`:

```python
if random_range_ratio == 1.0:
    return [upper] * count
lower = int(upper * random_range_ratio)
return [rng.randint(lower, upper) for _ in range(count)]
```

So ISL is uniform on **[819, 1024]** and OSL uniform on **[819, 1024]**, sampled
independently per request. The label is an UPPER BOUND. Mean ISL is 922, about 10% below the
label.

**BLIS (3), as this harness ran it**, used `ConstantSampler` at exactly 1024 and 1024, and
`max(24, 4 x concurrency)` requests. So: no variance, a 10% higher mean context, and a
different request count. The label was read as a value.

**The real benchmark (1)** IS specified, in SemiAnalysis's own repository, and this document
previously said otherwise. The correction matters because it removes most of the gap it claimed.

`SemiAnalysisAI/InferenceX`, `inferencex-e2e/benchmarks/single_node/srt_fixed_sequence.sh`, drives
the measurement:

```
run_benchmark_serving --input-len "$ISL" --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" --max-concurrency "$CONC"
```

which `benchmarks/benchmark_lib.sh` turns into

```
python3 -m infx.bench_serving.benchmark_serving --dataset-name random \
    --random-input-len $ISL --random-output-len $OSL --random-range-ratio $RATIO \
    --num-prompts $((CONC*10)) --max-concurrency $CONC --request-rate inf \
    --ignore-eos --num-warmups $((2*CONC)) --percentile-metrics 'ttft,tpot,itl,e2el'
```

and whose sampler, `infx/bench_serving/benchmark_serving.py`, is

```python
def sample_uniform(seq_len: int) -> list[int]:
    lower = int(seq_len * range_ratio)
    upper = seq_len
    return np.random.randint(lower, upper + 1, size=num_prompts).tolist()
```

`RANDOM_RANGE_RATIO` is `0.8` in every recipe that sets it: of 466 recipe files in the live tree,
50 set it and all 50 use 0.8.

**What this document got wrong.** It previously asserted that vLLM's `--random-range-ratio` is
SYMMETRIC -- `[len(1-r), len(1+r)]`, which is true of `vllm/benchmarks/datasets/utils.py` -- and
concluded that AISimulate's one-sided interval therefore differs from the real benchmark under
either reading. InferenceX does not use vLLM's client. It uses its own
`infx.bench_serving.benchmark_serving`, whose sampler is one-sided and identical to AISimulate's
`_sample_synthetic_lengths`. So the three workloads agree on the distribution, and the ~10%
mean-context gap this document attributed to the baseline does not exist.

### The protocol, now matched exactly

| Property | InferenceX (real) | AISimulate | BLIS (this harness) |
|---|---|---|---|
| ISL, OSL interval at `1024:1024` | `[819, 1024]` inclusive | same | same |
| Sampling | uniform, independent per request | same | same |
| Measured requests per point | `10 x CONC` | `10 x CONC` | `10 x CONC` |
| Warm-up requests per point | `2 x CONC`, discarded | not stated | `2 x CONC`, discarded |
| Arrival | `--request-rate inf` + `--max-concurrency` | all at t=0, concurrency gates | all at t=0, concurrency gates |
| Output length | `--ignore-eos`: exactly OSL | fixed | fixed |
| Prefix reuse | none | `cached_prefix_tokens` 0 | none |
| Draw sequence | NumPy RandomState | Python MT, seed 0 | Go RNG |

The warm-up was this harness's last real difference. It previously used `max(24, 4 x concurrency)`
completions with the leading half discarded -- a criterion derived here from the shape of the
transient. That converged, but it measured about `2 x concurrency` requests where the real harness
measures `10 x`, and it discarded a fraction rather than a fixed phase. Matching the harness
removes a difference that had to be argued for.

Effect: **10.39% to 10.41%** on the 192 vLLM points. The change is behaviour-neutral to two
decimal places, which is the useful result -- it confirms the earlier criterion was converged, and
it means no figure in this project rests on the harness's own choice of budget.

Three differences remain and are stated rather than hidden: the draw SEQUENCE differs, since
matching NumPy's RandomState from Go would mean shipping precomputed length vectors; `1k1k` and
`1k8k` have no recipe file in the live tree, so their ratio is confirmed only through the shared
sampler and harness rather than from a recipe; and only 53.2% of the snapshot's points were
measured under vLLM, the engine BLIS models.

### Consequence for BLIS

Matching AISimulate means adopting ITS workload: `random_range_ratio: 0.8` with AISimulate's
one-sided interval, `request_count = concurrency * 10`, seeds 0 and 42. That is what makes the
two SIMULATIONS comparable to each other against the same measurements.

It is not merely a variance question. Variable OSL means requests retire at different times,
so the resident batch churns continuously; constant OSL makes them finish in lockstep. The
resident batch is the quantity this whole experiment is about.

It also contaminates the attention finding. A 1.29x over-prediction measured while feeding
1024-token contexts, when the baseline fed a 922-token mean, partly reflects the workload
error rather than the coefficient. Re-fitting the coefficient on that basis would bake a
harness bug into the registry as calibration -- the worst available outcome.

## The two error definitions

AISimulate's is in `scripts/build_e2e_accuracy_overview.py:150-181`:

```python
_, anchor_measured, anchor_prediction = points[0]
errors = []
for _, measured, prediction in points[1:]:          # <-- ANCHOR EXCLUDED
    measured_shape   = measured / anchor_measured
    prediction_shape = prediction / anchor_prediction
    errors.append(abs((prediction_shape - measured_shape) / measured_shape) * 100)
```

Pooled across topologies, then a plain mean over all comparisons. Points are sorted by
concurrency, the anchor is the lowest, and a topology with fewer than 2 points is skipped.

This harness computed the same ratio but averaged over `points[:]` -- INCLUDING the anchor.
The anchor's error is exactly 0 by construction, since it is each side divided by itself. So
every figure this project reported was diluted by one free zero per sweep:

| Subset | AISimulate's definition | This harness's definition |
|---|---|---|
| all 447 | 11.55% over 364 | 9.41% over 447 |
| vLLM | 8.87% over 192 | 7.15% over 238 |

The arithmetic confirms it: 8.87 x 192/238 = 7.16, which is the 7.15 reported.

**The definition is settled by reproduction, not by reading.** Applying AISimulate's
definition to its own per-point data over the whole 1137-point snapshot gives **10.05%** on
878 comparisons, against a published `tpot_shape_error_pct` of **10.05%**. Exact to two
decimals. The anchor-inclusive variant cannot reproduce it.

So the 9.41% named as the target was itself the diluted figure. Under AISimulate's own
definition its whole-snapshot shape error is 10.05%, and on the 447-point subset 11.55%.

## What must change

1. The harness must average over `points[1:]`, matching AISimulate.
2. The BLIS workload must adopt AISimulate's: one-sided uniform at ratio 0.8,
   `concurrency * 10` requests, seeds 0/42.
3. Every number in RESULTS.md must be recomputed, and the old ones marked as superseded with
   the reason.
4. The attention over-prediction must be re-measured at the corrected context distribution
   before any registry change is considered.

Until 1 and 2 are done, no BLIS-vs-AISimulate figure from this worktree is apples to apples,
including the ones already recorded.

## The guarantee, as executable checks

`sim/kernelmodel/harness/parity_test.go` holds the checks that fail when the comparison stops
being apples to apples. Each was verified to fail on the defect it guards, and two of them
were themselves wrong on the first attempt in ways worth recording.

**The error definition is settled by reproduction.** AISimulate's definition, applied to
AISimulate's own per-point relatives over the whole 1137-point snapshot, must return the
`tpot_shape_error_pct` it publishes. It does: **10.05% on 878 comparisons against a published
10.05%**. The anchor-inclusive variant returns something else, so the check discriminates
rather than merely passing.

A first version of this check compared against the 447-point SCORED SUBSET and passed the
WRONG definition. On that subset the anchor-inclusive figure (9.41%) happens to sit nearer the
whole-snapshot 10.05% than the correct anchor-exclusive one (11.55%). That is a coincidence of
subsetting, and comparing a subset figure to a whole-snapshot published number was never
valid. The check now reads the full snapshot.

**The workload constants are asserted against literals restated from AISimulate's source**,
not against the package constants the harness uses. A first version compared
`w.RequestCount` to `concurrency * aisimulateRequestsPerUser` -- the same constant the code
reads -- so changing the constant changed both sides and the check still passed. Verified by
mutation: setting the ratio to 0.9 or the count to 4 cycles now fails.

**The distribution reaching BLIS is verified end to end.** 400 generated requests span exactly
[819, 1024] with mean 920.1 against an expected 921.5, through BLIS's `empirical` sampler.

## What is now matched, and what cannot be

| Property | Real measurement | AISimulate | BLIS (this harness) |
|---|---|---|---|
| ISL interval at the 1k label | unknown | [819, 1024] | [819, 1024] |
| OSL interval at the 1k label | unknown | [819, 1024] | [819, 1024] |
| Length sampling | unknown | uniform, independent per request | uniform, independent per request |
| Requests per point | unknown | concurrency x 10 | concurrency x 10 |
| Arrival | unknown | all at t=0, concurrency gates | all at t=0, concurrency gates |
| Prefix reuse | unknown | none (`cached_prefix_tokens` 0) | none |
| Draw sequence | unknown | Python MT, seed 0 | Go RNG, seed 42 |
| Error definition | — | anchor excluded | anchor excluded |

BLIS now matches AISimulate on every property the snapshot and AISimulate's source specify.
The two remaining differences are stated rather than hidden: the draw SEQUENCE differs (the
distribution does not), and the REAL benchmark's workload is unknown -- InferenceX records
only `isl` and `osl` integers, and vLLM's own `range_ratio` is symmetric where AISimulate's is
one-sided, so AISimulate's replay simulates a ~10% shorter mean context than the real runs
under either reading. That is a property of the baseline and it bounds how closely any
simulator can match these measurements.
