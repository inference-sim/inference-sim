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

**The real benchmark (1)** is NOT specified by anything available here. The InferenceX
`benchmark_results` rows carry `isl` and `osl` as plain integers and nothing else --
confirmed against the 1,217 cached vLLM-family rows, whose only workload fields are `isl`
and `osl`. No `range_ratio`, no dataset name, no seed.

This matters because vLLM's own `range_ratio` has DIFFERENT semantics from AISimulate's.
`vllm/benchmarks/datasets/utils.py:72-75`:

```python
input_low  = math.floor(real_input_len * (1 - input_range_ratio))
input_high = math.ceil(real_input_len * (1 + input_range_ratio))
```

Symmetric about the target, not below it. So:

| Workload | ISL interval at the 1k label | Mean ISL |
|---|---|---|
| Real, if run at vLLM default `range_ratio=0.0` | [1024, 1024] | 1024 |
| Real, if run at `range_ratio=0.8` | [204, 1844] | 1024 |
| AISimulate's replay | [819, 1024] | 922 |
| BLIS, as run here | [1024, 1024] | 1024 |

AISimulate's own simulated workload has a ~10% shorter mean context than whatever the real
benchmark used, under either reading of the real one. That is a property of the baseline, not
something this project can fix, and it is recorded because it bounds how well ANY simulator
can match those measurements.

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
