# Reproducing every figure in this work

Every number in `RESULTS.md` and `APPLES-TO-APPLES.md` is produced by a command listed here.
Nothing is transcribed by hand. If a figure in those documents disagrees with what a command
prints, the command is right.

## What you need

This repository, Go 1.24 or later, and the measurement corpora. Every other input is pinned to a
tagged release and reaches you without a separate checkout:

| input | release | how it arrives |
|---|---|---|
| blis-latency-kernel | `v0.2.0` | `go.mod`; its scenario fixtures (`testdata/aisimulate`) are read from the module cache |
| blis-schemas | `v0.3.0` | `go.mod` |
| blis-catalog | `0.2.2` | vendored subset at `testdata/catalog` |
| blis-registry | `v0.1.2` | vendored coefficient sets at `testdata/registry` |

The catalog and registry releases are the ones blis-latency-kernel `v0.2.0` pins in its own
`testdata/upstream.lock`; `TestTheVendoredCatalogAndRegistryAreTheLockedReleases` fails if the
vendored copies drift from that lock. To score against live upstream checkouts instead, clone the
tags and point the variables at them:

```
git clone --branch 0.2.2  --depth 1 https://github.com/inference-sim/blis-catalog.git
git clone --branch v0.1.2 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_CATALOG=$PWD/blis-catalog BLIS_REGISTRY=$PWD/blis-registry
```

or pass `-catalog`, `-registry` and `-scenarios` to the scoring commands.

### The measurement data

The scoring commands read measured latencies from `BLIS_MEASUREMENTS`, a directory holding
`aisimulate_e2e.json`, `aisimulate_summary.json`, `inferencex_absolutes.json` and
`inferencex_engine_settings.json`. They are extracted from third-party publications -- NVIDIA
AISimulate's end-to-end accuracy artifact and SemiAnalysis InferenceX's database dump -- which
neither this repository nor blis-latency-kernel redistributes. blis-latency-kernel's
`testdata/README.md` names each source, and its `scripts/extract_*` tools produce the files. A
command run without them refuses, naming the flags it needs.

```bash
export BLIS_MEASUREMENTS=/path/to/corpora
go test -tags scoring ./sim/kernelmodel/...   # the tests that read the corpora
```

On a laptop this takes about 35 s of wall time (34.6 s measured on a clean checkout of `e4d76827`),
well inside `go test`'s default 10-minute timeout. Its slowest test,
`TestNoScoredVLLMPointUsesAnInventedSetting` (about 18 s), checks every vLLM point's settings
without simulating any.

**NVIDIA's operator measurement tables** are large parquet collections in the
`ai-dynamo/aisimulate` repository. The registry's fitting and validation scripts read them from a
checkout of that repository; with `AISIMULATE` naming your checkout, pass
`--data "$AISIMULATE"/python/aisimulate/src/aisimulate_core/systems/data`.
Figures that need them are marked below.

### Which revision each figure was scored at

The headline tables in the next section were printed at commit `a5b9ac12`, at the tagged
releases above. Figures in the historical sections below were scored against pre-release
revisions of the kernel and registry (the kernel at `bd743a6`, the registry at `640a27e`) under
the earlier protocol, and are kept as the record of what was measured then.

## The headline comparison

Every scorer runs the `blis` binary for each point -- `blis replay` on the point's InferenceX
main phase, exactly as a user replays a trace -- so build it once and pass it with `-blis`:

```bash
cd inference-sim
go build -o blis main.go
go run ./cmd/metricscore -blis ./blis -framework vllm -gaps gaps.txt
go run ./cmd/metricscore -blis ./blis -framework vllm -config-tier measured -gaps gaps-measured.txt
go run ./cmd/kernelscore -blis ./blis -framework vllm -verbose
```

`metricscore` prints every arm (BLIS, AISimulate, AIC) on three tiers: STRICT,
MEASURED-OR-RESOLVED and ALL (approximate). `APPLES-TO-APPLES.md` defines them. For each tier the
output gives:

- its N;
- the sweeps of the two strict tiers, with the resolved settings that keep each out of STRICT;
- every prefix-caching point's scored and cold run side by side (information only);
- the per-chip, per-model and per-framework breakdowns over ALL.

`-gaps` writes the coverage report the tiers are decided from. Expected output, at `e4d76827`,
against the vendored catalog and registry and the corpora of `db-dump/2026-09-28` (mean absolute
error; BLIS / AISimulate / AIC):

| tier | sweeps | n (TPOT shape / mape) | TPOT shape | TPOT mape | TTFT shape | TTFT mape |
|---|---|---|---|---|---|---|
| STRICT | 0 | 0 | -- | -- | -- | -- |
| MEASURED-OR-RESOLVED | 10 | 52 / 62 | 11.38 / 8.73 / 9.15 | 16.79 / 19.85 / 19.08 | 13.96 / 23.22 / 22.34 | 23.47 / 42.96 / 23.77 |
| ALL (approximate) | 113 | 460 / 573 | 9.64 / 10.43 / 10.56 | 15.04 / 15.39 / 15.71 | 17.85 / 23.05 / 30.12 | 25.30 / 37.86 / 36.66 |

(TTFT n is 456 / 569 in ALL: four measured TTFT excursions are excluded for every arm.) `-config-tier measured` restricts the run to the 68 sweeps with an engine log. Its ALL row is 292 / 360 points at 10.60 / 11.91 / 12.00 (TPOT shape), 13.67 / 19.72 / 19.88 (TPOT mape), 22.72 / 31.54 / 31.28 (TTFT shape) and 28.36 / 45.07 / 39.03 (TTFT mape). Its MEASURED-OR-RESOLVED row is the one above.

`kernelscore` prints the TPOT-shape table for BLIS and AISimulate on the same tiers (11.38% against 8.73% on the 52 MEASURED-OR-RESOLVED points, and 9.64% against 10.43% on all 460), followed by the monotone, per-chip, signed and log-space breakdowns over all 460.

A full `metricscore -framework vllm` run took 51 minutes here, on a 10-core laptop alongside other jobs. It covers 113 sweeps: each concurrency point is a `blis replay` of `10 x concurrency` requests, and each of the 218 prefix-caching points is replayed twice (scored, and cold for information). The `-config-tier measured` run takes about 40 minutes.

Determinism: a point is a pure function of its inputs, so two runs agree exactly
(`TestARunIsDeterministicAtAFixedSeed`). The lengths are the fixed seed-0 vectors; the scorers'
`-seed` is `blis replay`'s, which draws only the synthetic prompts' token IDs, and with no shared
prefix a token ID moves no latency.

## Earlier comparisons (historical)

The four-estimator Hopper comparison and its audit, recorded in `RESULTS.md`, were produced by
`cmd/estimatorscore` against the roofline and trained-physics backends. Both backends and that
command were removed when BLIS became kernel-exclusive, so those figures cannot be re-run here;
`git log` holds the revision that produced them.

## The kernel's own scores, without the simulator

```bash
cd blis-latency-kernel
go run ./cmd/shape -testdata testdata/aisimulate      # concurrency-response shape, 447 points
go run ./cmd/score                                    # absolute inter-token latency, 18 points
go run ./cmd/worked-table                             # the five-row table in the design document
go run ./cmd/overlap-probe testdata/aisimulate/glm-5-h200-fp8-sglang-tp8.yaml
```

`cmd/shape` assumes resident batch equals client concurrency, which is the assumption the
simulator removes; the difference between its figure and `kernelscore`'s is the value of having a
scheduler.

## Reproducing the calibration

```bash
cd blis-registry
python scripts/fit_attention.py <data>/h200_sxm/attention/trtllm/1.3.0rc20 --chip h200
python scripts/fit_attention_prefill.py h200_sxm:h200
python scripts/fit_attention_by_kind.py --sku h200_sxm --chip h200
python scripts/fit_attention_by_kind.py --all
python scripts/fit_collectives.py --all <data>
python scripts/fit_recurrent.py <data>
python scripts/emit_primitives.py  <aisimulate>/systems ../blis-catalog > coefficients/cost-model-primitives.yaml
python scripts/emit_collectives.py <data> > coefficients/cost-model-collectives.yaml
```

Needs the parquet tables. The two `emit_*` scripts regenerate committed files; the diff should be
empty on an unchanged collection.

### The checks that make a fit trustworthy

```bash
python scripts/fit_attention_by_kind.py --check-regression
python scripts/validate_against_aisimulate_tables.py --check-known
python scripts/validate_against_aisimulate_tables.py
python validator/validate.py
python -m pytest validator/ -q
```

`--check-regression` asserts the per-kind fitter reproduces the committed unsuffixed
coefficients on the same rows -- a new fitter that cannot reproduce the old fit is a different
fit, and then nothing it produces can be trusted.

`--check-known` asserts four figures measured by hand before the gate existed, which is what
makes the gate trustworthy on a new one:

| What | Expected |
|---|---|
| decode attention, h200 gqa, corpus regime, all geometries | 0.795 |
| decode attention, h200 gqa, corpus regime, minimax per-rank geometry | 1.294 |
| decode attention, h200 gqa, corpus regime, evaluation group sizes | 1.064 |
| MoE grouped-GEMM, b200, minimax geometry, fastest lane | 0.625 |

Those three attention figures are the same coefficient measured over three populations. They
differ because the error depends on the GQA group size, not on the part, and conflating them
would justify a re-fit in the wrong direction.

## The prefill-attention axis probe

Marked in `RESULTS.md` as a defect found and deliberately not fixed. Needs the parquet
collections.

```bash
cd blis-registry
python scripts/probe_attention_prefill_axis.py --parts h200_sxm \
    --data "$AISIMULATE"/python/aisimulate/src/aisimulate_core/systems/data
python scripts/probe_attention_prefill_axis.py --residuals h200_sxm \
    --data "$AISIMULATE"/python/aisimulate/src/aisimulate_core/systems/data
```

The first prints the held-out A/B over four efficiency keys and the guard line, which must read
`OK`: the current key fitted on all rows has to reproduce the committed h200 entry (n 55096,
floor 26.5, scale 0.48). The second prints signed residuals by batch, which is what shows that no
single scalar key removes the trend.

The step-frequency figures that make the defect irrelevant here (98.33% of steps carry no prefill
request) came from temporary instrumentation in `Model.StepTime`, counting prefill requests per
step over a full `cmd/kernelscore -framework vllm` run. The instrumentation was removed; to
re-derive, count `Reqs[i].Scheduled > DecodeThreshold` per batch in a histogram.

## Reproducing the error definition

The one check that decides whether this comparison measures what it claims:

```bash
cd inference-sim
go test ./sim/kernelmodel/harness/ -run TestAISimulatesPublishedShapeErrorIsReproducible -v
```

It recomputes AISimulate's shape error from AISimulate's own per-point data, under AISimulate's
own definition -- anchor excluded, each side normalised to its own anchor -- over the whole
1,137-point snapshot, and requires 10.05% against a published `tpot_shape_error_pct` of 10.05%.
It also requires the anchor-INCLUSIVE variant to fail to reproduce it, so the test discriminates
between the two definitions rather than passing on either.

## Reproducing the workload and protocol claims

The per-request target lengths are a committed fixture. To regenerate it from InferenceX's own
calls (numpy required; `-I` keeps the interpreter from importing anything beside the script), and
to check the committed files against a fresh generation:

```bash
python -I scripts/gen_inferencex_workloads.py --corpus "$BLIS_MEASUREMENTS"/aisimulate_e2e.json
python -I scripts/gen_inferencex_workloads.py --check --corpus "$BLIS_MEASUREMENTS"/aisimulate_e2e.json
for s in 1 2 3 4; do
  python -I scripts/gen_inferencex_workloads.py --corpus "$BLIS_MEASUREMENTS"/aisimulate_e2e.json \
    --seed $s --out testdata/inferencex-workloads/seed$s.json
done
```

`--check` prints `29 points match`. The same vectors are recomputed in Go, without Python:

```bash
go test ./sim/kernelmodel/harness/ -run 'TestTheGoGenerator|TestTheFixtures|TestTheLoaderRefuses' -v
go test ./sim/kernelmodel/harness/ -run 'TestThePoolIsTheClientsSemaphore|TestAReplayedPoint|TestSummarize' -v
```

The first line checks an independent MT19937 and numpy-legacy `randint` against numpy's own
output, then every vector of every fixture against it. The second checks the concurrency gate on
the code `blis replay` runs, one point end to end through the binary, and the metric definitions.

To check the source claims against SemiAnalysis directly, with a clone of
`SemiAnalysisAI/InferenceX`:

```bash
git show 08d7a9cd:utils/bench_serving/benchmark_serving.py | sed -n '217p;228,235p;522,556p;595,632p;808,809p;1058p'
git show 08d7a9cd:utils/bench_serving/backend_request_func.py | sed -n '270p;319p'
git show f4a002ef:benchmarks/benchmark_lib.sh | sed -n '352,371p'
git show f4a002ef:.github/workflows/benchmark-tmpl.yml | grep -n RANDOM_RANGE_RATIO
for d in 2026-02-25 2026-03-01 2026-03-27 2026-03-28 2026-05-17 2026-05-30 2026-05-31 2026-06-04 \
         2026-06-05 2026-06-07 2026-06-26 2026-07-02 2026-07-06 2026-07-12 2026-08-04 2026-08-07; do
  c=$(git rev-list -1 --before=${d}T23:59:59 main)
  echo "$d: $(git grep -lE 'use-chat-template|--dsv4' $c -- benchmarks/single_node \
    | grep -vE '_mtp\.sh$|/agentic/' | tr '\n' ' ')"
done
```

The loop prints, per run date, the non-speculative recipes that pass a chat template: only
`qwen3.5_fp8_b300.sh` and, from 2026-06-26, two `minimaxm3_*_mi355x_atom.sh`; none is a scored
vLLM sweep.

and with numpy, that seed 0 gives the fixture's first `1024:1024` inputs and that a size-0 draw
leaves the generator untouched:

```bash
python -I -c "import numpy as np; np.random.seed(0); print(np.random.randint(819, 1025, size=5))"
# [ 991  866  936 1011  886]
```

## The test suites

```bash
cd blis-schemas         && go test ./...
cd blis-latency-kernel  && go test ./...
cd inference-sim        && go test ./sim/... ./cmd/...
cd blis-catalog         && python scripts/validate_catalog.py \
                        && python scripts/derive_graph.py --check \
                        && python -m pytest tests/ -q
cd blis-registry        && python validator/validate.py && python -m pytest validator/ -q
```

`derive_graph.py --check` re-derives all 28 model graphs from their committed configs and fails
on a stale or missing one, so a graph cannot drift from the config it came from.

## The design documents

In the `vllm` checkout, `docs/perf-model` holds four documents and seven verifiers. Each document
builds only if its verifiers pass:

```bash
cd vllm/docs/perf-model
python verify_arithmetic.py                 # 207 checks
python verify_calibration_facts.py          # 125
python verify_calibration_sources.py        # 371
python verify_kernel_design.py              # 521
python verify_kernel_implementation.py      #  40, runs cmd/shape and cmd/score and diffs them
python verify_worked_step_table.py          # runs cmd/worked-table and diffs section 2.1
python strategy_check.py
python build_html.py inference-cost-model
python build_html.py calibration-and-evaluation
python build_html.py latency-kernel-design
python build_html.py latency-kernel-implementation
python audit.py                             # all four rendered documents
```

`verify_kernel_implementation.py` is the one that keeps the documents honest about accuracy: it
runs `cmd/shape` and `cmd/score` and diffs their output against the tables, so a stale figure
fails the build. `SKIP_GO=1` skips those and says so rather than passing silently.

## Sensitivity: what the figures do not depend on

```bash
cd inference-sim
go run ./cmd/sensitivity -blis ./blis -framework vllm
```

Scores the monotone vLLM sweeps (TPOT shape, anchor excluded) under the scored configuration,
then varies one engine setting at a time -- `max_num_seqs` x2, x4 and /2, the token budget x2 and
/2 -- and then re-scores with the numpy seed-1..4 length fixtures, other workloads drawn from the
same intervals.

Expected draw rows at `e4d76827`, over 425 points:

| numpy seed | TPOT shape | TTFT shape |
|---|---|---|
| 0 (the benchmark's) | 8.60% | 17.64% |
| 1 | 8.60% | 17.81% |
| 2 | 8.58% | 18.01% |
| 3 | 8.58% | 17.79% |
| 4 | 8.56% | 17.92% |

A setting the runs passed is measured, so these rows bound only how much the score would move had
it been mis-stated; no value is chosen from them. The draw rows are other workloads, not repeated
observations of the benchmark's seed 0, and relax nothing.
