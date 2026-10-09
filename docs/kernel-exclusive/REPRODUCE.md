# Reproducing every figure in this work

Every number in `RESULTS.md` and `APPLES-TO-APPLES.md` is produced by a command listed here.
Nothing is transcribed by hand. If a figure in those documents disagrees with what a command
prints, the command is right.

## What you need

This repository, Go 1.24 or later, and the measurement corpora. Every other input is pinned to a
tagged release and reaches you without a separate checkout:

| input | release | how it arrives |
|---|---|---|
| blis-latency-kernel | `v0.1.0` | `go.mod`; its scenario fixtures (`testdata/aisimulate`) are read from the module cache |
| blis-schemas | `v0.2.2` | `go.mod` |
| blis-catalog | `0.2.1` | vendored subset at `testdata/catalog` |
| blis-registry | `v0.1.1` | vendored coefficient sets at `testdata/registry` |

The catalog and registry releases are the ones blis-latency-kernel `v0.1.0` pins in its own
`testdata/upstream.lock`; `TestTheVendoredCatalogAndRegistryAreTheLockedReleases` fails if the
vendored copies drift from that lock. To score against live upstream checkouts instead, clone the
tags and point the variables at them:

```
git clone --branch 0.2.1  --depth 1 https://github.com/inference-sim/blis-catalog.git
git clone --branch v0.1.1 --depth 1 https://github.com/inference-sim/blis-registry.git
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

**NVIDIA's operator measurement tables** are large parquet collections in the
`ai-dynamo/aisimulate` repository. The registry's fitting and validation scripts read them from a
checkout of that repository; with `AISIMULATE` naming your checkout, pass
`--data "$AISIMULATE"/python/aisimulate/src/aisimulate_core/systems/data`.
Figures that need them are marked below.

### Which revision each figure was scored at

The tables below record what each command printed when the figure was published. Those
figures were scored against pre-release revisions of the kernel and registry (the kernel at
`bd743a6`, the registry at `640a27e`); the commands are unchanged, and re-running them at the
tagged releases above gives the current figures. `docs/guide/latency-models.md` carries the
headline re-scored at the tags.

## The headline comparison

```bash
cd inference-sim
go run ./cmd/kernelscore -framework vllm
```

Prints the table in `RESULTS.md`: 192 vLLM points, the monotone subset, the per-framework and
per-chip-family splits, and the protocol it ran under. Add `-verbose` for every point.

Expected, at the commit this document ships with:

| Subset | n | BLIS | AISimulate |
|---|---|---|---|
| vLLM, all | 192 | 10.41% | 8.87% |
| vLLM, monotone only | 171 | 9.92% | 8.32% |
| Blackwell | 84 | 11.91% | 12.61% |
| Hopper | 108 | 9.24% | 5.96% |

A run takes roughly forty minutes: 83 sweeps, each concurrency point a full closed-loop
simulation of `12 x concurrency` requests.

Determinism: the seed is fixed (`-seed`, default 42) and a point is a pure function of its
inputs, so two runs agree exactly. `TestARunIsDeterministicAtAFixedSeed` asserts it.

The same command prints the signed and log-space reports. Expected:

| model | n | mean | median | mean abs | over | p10 | p90 |
|---|---|---|---|---|---|---|---|
| BLIS + blis-latency-kernel | 192 | +2.71% | +0.78% | 10.41% | 53% | -11.61% | +19.60% |
| AISimulate, same points | 192 | +5.39% | +4.02% | 8.87% | 71% | -6.83% | +19.77% |

| model | n | mean log | sd log | bias | floor |
|---|---|---|---|---|---|
| BLIS + blis-latency-kernel | 192 | +0.0165 | 0.1442 | +1.67% | 10.23% |
| AISimulate, same points | 192 | +0.0462 | 0.1122 | +4.73% | 7.73% |

plus the per-concurrency signed breakdown. The signed figures are the ones that killed the
batch-composition hypothesis; see `RESULTS.md`.

## Four estimators on the Hopper subset

```bash
cd inference-sim
go run ./cmd/estimatorscore
```

Scores blis-latency-kernel, AISimulate, roofline and trained-physics on the Hopper vLLM subset,
with the KV budget, engine settings and host per-token cost taken from the kernel for every arm so
that only the forward-pass model differs. Expected:

| estimator | n | mean | median | mean abs | over |
|---|---|---|---|---|---|
| blis-latency-kernel | 104 | +1.71% | -0.46% | 8.87% | 44% |
| AISimulate | 104 | +1.75% | +2.02% | 6.04% | 66% |
| roofline | 104 | +161.54% | +135.30% | 161.54% | 100% |
| trained-physics | 104 | -38.93% | -39.04% | 38.93% | 0% |

Roughly twenty minutes: 24 sweeps x 3 arms. It also prints the per-scenario breakdown and the one
point that could not be scored (`gpt-oss-120b-h100-fp4-vllm-tp2 1k8k` at concurrency 4 under
trained-physics, which completes no requests within the horizon).

`-hardware-config` and `-defaults` default to the repository's own `hardware_config.json` and
`defaults.yaml`, so run it from the repository root.

### The audit behind that table

```bash
go run ./cmd/estimatorscore -curves gpt-oss-120b-h200-fp4-vllm-tp4.yaml
```

A mean error cannot distinguish a bad model from a broken harness. This prints each arm's
predicted curve, the absolute latencies that explain its shape, and the admission facts the arms
must share. Expected:

| concurrency | measured | kernel | roofline | trained-physics |
|---|---|---|---|---|
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 1.1565 | 1.1329 | 1.7947 | 1.0027 |
| 16 | 1.3275 | 1.3579 | 3.1124 | 1.0082 |
| 32 | 1.8010 | 1.6830 | 4.9277 | 1.0191 |
| 64 | 2.3641 | 2.0878 | 6.7031 | 1.0408 |

with anchors of 3,336 us (kernel), 1,087 us (roofline) and 17,743 us (trained-physics), and an
admission block reporting 817,357 KV blocks and `12 x concurrency` completions, `identical across
arms: yes` at every point. A `NO` there means the comparison is contaminated and the table above
is not a step-time comparison.

### Checking the KV number itself

The arms sharing one KV budget does not make that budget correct. Deriving it a second way, through
`latency.CalculateKVBlocks`, is what found **inference-sim#1852**: the two paths agree to within
1.8% on `minimax-m2.5` (fp8) and disagree by 1.60x on `gpt-oss-120b` (mxfp4), because the legacy
path does not recognise mxfp4 and sizes those weights at bf16 -- 217 GiB against the config's own
54.3 GiB. At tp=2 it returns an error instead of a block count. The kernel is the correct side, and
no scoring path calls `CalculateKVBlocks`, so no figure here is affected.

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

## Reproducing the workload claim

```bash
go test ./sim/kernelmodel/harness/ -run "TestTheWorkloadMatches|TestTheLengthPDF" -v
```

Asserts the sampling interval, the measured request count and the warm-up count against literals
restated from InferenceX's source, not against the package constants the code uses -- a first
version compared a constant to itself and a mutation of it survived.

To check the claim against SemiAnalysis directly:

```bash
curl -sL https://raw.githubusercontent.com/SemiAnalysisAI/InferenceX/main/inferencex-e2e/benchmarks/single_node/srt_fixed_sequence.sh
curl -sL https://raw.githubusercontent.com/SemiAnalysisAI/InferenceX/main/inferencex-e2e/infx/bench_serving/benchmark_serving.py | sed -n '263,266p'
gh api repos/SemiAnalysisAI/InferenceX/git/trees/main?recursive=1 \
  --jq '.tree[].path' | grep 'srt-slurm-recipes.*yaml$' | while read p; do
    curl -sL "https://raw.githubusercontent.com/SemiAnalysisAI/InferenceX/main/$p" \
      | grep -oE "RANDOM_RANGE_RATIO: *'?[0-9.]+'?"
  done | sort | uniq -c
```

The last prints `50 RANDOM_RANGE_RATIO: '0.8'` -- every recipe that sets it uses 0.8.

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
go run ./cmd/sensitivity -framework vllm
```

Varies each engine setting the snapshot does NOT publish and reports the effect. Against a gap of
about 1.5 points, every one is worth under half a point: `max_num_seqs` x4 moves the score -0.17,
halved +0.36, the token budget under 0.05 either way, and removing the KV bound on admission
0.00. That is the evidence for the claim that the residual is a modelling gap rather than a
consequence of unstated configuration.
