# Reproducing every figure in this work

Every number in `RESULTS.md` and `APPLES-TO-APPLES.md` is produced by a command listed here.
Nothing is transcribed by hand. If a figure in those documents disagrees with what a command
prints, the command is right.

## What you need

Five checkouts, as siblings. Four are public in the `inference-sim` org; the fifth is this
simulator.

```
git clone https://github.com/inference-sim/blis-schemas.git
git clone https://github.com/inference-sim/blis-latency-kernel.git
git clone -b modeling https://github.com/inference-sim/blis-catalog.git
git clone -b modeling https://github.com/inference-sim/blis-registry.git
git clone https://github.com/inference-sim/inference-sim.git     # this repository
```

Go 1.24 or later. Python 3.11 or later with `pyarrow` and `pyyaml` for the registry scripts.

The catalog and registry are read at RUN TIME by path, not imported, so the scoring commands take
their locations as flags. The defaults in each command assume `~/Documents/Projects/<repo>`; pass
`-catalog` and `-registry` if yours are elsewhere.

### The measurement data

Two sources, with different access:

**NVIDIA's published accuracy snapshot** is committed, at
`blis-latency-kernel/testdata/measurements/aisimulate_summary.json` (584 KB, the full nested
summary) and `aisimulate_e2e.json` (the extracted 447-point corpus). Nothing further is needed to
reproduce the scores. To refresh them from NVIDIA:

```bash
gh api repos/ai-dynamo/aisimulate/actions/workflows/360006241/runs?status=success \
  --jq '.workflow_runs[0].id'
# then, for that run id:
gh api repos/ai-dynamo/aisimulate/actions/runs/<id>/artifacts \
  --jq '.artifacts[] | select(.name|startswith("e2e-accuracy-web")) | .id'
gh api repos/ai-dynamo/aisimulate/actions/artifacts/<artifact-id>/zip > web.zip
unzip web.zip                      # yields summary.json
python blis-latency-kernel/scripts/extract_aisimulate_e2e.py summary.json \
    -o blis-latency-kernel/testdata/measurements/aisimulate_e2e.json
python blis-latency-kernel/scripts/gen_aisimulate_scenarios.py \
    blis-latency-kernel/testdata/measurements/aisimulate_e2e.json \
    -o blis-latency-kernel/testdata/aisimulate
```

**NVIDIA's operator measurement tables** are NOT committed: they are large parquet collections in
the `ai-dynamo/aisimulate` repository. The registry's fitting and validation scripts read them
from a checkout, and default to `/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data`.
Pass `--data <path>/systems/data` for your own. Figures that need them are marked below.

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
