# Example scenarios

Ready-to-run scenario files for the README quickstart and the feature walkthroughs.

Point `--scenarios` at this directory:

```bash
./blis run --scenarios examplescenarios \
  --scenario deepseek-v4.1-flash-h100-tp8-dspark5.yaml \
  --catalog $PWD/blis-catalog --registry $PWD/blis-registry \
  --rate 10 --num-requests 100
```

Each file is a [`blis-schemas`](https://github.com/inference-sim/blis-schemas) `Scenario` +
`Deployment` pair, runnable against the pinned catalog (`blis-catalog` `0.3.0`) and registry
(`blis-registry` `v0.2.0`). Nothing here is fetched at run time; BLIS needs only the binary, a
catalog, and a registry. The scenarios that use speculative decoding pin their acceptance in
the file, so `blis run` needs no `--speculative-*` flag.

## The scenarios

| File | Model / shape |
|---|---|
| `deepseek-v4.1-flash-h100-tp8-dspark5.yaml` | DeepSeek-V4.1-Flash, 1×H100, TP8 EP, DSpark K=5 — the quickstart |
| `gpt-oss-120b-h100-fp4-vllm-tp2.yaml` | gpt-oss-120b (MoE), 1×H100, TP2, nvfp4 |
| `gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml` | gpt-oss-120b (MoE), 2×H100, TP2 — cluster fixture (up to 8 instances) |
| `glm-5-h200-3p1d-ib.yaml` | GLM-5, 4×H200, 3 prefill + 1 decode (P/D disaggregation) |
| `deepseek-v4-pro-b300-fp4-vllm-tp4-mtp2.yaml` | DeepSeek-V4-Pro (MoE), 1×B300, TP4, nvfp4, MTP K=2 |
| `kimi-k3-h200-tp8dp4ep32-balanced.yaml` | Kimi-K3 (MoE), 4×H200, TP8 DP4 EP32, MTP K=2 |

Each file's header records where its deployment came from.

### Workload shapes

The scenario fixes only the *deployment*. The workload — arrival rate, request count, token
lengths — is set on the command line. gpt-oss-120b's scored single-turn shapes include **8k/1k**
(ISL 8192 / OSL 1024), run as:

```bash
./blis run --scenarios examplescenarios \
  --scenario gpt-oss-120b-h100-fp4-vllm-tp2.yaml \
  --catalog $PWD/blis-catalog --registry $PWD/blis-registry \
  --prompt-tokens 8192 --prompt-tokens-min 8192 --prompt-tokens-max 8192 --prompt-tokens-stdev 0 \
  --output-tokens 1024 --output-tokens-min 1024 --output-tokens-max 1024 --output-tokens-stdev 0 \
  --rate 10 --num-requests 100
```

## Notes

The `--rate`/`--num-requests` in the quickstart is a synthetic open-loop workload, not a
benchmark reproduction. For deployments authored from a recipe (Kimi-K3), settings BLIS does not
model are dropped and noted in the file header (e.g. `enforce-eager`; an unset `max-model-len`
is given a concrete BLIS sizing value). Absolute latencies for deployments outside the kernel's
scored set are estimates; see [Model compatibility](../docs/reference/models.md).
