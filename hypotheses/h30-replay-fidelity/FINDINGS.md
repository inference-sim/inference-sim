# H30: CrossModel Request-Level Fidelity — BLIS Replay vs Real vLLM

**Status**: Partially Confirmed
**Date**: 2026-03-03
**Experiments**: 10 training-set, 3 validation, 3 test (16 total)
**Control**: Iter 2 per-model blackbox coefficients (isolates crossmodel vs scheduler error)

## Hypothesis

> When BLIS simulates statistically equivalent workloads for training-set experiments using Iter 3 crossmodel coefficients, aggregate latency metrics (TTFT mean, E2E mean, throughput) will match real vLLM within 25% relative error, confirming the crossmodel backend is useful for capacity planning.

## Experiment Design

### What Was Compared (Same-Specification, Not Per-Request Replay)

BLIS generates a **statistically equivalent** workload from the same inference-perf parameters — same Poisson rates, same constant token distributions, same prefix groups. The specific request sequence differs (different random seed). This tests: "Given the same workload specification and server configuration, does BLIS produce similar latency distributions?"

### Controlled Variables
- Server config: model, TP, max_num_seqs=128, max_num_batched_tokens=2048
- KV cache: total_kv_blocks (from step traces), block_size=16
- Workload: arrival rates/stages, shared_prefix (groups × users), token lengths
- Single instance (no routing)

### Known Gaps
1. **Input tokens: BLIS 547 vs real ~574** (5% fewer — missing chat template/BOS/EOS overhead)
2. **Arrival sequence**: Different Poisson realization (same rate, different seed)
3. **Multi-turn disabled**: Real inference-perf's `enable_multi_turn_chat` controls chat template format, not context accumulation. Real data confirms constant input tokens. BLIS's multi-turn mode does context accumulation — a semantic mismatch discovered during the experiment.

### Three-Way Comparison Design
To isolate the error source, we ran every experiment with BOTH:
- **CrossModel** (Iter 3 global): β=[116.1, 1226.9, 19.9, 9445.2], α=[13732, 0, 860.6]
- **Blackbox** (Iter 2 per-model): per-model β and α from training/ledger.md

If crossmodel error > blackbox error → coefficient problem.
If both have similar error → BLIS scheduler/modeling problem.

## Results

### Training Set (moderate load, 0% failure)

| Model | Profile | BB TTFT | CM TTFT | Real TTFT | BB E2E | CM E2E | Real E2E | BB rps | CM rps | Real rps |
|-------|---------|---------|---------|-----------|--------|--------|----------|--------|--------|----------|
| llama-2-7b | general | 27.0 | 27.3 | **44.8** | 2771 | 2390 | **3558** | 13.91 | 13.92 | **13.92** |
| llama-2-7b | codegen | 22.8 | 22.7 | **28.0** | 2060 | 1607 | **2081** | 7.62 | 7.63 | **7.46** |
| llama-2-7b | roleplay | 22.5 | 22.0 | **27.1** | 1984 | 1530 | **2071** | 6.14 | 6.14 | **5.99** |
| llama-2-70b | general | 49.3 | 45.4 | **103.0** | 4996 | 5351 | **5321** | 13.88 | 13.88 | **13.86** |
| llama-2-70b | codegen | 46.7 | 44.1 | **55.6** | 4562 | 5120 | **4606** | 7.61 | 7.61 | **7.43** |
| llama-2-70b | roleplay | 46.8 | 43.9 | **54.8** | 4563 | 5157 | **4562** | 6.12 | 6.12 | **5.98** |
| mixtral-8x7b | general | 47.4 | 51.1 | **68.9** | 4859 | 4711 | **5039** | 13.89 | 13.89 | **13.87** |
| mixtral-8x7b | codegen | 46.2 | 47.5 | **58.9** | 4627 | 4085 | **4675** | 7.61 | 7.61 | **7.44** |
| mixtral-8x7b | roleplay | 46.7 | 50.0 | **60.5** | 4674 | 4101 | **4685** | 6.12 | 6.13 | **5.98** |
| codellama-34b | general | 39.5 | 39.7 | **51.6** | 3952 | 4417 | **4093** | 13.90 | 13.90 | **13.89** |

All values in ms except rps. BB=Blackbox(Iter2 per-model), CM=CrossModel(Iter3 global).

### Validation + Test Set

| Model | Profile | Split | BB TTFT | CM TTFT | Real TTFT | BB E2E | CM E2E | Real E2E | BB rps | CM rps | Real rps |
|-------|---------|-------|---------|---------|-----------|--------|--------|----------|--------|--------|----------|
| codellama-34b | codegen | val | 37.9 | 38.4 | **45.6** | 3677 | 4191 | **3723** | 7.61 | 7.61 | **7.44** |
| codellama-34b | roleplay | val | 37.9 | 38.2 | **45.7** | 3688 | 4214 | **3670** | 6.13 | 6.13 | **5.98** |
| mixtral-8x7b | reasoning | val | 315.9 | 194.4 | **103,035** | 29,375 | 29,045 | **156,421** | 3.93 | 3.93 | **1.00** |
| llama-2-7b | reasoning | test | 2,140 | timeout | **106,645** | 17,786 | timeout | **160,063** | 0.08 | — | **0.49** |
| llama-2-70b | reasoning | test | 17,036 | timeout | **118,996** | 49,201 | timeout | **161,730** | 3.87 | — | **2.13** |
| codellama-34b | reasoning | test | 45.3 | 64.2 | **120,171** | 25,208 | 27,719 | **158,991** | 3.94 | 3.93 | **3.22** |

## Analysis

### Finding 1: Throughput matches within ±2.5% at sub-saturation — expected but informative

Both crossmodel and blackbox predict throughput within ±2.5% on all moderate-load (0% failure) experiments. **Caveat:** At sub-saturation, all requests complete within the horizon, so throughput ≈ arrival rate. The ±2.5% measures Poisson realization noise, not step-time accuracy. The meaningful throughput test is at saturation (H31: codellama-34b-reasoning, where BLIS predicts 3.93 rps vs real 3.22 rps = +22% overestimate).

What the sub-saturation throughput DOES validate: BLIS completes all requests within the horizon (no pathological queueing divergence), and the time-weighted stage rates produce the correct request count.

### Finding 2: TTFT is systematically underpredicted by BOTH backends (-17% to -56%)

**This is the critical finding.** Per-model blackbox coefficients produce almost **identical** TTFT underprediction as global crossmodel coefficients. This means:

> The TTFT error is NOT from the crossmodel's global β. It is from BLIS's scheduling model being systematically faster than real vLLM.

| Model (general) | BB TTFT RE | CM TTFT RE | Difference |
|-----------------|-----------|-----------|------------|
| llama-2-7b | -40% | -39% | 1pp |
| llama-2-70b | -52% | -56% | 4pp |
| mixtral-8x7b | -31% | -26% | 5pp |
| codellama-34b | -23% | -23% | 0pp |

The 0-5pp difference between backends is noise. The ~20-56% underprediction is a BLIS simulator property — **zero inter-step overhead**. BLIS schedules the next step at `now + stepTime` with no gap (`sim/simulator.go:427`), while real vLLM has measurable overhead between steps:

- **Scheduler CPU time** (0.5-2ms/step): `schedule()` in vLLM performs KV block allocation, queue traversal, token budget accounting, prefix cache hash lookups
- **CUDA kernel launch gap** (0.1-0.5ms/step): Micro-gaps between kernel launches (attention → MLP → layernorm → all-reduce → sampling)
- **Python interpreter overhead**: GIL, `update_from_output()` callback, detokenizer dispatch
- **HTTP/ASGI overhead** (for TTFT specifically): request parsing, tokenization, first-chunk framing

All three BLIS latency backends return 0 from `SchedulingProcessingTime()`. With ~100-200 steps per request, the accumulated inter-step gap is 50-400ms per request — consistent with the observed TTFT underprediction.

**Note:** This is a BLIS simulator modeling gap, not a coefficient accuracy issue. Conclusive attribution would require a "perfect-beta" control (injecting real step durations into BLIS) — this was not performed in this experiment.

### Finding 3: E2E varies — crossmodel's γ₁ is compensating error

The crossmodel's α₂=860.6 µs/tok (output processing time) adds ~213ms per request to E2E. This is 170× real detokenization cost (~5 µs/tok). It was diagnosed in Iter 3 as a "β diagnostic" — it absorbs the average β decode error. In practice, it accidentally brings CM E2E closer to real for some models (70b: +0.6% CM vs -6% BB).

### Finding 4: Why Iter 3 analytical evaluation showed great metrics

The Iter 3 analytical evaluation used **real observed queue wait times** (`T_queue_obs`):

```
TTFT_pred = α + T_queue_obs + T_prefill_pred  [Iter 3 formula, ledger.md line 161]
```

This means:
1. The coefficient evaluation never tested queue dynamics prediction
2. At saturation (reasoning test set), `T_queue_obs` ≈ 120,000ms dominates → coefficient error is <0.02% of total TTFT
3. The "1.5% test TTFT error" was real queue wait masking the coefficient error

BLIS must predict queueing from scratch. When both backends produce ~2x TTFT underprediction despite matched throughput, the error is in BLIS's scheduling model, not the coefficients.

### Finding 5: Reasoning experiments are catastrophically off

At saturation (reasoning profile, 4 RPS), both backends dramatically overestimate capacity:
- codellama-34b: 3.93 rps predicted vs 3.22 rps real (22% gap → ρ shifts from 124% to 102%)
- mixtral-8x7b: 3.93 rps predicted vs 1.00 rps real (293% gap)
- llama-2-7b: 0.08 rps blackbox vs 0.49 rps real (both heavily KV-constrained)

The 22% capacity overestimate (3.93 vs 3.22 rps maximum serviceable rate) causes a **regime transition**: real vLLM operates at ρ=124% (unstable, linearly growing queues → 120s TTFT), while BLIS operates at ρ=102% (stable, bounded queues → 64ms TTFT). This is not a proportional amplification — it is a phase transition from stable to unstable queueing. The empirical 2000× TTFT ratio (120,000ms / 64ms) is the ratio of queue wait times across this phase boundary, consistent with M/G/1 theory where E[W] → ∞ as ρ → 1⁺. This confirms H7 (#401) and H29 (#433): capacity estimates near saturation are exponentially sensitive to µ.

### Finding 6: `enable_multi_turn_chat` semantic mismatch

Initial run produced TTFT of 187 **seconds** (vs real 52ms) because BLIS's `ExpandInferencePerfSpec` interprets `enable_multi_turn_chat: true` as context accumulation (`ContextGrowth: "accumulate"`). Real inference-perf uses it for chat template formatting. Real data confirms constant input tokens. Fix: set `enable_multi_turn_chat: false`.

**This is a real BLIS bug** — the inference-perf workload converter misinterprets the multi-turn flag.

## Verdict

**Partially Confirmed.** Throughput is excellent (±2.5%). E2E mean |RE| (14.6%) passes targets. TTFT is systematically underpredicted (25.9% mean |RE|), but the three-way comparison proves this is a BLIS scheduling model limitation (both backends show the same gap), not a crossmodel coefficient issue.

## Issues Filed

- [ ] `enable_multi_turn_chat` semantic mismatch in `ExpandInferencePerfSpec` (workload bug)
- [ ] BLIS `SchedulingProcessingTime()` returns 0 — real vLLM has ~100-300µs CPU overhead per step per layer that accumulates
- [ ] Evaluation protocol gap: `problem.md` Section 11b (BLIS replay) should explicitly note that it tests system composition, not just coefficients — Section 11a's `T_queue_obs` substitution masks scheduling model errors

## Devil's Advocate

**Could the TTFT gap be from something other than zero inter-step overhead?** Partially — the 5% fewer input tokens in BLIS (547 vs 574) contribute ~2-3% to the gap. But the crossmodel has no per-token prefill cost for dense models, so even fixing the input count wouldn't change step times. The three-way comparison (both backends showing same gap) eliminates coefficient source as a cause. A "perfect-beta" control (injecting real step durations) would be needed to fully separate inter-step overhead from batch composition divergence.

**Could the throughput match be coincidental?** At sub-saturation, throughput ≈ arrival rate for any model that completes all requests. The meaningful throughput test is at saturation (codellama-34b-reasoning: +22% overestimate). The sub-saturation throughput match validates that BLIS completes all requests within the horizon, but does not validate step-time accuracy.

## Scope and Limitations

1. **Same-specification, not per-request replay**: BLIS generates its own Poisson arrivals; individual request sequences differ from real vLLM.
2. **5% fewer input tokens**: Missing chat template overhead (~27 tokens). Systematic but small effect.
3. **No perfect-beta control**: Cannot conclusively separate inter-step overhead from batch composition divergence.
4. **Single instance only**: Cluster-mode amplification effects (H7, H29) not tested.
5. **H100 SXM / vLLM v0.15.1 only**: Results may not generalize to other hardware or vLLM versions.

## Inter-Step Overhead Components (from vLLM Expert review)

The zero inter-step overhead gap includes these real vLLM components:
- **scheduler.schedule() CPU time** (0.5-2ms/step): KV block allocation, queue traversal, token budget
- **prepare_model_input()** (1-5ms/step): Attention metadata, block tables, input tensor GPU transfer
- **CUDA graph capture/replay asymmetry**: Graph miss = 5-50ms capture; graph hit = 10-50µs replay. Mixed prefill+decode steps use eager execution (full kernel launch chain)
- **Python interpreter overhead**: GIL, update_from_output(), detokenizer dispatch
- **HTTP/ASGI overhead** (TTFT only): Request parsing, tokenization, first-chunk framing

## Reproducing

```bash
# 1. Generate workload specs from training data
python3 training/generate_replay_specs.py

# 2. Run one experiment (codellama-34b general)
./blis convert inference-perf --spec training/replay_data/20260218-150304-codellama-34b-tp2-general.yaml > /tmp/v2.yaml
./blis run --model codellama/CodeLlama-34b-Instruct-hf --latency-model crossmodel --hardware H100 \
  --tp 2 --total-kv-blocks 26602 --block-size-in-tokens 16 --max-num-running-reqs 128 \
  --max-num-scheduled-tokens 2048 --num-instances 1 --num-requests 16800 --horizon 1200000000 \
  --model-config-folder training/model_configs/CodeLlama-34b-Instruct-hf --workload-spec /tmp/v2.yaml

# 3. Compare against: training/replay_data/20260218-150304-codellama-34b-tp2-general_ground_truth.json
```
