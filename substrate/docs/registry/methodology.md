# How a coefficient in this registry is defined, fitted and checked

This document states the rules. `band-selection.md` records one model-selection decision
made under them; `vllm/docs/perf-model/hypothesis-log.md` records the measurements, the
negative results and the retractions.

## 1. What a coefficient is

A coefficient is a **measured constant of a law the kernel evaluates** — never a fudge
factor and never a per-model correction. Each entry carries:

| field | meaning |
|---|---|
| `value` | the number |
| `units` | stated explicitly, because the single commonest error in this project is a unit slip |
| `method` | `measured`, `vendor_spec`, `assumed`, or `not_charged` (a deliberate zero) |
| `fitted` | whether a script produced it |
| `scope` | the deployment dimensions it applies to: hardware, model, TP, EP, nodes |
| `sources` | the exact file the value came from, with row counts |
| `rationale` | why this value, what it replaced, and what rejecting it would cost |

Three rules follow from `method`:

* An `assumed` coefficient is a **declared gap**, not a measurement. §6.0 counts them by
  kind; the ones nothing in this project's data could replace are the seven host
  overheads and the four CUDA-graph capture sizes (§11). Any claim resting on one has to
  say so.
* A `vendor_spec` coefficient is NVIDIA's observation, not this project's. `hbm_derate`
  at 0.8 is the example.
* A `not_charged` coefficient is a deliberate zero, so a consumer can tell "no cost" from
  "cost unknown". `cudagraph_capture_bytes_none` is the one instance.

**A value and its citation move together.** Editing one without the other is how a
registry comes to describe a fit nobody ran, and the validator re-derives values from the
sweep its citation names, so a stale citation fails rather than lies.

**A coefficient with no reader is not live.** `dtypeFit` in the kernel can return only
three dtype suffixes — `bf16`, `fp8`, `nvfp4` — so the 14 `gemm_*_fp8_block` entries are
unreachable. That was discovered by grepping the kernel for the key, not the registry.
Check both.

Each unread family is either given a reader or removed, and the decision is recorded
(inference-sim/blis-registry#36):

* **Removed:** `hbm_constant_latency` and `p2p_latency`, the two NVIDIA-simulator constants
  whose fixed cost the kernel's measured attention and collective floors already carry, so
  a reader would double-count; and `moe_routing_imbalance_p90`, a tail a mean-step model
  does not price (its value is stated in each median's rationale). A property test keeps
  them out.
* **Kept, with a named reader:** the `gemm_*_fp8_block` ramps. They are what vLLM runs for
  a 128×128 block-scaled FP8 checkpoint (DeepGEMM and FlashInfer block-scaled kernels), and
  their fit differs from per-tensor FP8 (h100: 0.749 / 160 against 0.648 / 131). The kernel
  change that selects a ramp by weight format and chip, blis-latency-kernel#34, is what
  reads them.
* **Committed ahead of their reader**, each naming it: `cost-model-memory`
  (blis-latency-kernel#22); the weight-only Marlin ramps, the indexer pair and the
  host-gather rate (§10.5), read once blis-latency-kernel#44 lands (it closes #34 and #38
  and adds the role-keyed attention lookup). The V4.1 sparse rate's name is looked up by
  kernel `main` too, but a V4.1 scenario fails at construction there first (#36), so it is
  read once #44 lands as well.
  The reference overview lists every one, so "committed" is never read as "live".

## 2. The lane rule: fit the engine you predict

AISimulate ships five framework lanes (`vllm` 332 files, `sglang` 313, `trtllm` 217,
`nccl` 23, `oneccl` 2). The lane is part of the path, so choosing it is explicit — and
the lanes are not interchangeable:

| operator | finding |
|---|---|
| decode attention | vLLM/TRT-LLM differ 1.30-1.35x on Hopper and 0.83-0.85x on Blackwell. The SIGN flips, because `vllm/v1/attention/backends/fa_utils.py:99-107` dispatches on compute capability — FA3 at major 9, FA4 at major 10. |
| prefill attention | vLLM runs at 0.79-0.81x of TRT-LLM's on all five parts, concentrated at short prompts: 0.505x at `isl=1` against a ~0.87x plateau above 128. |
| MoE | 1.38-1.41x Hopper, 0.86-0.90x Blackwell, with entirely disjoint `kernel_source` values. |
| GEMM | the lanes AGREE to 0.4-3.6% over 100,668-130,156 shared shapes — same cuBLAS/CUTLASS kernels — so GEMM stays on the larger TRT-LLM sweep. |
| all_gather, reduce_scatter, broadcast | NCCL **is** the vLLM path. `CustomAllreduce` implements all-reduce only; `should_custom_all_gather` and `should_custom_reduce_scatter` gate and fall through. AISimulate has no vLLM data for them because there is nothing to measure. |

So "use the vLLM lane" is a conclusion to be tested per operator, not a blanket rule.
Of 654 lane-citing coefficients, 572 are not on the vLLM lane and most of those are
RIGHT:

| citation | n | verdict |
|---|---|---|
| nccl, all_gather / reduce_scatter / all_to_all | 414 | correct — not implemented by `CustomAllreduce` |
| nccl, all_reduce | 84 | the vLLM lane exists; applied where it helps (see §5) |
| trtllm, gemm | 42 | correct — lanes agree to 0.4-3.6% |
| trtllm, attention | 12 | sliding-window and L40S/A100 gaps; see the absences in §5 |
| trtllm, moe | 12 | the vLLM lane differs by architecture; untried |
| sglang / trtllm, recurrent | 4 | vLLM `kda` and `linear_attention` data exists; untried |
| sglang, gemm / moe | 4 | A100's only lane for those families |
| **vllm** | **82** | decode and prefill attention, all five served parts, plus A100 |

**One collection per part, per family.** Globbing collections mixes engine versions and
silently changes the row set — it once produced 66,148 rows where the committed fit used
40,367. A difference between two parts should be the silicon, not a software release.

## 3. The three-level data separation

> [Fitting principles](fitting-principles.md) §3.1 supersedes this section's "evaluation only"
> rule for end-to-end rows: InferenceX end-to-end points are split 40/60 into fit and held-out sets.

| source | role | why that role |
|---|---|---|
| AISimulate per-operator sweeps | **fitting** | per-kernel, lane-labelled, and the only data that isolates one primitive |
| HF FPM whole-forward (Apache-2.0) | **validation, model selection, and fitting a COMPOSED term** | one synchronized forward pass at a known batch and KV composition — the quantity the kernel composes, with no scheduler |
| InferenceX measured rows | **evaluation only** | end-to-end with a scheduler and a client; never enters a fit or a selection |
| InferenceX remaining rows | reporting only | their engine settings are not stated per run |
| vLLM start-up logs in the vLLM issue tracker | **anchoring an `assumed` magnitude only** | vLLM's own measurement of CUDA-graph capture memory, on uncontrolled deployments; the only public record of that quantity (§11) |

**What FPM may and may not fit, and why the rule changed.** An earlier revision of this
table read "validation and model selection" and forbade FPM from entering any fit. The
reason given was a confound: model and system were perfectly collinear, so a model holdout
was also a chip holdout and a change across it could be attributed to neither. **That is
no longer true of the dataset.** Derived from the catalog index rather than recalled:

| model | systems it appears on |
|---|---|
| DeepSeek-V4.1-Flash | b200-sxm, b300-sxm, gb200, gb300, h200-sxm |
| DeepSeek-V4-Pro | b200-sxm, b300-sxm, gb300 |
| MiniMax-M2.7 | b200-sxm, h200-sxm |
| GLM-5.2-NVFP4 | b200-sxm, gb200 |

Four of seven models now span multiple systems, and b200-sxm carries six models against
gb300's three, so **both axes vary independently**. A chip holdout is no longer a model
holdout. The confound that justified the prohibition has dissolved, and keeping the
prohibition would now cost accuracy for no methodological gain.

The distinction that remains, and it is a real one, is **what kind of quantity each
dataset constrains**:

* **AISimulate isolates ONE primitive.** A GEMM sweep varies M with everything else held,
  so it identifies that primitive's two parameters and nothing else. This is the only
  data that can fit a per-primitive coefficient, and every `measured` entry in this
  registry still comes from it.
* **FPM constrains a COMPOSITION.** One `latency_ms` per `(batch, prefill_tokens,
  kv_tokens)` is the sum over every layer and every resource. Fitting a per-primitive
  coefficient against it is under-determined: many assignments across the primitives
  produce the same total, so the fit would attribute to one term whatever every other
  term gets wrong. That is not a provenance rule, it is an identifiability fact, and it
  does not change with the dataset.

So FPM **may** fit a term that is itself about composition — the overlap-band choice
(`docs/band-selection.md`), the attention-DP funnel factor, a per-step host constant —
because those are properties of the whole forward pass and FPM measures the whole forward
pass. It **may not** fit an attention rate or a GEMM asymptote, because nothing in a
whole-pass total identifies those separately.

Three obligations on any FPM-fitted term, so the looser rule does not become a looser
standard:

1. **Hold out along a dimension the term must generalise across** — a chip, a model, or a
   parallelism — and report train and holdout error separately. §4 lists the splits the
   dataset now supports.
2. **Cite FPM explicitly**, by dataset revision SHA, since the dataset is rewritten and
   has no tags.
3. **Keep the end-to-end check.** FPM is not the evaluation corpus, so a term fitted on it
   still has to face InferenceX, and the standing rule below still governs.

No committed coefficient cites FPM today. This records what is permitted, so the next
composed term does not have to argue the rule from scratch.

`scripts/select_overlap_band.py` enforces the last two by refusing any path under an
`inferencex` or `semianalysis` directory, and a new fitter should import that guard rather
than restate it.

FPM is entirely vLLM — all 85,484 rows carry `backend: vllm` — and splits as 12,522 pure
decode, 5,226 pure prefill, 67,736 mixed prefill+decode.

**The constraint that governs everything else: a better fit to a component benchmark can
be a worse model.** Six independently verified corrections each improved a per-primitive
or per-step measurement and made the end-to-end score worse, because the kernel's accuracy
rests on partially cancelling errors. So **no per-primitive fit ships without an
end-to-end check**, and a correction whose offsetting term is unidentified waits.

The sixth is the clearest case and is worth stating in full, because it shows the rule
catching something that every other gate passed. `attention_decode_floor_mla` was fitted
from AISimulate's `mla_generation_module_perf` tables, committed as `method: measured,
fitted: true`, and the fit was clean — it reproduced from its own writer, satisfied the
schema, and sat inside every physical bound then checked. It was also wrong about *which
quantity it measured*: those tables are MODULE measurements covering the whole MLA block
including its down-projections, and blis-catalog prices `qkv_proj` and `o_proj` as separate
`GEMM` nodes in the same layer, so the kernel charged the projection weight read twice.
`scripts/fit_attention_mla.py` had already recorded the arithmetic (DeepSeek-V3's
projections are 293.6 MB at fp8, 61.2 µs at H200's 4.80 TB/s, against a 44.0–63.8 µs
measured module floor) as its reason for being rejected; the pair was nonetheless committed
later from the same tables.

Only the end-to-end check found it. Scored as a 2×2 over the two terms on the InferenceX
corpus (573 points, vLLM lane):

| floor | rate | overall TPOT mean\|e\| | kimi-k2.5 TPOT |
|---|---|---|---|
| part-wide | fitted | **14.98%** | **6.38%** |
| part-wide | part-wide | 14.90% | 6.91% |
| fitted (module) | part-wide | 17.32% | 15.38% |
| fitted (module) | fitted | 17.43% | 14.57% |

The **rate** is sound — 0.80–1.13× of each part's part-wide rate, the right order for a
latent-cache read, and the only arm that improves kimi-k2.5 on its own. The **floor** was
the entire regression: either arm carrying it lands near 15% on kimi-k2.5 and 17.3%
overall, flipping the kernel from beating AISimulate's 15.39% to losing to it.
`scripts/correct_mla_floor.py` commits the floor as `assumed` from the same part's measured
attention-kernel floor, and three gates now hold it: that script's `--check`,
`relane_attention_mla.py --check-rate` for the half still fitted, and a property test
bounding any kind-specific floor at 3× its part-wide sibling.

The generalizable lesson is not "distrust module tables" but **a fit can be clean and still
answer a different question than the coefficient asks**. Provenance of the number is not
provenance of the quantity, and only a check at the level the model is used at can tell
them apart.

## 4. Train, validate, evaluate

Fitting splits are taken along a dimension generalization actually has to cross, not at
random — a random row split puts near-duplicate shapes on both sides and reports a
generalization that was never tested.

* **GEMM shape ramp**: whole `k` VALUES held out, so a test shape has a reduction depth
  the fit never saw. Train and test agree to within 0.04x on every part and dtype.
* **FPM topology**: MiniMax-M2.7 on h200_sxm spans six topologies (`dep/tp1/ep2`,
  `dep/tp1/ep4`, `pure_tp/tp2/ep1`, `pure_tp/tp4/ep1`, `tep/tp2/ep2`, `tep/tp4/ep4`), so a
  parallelism-dependent term can be validated with model and chip fixed.
* **FPM regime**: fit on pure-prefill and pure-decode rows, validate on the 67,736 mixed
  rows, whose batch composition varies independently of concurrency.

**A confound that WAS true and no longer is**: an earlier revision recorded model and
system as perfectly collinear in FPM — DeepSeek-V4-Pro only on gb300, GLM-5.2 only on
b200_sxm, MiniMax-M2.7 only on h200_sxm — so a model holdout was also a chip holdout. The
dataset has since grown and that is false: DeepSeek-V4.1-Flash spans five systems,
DeepSeek-V4-Pro three, MiniMax-M2.7 and GLM-5.2-NVFP4 two each, and b200-sxm carries six
models. Both axes now vary independently, which is what makes the holdouts above
attributable and what §3 relies on in permitting FPM to fit a composed term.

The confound is recorded rather than deleted because the figures it produced are still in
this document's history, and a reader comparing revisions needs to know which claim held
when. Re-derive the current state from the catalog index rather than trusting either
version: the dataset is rewritten in place.

Errors are reported for train, validation AND evaluation. A fit quoted only on its
training data is not a result.

## 5. Per-part provenance

> Counts in this section were taken when it was written. The current counts, generated
> from `coefficients/` on every documentation build, are on the
> [reference pages](reference/index.md).

Coefficient counts by citation lane:

| chip | total | nccl | vllm | trtllm | sglang | other |
|---|---|---|---|---|---|---|
| h200 | 100 | 63 @ 2.29.2 | 13 @ 0.24.0/0.25.0 | 10 @ 1.3.0rc20 | — | 14 |
| h100 | 104 | 63 @ 2.29.2 | 13 @ 0.24.0/0.25.0 | 12 @ 1.3.0rc20 | 2 @ 0.5.16 | 14 |
| b200 | 102 | 63 @ 2.29.2 | 13 @ 0.24.0/0.25.0 | 12 @ 1.3.0rc20 | — | 14 |
| b300 | 102 | 63 @ 2.29.2 | 13 @ 0.24.0/0.25.0 | 12 @ 1.3.0rc20 | — | 14 |
| gb200-nvl72 | 78 | 48 @ 2.29.2 | 4 @ 0.25.0 | 12 @ 1.3.0rc20 | — | 14 |
| l40s | 100 | 72 @ 2.27.3 | 4 @ 0.24.0 | 10 @ 1.3.0rc20 | — | 14 |
| a100-sxm | 94 | 63 @ 2.27.3 | 13 @ 0.14.0 | — | 4 @ 0.5.10 | 14 |
| a100-80 | 174 | 126 @ 2.27.3 | 22 @ 0.14.0 | — | 4 @ 0.5.10 | 22 |

`other` is the non-AISimulate set: host overheads, HBM derate, communicator sizes.

### Version pins, stated rather than averaged away

A100's vLLM collection is **0.14.0** and L40S's is **0.24.0**, against 0.25.0 on the
Hopper and Blackwell parts. Those are the newest vLLM collections AISimulate ships for
those SKUs. Pinning one collection per part keeps a difference between parts attributable
to silicon; it also means A100's coefficients describe an older engine, and that is a
property of the data rather than a choice.

### A100-80 is an alias

`a100-80` and `a100-sxm` are the same silicon — identical `TFlopsPeak` 312,
`BwPeakTBs` 2.039, `MemoryGiB` 80, `SMCount` 108 — and the catalog's own chip file says
"Alias for A100-SXM". Both names appear in `hardware_config.json`, so a BLIS run can
request either.

Aliasing is done two ways, deliberately:

* Where the coefficient NAME has no chip suffix, `a100-80` is added to `a100-sxm`'s
  **scope**, so one measurement serves both names and a refit cannot update one and miss
  the other. The kernel matches scope by membership (`resolve.Scope.Admits`).
* The collective names DO carry a chip suffix, because the kernel builds the key as
  `collective_<kind>_<op>_<dtype>_<rank>rank_<chip>` with `-` replaced by `_`. A
  membership scope cannot satisfy that, so 72 measured `*_a100_80` entries carry
  A100-SXM's values with the alias stated in each rationale (24 more are the derived
  16-rank widths). **Re-fit both together.** Every `*_a100_sxm` entry is scoped to
  `a100-sxm` alone: an `a100-80` deployment builds the `_a100_80` name and never looks
  them up, so the 72 that listed it in their scope said nothing.
* The descriptor entries (`hbm_derate`, `host_link_bandwidth`, the NCCL buffer sizes,
  `engine_workspace_bytes`) come from `systems/a100_sxm.yaml` alone, scoped to both names.
  An earlier revision also emitted `systems/a100_pcie.yaml` for `a100-80`, which put two
  entries of every descriptor name in scope for it; the values were equal, so nothing was
  mispriced, but the kernel kept whichever came later and a refit of one copy would have
  moved the answer silently.

### Deliberate absences

Absent is not the same as missing:

* `gemm_*_fp8` and `gemm_*_fp8_block` on A100 — the part has `TFlopsFP8: 0`, so
  `dtypeFit` returns the `bf16` suffix. A fraction of a rate the silicon does not have
  would be worse than nothing.
* `attention_decode_*_swa` on A100 — vLLM 0.14.0's schema has no `window_size` column;
  the sliding-window sweep postdates it. The kernel falls back to the unsuffixed pair.
* int8 all-reduce on A100 stays on NCCL — `allreduce_dtype` in the custom_allreduce sweep
  is bfloat16 across all 138 rows, so there is no vLLM int8 measurement. Unmeasured is not
  unchanged.
* L40S sliding-window decode is on the vLLM lane (`vllm/0.24.0`, its newest vLLM attention
  collection) **by decision, at a measured cost**. Commit 5be809b moved it for lane
  correctness — the alternative sums two engines in one step for every windowed model on
  the part — knowing its held-out folds favour TRT-LLM (§7.1: 2.073x against 1.681x, 0 of
  2 folds won). The trade is recorded here rather than reverted. An earlier revision of
  this bullet said it stayed on TRT-LLM; the committed entries never did after 5be809b.
* L40S all-reduce stays on NCCL — the vLLM-graph floors agree with NCCL to within 13%,
  because L40S has no NVLink (PCIe Gen4, 32 GB/s) and `custom_allreduce` needs peer-to-peer
  NVLink to help. A100-SXM has NVLink 3 at 300 GB/s and its floors drop 2.1-3.8x, which is
  the same mechanism that moved the Hopper and Blackwell parts.

## 6. Reproducing any value

Every fitted coefficient names its script in the rationale. The fitters are pure: given
the same collection they print the same numbers, and the validator re-runs them.

    scripts/fit_attention.py <data>/<sku>/attention/<lane>          --chip <chip>
    scripts/fit_attention_prefill.py --collection <lane> <sku>:<chip>
    scripts/fit_attention_by_kind.py --collection <lane> --sku <sku> --chip <chip>
    scripts/fit_collectives_vllm.py <data>/<sku>/comm/vllm/<version>
    scripts/fit_collectives.py <data>/<sku>/comm/nccl/<version>
    scripts/fit_gemm_envelope.py <data>/<sku>/gemm/<lane>
    scripts/fit_gemm_shape_ramp.py [--emit] <sku>:<chip>
    scripts/probe_cudagraph_dispatch.py --settings S --corpus C
    scripts/emit_memory.py [--check]
    scripts/mine_vllm_cudagraph_memory.py {fetch,extract} CACHE_DIR

Two traps these scripts exist to avoid, both of which produced wrong answers by hand:

* **`latency` is MILLISECONDS** in every AISimulate parquet. Reading it as seconds gives
  2,070 TB/s on a 4.8 TB/s part, or a 143,314 GB/s collective peak. The error announces
  itself if you check the number against the datasheet, and not otherwise.
* **FPM's `total_kv_read_tokens` is summed OVER THE BATCH**, while the kernel takes a
  per-sequence context. Joining them directly is a 256x KV error at batch 256; the join is
  on `(batch, batch*ctx == kv_total)`.

### 6.0 What is reproducible, and what is not

Five kinds of value live in this registry. Counted from the coefficient files
themselves (1,143 entries in six sets):

| `method` | entries | reproducible by |
|---|---|---|
| `measured` | 844 | a named fitter in `scripts/`, re-run by the validator |
| `vendor_spec` | 57 | NVIDIA's own descriptor or simulator source, transcribed by a writer with `--check` |
| `assumed` | 239 | a derivation script where one exists; otherwise nothing — each states its reasoning and what would replace it |
| `copied` | 1 | nothing to reproduce: another scope's value carried, `copied_from` naming it -- an assumed transfer, not a derivation (DeepSeek-V4.1's `host_launch_eager_per_kernel` off h200, §10.6) |
| `not_charged` | 2 | nothing to reproduce: a declared zero (`cudagraph_capture_bytes_none`; DeepSeek-V4.1's `host_admission_per_request`, §10.6, an inferred zero the schema requires to carry this method) |

The 239 `assumed` entries are of four different kinds, and the difference matters
to a reader weighing one:

* **Derived from this project's own measurements** — 212 entries, each
  regenerated by a script. 204 collective triples close rank-width and part gaps
  (`extrapolate_collective_width.py`, `extrapolate_by_generation.py`, each
  holdout-validated, §8.5); A100's two sliding-window attention entries scale
  its own measured full-attention floor and rate by its kernel sibling's
  windowed/full ratio (`extrapolate_by_generation.py`); six `attention_decode_floor_mla` entries are set from
  the part's own measured attention floor (`correct_mla_floor.py`, §3).
* **Derived from committed siblings** — 8 entries, all `fit_deepseek_v41_h200.py` (§10.5),
  four per part on h200 and h100, each from that part's own siblings and descriptor:
  the indexer-scan rate at NVIDIA's empirical HBM rate (BwPeak × `hbm_derate`, not a
  bound; at most 1.25x pessimistic, its optimism unbounded), the weight-only MXFP8 ramp at the measured BF16 envelope
  (an upper bound on the asymptote; the knee is not a bound), and the engram host-gather
  rate at the PCIe 5.0 line rate (an upper bound). Each states its direction where one is
  known, and each is re-derived from its siblings by a property test that needs no sweep.
* **Fitted on a neighbouring kernel's measurements** — 6 entries, all
  `fit_deepseek_v41_prefill.py` (§10.6): V4.1's prefill indexer, fitted on the paged
  logits kernel's prefill-shaped rows because vLLM runs the ragged kernel there, which is
  measured only on the SGLang lane, with the lane decision open (§10.6 states the gap).
* **Not derivable from any dataset this project reads** — 13 entries. The seven
  host overheads in `cost-model-host-overheads.yaml`: no dataset measures
  client-observed host time (AISimulate times kernels, FPM times one synchronized
  forward pass — all 42 of its columns were enumerated, and there is no `ttft`,
  `e2e`, `client`, `queue` or `host` column — and InferenceX is evaluation-only).
  And the four capturing `cudagraph_capture_bytes_<mode>` entries in
  `cost-model-memory.yaml`, whose magnitude is anchored on vLLM's own start-up
  logs (§11); their anchor is reproducible from a committed CSV, but the samples
  are uncontrolled deployments, which is why the method stays `assumed`. And
  DeepSeek-V4.1's `host_launch_eager_per_kernel`, backed out of a vendored end-to-end
  enforce-eager anchor (`fit_deepseek_v41_host.py`, §10.6), and its
  `host_admission_per_token`, a front-end timing taken on an Apple M1 Max and transferred to
  every part, because no serving node's CPU was measured (§10.6).

Each of those thirteen carries the measurement that would replace it.
`host_admission_per_request` names the exact experiment, a client-observed TTFT
measurement at 1-token and 1,024-token prompts against a running vLLM server at
concurrency 1 with a warm cache, whose intercept is the coefficient with no
evaluation data involved. The capture entries name one log line from one
start-up of the target deployment.

This boundary is stated rather than smoothed over because the largest single
accuracy movement in this work — TTFT mape 52.94% to 31.47% on the measured
tier — comes from one of the seven.

### 6.1 Reproducing the end-to-end evaluation tables

Reproducing a coefficient needs only a fitter and its collection. Reproducing an
accuracy FIGURE needs four things pinned, and each of them has silently produced a
wrong number during this work:

| what | why it matters |
|---|---|
| the scorer checkout | `cmd/metricscore` is on the `kernel-exclusive` branch of the `sriumcp/inference-sim` fork (48ca63f at the time of writing). `inference-sim/inference-sim` carries neither that branch nor `cmd/metricscore`, and a checkout of its `main` fails with `stat cmd/metricscore: directory not found` |
| `-registry` | defaults to the sibling `blis-registry` checkout, so scoring a branch without passing it explicitly scores a different registry |
| `-catalog` | the figures move with the catalog revision; a comparison across catalog revisions once produced an apparent 0.03pp "effect" that was two catalog commits |
| `-config-tier`, `-framework`, `-length-range-ratio` | each changes every figure. On one kernel revision the same command with and without `-length-range-ratio 1.0` gives 11.62/12.05/15.23/52.94 against 11.22/12.19/16.22/57.13 |

`scripts/compare_registries.py` pins all four by construction and prints the
revisions it used. Beyond comparing two registries it records and checks absolute
figures:

    # write the current figures, with revisions and flags, to docs/evaluation-baseline.json
    python scripts/compare_registries.py --record --scorer /path/to/kernel-exclusive/checkout

    # fail if any figure, or any n, has moved since
    python scripts/compare_registries.py --check  --scorer /path/to/kernel-exclusive/checkout

`docs/evaluation-baseline.json` is the machine-checkable record. It holds the four
headline figures and their point counts per tier, plus the framework, the
length-range-ratio and the scorer and catalog revisions, because a figure is a
claim about that whole triple rather than about the registry alone. `--check`
compares the revisions too and says when they have moved, so a drift is
attributable rather than merely visible.

A figure quoted without its flags cannot be checked, and this is the reason every
table in the companion documents carries its invocation.

### 6.1.1 The scorer's kernel is pinned, and the pin is not this registry's concern

One asymmetry in the reproduce recipe is worth stating because it looks like an
oversight and is not. The scoring branch clones the catalog and the registry at
release tags (`blis-catalog` `0.2.1`, `blis-registry` `v0.1.1` at 48ca63f), because both
are read at RUN TIME by path and the scorer takes `-catalog` and `-registry` flags. It
does not clone `blis-latency-kernel` at all: the kernel is a COMPILED dependency pinned
by module version in `go.mod`. An earlier revision of this section said the catalog and
registry were cloned with `-b modeling`; none of the three repositories has a
`modeling` branch.

The consequence for a reader: changing this registry changes the figures
immediately, while changing the kernel does not until its pin is advanced. At the
time of writing the scoring branch pins blis-latency-kernel `v0.1.0` and
blis-schemas `v0.2.2`. A figure recorded by
`compare_registries.py --record` therefore states the scorer revision alongside
the registry, and `--check` reports when it has moved, because a figure is a claim
about the pair.

A local `replace` directive pointing at on-disk kernel checkouts is how the
scoring branch's own plan describes linking the two, and it is explicitly
worktree-only: committing one would break the build for anyone following the
clone recipe.

### 6.2 Which tier to weigh

`inferencex_engine_settings.json` records **68 sweeps with a captured engine-args
log and 136 without**, each of the latter listed in its `incomplete` array with
`reason: "no engine-args log"`. For those 136 the engine configuration is resolved
from vLLM's defaults rather than observed, so an error there is as likely to be a
wrong assumed configuration as a wrong model.

The measured tier is the only one where the simulated deployment is ground truth.
It is therefore the tier on which a coefficient change can be attributed to the
model, and the tier to weigh when judging one. The resolved tier answers a weaker
and still useful question — how well a predictor does when it must guess the
configuration too — and mixing the two in one average mixes two kinds of
evidence, which is why the scorer takes `-config-tier` and the baseline records
them separately.

### 6.3 Validation gates

`pytest validator/` runs the registry's **derivation / value-preservation** gates — schema
and provenance validation now live in blis-schemas (the Go validator every consumer loads
through; see the registry README), not here. The derivation gates that matter:

* **independent re-derivation** — re-runs the fitter and compares to the registry, so a
  value edited without its citation fails.
* **single-sourced citations** — every source in the AISimulate sets must cite AISimulate.
* **lane pinning** — the all-reduce test pins the `vllm_graph` lane specifically, because
  eager is SLOWER than NCCL and accepting either would accept a 3.4x mispricing.
* **memory occupancy** — `emit_memory.py --check` re-renders `cost-model-memory.yaml` byte for
  byte when the AISimulate tree is present; its capture half re-derives from the committed
  CSV in CI with no tree at all, and property tests hold every vLLM `cudagraph_mode` to an
  entry, no capturing mode above `FULL_AND_PIECEWISE`, and the activation multiple
  non-increasing with width and at least as large for MoE as for dense (§11).
* **skips are explicit** — a gate that cannot run says why. The three-factor GEMM ramp's
  gate skips while the registry carries the one-factor form, and activates the moment the
  coefficients land.

## 7. Held-out validation of the lane decisions

§4 states the splitting principle. This section records the folds actually run, the
numbers they produced, and the decisions they support or refuse. Every figure here is
reproduced by the script named beside it; none is a residual on the rows that were fitted.

### 7.1 Sliding-window attention: hold out one window width

`scripts/holdout_attention_swa.py`. The window is the dimension that defines the kind —
`kv_bytes` reads `min(context, window)` positions — so a held-out width is a byte count the
fit never saw. Reported as the median over folds, because the widths are heavily unbalanced
(window 128 carries 82% of h200's rows, so that fold trains on an eighth of the sweep while
the others hold out 3-6%); a mean over folds this uneven would describe none of them.

| part | vLLM collection | vLLM median held-out | TRT-LLM median held-out | shared folds won by vLLM | decision |
|---|---|---|---|---|---|
| h200 | vllm/0.25.0 | **1.457x** | 1.754x | 3/3 | relane |
| h100 | vllm/0.25.0 | **1.433x** | 1.787x | 2/2 | relane |
| b200 | vllm/0.25.0 | **1.344x** | 1.475x | 2/3 | relane |
| b300 | vllm/0.25.0 | **1.317x** | 1.498x | 2/2 | relane |
| gb200-nvl72 | vllm/0.25.0 | **1.276x** | 1.358x | 3/3 | relane |
| l40s | vllm/0.24.0 | 2.073x | **1.681x** | 0/2 | **relane, at a stated cost** (5be809b) |

Twelve of thirteen shared folds favour vLLM on the five relaned parts; the single exception
is b200 at window 2048 (1.259x against 1.244x). The vLLM sweep also covers five widths
where TRT-LLM covers two or three, so the TRT-LLM fit was never tested at 512 or 1024 on
any part.

L40S is the documented exception and has held-out evidence rather than a residual: its
vLLM fit is worse on BOTH folds. It was relaned anyway, deliberately, in 5be809b: the
decision rule there was lane correctness — one engine per predicted step — with the fit
penalty recorded rather than hidden, and `relane_attention_swa.py --check` now maintains
the L40S and GB300 pairs alongside the five 0.25.0 parts. An earlier revision of this
paragraph and of that script's header said L40S stayed on TRT-LLM. Note that L40S has no `vllm/0.25.0` attention collection
at all — `vllm/0.24.0` is its newest — so the comparison uses that, and an earlier reading
of "no vLLM data" for this part was wrong.

### 7.2 KDA: hold out the head-count geometry

`scripts/fit_recurrent.py --group-by kernel+heads`. vLLM's KDA sweep varies `num_k_heads`
over 12/24/48/96 where SGLang's holds it at 12. Pooling those four geometries into one fit
gives geo-err **1.733x**; fitted per geometry the same rows give 1.169x to 1.196x. The
pooled figure is an artefact of the pooling, not a property of the lane:

| lane | num_k_heads | n | floor | rate | geo-err |
|---|---|---|---|---|---|
| sglang/0.5.16 (committed) | 12 | 11 | 5.1 us | 1.50 tok/us | **1.160x** |
| vllm/0.1.dev19262 | 12 | 11 | 5.4 us | 1.75 tok/us | **1.169x** |
| vllm/0.1.dev19262 | 24 | 11 | 5.3 us | 1.00 tok/us | 1.196x |
| vllm/0.1.dev19262 | 48 | 11 | 3.6 us | 0.50 tok/us | 1.191x |
| vllm/0.1.dev19262 | 96 | 11 | 3.6 us | 0.25 tok/us | 1.188x |

At the matching geometry — `num_k_heads: 12`, which is what SGLang measures — the two lanes
are **equivalent** (1.160x against 1.169x, a difference of 0.009). An earlier claim that the
vLLM lane fits 1.49x worse was wrong, and wrong because of the pooling above.

This does not by itself relane KDA. The vLLM collection is a dev build
(`0.1.dev19262+gb6bbf29dd`), Kimi-K3 appears in no InferenceX tier and in no FPM artifact
with a KDA node, so the change is unevaluable end to end and the decision rests on fit
quality alone, where the two lanes tie.

### 7.3 GEMM ramp: hold out whole K values, and the largest M values

`scripts/holdout_gemm_ramp.py`. Two folds, and the second one found a defect in the FORM
rather than in the lane.

**The ramp is the wrong shape above M ~ 4096.** `eff(M) = eps_max * M / (M + M_half)` is
monotone and saturating, but the measured envelope PEAKS mid-sweep and then declines — on
h200 bfloat16, 0.956 at M=3329 falling to 0.875 at M=32768 on the vLLM lane, and 0.870 at
M=4097 falling to 0.839 on TRT-LLM. Both lanes show it, so it is a property of the kernel
and not of either engine. Holding out the three largest M values and extrapolating:

| held-out M | predicted | measured | error |
|---|---|---|---|
| 8192 | 0.987 | 0.930 | +0.057 |
| 16384 | 0.993 | 0.911 | +0.082 |
| 32768 | 0.997 | 0.875 | **+0.121** |

So `eps_max` is fitted to an asymptote the kernel never reaches, and a fit that has not seen
large M over-predicts there by up to 12 points of efficiency.

**Why this is recorded and not fixed.** A decode step's M is the token count in the batch,
and the corpus runs M = 1 to 2048 with 97.2% of points (1,036 of 1,066) at M <= 128. The decline begins above
M ~ 4096, outside the range any evaluated deployment reaches. Scoring the same fit inside
and outside the operating range:

| lane | range | n | rms | mean signed |
|---|---|---|---|---|
| vllm/0.25.0 | M <= 128 (97% of corpus steps) | 30 | 0.0585 | +0.0509 |
| vllm/0.25.0 | M <= 2048 (all corpus steps) | 52 | 0.0745 | +0.0009 |
| trtllm/1.3.0rc20 | M <= 128 | 30 | **0.0474** | **+0.0393** |
| trtllm/1.3.0rc20 | M <= 2048 | 52 | **0.0538** | +0.0040 |

**This refuses a vLLM-lane GEMM relane.** In the range that matters TRT-LLM fits better on
both measures, which is consistent with §2's finding that the lanes agree to 0.4-3.6% on
shared shapes — the same cuBLAS/CUTLASS kernels underneath — so the larger TRT-LLM sweep
wins on determination rather than on engine. The lane rule is "fit the engine you predict"
only where the engine changes the kernel, and for GEMM it does not.

Both lanes over-predict efficiency at small M (+0.039 and +0.051), which UNDER-predicts
decode latency on the shapes that dominate the corpus. That is a real, quantified direction
for future work, and it is a form problem: the fix is a non-monotone ramp or a small-M
correction term, not a different collection.

### 7.4 What FPM could and could not adjudicate

FPM is the validation source in §3, but its reach is narrower than the coefficient set:

* **SWA**: unusable. No FPM model carries a sliding-window node. Four of the six models in
  the FPM catalog have a graph in blis-catalog — MiniMax-M2.7 (`gqa`), GLM-5.2
  (`gqa`, `sparse_mla`), DeepSeek-V4-Pro (`gqa`, `mla`, `sparse_mla`) and Kimi-K3
  (`kda`, `mla`) — and none declares `kind: swa`; the other two
  (DeepSeek-V4.1-Flash, DeepSeek-V4-Flash-0731) have no catalog graph, so the kernel cannot
  price them either way. The relane in §7.1 therefore rests on held-out AISimulate folds
  plus provenance, with no FPM arbiter, and that is weaker evidence than a GEMM change
  would carry.
* **KDA**: unusable for the same reason; Kimi-K3's FPM artifacts carry no KDA-node forward
  pass in the local snapshot.
* **GEMM**: usable in principle — each of the four graphed FPM models carries dense `GEMM`
  nodes (3 on MiniMax-M2.7, 9 on GLM-5.2, 18 on DeepSeek-V4-Pro, 17 on Kimi-K3) — but the
  AISimulate K and M folds above already refuse the relane on the shapes the corpus runs,
  so no FPM run was needed to decide it. An FPM fold remains the right arbiter for a change
  to the ramp's FORM, which §7.3 identifies and does not attempt.

## 8. Lane provenance: every coefficient family, and every remaining borrow

The lane rule in §2 is applied per operator. This section is the resulting census, so a
reader can tell at a glance which engine's measurements each coefficient rests on, and
which four are borrowed because no alternative exists.

### 8.1 The census

> Counts in this section were taken when it was written. The current counts, generated
> from `coefficients/` on every documentation build, are on the
> [reference pages](reference/index.md).

| lane | n | share | what it is |
|---|---|---|---|
| `nccl` | 498 | 68.8% | all_gather, reduce_scatter, all_to_all, and the int8 all-reduce |
| `vllm` | 152 | 21.0% | attention, GEMM, MoE imbalance, KDA, and the fp16 all-reduce |
| descriptor | 64 | 8.8% | NVIDIA's own `systems/<sku>.yaml` — HBM derate, PCIe, NCCL buffers (p2p and the memory-access constant then; both since retired) |
| none | 6 | 0.8% | host overheads, all `method: assumed` |
| `trtllm` | 2 | 0.3% | mamba2 — **borrow** |
| `sglang` | 2 | 0.3% | A100 MoE imbalance — **borrow** |

**NCCL is not a borrow.** It is vLLM's own path for every collective except all-reduce:
`CustomAllreduce` implements all-reduce only, and `should_custom_all_gather` /
`should_custom_reduce_scatter` gate and fall through. AISimulate ships no vLLM data for
those operators because there is nothing to measure. Of the 654 data-fitted coefficients,
**650 (99.4%) rest on vLLM's actual execution path** — 152 vLLM-lane plus 498 NCCL. The
remaining four are the borrows in §8.3.

*Since this census* (#36, §10.5): the `moe_routing_imbalance_p90` family is retired, so the
A100 borrow is one entry and the vLLM lane loses seven; the descriptor rows are 48 now that
`hbm_constant_latency`, `p2p_latency` and the a100_pcie duplicates are gone; and the vLLM
lane gains ten entries (the Marlin ramp on h200, h100 and gb300, the V4.1 sparse rate and
indexer floor on h200 and h100).
The reference pages carry the current counts.

### 8.2 Per-family detail

| set | family | n | lanes |
|---|---|---|---|
| attention | `attention_decode_{floor,rate}` | 14 | vllm 0.25.0 ×10, 0.24.0 ×2, 0.14.0 ×2 |
| attention | `attention_decode_*_swa` | 12 | vllm 0.25.0 ×10, 0.24.0 ×2 |
| attention | `attention_prefill_{floor,work_scale}` | 14 | vllm 0.25.0 ×10, 0.24.0 ×2, 0.14.0 ×2 |
| collectives | all-reduce fp16 (floor, peak, transition) | 69 | **vllm** 0.24.0 ×36, 0.14.0 ×18, nccl ×15 |
| collectives | every other collective | 429 | nccl 2.29.2 / 2.27.3 |
| primitives | `gemm_{eps_max,m_half}` | 52 | vllm 0.27.1 ×32, 0.25.0 ×12, 0.24.0 ×6, 0.14.0 ×2 |
| primitives | `moe_routing_imbalance_median` | 8 | vllm ×7, **sglang 0.5.10 ×1 (A100)** — the p90 family was retired in #36 |
| primitives | `gemm_{eps_max,m_half}_w4a16_mxfp4_grouped` | 6 | vllm 0.25.0 ×6 (h200, h100, gb300), §10.5 |
| primitives | `gemm_{eps_max,m_half}_w8a16_mxfp8`, `host_gather_rate_uva` | 6 | `assumed` bounds (h200, h100), §10.5 |
| primitives | descriptors | 48 | AISimulate `systems/<sku>.yaml` |
| attention | `attention_decode_rate_sparse_mla`, `attention_decode_*_block_index_scores` | 6 | vllm 0.25.0 ×4 + 2 `assumed` (h200, h100; deepseek-v4.1-flash), §10.5 |
| attention | `attention_decode_{launch_floor,split_floor,serial_rate,per_token,throughput_rate}_sparse_mla` | 10 | vllm 0.25.0 ×10 (h200, h100; deepseek-v4.1-flash), §10.7 |
| attention | `attention_prefill_*_sparse_mla`, `attention_prefill_*_block_index_scores` | 14 | vllm 0.25.0 ×8 + 6 `assumed` proxy (h200, h100; deepseek-v4.1-flash), §10.6 |
| primitives | `mhc_{floor,rate}_{pre,post}` | 8 | vllm 0.25.0 ×8 (h200, h100; deepseek-v4.1-flash), §10.6 |
| recurrent | `recurrent_decode_*_kda` | 6 | vllm 0.1.dev19262 |
| recurrent | `recurrent_decode_*_mamba2` | 2 | **trtllm 1.3.0rc20** |
| host-overheads | all part-wide | 7 | none — `method: assumed` |
| host-overheads | deepseek-v4.1-flash | 4 | eager launch `assumed` on h200 (end-to-end anchor) and `copied` to the other eight parts (an assumed transfer), admission per token `assumed` (a front-end timing on an Apple M1 Max, transferred), per request `not_charged` 0.0 (an inference from what was timed, bounded end to end; blis-schemas requires `not_charged` for a zero), §10.6 |

Regenerate this table rather than editing it: every count above is derivable from
`coefficients/` with a group-by on the family stem, and a retyped count is the defect
class this registry has caught most often.

Where a part has no collection at the newest version, it uses the newest it has: L40S's
attention and GEMM come from `vllm/0.24.0` because 0.25.0 does not cover it, and A100's
from `vllm/0.14.0`. The citation on each entry names the collection it was fitted on.

### 8.3 The four borrows, and why each is unavoidable

**mamba2 (2 entries, trtllm/1.3.0rc20).** `mamba2_perf.parquet` exists on the TRT-LLM lane
and nowhere else — checked across all eight SKUs in the tree. The vLLM `linear_attention`
collections carry `gdn_perf.parquet` only. A strict vLLM-only registry could not price
Nemotron-3 hybrids at all. The entries remain a documented LOWER BOUND regardless of lane:
the sweep has only `causal_conv1d_fn` and `causal_conv1d_update`, so the selective-scan
kernel that does the rest of a Mamba2 layer is in no collection.

**A100 MoE imbalance (1 entry since the p90 family was retired; 2 when written, sglang/0.5.10).** A100's only vLLM MoE collection
(`vllm/0.14.0`) contains `power_law_1.01` and `power_law_1.2` rows and **no `balanced`
rows at all**. The coefficient is a ratio of skewed to balanced latency at identical
shape, so with no balanced measurement there is no ratio to take. SGLang's sweep carries
all three distributions. A100's GEMM entries DID move to the vLLM lane — that sweep is
usable — so this is the narrowest possible borrow.

### 8.4 What validates what

The three-level separation in §3 assigns FPM the validation role, but FPM's reach is
narrower than the coefficient set and it is worth being exact about where it applies.

FPM's 59 records are 41 vLLM and 18 SGLang, across five GPU families: B200 (17),
GB300 (21), H200 (13), GB200 (4), B300_SXM (4). So:

* **GEMM and MoE**: validatable. Every FPM model carries dense `GEMM` nodes, and the MoE
  models carry `GroupedGEMM`, on chips the catalog prices.
* **Attention, full and windowed**: GQA and MLA are validatable; **sliding window is not**
  — no FPM model declares a `kind: swa` node. The SWA relane therefore rests on held-out
  AISimulate folds (§7.1) plus lane correctness, with no FPM arbiter.
* **KDA**: the data exists and is now reachable, but one gap remains and it is not the
  one previously recorded here. Four Kimi-K3 artifacts are all vLLM and all on **GB300**,
  two of them on `vllm/0.1.dev19262` — the same collection the kernel fit used, which is
  the cleanest possible pairing. An earlier revision of this section said the catalog had
  no `gb300` chip; it has carried one since blis-catalog#23, and this registry now carries
  102 GB300 coefficients including the KDA pair. The dataset's own
  `catalog/index.json` lists both artifacts as `aic_fpm_forward_perf` with
  `phases: [decode, prefill]`, 5,266 and 4,874 rows at `tep8`.

  What blocks the scoring now is a **rank width**, not a chip. The `tep8` artifact is
  tp=8, moe_ep=8, and NCCL sweeps GB300 at 2 and 4 ranks only because a Grace-Blackwell
  tray is four GPUs — so `all_gather`, `reduce_scatter` and `alltoall` carry
  `method: assumed` 8-rank floors (§8.5) rather than measurements. All-reduce IS measured
  at 8 and 16 ranks, from vLLM's own custom kernel. The second artifact,
  `vllm/0.29.0/tp8-dcp8`, is `moe_ep=1` / `pure_tp`, so it exercises no expert
  all-to-all and is the cleaner first cell to score.

  Which MoE kernel that run used matters, and it is not stock v0.31.0's. The FPM metadata
  for the `0.1.dev19262/tep8` leaves records `moe_quant_mode: w4a16_mxfp4` as the
  effective runtime identity (`fpm_identity_correction`, policy
  `runtime-dtype-annotation-v1`): the Kimi-K3 preview build ran its MXFP4 experts on
  MarlinExperts W4A16, because its CUTLASS path rejected SiTU (AISimulate
  `collector/vllm/collect_moe.py`). Stock v0.31.0 would instead select
  `CutlassExpertsMxfp4` (W4A4) on SM100. So scoring that run needs the kernel to price
  Marlin W4A16 experts at the BF16 rate class, which is what
  `gemm_*_w4a16_mxfp4_grouped` on gb300 measures (§10.5). blis-latency-kernel#44 selects
  ramps by weight format and chip, which on gb300 is the FP4 class -- right for stock
  v0.31.0 -- so this entry stays unread until a deployment can state that its MoE backend
  is Marlin. The scoring attempt recorded on
  #25 also over-predicted by ~2x with the FASTER FP4 rate class, and making the KDA read
  free left that unchanged, so the MoE/GEMM composition is a second blocker. #41 carries
  the validation forward.

  Note also that FPM carries no per-kernel or per-node column at all — the truth
  parquets hold one `latency_ms` per `(batch_size, total_prefill_tokens,
  total_kv_read_tokens)`. "No KDA-node forward pass" describes something FPM never had
  for any model; the right question is only whether a Kimi-K3 whole-forward measurement
  exists on a chip the kernel can price, and it does.
* **mamba2**: no FPM artifact carries a Mamba2 forward pass, and no InferenceX tier
  carries Nemotron-3. Unvalidatable from either direction.
* **A100 and L40S**: absent from FPM and from InferenceX entirely. Their coefficients are
  fitted and shipped but carry no end-to-end check, which is stated rather than implied.

### 8.5 Rack-scale rank widths, and the one place this registry extrapolates

A Grace-Blackwell tray is four GPUs, so AISimulate's NCCL sweeps cover **2 and 4 ranks
only** on GB200-NVL72 and GB300 — against 2, 4 and 8 on every SXM part. `tp=8` is an
ordinary deployment on both, so three of the four collectives have no 8-rank measurement
on any lane. All-reduce does: vLLM's own `custom_allreduce_perf.parquet` sweeps GB300 at
2, 4, 8 and 16 ranks, and since §5 already puts every fp16 all-reduce on that lane, the
8-rank figure is a first-class measurement rather than a borrow.

For the other three, `all_gather`, `reduce_scatter` and `alltoall`, GB300 ships 18
entries at `method: assumed`. This is the only extrapolation in the registry, and it is
constructed to the standard `cost-model-host-overheads.yaml` set: dimension derived,
magnitude caveated, replacement experiment named.

**The model that was refuted first.** Ring hop-count predicts the 4→8 floor ratio as
`(2·7/8)/(2·3/4)` = **1.167** for all-reduce. The measured median across the seven parts
swept at both widths is **1.703**. Hop count is wrong by 46%, because a collective floor
is dominated by per-rank synchronisation and launch cost rather than by hops. Grounding
on textbook ring algebra would have under-priced these by a third; the refutation is
recorded because the wrong model is the plausible one.

**The model that holds.** `floor(n) = a + b·(n−1)`, with `a` and `b` fitted on each
part's own measured widths {2, 4} and evaluated at n=8. Validated as a holdout on the
NVLink parts that *do* carry 8-rank data:

| operator | n | pred/meas median | range |
|---|---|---|---|
| `all_reduce` | 14 | 0.987 | 0.953–1.041 |
| `all_gather` | 14 | 1.017 | 0.954–1.090 |
| `reduce_scatter` | 14 | 1.018 | 0.967–1.107 |
| `alltoall` | 14 | **1.128** | 0.737–1.294 |

`alltoall` is the stated exception and its entries say so: its floor barely grows with
width (median 4→8 ratio 1.090), so a two-point linear fit amplifies noise.

**The reference class excludes L40S and RTX Pro 6000**, whose 8/4 rate ratios reach
2.7×–4.2×. Those parts have no NVLink, so an 8-rank group crosses a materially different
fabric. GB300 is full-NVSwitch: its descriptor states `inter_node_bw == intra_node_bw ==
900 GB/s`, so crossing a tray boundary *inside the rack* is not a bandwidth change at
all — only the rack boundary is, and that is a 9× step to InfiniBand. Including PCIe
parts would have inflated the extrapolation.

**Rates are not extrapolated.** The 8-rank peak and transition rates carry the part's own
**4-rank measured** values. A geometric decay fitted on {2,4} under-predicts the measured
8-rank rate by a median 0.84×–0.95×, and the ring-bandwidth law by 0.91×–0.94× — both
biased low rather than centred, so neither is a sound estimator. Carrying the 4-rank
figure gives a rate that is *above* the true 8-rank rate, which is an upper bound on
throughput and therefore a lower bound on transfer time. That direction is stated in each
entry rather than left for a reader to infer.

**What replaces all 18 at once:** one 8-rank GB300 NCCL sweep.

**The kernel enforces the boundary.** `blis-latency-kernel` reads the measured widths from
the registry per chip rather than from a constant, and a width inside the sweep range that
this part was never measured at is an **error naming the operator, width and chip** — not
a silent clamp to a narrower floor. Before that change a GB200 `tp=8` group was priced at
the 4-rank coefficient, understating an 8-rank all-reduce by 1.53×–1.91× across the parts
measured at both widths.

## 9. The FPM mixed rows, and what they found

§8.4 noted that FPM's reach is narrower than the coefficient set. It was also being
under-used: 79.2% of FPM is MIXED prefill+decode rows (67,736 of 85,484), and none had
ever been scored. `scripts/score_fpm_mixed.py` closes that, and the result is the largest
single finding in this work.

### 9.1 What the coordinates mean, verified in AISimulate's own source

FPM's coordinate system is `iteration_totals_balanced_v1`
(`crates/core/src/perfmodel/perf_database/fpm_forward.rs:59`), and the totals are
**per-rank iteration totals**, not global ones. AISimulate builds a query as

```rust
// crates/core/src/perfmodel/operators/fpm_forward.rs:203-218
FpmPhase::Prefill => vec![b, b * s as f64, b * prefix as f64],
FpmPhase::Decode   => vec![b, b / w * s as f64],
```

so `total_prefill_tokens = batch × per_request_prefill` and
`total_kv_read_tokens = batch × per_request_context`. Dividing by `batch_size` to recover
the per-request shape the kernel takes is therefore the correct inverse, and is what
`select_overlap_band.py` already documents. The source warns that the round trip adds
integer-division rounding, which is why the scorer medians per grid point rather than
per row.

For a data-parallel cell the recorded latency is the **maximum across DP ranks**
(`collector/fpm_forward/native_artifact.py:704`,
`expected_wall_time = max(value for _, value in wall_times)`), not a mean — so a DP cell's
measurement is gated by its slowest rank.

### 9.2 Six cells scored, and the error is strongly topology-dependent

NoOverlap edge, all mixed grid points, against the committed registry:

| cell | parallelism | n | mean abs | signed | over |
|---|---|---|---|---|---|
| m27 pure_tp4 | tp4, ep1 | 5,986 | 15.63% | **−13.63%** | 13.0% |
| m27 tep2 | tp2, ep2 | 5,782 | 20.13% | **−14.86%** | 20.8% |
| m27 tep4 | tp4, ep4 | 5,986 | 25.58% | **−25.24%** | 3.1% |
| m27 dep2 | tp1, dp2, ep2 | 5,383 | 34.58% | **−33.96%** | 2.6% |
| m27 dep4 | tp1, dp4, ep4 | 5,918 | 47.14% | **−46.67%** | 1.6% |
| glm-5.2 tep8 | tp8, ep8 | 5,910 | 49.79% | **−45.04%** | 10.5% |

Every cell under-predicts, and the error spans 13% to 47% — a 3.4× range across topologies
of the same model on the same chip. That is not scatter; it is a missing term that scales
with parallelism.

The error is also **flat across the prefill share of the batch** on the tep4 cell
(−25.05% below 25% prefill, −22.57% at 25–75%, −28.40% above 75%), so it is not a
prefill-specific or a decode-specific defect. The whole forward pass is under-priced.

### 9.3 The mechanism: attention-DP funnels every rank's tokens into the experts

With attention data parallelism, all DP ranks' tokens are concatenated before expert
routing, so the MoE grouped GEMM sees `dp × tokens` rather than one rank's share. Three
independent sources agree:

* **vLLM**: `fused_moe/routed_experts_capturer.py:115-117` — "``n == total`` (naive
  dispatch): all DP ranks' tokens are **concatenated before routing**". `naive` and
  `allgather_reducescatter` are that path, and `allgather_reducescatter` is vLLM's default
  (`config/parallel.py:195`).
* **AISimulate**: `crates/core/src/perfmodel/operators/moe.rs:287` —
  `let num_tokens = num_tokens.saturating_mul(self.attention_dp_size.max(1));` with the
  comment "Attention-dp scales up the total input tokens (all dp ranks all-gather into one
  shared expert pool)".
* **blis-latency-kernel**: `kernel.go:402` —
  `routedPerRank := tokensF * float64(l.TopK) * k.localExpertShare`. **No DP factor.**
  `DP` reaches exactly one decision, `SequenceParallelMoE`
  (`internal/resolve/layout.go:106`), and that rule requires `tp > 1 && dp > 1`
  (`blis-schemas rules/rules.go:196`) — so on the two `dep` cells, which carry `tp=1`,
  DP influences no pricing term at all.

The two `dep` cells are the two worst MiniMax results and they order by `dp` (−33.96% at
dp=2, −46.67% at dp=4), which is the direction and roughly the magnitude a missing `×dp`
on the routed term predicts.

**What this does NOT explain.** `glm-5.2 tep8` is dp=1 and still −45.04%, so a missing DP
factor cannot be the whole account. GLM-5.2 differs from MiniMax-M2.7 in more than
topology — it carries `sparse_mla` attention and a separate indexer GEMM — so that cell
needs its own diagnosis rather than being folded into this one. Stating the limit is the
point: one mechanism explains the MiniMax ordering, and a second, unidentified term is
still live on GLM.

### 9.4 A committed figure this corrected

`docs/band-selection.md` reports NoOverlap's signed mean as −3.44% over 219 points and
calls it "nearly unbiased". Scoring every pure-decode grid point in the m27 tep4 artifact
rather than the 45 the old band sweep reached gives **−22.93%** (1,500 points), or
**−19.27%** restricted to batches within the scenario's declared `max_num_seqs` of 256
(1,048 points). FPM carries 1,557 distinct decode grid points for that cell, so the
committed figure rested on under 3% of the evidence available for it, and the deficit
grows with batch (−12.8% at batch 8 to −31.3% at 512) in a way a 45-point slice can miss.

The band CHOICE is unaffected — NoOverlap beats Overlap on every slice measured here, by
about 7pp on the mixed rows. The claim of near-unbiasedness is withdrawn.

## 10. Sparse MLA: the byte count is the fix, and the rate is not fittable

Commit 587c1be recorded sparse MLA as **unfittable**, and the kernel's `new.go` said
`sparse_mla` was "deliberately NOT mapped … the registry carries no fit for it", so every
layer of deepseek-v4-pro was priced as if it read the whole KV cache. Checking that against
the data rather than inheriting it produced a split answer: the **byte count** was the
defect and is now fixed in the kernel, while a per-kind **rate** turns out not to be
identifiable from this dataset at all. Both halves are recorded because the negative one is
the more useful.

### 10.1 The table that is not a module measurement

The `sparse_attention` family ships **thirteen** tables per part. Twelve floor between
37.7 µs and 1,186 µs on h200 — at or above the **66.2 µs** it costs to read the `csa4_moe`
layer's 317.5 MB of projections at 4.80 TB/s, which blis-catalog prices as nine separate
`GEMM` nodes. Those are module measurements, and fitting a kernel coefficient from one is
the defect `scripts/correct_mla_floor.py` had to undo (§3, sixth entry).

One is not: `dsv4_hca_attn_module_perf` floors at **9.6 µs** on the `FLASHMLA_SPARSE_DSV4`
lane — *below* this part's 13.5 µs GQA attention floor and 7× below the projection read.
35,262 rows across all six NVIDIA parts, and uniquely in the family a `compress_ratio`
column.

### 10.2 The byte count, chosen by search rather than assumed

Candidate forms were grid-searched against (floor, rate) on three parts. The discriminator
is whether the optimum is **interior**: a form that under-counts bytes forces the rate to
the grid ceiling, which is how a wrong byte count announces itself.

| form | h200 error | rate as fraction of peak |
|---|---|---|
| full context — *the old fallback* | 1.698 | **1.00 — clamped on every part** |
| `topk=128` only | 1.309 | 0.07 |
| `topk=128 + (step−128)/128` | **1.301** | 0.08 |
| `topk=1024 + (step−1024)/4` | 1.483 | **1.00 — clamped** |

The winner is exactly the geometry blis-catalog declares for `csa128_moe` (`window: 128`,
`compress_ratio: 128`) — resolved from the catalog, not tuned. **The full-context form
clamping at 1.00 of datasheet peak on all three parts is the quantitative statement that
the old fallback could not describe this kernel**: it needed a physically impossible
bandwidth to cover the measured time.

The physics is directly visible. At batch 1 on h200 latency is flat at 9.8–13.7 µs from
step 0 to 16,384 and reaches only 30.6 µs at step 1,048,575 — a 1 M-token context costs
about as much as a 16 K one.

### 10.3 Why a per-kind rate is NOT fittable, which is the finding

A rate was fitted — 0.083–0.111 of datasheet peak across six parts, errors 1.30×–1.36×
with head and tail balanced — and then **withdrawn**, because the end-to-end check refused
it:

| configuration | overall TPOT | deepseek-v4-pro TPOT |
|---|---|---|
| neither (full context, part-wide rate) | 14.98 % | 10.42 % |
| **sparse byte count only** | 15.04 % | **8.72 %** |
| sparse byte count + sparse rate | 14.95 % | 14.36 % |

The byte count earns **1.7 points** on the affected model. The rate costs **5.6**.

The mechanism is not noise, and it is specific. deepseek-v4-pro **alternates two sparse
geometries** — 30 `csa128_moe` layers (window 128, ratio 128) and 30 `csa4_moe` layers
(index_topk 1024, ratio 4) — whose selected-token counts differ by **32×** at a 1 M context
(8,319 against 262,912). The fitted rate came from `dsv4_hca_attn_module_perf`, which states
`compress_ratio: 128` on every row, so it describes `csa128_moe` alone. Charging it to all
60 layers applies a rate fitted on the small read to the large one.

**And this dataset cannot supply the missing half.** `compress_ratio: 4` appears only in the
`dsv4_csa_*` tables, and every one of those is a module measurement (floors 61.4–80.6 µs).
Fitting a `csa4_moe` rate from them would re-import the projection double-charge. A rate
covering half a model's layers is not a coefficient — it is a bias with a provenance string,
so no part-wide `attention_decode_rate_sparse_mla` ships. (One ships scoped to
DeepSeek-V4.1-Flash on h200, whose every sparse layer is inside the measured range; §10.5
says why that is a different question.)

`scripts/fit_attention_sparse_mla.py` is kept as a negative result, the way
`fit_attention_mla.py` and `probe_moe_roofline.py` are: the fit is clean, physical, and
wrong for this cost model, and deleting it would leave the next person to re-derive that.

### 10.4 What shipped instead

The fix is in the kernel, not the registry, and it needs no fitted constant:
`selectedKVTokens` bounds a sparse layer's decode read **per layer and per request** from
the catalog's own `index_topk`, `window` and `compress_ratio`. Sparse layers keep the
part-wide measured floor and rate, now charged against the right number of bytes.

Per-request rather than batch-wide because `min(sum) ≠ sum(min)`: a batch holding one
request below the top-k and one far above it gets both wrong if the bound is applied to the
aggregate. Per-layer because the two geometries above differ by 32×.

No schema change was needed — blis-schemas v0.2.0 already carries `index_topk` and
`compress_ratio`; the kernel's layer plan simply was not reading them.

The generalizable point, and it is the same one §3 makes from the other direction: a
coefficient can be *measurable* and still not be *identifiable for the thing it will be
charged against*. Here the quantity was measured cleanly on one of two layer geometries,
and only the end-to-end check could tell that this made it unusable.

### 10.5 DeepSeek-V4.1-Flash on H200: what vLLM v0.31.0 runs, and what is measured of it

inference-sim/blis-registry#37, #38 and #39 asked for the coefficients the kernels vLLM
v0.31.0 runs for DeepSeek-V4.1-Flash on SM90 need. Every kernel identity below is read from
the v0.31.0 tag (`db9527a4`); every number is written by
`scripts/fit_deepseek_v41_h200.py` or `scripts/fit_marlin_grouped_ramp.py`, both with
`--check`.

**No vLLM 0.31 sweep exists, so the timed path was compared instead.** AISimulate's newest
vLLM collection for h200 is `vllm/0.25.0`. For each kernel used here, the source the two
tags build was diffed, and what the 0.25.0 timing does and does not contain is stated:

| what | v0.25.0 → v0.31.0 | effect on the fit |
|---|---|---|
| FlashMLA SM90 sparse decode (`flash_mla_with_kvcache`) | vllm-project/FlashMLA `a6ec2ba` → `0eee43b`: namespace, include paths, `MODEL1` → `V4`; the 584 B `BYTES_PER_TOKEN` assert unchanged | none |
| FlashMLA split-KV combine, same call | grid `(b, s_q, h)` → `(b·s_q, 1, h)`; 192/224/256-split cases added | inert at one query token per request; the new cases are reached only past 160 splits |
| FlashMLA host API | rewritten; the split workspace is skipped for unsplit requests (`0397728d`) | host-side, not in the device time |
| top-k → cache-slot translation | V4.1 launches `compute_global_topk_indices_and_lens` on every compressed (ratio 1 or 2) decode layer (`nvidia/flashmla.py:181-188`, under `if not swa_only`); the 0.25.0 ratio-128 rows take indices from metadata | **a known gap**: one small Triton launch per layer the rate does not price |
| DeepGEMM `fp8_fp4_paged_mqa_logits` | deepseek-ai/DeepGEMM `a6b593d` → vllm-project/DeepGEMM `e1f418c` (v0.31.0 builds vLLM's fork): renames; bounds handling for zero-length contexts and block-table rows past a request's pages | none |
| Marlin MoE template (`marlin_moe_wna16`) | the runtime act-order path removed (#54809), compile-time dead for MXFP4's group-32 instantiation | none |
| Marlin MoE Python (`fused_moe/experts/marlin_moe.py`) | 10 commits; #48144 sets the M used for block-size selection under EP to `ceil(M · local / global experts)` | inert at ep 1, where the ramp is fitted; the 0.25.0 rows at ep > 1 may not describe v0.31.0 wherever it changes the chosen block size |

**What ships.**

| entry | scope | method | value | from |
|---|---|---|---|---|
| `attention_decode_rate_sparse_mla` | h200, deepseek-v4.1-flash | measured | 1,123,200 B/µs (0.234 of peak), floor inherited (13.5 µs) | `dsv4_hca_attn_module_perf`, V4-Flash rows: the `h64` FlashMLA instance and 584 B record V4.1 runs |
| `attention_decode_floor_block_index_scores` | h200, deepseek-v4.1-flash | measured | 4.3 µs | `dsv4_paged_mqa_logits_module_perf`, the kernel itself under CUDA graphs |
| `attention_decode_rate_block_index_scores` | h200, deepseek-v4.1-flash | assumed | 3,840,000 B/µs, NVIDIA's empirical HBM rate (BwPeak × `hbm_derate`), not a bound | see below |
| `gemm_{eps_max,m_half}_w4a16_mxfp4_grouped` | h200, gb300 | measured | 0.245 / 26 (h200), 0.114 / 55 (gb300), of the BF16 peak | vLLM's `MarlinExperts` MXFP4 rows, fitted under the kernel's routed-term composition. On gb300 it applies only where the MoE backend is Marlin W4A16 (the Kimi-K3 preview build FPM measured), not to stock v0.31.0 |
| `gemm_{eps_max,m_half}_w8a16_mxfp8` | h200 | assumed | the BF16 envelope: an upper bound on the asymptote; the knee is not a bound | no dense Marlin sweep exists on any part |
| `host_gather_rate_uva` | h200 | assumed | 57,764 B/µs: the PCIe 5.0 raw rate × 128/130 line code × 264/288 | no host-gather measurement exists |

**On H100 the same six entries ship, each fitted from H100's own rows.** H100 SXM is the
same GH100 die and SM90 kernels as H200 with 0.70× its HBM (3.35 against 4.8 TB/s), and
AISimulate's `h100_sxm` carries the same three tables. The writers fit every part they list
from that part's sweep, catalog descriptor and committed siblings; no h200 number enters an
h100 entry.

| entry | h100 value | h200 value | fit |
|---|---|---|---|
| `attention_decode_rate_sparse_mla` | 1,105,500 B/µs (0.330 of peak), floor inherited (14.5 µs) | 1,123,200 (0.234) | 1.198x, signed 1.057, n = 207; held out ≥ 5 blocks 1.175x (signed 0.853, n = 16); at 10 blocks 1.069x (n = 4) |
| `attention_decode_floor_block_index_scores` | 4.9 µs | 4.3 | 1.086x, n = 414; 246 floor-dominated points 1.067x; per checkpoint 4.95 / 4.90 |
| `attention_decode_rate_block_index_scores` | 2,680,000 B/µs, BwPeak × `hbm_derate` (3.35 × 0.8) | 3,840,000 | `assumed`; the sweep's slope is 6.44 TB/s-equivalent, 1.92× H100's peak |
| `gemm_{eps_max,m_half}_w4a16_mxfp4_grouped` | 0.282 / 15 | 0.245 / 26 | 1.140x, signed 1.039, n = 27; int4 holdout 1.161x, signed 1.068 |
| `gemm_{eps_max,m_half}_w8a16_mxfp8` | 1.0 / 170, H100's BF16 envelope | 0.992 / 110 | `assumed` |
| `host_gather_rate_uva` | 57,764 B/µs, H100's own `host_link_bandwidth` × 128/130 × 264/288 | 57,764 | `assumed` |

**Neither fitted pair scales with bandwidth, and the rows say so directly.** On the 207
FlashMLA rows both parts measure, H100 latency is a median 1.05× H200's (1.13× over the
slowest tenth), where an HBM-bound read would be 1.43×; on the 27 Marlin token counts it is
0.98–1.06×. So the sparse rate is close to H200's (0.98×) and a larger fraction of H100's
peak: at reads of at most 10 blocks per request it is a cost per byte, not a bandwidth, and
must not be rescaled between parts by BwPeak. The Marlin ramp differs from H200's because
its memory half is charged at H100's lower BwPeak × `hbm_derate`, and the SM half carries
what is left; the pair holds only with that part's own memory half. The sparse rate's batch
bias has the same shape on H100 (signed 1.02 at batch 1, 1.44 at 128, 0.59 at 1,024), and
the per-request and occupancy forms again leave it (1.186x, 1.179x). The pinned 14.5 µs floor
over-prices batch-1 one-block reads (10.1–10.6 µs) by 37–44%.

**No end-to-end check is possible on H100.** No forward-pass dataset carries
DeepSeek-V4.1-Flash on h100 under any engine, and the h200 rows FPM does carry are SGLang
(next paragraph), so nothing arbitrates the h100 entries either; each rationale says so.

**The sparse rate's error depends on batch, and V4.1 runs where the table has no rows.**
The overall 1.182x hides a batch-dependent bias. Signed, predicted over measured, by batch:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| signed | 0.99 | 1.04 | 1.07 | 1.09 | 1.08 | 1.05 | 1.25 | 1.45 | 1.10 | 0.80 | 0.65 |

* **Not a missing term this form can carry.** A per-request term, `floor + c · batch +
  bytes / rate` (1.179x), and an occupancy form, `floor + a · ceil(batch · blocks / P)` with
  P fitted at 64 (1.173x), leave the same pattern, so it is batch-dependent scheduling inside
  the call. The writer fits and prints both.
* **Inside the measured range on the block axis only.** V4.1 reads at most 640 tokens per
  query, 10 FlashMLA blocks (2 window + 8 top-k), inside the 1–130 the table covers. But
  the sweep's batch × context ≤ 1 M grid pairs 10 blocks only with batch ≤ 8, while V4.1
  reads 10 blocks at any context past about 1k tokens, at whatever batch it serves. Above
  batch 8 the rate's error is what the table says about neighbouring cells: at batch 64 and
  up, 10–45% slow or 20–35% fast.
* **The holdout is in the small-batch region too.** Trained on requests of at most 3 blocks
  and tested on 5 or more, it predicts 1.210x, signed 0.83 — but all 16 held-out rows are at
  batch ≤ 16.

**The pinned floor over-prices the shortest reads.** The floor is this part's 13.5 µs
part-wide attention floor, which the kernel inherits for a kind with a rate and no floor.
Batch-1 rows of at most one block measure 9.8–11.8 µs, so pinning costs 14–38% there. A free
floor lands at 13.3 µs (1.181x against 1.182x) only because a linear form cannot follow the
step between one and two blocks. On V4.1 a read of one block occurs on the window-only
layers (0, 1, and the DSpark drafts) at contexts up to 64 tokens, and on a ratio-2 layer at
context 1, where the indexer emits `min((pos + 1) // ratio, topk)` = 0 compressed tokens; past
that every compressed layer reads at least two blocks.

**Why a sparse rate ships for V4.1 when §10.3 withdrew one for V4-Pro.** The withdrawn rate
was fitted on Pro's 128-head rows — the `h128` instance, which Pro runs only below TP 2,
since FlashMLA pads `n_local_heads` — and was to be charged against the kernel's selection
rule for Pro's csa4 layers: up to 262,912 tokens at a 1 M context, outside every measured
row on the block axis (vLLM itself reads far less there; blis-latency-kernel#32). Every V4.1
sparse read is on the `h64` instance at every TP width and inside the block axis. The entry
is scoped to this model, so no other deployment moves.

**Why the indexer's rate is assumed when its floor is measured.** The collector gives every
query row one block table, so the unique key footprint is at most 34.6 MB, and the fitted
throughput, 5.35 TB/s-equivalent, exceeds the 4.8 TB/s HBM peak — impossible from HBM, and
consistent with the shared keys being served from cache. A fit of that slope would be clean
and wrong in the §3 sense. The floor is a launch cost: the 238 points whose work term is
under 5% of it identify it at 1.050x, and each checkpoint alone gives 4.29 and 4.28 µs. In
serving each request's keys are its own, read at `ceil(context / ratio) × 132 B`; V4.1's
ratios are 1 and 2, so the scan grows with context while the attention it feeds does not.
The committed rate is NVIDIA's empirical HBM fraction (BwPeak × `hbm_derate`), not a bound:
the real rate cannot exceed BwPeak, so it can be pessimistic by at most 1.25x (scan time
over-priced), while how optimistic it can be is not bounded. The writer refuses to keep
assuming it if a future sweep's throughput stops exceeding the HBM peak.

**Why the Marlin ramp is fitted in the kernel's composition, and what it is coupled to.** A
weight-only GEMM is two costs, a BF16 `mma.sync` compute and a narrow weight read. The
kernel sums them for the routed term, and fitting the ramp as what remains after the
kernel's own memory half is the only way it composes back to the measurement. That couples
it to blis-latency-kernel's routed term as #44 (`483f6aa`) implements it:

* compute and memory summed, the ramp evaluated at the step's token count, compute at the
  BF16 peak, memory as `ExpertsTouched` experts' stored bytes at BwPeak × `hbm_derate`,
  imbalance and SM derate 1;
* FLOPs the rank's own, the intermediate width divided by tp, which #44 applies on this ramp;
* no Marlin padding: #44 also scales routed FLOPs and bytes by Marlin's round-up of
  `inter / tp` to a multiple of 128 (V4.1 at TP8: 288 → 384). Fitted at tp 1, where that pad
  is 1, the fit has not checked it; it is the consumer's addition;
* stored bytes per weight as blis-schemas v0.3.0's `StoredBytes` gives them for MXFP4,
  0.5 + 1/32, which #44 charges. At payload-only bytes (0.5, what kernel `main` `e9cf217`
  charges) the fit would be 0.240 / 28 on h200, 0.114 / 58 on gb300.

A change to any of those — sum versus max, the ramp's argument, the byte count — needs a
refit. The fit holds out on vLLM's int4 Marlin MoE, a different geometry and weight type:
1.196x on h200 and 1.266x on gb300. Its error grows with expert and tensor parallelism; the
committed BF16 ramp does the same on BF16 Triton experts (on h200, 2.83x at ep 8 and 8.33x
at ep 32, printed by the writer's form check), so that growth is a property of the
routed-term form — though #48144 above means the 0.25.0 ep > 1 rows may not be v0.31.0's.

For scale: kernel `main` prices MXFP4 experts on H200 at the FP8 peak with the FP8 ramp.
That is 1,456 TFLOPS effective where Marlin sustains about 242.

**What does not ship, and why.**

* **A ramp for `CutlassExpertsMxfp4`**, what stock v0.31.0 runs for compressed-tensors
  MXFP4 experts on Blackwell. AISimulate carries no such rows on any Blackwell part.
* **Prefill pairs for either kernel.** No kernel-level table carries V4.1's prefill path
  (`flash_mla_sparse_fwd` over a gathered compressed region, `fp8_fp4_mqa_logits`), and
  the kernel has no per-kind prefill name to read one with.
* **SGLang's `dsv41_module_perf`** (h200). It is eager, a module timing (`MQALayer.forward`
  with projections), at batch ≤ 3 and context ≤ 2,049, and flat at 0.8–2.0 ms. That is
  launch-bound, not a kernel rate.

**No end-to-end check exists for vLLM, and that is stated rather than waived.** At FPM
revision `03e006de`, DeepSeek-V4.1-Flash on h200, b200, b300, gb200 and gb300 is measured
under **SGLang only** -- another engine, another MoE kernel (`w4a16_mxfp4_humming` on h200)
and another checkpoint revision (`fb2764a5` against the catalog's `52544481`). The only vLLM
V4.1 leaf is a historical GB200 dev build labelled "comparator". So FPM cannot be the arbiter
§3 requires for V4.1's vLLM pricing; an earlier version of this section said it could.
blis-registry#42 stays open with no arbiter until a vLLM V4.1 forward-pass measurement
exists. Separately, blis-latency-kernel `main` cannot
price the model: it refuses the `fp8_ds_mla` cache (#36), charges 20,480 KV bytes per token
against roughly 1,790 (#30), drops the 128-token window from the sparse read (#32) and
does not pad FlashMLA's heads (#39). blis-latency-kernel#44 closes all four. Three
consumer-contract points go with the entries, each stated in them:

* **Multi-token decode.** The sparse rate was fitted at one query token per request. Under
  DSpark speculation a request decodes 1 + num_spec_tokens query tokens, and FlashMLA treats
  each as its own query; how much of that read the tokens of one request share is
  unmeasured. #44 charges the read per query row and records that as an assumption
  (`sparse_read_per_query`), which is what this entry asks of a consumer.
* **Routed FLOPs under pure TP.** This ramp's FLOPs are each rank's own (`inter / tp`). #44
  divides routed FLOPs by tp on the weight-only Marlin ramps, as this fit requires, and keeps
  its documented over-charge on the native ramps, where it cancels a measured error;
  blis-latency-kernel#47 tracks removing it there. A consumer that charged this ramp the
  whole expert would price its compute half tp times too slow.
* **gb300's Marlin ramp stays unread** while ramps are chosen by format and chip, which on
  gb300 is the FP4 class — right for stock v0.31.0. It is for scoring the Kimi-K3 preview
  build's FPM run, for blis-registry#41 (the Kimi-K3/GB300 FPM check of the KDA pair).

Every V4.1 attention read in blis-catalog 0.3.0 is `sparse_mla`, the window-only layers (0,
1, and the DSpark drafts at a 133-token window) included, all inside the block axis above.

### 10.6 DeepSeek-V4.1-Flash prefill attention, prefill indexer, mHC and host terms

Before these entries the kernel priced three V4.1 operators with numbers fitted on a
different kernel or with none: the sparse-MLA PREFILL attention and the indexer's prefill
logits with the generic full-attention prefill pair (`attention_prefill_floor` /
`attention_prefill_work_scale`, fitted on `context_attention_perf`), and mHC as Elementwise
bytes at the HBM rate with no floor -- 0.07 ms a one-token decode step against 1.05 ms
measured, one of the two terms behind V4.1's 19-30% fast low-load ITL. Every number below is
written by `scripts/fit_deepseek_v41_prefill.py` or `scripts/fit_deepseek_v41_host.py`,
both with `--check`; each part is fitted from its own rows. The per-call forms are stated in
each entry's rationale.

| entry (scope: h200 / h100, deepseek-v4.1-flash) | method | h200 | h100 | fit (h200) |
|---|---|---|---|---|
| `attention_prefill_{floor,per_request,per_token,rate}_sparse_mla` | measured | 14.8 µs, 1.323 µs, 0.049 µs, 3,159,685 B/µs | 15.5, 1.325, 0.0571, 3,509,505 | 1.067x, signed 1.003, n = 1,642 (not V4.1's operating point: signed 0.81 at 10 blocks); held out on blocks 1.137x (signed 0.91), on chunk length 1.036x |
| `attention_prefill_{floor,per_token,rate}_block_index_scores` | **assumed** (proxy kernel) | 4.1 µs, 0.00466 µs, 7,298,637 B/µs | 4.7, 0.00441, 7,364,876 | 1.066x, signed 0.969, n = 3,504; held out on context 1.054x, on rows 1.142x |
| `mhc_{floor,rate}_pre` | measured | 8.9 µs, 1,799,384 B/µs | 9.3, 1,325,488 | 1.052x, n = 70; across hidden 1.103x / 1.073x |
| `mhc_{floor,rate}_post` | measured | 3.0 µs, 1,850,611 B/µs | 3.1, 1,306,352 | 1.091x, n = 70; across hidden 1.178x / 1.109x |

**Sparse-MLA prefill is fitted on the kernel table's prefill rows, in four terms.**
`dsv4_hca_attn_module_perf` times `forward_mqa` alone; its `isl > 1` rows run the prefill
branch (SWA and compressed gather, index combine, `flash_mla_sparse_fwd`). A floor plus a
work rate over the decode fit's block-and-record count gives 1.390x with a strong batch bias
(signed 0.15 at batch 1,024); a per-request and a per-query-token term remove it (1.067x,
every batch within 0.96-1.05 signed). The per-token term, 0.049 µs, is the cost of a query's
own traffic -- 64 padded heads of 512 BF16 in and out -- at about the part's empirical HBM
rate. The fit uses only rows reading at most V4.1's 10 blocks a query. Its weak point is
V4.1's operating point: the only 10-block rows are short chunks at step 65,536, under-priced
1.31x (signed 0.81), and long chunks at 10 blocks are not in the table; long-chunk rows at
11-67 blocks extrapolate at 1.05x. **So the 1.067x headline is not V4.1's error.** It is
dominated by the rows V4.1 does not run: 980 of the 1,642 rows are at 3 blocks, 20 at 10 and
none at 8-9. At V4.1's read the evidence is signed 0.81 in sample (under-priced) and, trained
on fewer blocks, signed 0.91 held out on 7-10 blocks; the honest statement of this entry's
error at V4.1's operating point is "about 10-20% fast, from 71 rows", not 1.067x.

**These entries ship without the end-to-end check §3 and the evidence page ask for, as an
explicit exception.** No vLLM forward-pass measurement of V4.1 exists on any part (§10.5:
FPM's V4.1 rows are SGLang), so the rule cannot be met by waiting for the data on hand; the
alternative is to keep pricing V4.1 with entries fitted on other kernels, which is the known
larger error. Each entry therefore says it has no arbiter, and blis-registry#42 stays open
until a vLLM V4.1 forward-pass measurement exists. The SM90 sparse prefill device code is unchanged between
the FlashMLA pins of vLLM 0.25.0 and 0.31.0 (renames only). V4.1's ratio-1/2 layers gather
their whole compressed stream before the kernel; the ratio-128 rows hold that gather only at
1/128 of the context, so it stays with the catalog's `prefill_dequant_gather` node.

**The indexer's prefill is a proxy, and is labelled so.** vLLM v0.31.0's prefill indexer runs
DeepGEMM's ragged `fp8_fp4_mqa_logits`; the entries are fitted on the paged
`fp8_fp4_paged_mqa_logits` at prefill shapes. Unlike decode, the sweep's shared block table
is representative here (every query row of a request reads the same keys), and the fitted
throughput is compute-like (906 TFLOPS at 64 heads). The 32-head V4.1 indexer is scaled
linearly from 64, an assumption. **The ragged kernel is measured, on another lane:**
SGLang 0.5.14's `glm5_mqa_logits_module_perf` times `deep_gemm.fp8_mqa_logits` at GLM-5's
indexer geometry -- 32 index heads x 128, FP8 keys, ratio 1, V4.1's head count. Evaluated
as the kernel charges it, the proxy is 1.50x there, signed 0.93 (h200) and 0.90 (h100);
signed 0.89 / 0.82 on long chunks over long prefixes and 1.18 / 1.09 at V4.1's 8,192-token
chunk. An earlier review read the table's `num_heads` (GLM-5's 64 attention heads) as index
heads and found the proxy over-pricing by 1.4-1.7x; at the config's 32 it does not. The
ragged rows are not fitted because the lane decision is open: SGLang 0.5.14 calls upstream
DeepGEMM's `fp8_mqa_logits`, vLLM v0.31.0 its fork's `fp8_fp4_mqa_logits` on the FP8 path,
and the two pins' device code has not been compared. `fit_deepseek_v41_prefill.py` prints
the comparison and each rationale states it.

**mHC is measured per site and interpolated by bytes.** `mhc_module_perf` times TileLang
`mhc_pre` and `mhc_post` (two sites a row) at hidden 4096 and 7168; W = tokens x hidden x 8 B
carries the hidden size, so one floor and rate per component fit both, and each hidden size
predicted from the other lands within 1.06-1.18x. The pre includes the TF32 prenorm GEMM, so
the catalog's `hc_*_fn` GEMM must not also be charged. On SM90 v0.31.0 runs the fused delayed
post+pre, which above 32 tokens is the same two kernels and at <= 32 folds the post into the
GEMM -- so at decode sizes the pair is an upper bound by about one launch.

**Two terms were looked for and do not ship.** *Routed MoE at V4.1's geometry:* no vLLM
collection holds a hidden 5120 / inter 2304 / 384-expert / top-6 W4A16 row on h200 or h100.
The only rows at that geometry are SGLang `dev-1aa0e962` (`flashinfer_cutlass` at moe_tp 2,
`mxfp4_humming` at moe_tp 4, ep 1, 15 token counts each) -- other kernels and another
sharding than vLLM v0.31.0's Marlin at EP8 -- so the Kimi-K3-geometry Marlin ramp stays, with
its evidence class. *The compressor:* no table isolates it; the csa/hca context-module tables
time the whole attention module, projections included, and a residual from them would be a
module-derived term of the kind §10.1 rejects. It stays at the catalog's bytes.

**Host terms.** `host_launch_eager_per_kernel` (92.87 µs, `assumed`) is V4.1's per-launch
Python dispatch in an eager (cudagraph NONE) step -- vLLM v0.31.0 cannot torch.compile this
model, so NONE is plain eager PyTorch. It is backed out of an internal enforce-eager H200
serving run of the same model (ITL flat at 93.8-101.6 ms from c1 to c64). The run's report
is not redistributed: `validator/testdata/anchors/v41_h200_enforce_eager_anchor.json` holds
only what the fit needs -- the plateau's levels, median, minimum and maximum ITL, the setup
and the report's sha256 -- and `--check` re-derives it from a local copy of the report when
`V41_EAGER_ANCHOR_REPORT` names one, and checks its internal consistency otherwise. The
division is by the KERNEL's own launch count and device time for that step, a recorded input:
`v41_kernel_eager_basis.json`, written by `--record-device` from `diagnosis/kdevice` (public
kernel API) with clean kernel, registry and catalog commits. `--check` refuses a basis
recorded for another anchor, scenario, host composition or step type: the anchor is a decode
step, so the count must be the decode-only 972 (a step with a prefill row also launches the
graph's 46 `phase: prefill` gathers, 1,018; kernel 993ddeb used 1,018 on every step, which
gave 89.4 µs). **The composition is additive**, as the kernel's is: step = launches x (eager +
dispatch) + device, so e = (ITL - device) / launches - `host_launch_per_kernel` =
(96.56 - 3.18 ms) / 972 - 3.2 µs = 92.87 µs. A launch-overlap form, step ~ max(launches,
device) (vLLM enqueues eager kernels asynchronously), was tried in the kernel and gave 96.14
µs here, and was REJECTED: it does not reproduce the H200 DSpark arm's capture-ceiling
collapse at c32, whose measured throughput the additive form matches (blis-latency-kernel
07ed424, `TestTheEagerDecodeAnchorIsAdditive`, which pins the constant at 92.75-93.7 µs). The
basis records the composition and `--check` refuses any other. The committed basis is kernel
07ed424 (972 launches, 3.18 ms with the mHC entries). The constant is valid only with the
basis and the composition it was divided by. It is derived on H200 nodes only:
the h200 entry is `assumed` (backed out of the anchor), and the other eight parts, h100
included, carry it as a separate `copied` entry (`copied_from: hardware=h200,...`). That copy
is an ASSUMPTION about host CPU and driver launch cost, not a derivation; an H100
enforce-eager ITL anchor (same model, TP8, no speculation, c1-c64) would replace it. The host
set's coverage rule is checked per name and model across entries, so the split is allowed.
`host_admission_per_token` (2.67 µs, `assumed`) is a timing of vLLM v0.31.0's own V4.1 front
end -- render, tokenize, block hash -- taken on the diagnosis workstation (Apple M1 Max, one
Python thread) with synthetic text. No part it is scoped to matches that host, so on every
part it is an assumed transfer of the measurement, labelled as the eager copy is; the same
timing on the serving node's CPU would replace it. And `host_admission_per_request` is 0.0,
`not_charged` only because blis-schemas requires that method for a zero value: what was timed has no fixed term above noise, but the bench does not time HTTP
handling, validation, the IPC to EngineCore or scheduler admission, so the zero is an
inference bounded only end to end (an 8,030-token c1 TTFT of 37.15 ms in total), not a
structural zero. These two model-scoped entries sit after their
part-wide namesakes in the same set, the one arrangement
`test_a_model_scoped_entry_never_shadows_a_part_wide_one` allows: last-in-file-order and
scoped-wins resolve it the same way.

### 10.7 DeepSeek-V4.1-Flash decode: a two-branch FlashMLA form, and what stays assumed

The single decode rate (§10.5) is 1.18-1.20x with a batch bias it cannot carry: signed
1.45-1.54 at batch 128 and 0.59-0.80 at 1,024. The FlashMLA decode rows show two regimes.
At small batch the call is latency-bound and flat in batch to about 128 rows: one block
costs about 10.2 µs (h200) / 10.4 µs (h100), any read of two or more blocks adds about
3.5 µs at once (the split-KV combine launch), then 0.2-0.3 µs a block. At large batch it is
linear in rows through the origin: one cost per query row and one per block read. So
`scripts/fit_deepseek_v41_decode.py` fits, per part, on the rows of at most V4.1's 10
blocks:

    t = max(launch_floor + split_floor x [B > 1] + B x 37,376 B / serial_rate,
            sum over query rows (per_token + blocks x 37,376 B / throughput_rate))

| part | launch floor | split | serial rate | per token | throughput rate | fit | held out rows ≤ 256 → ≥ 512 | held out blocks ≤ 4 → ≥ 6 |
|---|---|---|---|---|---|---|---|---|
| h200 | 10.0 µs | 3.1 µs | 115,005 B/µs | 0.05437 µs | 1.30 TB/s | 1.061x (single rate 1.176x) | 1.038x (single rate 1.375x) | 1.059x |
| h100 | 10.3 µs | 3.6 µs | 145,351 B/µs | 0.06567 µs | 1.27 TB/s | 1.060x (single rate 1.193x) | 1.028x (single rate 1.502x) | 1.058x |

The per-token term lands near the prefill fit's own (0.049 / 0.057 µs), an independent
check that it is each row's query and output traffic. The weakness is stated in each
entry: the throughput branch is identified only by the batch ≥ 256 rows, which are at 1-3
blocks. Trained on rows ≤ 64 the form predicts ≥ 128 at 1.8-1.9x, worse than the single
rate (1.3x), because no training row reaches that branch. At 10 blocks the throughput
branch binds above about 47 query rows per rank, a region the sweep's batch x context
≤ 1M grid never measures; there the per-block cost is extrapolated from 1-3 blocks, as the
single rate's is. Batches 16-64 at 4+ blocks are under-priced (signed 0.83-0.93).

The names are new. `attention_decode_floor_sparse_mla` is deliberately not used: the
kernel reads that name today as the floor of the single-rate pair, which was fitted over
the inherited part-wide floor, and would silently re-price it. The kernel uses the form only
when all five resolve.

**What stays `assumed`, and why.**

* *The indexer's decode scan rate* (`attention_decode_rate_block_index_scores`). Both
  kernel-level sweeps of `fp8_fp4_paged_mqa_logits` -- vLLM 0.24.0/0.25.0 and SGLang
  0.5.14 -- give every query row one shared block table, and none flushes L2. The largest
  unique footprint, one request at a 1M context, is 34.6 MB, inside the 50 MB L2, and its
  implied throughput is 7.2 TB/s-equivalent on h100 (above the 3.35 TB/s HBM peak) even at
  batch 1. No row reads its keys from HBM, so no row identifies the HBM-bound scan V4.1
  runs in serving, where it is the one decode term that grows with context (ratio 1 and 2
  over all 40 layers' index sources). Its per-row term is identified (about 0.006-0.01 µs a
  row at 64 heads, past 256 rows) but is under 3 µs a layer at V4.1's batches and is not
  shipped.
* *The decode top-k.* vLLM v0.31.0 runs `cooperative_topk` up to 64 rows and
  `persistent_topk` beyond (`indexer_topk.py`, `index_topk` 512 on SM90). AISimulate
  times only SGLang's `topk_transform_512` (`dsv4_csa_topk_calib_perf`) and its
  GLM-5 fused variant, different kernels with mode switches visible in the rows (0 µs,
  then a 48-85 µs plateau, then ~10 µs past 128k). Nothing measures vLLM's kernels, and the
  catalog keeps pricing the logits write and top-k read as `index_logits_topk` bytes.
* *`compute_global_topk_indices_and_lens`* and V4.1's candidate select/mask: unmeasured.
* *Host and launch terms.* AISimulate carries no host, launch or graph-replay rows on any
  part; every operator row is a device timing under CUDA-graph replay. The collective
  floors on h100 and h200 are already `measured` (NCCL and vLLM custom all-reduce), so
  they are not among the assumed terms. The only public timing of an engine's Python path
  found is FPM's GLM-5.3-Flash vLLM 0.31.0 fidelity pairs on GB300 (next paragraph);
  nothing times graph replay or eager launch on Hopper.

**Held-out evidence for a per-request host term, not used to fit anything.** Each of FPM's
five GLM-5.3-Flash vLLM 0.31.0 GB300 leaves (FP8 TP2/TP4, NVFP4 TP1/TP2/TP4, checkpoint
`eb9eb208`, FULL_AND_PIECEWISE, async scheduling off) ships `source-fidelity.json`, pairing
each held-out point's FPM step time with a "clean" timer that excludes `schedule()`, the
scheduler-output send and `update_from_output`. Re-derived here from the five files: an OLS
of decode (FPM - clean) on batch over 77 points a leaf gives 20.4, 24.9, 26.4, 27.6 and 32.6
µs per request (intercepts 16-282 µs); the SGLang 0.5.20 control leaf gives -1.2. It does
not hold at small batch: fitted on batch ≤ 8 alone the slopes range from -21 to +33 µs, and
the residual at batch ≥ 16 is up to 1 ms, so the per-request reading rests on batch 16-32.
Caveats: Grace (Arm) hosts, FPM and clean are separate runs, batch ≤ 32, and the clean
timer's boundary is described only in words. It is the first in-version public timing of a
per-request scheduler cost, and it agrees in magnitude with `fit_host_step_per_request.py`'s
form; it is recorded as held-out evidence for that term on gb300 and moves no value.

## 11. Memory occupancy: what composes, and where each term comes from

`cost-model-memory.yaml` exists because a consumer was carrying three per-rank memory
magnitudes as constants in its own source — CUDA-graph capture at 512 MiB, collective
buffers at 392 MiB, activation scratch at "a few live buffers" (×4) — where no provenance
readout could see them (inference-sim/blis-registry#33). They set how many sequences
fit, so they are not soft numbers.

### 11.1 The composition, taken from NVIDIA's capacity path

AISimulate's KV-capacity estimate (`aisimulate_core/sdk/memory.py`) charges, per rank and
outside the KV budget:

    weights + activation + nccl_communicator_bytes_<N>rank + engine_workspace_bytes
            + cudagraph_capture_bytes_<mode>

Every term **adds**. Two questions the consumer could not answer from its side are
settled by the source rather than by preference:

* `nccl_communicator_bytes_<N>rank` is the **whole** collective-buffer reservation.
  `base_backend.py` sums `nccl_mem` and `other_mem`; neither contains the other.
* `engine_workspace_bytes` (`misc.other_mem`) does **not** include CUDA-graph capture.
  This registry's own rationale previously said it did; AISimulate annotates `other_mem`
  as "excludes CUDA graph reservation" (`support/config_profile.py`) and takes the graph
  reservation as a separate argument (`capacity.py`, `cuda_graph_reserved_bytes`). Adding
  a capture term on top therefore does not double-count. The rationale is corrected in
  `emit_primitives.py` and in all nine committed entries.

### 11.2 Activation scratch — `vendor_spec`, NVIDIA's table

    activation = max(activation_buffer_count_<family>_<N>rank · max_num_batched_tokens · h · 2,
                     activation_scratch_floor_bytes)

The multiple is AISimulate's `ACTIVATION_COEFFICIENTS`, which the vLLM backend reuses from
TRT-LLM's; `emit_memory.py` reads it from the source with `ast` and refuses if vLLM stops
aliasing it. Two rows are registered because the kernel can tell only whether a model
routes to experts: dense (the LLAMA row, 11 / 6.5 / 5 / 5 at 1 / 2 / 4 / 8 ranks) and MoE
(22 / 13 / 10 / 10). `h` is `num_attention_heads × head_dim`, equal to `hidden_size`
unless a config declares a different `head_dim`; `N` is `min(tp, 8)`.

Three scoping decisions, each a finding rather than a convenience:

* **The `2` is the model dtype, not the served one.** The issue proposed sizing scratch at
  the served width (1 byte for fp8). vLLM's quantized linears return
  `out_dtype=x.dtype` (`vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`): the fp8
  operand is a transient copy, and the residual stream stays bf16. Halving it would
  understate an fp8 deployment.
* **Not scoped by graph mode.** vLLM measures the activation peak in an eager profile run
  and the capture separately (`gpu_worker.py`); a mode-scoped multiple would charge
  capture twice.
* **Not scoped by hardware**, because the source states no per-part value; the scope lists
  every catalog part, under the same coverage test as the host set.

**Engine.** These are the vLLM figures. SGLang's table is about 1.3× higher with a further
15% overhead, and SGLang does not charge activations inside its static pool. A deployment
states no engine identity and the consumer prices vLLM, so no SGLang row is registered.

**Not registered: the MoE dispatch workspace.** AISimulate adds
`tokens · h · num_experts · top_k / ep / 128 · 4` bytes for its DeepSeek-derived families
and Kimi-K2.5 — 3,758,096,384 bytes for DeepSeek-V3 at 8,192 tokens and ep 1. That is a
formula over catalog facts and the fp8 block size, not a learned magnitude, so it belongs
to the consumer's composition; it is recorded here so it is not lost.

### 11.3 CUDA-graph capture — `assumed`, anchored on vLLM's own measurements

No dataset this project reads measures it. AISimulate takes it from the caller with a zero
default; the FPM dataset records each run's `cudagraph_mode` and capture sizes in its
resolved configs but no memory figure (all 417 of its JSON, text and TSV files under 3.5 MB at
revision `68fa3add` were read), and its SGLang raw evidence was collected with prefill capture off.

vLLM measures it on every start-up and logs it, and users paste those logs into issues.
`mine_vllm_cudagraph_memory.py` collected 506 samples from 677 vLLM issues into
`docs/vllm-cudagraph-capture-samples.csv`, which is committed so the anchor is reproducible
offline. The medians on catalog parts are 0.80 GiB for `PIECEWISE` (n = 18) and 1.08 GiB
for `FULL_AND_PIECEWISE` (n = 22); the replaced constant was 0.50 GiB. The **dimension**
is vLLM's profiler (`gpu_model_runner.profile_cudagraph_memory`): one first capture per
mode, sized by the largest shape, plus at least 1 MiB per further graph, with `PIECEWISE`
and `FULL` overlaid in one pool. So the cost is dominated by the largest captured shape,
not proportional to the number of shapes.

`FULL` and `FULL_DECODE_ONLY` have too few samples (one and four logs, only one of them on a catalog
part),
so they are charged the `FULL_AND_PIECEWISE` figure as an **upper bound** — they capture a
subset of its graphs into the same pool — with the direction declared. `NONE` is
`not_charged` at zero.

**Why this source is admissible here and nowhere else.** §3's separation is about fitting:
an uncontrolled sample cannot identify a per-primitive constant. These entries are not
fitted and are not `measured`; they are `assumed` entries whose magnitude is oriented by
the only public record of the quantity, in the pattern `host_admission_per_request` set.
None of the samples is from InferenceX or FPM, and the single-sourcing tests
(`test_generated_from_aisimulate.py`) still govern the primitive and collective sets,
which this set is kept out of for exactly that reason.

### 11.4 What it changes on the issue's fixture

On `deepseek-v3-h200-fp8-sglang-tp8` (tp 8, `hidden_size` 7168, 8,192 batched tokens,
`PIECEWISE`), resolved through `blis-latency-kernel`'s own `resolve.Load`:

| term | hardcoded | registry |
|---|---|---|
| activation scratch | 469,762,048 (×4) | 1,174,405,120 (MoE 8-rank ×10) |
| CUDA-graph capture | 536,870,912 | 858,783,744 |
| collective buffers | 411,041,792 | 411,041,792 (`nccl_communicator_bytes_8rank`) |
| engine workspace | not charged | 3,758,096,384 |
| **total** | **1,417,674,752** | **6,202,327,040** |

The set is committed **ahead of its reader**: the kernel reads none of it yet, and
inference-sim/blis-latency-kernel#22 is sequenced to land after it.

## 12. Speculative acceptance profiles

`acceptance/inferencex-golden-al.yaml` holds the acceptance lengths a simulator advances
speculating requests by (inference-sim/blis-registry#44), in blis-schemas#65's format. It
is not a coefficient set: acceptance does not price a step, it sets how many tokens a step
yields. The generated tables and the set's caveats are on the
[acceptance reference page](reference/acceptance.md).

### 12.1 One source, transcribed

Every value comes from SemiAnalysis InferenceX's
`inferencex-e2e/infx/golden_al_distribution/*.yaml`, vendored under
`validator/testdata/inferencex/` at the commit `scripts/emit_acceptance_profiles.py` cites.
The emitter reads each acceptance length, and every provenance field a file's header
states (the source Actions run, benchmark, category, sampling, image, TP and engine
settings, including `speculative-config: a=b | c=d` lines), and types none of them. Every
header is read, carried or not: a piece stating a value, as `key: value` or `key=value`
with any spelling of key, that the emitter does not know how to transcribe fails the run,
so a setting cannot be silently dropped. Only named prose lines are exempt. A header that
states `method:` must state the curve's method; one that does not is checked against its
"Measured on (..., vLLM <method>)" line. An acceptance length is `measured`; the
one file InferenceX marks provisional is `copied`, with `copied_from` naming the measured
profile it repeats, and the emitter checks that the two values agree.

### 12.2 Model names: resolved, or refused

InferenceX keys a curve by the lower-cased basename of the Hugging Face repository it
served (`MODEL_KEY` in its SPEED-Bench collectors). The emitter resolves that key to the
catalog model whose `source.repo` has that basename, against the vendored copy of every
catalog `model.yaml`. A key that resolves to none is not carried, and the set's caveats
name it. A quantised checkpoint is a different catalog model, so InferenceX's
`qwen3.5-397b-a17b-nvfp4` does not become the catalog's bf16 `qwen3.5-397b-a17b`.

One alias is allowed, with its evidence in the emitter: `glm-5.3-fp8`, the provisional
curve's key, is InferenceX's name for `zai-org/GLM-5.3`, the catalog's `glm-5.3`. The
emitter fails if the catalog stops recording that repository for it.

### 12.3 Curves a key cannot tell apart

A profile is keyed by model, method, K, thinking mode and category; it cannot state the
drafter or the draft sampler. Where InferenceX publishes two curves that differ only in
those, at most one can be carried. The emitter follows InferenceX's own resolver
(`curves.py`, `curve_name`): MiniMax-M3's base EAGLE3 curve is carried and the GQA-drafter
variant is not, and neither Kimi-K3 DSpark curve is carried, because the resolver itself
refuses to choose between greedy and probabilistic draft sampling. Every file is
classified in the emitter's `CURVES` table, and an unclassified file fails the run.

### 12.4 Hardware: stated, or read from the run

A measured profile must name the part it was measured on. Most headers state it ("B300,
vLLM MTP"). The DeepSeek-V4.1-Flash file states none, and its collecting job ran on
InferenceX's `b200-nscale` cluster, launched through the B200 benchmark base, so its
profiles say `b200`. blis-registry#44 describes the curve as measured on B300; the run
contradicts that, and the run is what is recorded. The emitter's `RUN_HARDWARE` table
holds each run's runner, read through the GitHub API, and is checked against every header
that also states a part. Files that name no engine image say so in `engine_image`: their
runs' logs have expired, so the image cannot be recovered.

The collectors are named `<prefix>_fp4_b300_vllm.sh` because `speedbench-al.yml` requires
that name and labels every collection `PRECISION: fp4`; the fp4 is not what they serve.
Each serves its checkpoint's own weights, and the DeepSeek-V4-Pro and GLM-5.2 collectors
set `--kv-cache-dtype fp8`, which their profiles record as `kv_cache_dtype`. The
emitter's `COLLECTORS` table holds what each collector serves, and a curve whose
collector it has not read fails the run.

### 12.5 Gates

`emit_acceptance_profiles.py --check` runs in every CI job
(`validator/test_acceptance_profiles.py`), because its inputs are vendored. The `rederive`
job also checks the vendored curves and catalog models against InferenceX and blis-catalog
at their pinned commits, and re-runs the emitter against the upstream checkout. Schema
validation is blis-schemas' `cmd/validate-registry`, which reads `acceptance/` from
blis-schemas#74 on. CI's `schema-validate` pins that pull request's head commit for now; a
release tag carrying it must replace the commit before this merges.
