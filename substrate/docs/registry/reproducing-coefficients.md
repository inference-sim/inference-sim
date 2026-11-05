# Reproducing every coefficient from the original datasets

This guide takes a reader from two public datasets to the coefficient values this
registry ships. It exists because a fitted constant whose fit cannot be re-run is not a
measurement, it is a number someone wrote down. Two earlier sets in this registry — the
per-GPU `roofline-*` MFU sets and `trained-physics` — were removed precisely because they
could not be re-derived: `trained-physics` carried fifteen entries, eleven of them
`fitted: true`, with **no `sources` block on any entry** and no fitter in `scripts/`; the
five `roofline-*` sets carried two `method: literature` entries each, citing an issue
number rather than a dataset. Both are recoverable from git history if ever needed.

For *why* the coefficients take the forms they do, and for the train/validate separation
that governs which dataset may be used for what, read
[`methodology.md`](methodology.md) first. This document is the mechanical companion: where
the data comes from, and which command regenerates which file.

## 1. The two datasets

### AISimulate — per-operator sweeps and system descriptors

| | |
|---|---|
| Repository | <https://github.com/ai-dynamo/aisimulate> |
| Docs | <https://ai-dynamo.org/aisimulate/> |
| Licence | Apache-2.0, with one MIT third-party file carved out in the `LICENSE` NOTICE and attributed in `THIRD_PARTY_NOTICES.md`. GitHub's license API reports `NOASSERTION` because of that preamble; the text itself says "Except for the third-party material identified in that notice file, this codebase is licensed under the Apache License 2.0", and source files carry `SPDX-License-Identifier: Apache-2.0`. |
| Provenance | Successor to [`ai-dynamo/aiconfigurator`](https://github.com/ai-dynamo/aiconfigurator); method described in arXiv:2601.06288. |
| Fetch | `git clone https://github.com/ai-dynamo/aisimulate` — the sweeps are checked into git as Parquet. Git LFS is needed only for retained legacy `.txt` assets, which no fit in this registry reads. `pip install aisimulate` also ships the tree inside the wheel. |
| Pin | Every committed value re-derives at `2f0f4cd8606d580567b253d02f3d699fe36db865` (2026-10-10), with blis-catalog at `0.5.0` (`b5550f7`; the commit `validator/testdata/catalog` vendors). The `rederive` CI job checks out both; bump either deliberately, with the diff the writers produce. A catalog revision that changes a part's datasheet figure moves every rate fitted as a fraction of it. |
| Python | `pip install -r requirements-fit.txt` — the validator's dependencies plus `numpy`, `pandas` and `pyarrow`, which the fitters need to read Parquet. |

The two trees the fits read are **not** at the repository root:

```
python/aisimulate/src/aisimulate_core/systems/<sku>.yaml          # system descriptors
python/aisimulate/src/aisimulate_core/systems/data/<sku>/<op>/<framework>/<version>/*.parquet
```

Export that second path once; every script below honours it:

```bash
export AISIMULATE_DATA=<aisimulate>/python/aisimulate/src/aisimulate_core/systems/data
export BLIS_CATALOG=<path-to-blis-catalog>
```

**Dataset SKU directories are not catalog chip names.** The sweeps use
`h200_sxm`, `h100_sxm`, `b200_sxm`, `b300_sxm`, `gb200`, `a100_sxm`, `l40s`; the catalog
calls the same parts `h200`, `h100`, `b200`, `b300`, `gb200-nvl72`, `a100-sxm`, `l40s`.
The mapping lives in each fitter's `SKUS` table and is passed as `sku:chip` pairs. There
is no `gb200_nvl72` directory.

Framework lanes present in the data tree are `vllm`, `sglang`, `trtllm` and, for `comm`,
`nccl`. Which lane is admissible for which operator is a methodology question, not a
mechanical one — see the lane table in [`methodology.md`](methodology.md).

### FPM — whole-forward-pass measurements

| | |
|---|---|
| Dataset | <https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset> ("AISimulate FPM Dataset"; FPM = Forward Pass Model) |
| Licence | Apache-2.0, same NVIDIA text and MIT carve-out as the code repository |
| Pinning | **By commit SHA — the dataset has no tags** and its history is actively rewritten. Cite a SHA, never `main`. |
| Fetch | `huggingface_hub.snapshot_download(repo_id="nvidia/aisimulate-fpm-dataset", repo_type="dataset", revision=<sha>)` |

Layout is
`data/<org>--<model>/<system>/<framework>/<version>/<parallelism>/{manifest.json,fpm/,measurements/}`.
The per-leaf measurement Parquets under `measurements/` are what this registry validates
against; the three configs the HuggingFace datasets-server exposes
(`catalog/configurations`, `fpm/files`, `measurements/files`, 326 rows in total) are
**file indexes, not measurements** — a reader who stops at those will conclude the dataset
is three hundred rows.

At the revision this registry validated against, the measurement Parquets hold **85,484
rows, every one `backend: vllm`**, splitting by the KV axis into 12,522 pure decode
(`total_prefill_tokens == 0`), 5,226 pure prefill (`total_kv_read_tokens == 0`) and 67,736
mixed. Derive these counts rather than quoting them: the dataset is rewritten, and the
`workload_kind` column labels any row carrying prefill tokens as `prefill`, which is a
coarser split than the one fitting needs.

FPM is used for validation and model selection. It may fit only a term that is itself a
property of the whole forward pass, never a per-primitive constant, and no committed
coefficient cites it. [`methodology.md`](methodology.md) §3 gives the argument, including
why an earlier, stricter rule was relaxed.

## 2. What each dataset's units are

Two traps, both of which have produced wrong numbers in this project before:

* **`latency` in every AISimulate Parquet is MILLISECONDS.** Reading it as seconds gives a
  1000× error that happens to look plausible on some parts.
* **FPM's `total_kv_read_tokens` is summed OVER THE BATCH**, while the kernel takes a
  per-request context length. Dividing by `batch_size` is required.

## 3. Regenerating each coefficient set

Run from the repository root with `AISIMULATE_DATA` and `BLIS_CATALOG` exported. Two sets
are fully generated and are compared by diff; the rest are fitted per part, and each
printed value should match the committed entry.

Entry counts quoted in this section were taken when each passage was written, and sets
have grown since. The current count of every set, by method, is on the
[reference pages](reference/index.md), which are generated from the committed files.

### `cost-model-primitives.yaml` — per-family writers

No single command regenerates this set; each family has an owner (table below), and
`emit_primitives.py` owns only the `vendor_spec` descriptor entries. Those it reads from
AISimulate's own `systems/<sku>.yaml` — the keys are `mem_bw_empirical_scaling_factor`,
`pcie_bw`, `misc.nccl_mem` (the rank-count→bytes map) and `misc.other_mem`, several of
which NVIDIA annotates in-file as "nonofficial correction based on observations". It takes
the `systems` directory, not `systems/data`:

```bash
python scripts/emit_primitives.py \
    <aisimulate>/python/aisimulate/src/aisimulate_core/systems "$BLIS_CATALOG"
```

and its output is a reference to compare the descriptor entries against, not a file to
commit whole (see the warning at the end of this section). Two descriptor keys are
deliberately not emitted — `mem_empirical_constant_latency` and `p2p_latency` — because
nothing in BLIS reads them and a reader would double-count the measured floors
([methodology §1](methodology.md#1-what-a-coefficient-is)). `a100_pcie` is skipped and the
`a100_sxm` descriptor is scoped to both A100 names
([methodology §5](methodology.md#a100-80-is-an-alias)).

The GEMM envelope re-derives, and is checked against the committed value, with its owner:

```bash
python scripts/relane_gemm_envelope.py --check
```

### `cost-model-collectives.yaml` — generated

This set is generated in two passes, because its entries come from two different lanes.

```bash
python scripts/emit_collectives.py "$AISIMULATE_DATA" > /tmp/collectives-nccl.yaml
```

That regenerates the NCCL-sourced entries. Checked against the committed file, **all 435
entries that cite an NCCL collection reproduce with no value difference.** The generator
is pinned to the collections those entries cite (`PINNED_NCCL`: 2.27.3 for A100 and L40S,
2.29.2 elsewhere) rather than taking the newest directory. That pin matters: AISimulate has
since added NCCL 2.30.7, which carries 60 rows under a narrower schema with no `device` or
`version` column where 2.29.2 carries 504 under the full one, so "newest wins" would both
crash and silently re-cite the entries. Set `NCCL_COLLECTION` to move a fit deliberately.

The remaining 45 entries are all `all_reduce`, and they do **not** come from NCCL. vLLM
uses its own one-shot/two-shot kernel for any tensor-parallel all-reduce that fits the
custom kernel's size limit, which AISimulate measures separately; on matched
`(num_gpus, message_size)` pairs the custom kernel under CUDA graph is the faster by a
median factor of 0.635, so an NCCL-derived coefficient over-charges a graph-mode
all-reduce by about 1.6×. Those 45 ship from the vLLM lane:

```bash
python scripts/fit_collectives_vllm.py --all "$AISIMULATE_DATA"
```

Re-running the NCCL pass alone will therefore differ from the committed file on exactly
those 45 entries, and on nothing else. A reproducer should expect that.

Every entry is `method: measured`. The 208 with `fitted: true` are curve fits; the
remainder are order statistics — a floor is the minimum over the size-independent region,
not a regression — which is why they are `fitted: false` while still being measurements
with their own Parquet citation.

The 72 `_a100_80` entries are the one group the NCCL pass cannot emit from this checkout:
`a100_pcie` ships no `comm/` tree. They are scoped to `a100-80` and carry the A100-SXM
measurement — 63 citing `a100_sxm/comm/nccl/2.27.3`, the other 9 the vLLM custom-all-reduce
sweep — so each names the sweep it actually comes from rather than claiming an A100-PCIe
measurement that does not exist. Treating the two A100 80GB parts as one for collectives is
a modelling choice, recorded in each entry's rationale.

### `cost-model-attention.yaml` — fitted per part

Decode (`attention_decode_floor`, `attention_decode_rate`), fitted on the vLLM lane:

```bash
python scripts/fit_attention.py "$AISIMULATE_DATA"/h200_sxm/attention/vllm/0.25.0 --chip h200
```

which prints `attention_decode_floor 13.5 us` and `0.54 of 4.800 TB/s` — that is the
committed `attention_decode_rate` of 2,592,000 bytes/µs — over 34,818 full-attention
points, the count its citation records.

Prefill (`attention_prefill_floor`, `attention_prefill_work_scale`):

```bash
# Five parts have a vLLM 0.25.0 context-attention sweep.
python scripts/fit_attention_prefill.py --collection vllm/0.25.0 \
    h200_sxm:h200 h100_sxm:h100 b200_sxm:b200 b300_sxm:b300 gb200:gb200-nvl72

# L40S and A100 do not. Each ships from the newest vLLM lane that has one,
# which is what their `sources` entries cite.
python scripts/fit_attention_prefill.py --collection vllm/0.24.0 l40s:l40s
python scripts/fit_attention_prefill.py --collection vllm/0.14.0 a100_sxm:a100-sxm
```

Expected output, matching the committed entries and the point counts in their citations:

| part | lane | floor µs | work_scale | points |
|---|---|---|---|---|
| h200 | vllm/0.25.0 | 18.5 | 0.44 | 27,693 |
| h100 | vllm/0.25.0 | 19.5 | 0.44 | 27,693 |
| b200 | vllm/0.25.0 | 16.5 | 0.38 | 27,574 |
| b300 | vllm/0.25.0 | 15.0 | 0.44 | 27,574 |
| gb200-nvl72 | vllm/0.25.0 | 17.0 | 0.38 | 27,574 |
| l40s | vllm/0.24.0 | 13.0 | 0.50 | 26,267 |
| a100-sxm | vllm/0.14.0 | 19.5 | 0.50 | 5,457 |

`--collection` defaults to `trtllm/1.3.0rc20`, the lane an earlier revision of these
coefficients was fitted on, so that fit also stays reproducible; pass the lane explicitly
for the values currently committed. Only full attention is fitted — rows with a non-zero
`window_size` read a bounded number of bytes, and fitting them together would fit one
curve to two byte counts.

### `cost-model-recurrent.yaml` — fitted per family

```bash
# KDA (Kimi-K3), h100 / h200 / gb300. Re-derives all six entries and diffs them
# against the committed file; --insert CHIP adds a part the set does not carry yet.
python scripts/relane_recurrent_kda.py --check

# MAMBA2 (Nemotron-3-Ultra). The model filter is required to reproduce the committed
# values: this sweep carries six geometries and pooling them fits none of them.
python scripts/fit_recurrent.py \
    "$AISIMULATE_DATA"/h100_sxm/linear_attention/trtllm/1.3.0rc20/mamba2_perf.parquet \
    --model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
```

The second prints `n=11 floor=3.5us rate=8.50 tok/us geo-err 1.120x`, which is both
committed `*_mamba2` value and the error its rationale states. Without `--model` the same
command pools 66 rows across six models and prints 4.2/23.25 — a number that is in the
file's history nowhere and should not be mistaken for a refit.

The KDA command prints `h100 3.8us/0.235`, `h200 4.3us/0.237` and `gb300 7.4us/0.457`,
which are the committed values. **Do not fit KDA from `kda/sglang/0.5.16` or take the
`kda_fused_decode` row**: an earlier revision of this guide did, and both are wrong.
`fused_kda_decode` is AMD-only — `vllm/models/kimi_k3/amd/ops/kda_decode.py` gates on
`gfx942`/`gfx950` — so it runs on no CUDA deployment, and vLLM's NVIDIA path
(`nvidia/kda.py:980-993`) runs `causal_conv1d_update` *then*
`fused_recurrent_kda_packed_decode`, so a layer costs the **sum**. The superseded recipe
gives floor 5.1 µs and rate 1.50 tok/µs, a rate 6.4× too fast.

What this data does and does not cover matters more here than anywhere else in the
registry, and the script's header says it: KDA (Kimi-K3) is complete, carrying the
convolution and the recurrent scan separately plus a fused decode variant. MAMBA2 is
**not** — the sweep has only `causal_conv1d_fn` and `causal_conv1d_update`, so the
committed `*_mamba2` pair is the convolution alone and is a documented LOWER BOUND on a
Mamba2 layer, not its cost. The selective-scan kernel is in no collection in the tree.

### Sliding-window attention — `*_swa`

The windowed entries are fitted by a separate script, because a windowed kernel reads a
number of bytes bounded by the window rather than by the context.
`scripts/relane_attention_swa.py` owns them and reads its values from
`scripts/fit_attention_by_kind.py`:

```bash
python scripts/relane_attention_swa.py --collection vllm/0.25.0 --check
```

The windowed entries were relaned from TRT-LLM to vLLM (methodology §7.1), and each
entry's citation names the collection it was fitted on: `vllm/0.25.0` on the Hopper and
Blackwell parts, `vllm/0.24.0` on L40S, whose newest vLLM collection that is. Pass the
collection the entry cites; the A100 pair is `assumed` and is written by
`extrapolate_by_generation.py` instead.

Using the context length for both kinds is what made an early windowed fit land at 2.00 of
datasheet peak — a physically impossible rate, and the reason the two kinds are fitted
apart rather than together.

### InferenceX CollectiveX run bodies (`$COLLECTIVEX_DATA`)

Three writers read CollectiveX run bodies: `fit_deepep_ll_graph.py`, `fit_host_link.py` and
`fit_deepep_collectivex.py`. Fetch the run list and each body from

```
https://inferencex.semianalysis.com/api/v1/collectivex/runs?version=1
https://inferencex.semianalysis.com/api/v1/collectivex/runs/<run_id>?version=1
```

and save each body as `cx_run_<run_id>.body`. The query must be the bare integer
`version=1`: without it, or with `v1`, the API answers `{"error":"Unknown version"}`. The
API keeps only its latest 15 runs, so a cited run can stop being fetchable. Each writer
checks that the bodies its committed entries cite are present and hash to the cited sha256.
If they are not, it exits 3 rather than re-fit and re-cite the entries on other runs, and
`validator/test_deepep_ll_graph.py` skips with "the cited CollectiveX bodies are not
present". The entries `fit_deepep_ll_graph.py` and `fit_host_link.py` wrote cite runs the
API no longer serves (33356406487, 36334392363, 36367074689, 36402856875, and the
swap_blocks bodies sha256 `26cad6ccbdaa`). They stay as committed, and those two gates skip
until the bodies are restored.

### Data freshness, checked 2026-10-30

| source | newest public | what the committed sets cite | outcome |
|---|---|---|---|
| AISimulate | `ai-dynamo/aisimulate` `7980e9500c00d9f85d7f296451d12768a743c70e` (2026-10-29) | `2f0f4cd` (main pin), `8ae6b658` (NVFP4 grouped, parked), `95b7f839` (DeepEP) | every parquet the Muse-relevant entries cite is byte-identical at `7980e95`. New since the pin: vLLM 0.30.0/0.31.0 gemm, moe and comm tables. h100/h200 all-reduce relaned to 0.31.0 (held-out better at every width). The 0.31.0 dense gemm tables and the 0.30.0 Marlin rows were scored on held-out points and do NOT beat the committed ramps, so they stay (see the PR) |
| FPM | `nvidia/aisimulate-fpm-dataset` `dbfcdd9eb62ea733efbc3f0cbd78277f2960061a` (2026-10-19) | `03e006de` (validation only, cited in attention rationales) | FPM fits no committed per-primitive value (§1), so nothing to refit |
| InferenceX CollectiveX | 20 runs served, newest `1793344844` (source_sha `54135009`) | `1793303708` (GB200/H200 DeepEP); 33356406487 and others (8-rank LL graph, host link), no longer served | every GB200/H200 deepep-v2 series in the new runs equals the cited run point for point; the writer now keeps the cited run |
| InferenceX OperatorX | `operatorx/compare?op=moe` responses, re-fetched: byte-identical to 2026-10-29 | runs 36292796863 / 36292840239 (InferenceX `8f03d27`) | nothing newer |
| InferenceX repo / db dump | `SemiAnalysisAI/InferenceX` `ea8b34394a603396d681e3c3d466d81cc885779b`; `InferenceX-app` `db-dump/2026-09-28` | evaluation only (§3) | no coefficient fitted from it |
| nccl-tests#309, route179.dev | issue last updated 2025-06-19; page sha256 unchanged | `cost-model-fabric` | unchanged |

### `cost-model-host-overheads.yaml` — not derived from either dataset

All seven entries are `method: assumed` and none is `fitted`. These are host-side costs
that neither dataset isolates — the script header notes that several could move to
`measured` without cluster time, and should. Three of them cite the removed `trained-physics` set as an
order-of-magnitude anchor only — the values there were fitted against a different
functional form, and a coefficient is valid only for the form it was fitted against. The
citations name the git SHA where that file can still be read.

### `cost-model-memory.yaml` — AISimulate's memory model, and vLLM's own logs

`scripts/emit_memory.py` writes the whole file, and `--check` gates it.

* The activation multiples and floor are `vendor_spec`, read with `ast` from AISimulate's
  `aisimulate_core/sdk/backends/` (located from `$AISIMULATE_DATA`, two directories up).
  Verified against AISimulate `70305030`.
* The CUDA-graph capture bytes are `assumed`, the median per mode of the committed
  `docs/vllm-cudagraph-capture-samples.csv`. That CSV is regenerated from the vLLM issue
  tracker, which needs an authenticated `gh` and changes as issues are filed — so a
  re-mine is a deliberate data update, committed with its diff, not part of `--check`:

```bash
python scripts/mine_vllm_cudagraph_memory.py fetch   /tmp/vllm-cg-cache
python scripts/mine_vllm_cudagraph_memory.py extract /tmp/vllm-cg-cache \
    > docs/vllm-cudagraph-capture-samples.csv
python scripts/emit_memory.py            # then review the value diff
```

The capture half needs no AISimulate tree, so
`test_capture_bytes_re_derive_from_the_committed_samples` runs it in CI. Methodology §11
gives the composition these terms enter and why each is scoped as it is.

### Every committed value, and the one command that regenerates it

The table below is the reproducibility contract: each family names the script that owns
it, and each of those scripts takes `--check`, which re-runs its own fitter and exits
non-zero if the committed file would change. `validator/test_writers_are_idempotent.py`
runs them all, and `test_every_fitted_family_has_a_writer` fails if a fitted family
appears with no owner — so a value that cannot be regenerated cannot be added.

| family | writer | source |
|---|---|---|
| `gemm_eps_max_*`, `gemm_m_half_*` | `relane_gemm_envelope.py` | AISimulate `gemm`, vLLM lane |
| `moe_routing_imbalance_median` | `relane_moe_imbalance.py` | AISimulate `moe`, vLLM lane |
| `gemm_*_w4a16_mxfp4_grouped` | `fit_marlin_grouped_ramp.py` | AISimulate `moe`, vLLM `MarlinExperts` MXFP4 rows, h200, h100, gb300, b200 and b300; see methodology §10.5 for where each applies |
| `gemm_*_w4a16_nvfp4_grouped` (h100, h200) | `fit_marlin_nvfp4_grouped_ramp.py --check` (`$OPERATORX_DATA`) | InferenceX OperatorX MoE, MiniMax-M3, `MarlinExperts` NVFP4 (E4M3 scale per 16, 0.5625 B/weight) |
| `attention_decode_{floor,rate}` | `relane_attention_decode.py` | AISimulate `attention`, vLLM lane |
| `attention_decode_*_swa` | `relane_attention_swa.py` | same, windowed rows |
| `attention_decode_*_mla` | `relane_attention_mla.py` | AISimulate `mla` module tables |
| `attention_decode_rate_mla` | `relane_attention_mla.py --check-rate` | same; the FLOOR half is owned by `correct_mla_floor.py`, see methodology §3 |
| `attention_decode_floor_mla` | `correct_mla_floor.py` | derived from each part's own `attention_decode_floor` (the module tables are rejected for it) |
| `attention_decode_rate_sparse_mla` | `fit_deepseek_v41_h200.py` — scoped to deepseek-v4.1-flash on h200 and h100 only, each fitted from its own part's rows | AISimulate `sparse_attention`, the FlashMLA kernel table; no part-wide rate is fitted, because for deepseek-v4-pro it is identifiable for only one of two sparse geometries (methodology §10.3, §10.5) |
| `attention_decode_{launch_floor,split_floor,serial_rate,per_token,throughput_rate}_sparse_mla` | `fit_deepseek_v41_decode.py` — deepseek-v4.1-flash on h200 and h100 | the same FlashMLA kernel table's decode rows, at most 10 blocks (methodology §10.7) |
| `attention_decode_*_block_index_scores` | `fit_deepseek_v41_h200.py` | the floor from AISimulate's indexer-kernel table; the rate `assumed` at NVIDIA's empirical HBM rate, BwPeak × `hbm_derate` (methodology §10.5) |
| `gemm_*_w8a16_mxfp8`, `host_gather_rate_uva` | `fit_deepseek_v41_h200.py` (`assumed`) | upper bounds derived from committed siblings; no sweep measures them |
| `attention_prefill_*` | `relane_attention_prefill.py` | AISimulate `attention` context rows |
| `attention_prefill_*_sparse_mla`, `attention_prefill_*_block_index_scores`, `mhc_*` | `fit_deepseek_v41_prefill.py` — scoped to deepseek-v4.1-flash on h200 and h100 | AISimulate `sparse_attention` FlashMLA prefill rows and the paged-logits kernel at prefill shapes (a proxy, `assumed`); `mhc/vllm/0.25.0` (methodology §10.6) |
| deepseek-v4.1-flash `host_launch_eager_per_kernel`, `host_admission_*` | `fit_deepseek_v41_host.py` | vendored `validator/testdata/anchors/` (the derived inputs of an internal enforce-eager report, via `--extract-anchor`, re-verified when `V41_EAGER_ANCHOR_REPORT` names a local copy; the kernel basis; the front-end timing) |
| `recurrent_decode_*_kda` | `relane_recurrent_kda.py` | AISimulate `kda`, vLLM lane |
| `recurrent_decode_*_{gdn,mamba2}` | `relane_recurrent_family.py --family X` | AISimulate `linear_attention` |
| `collective_*` (measured widths) | `emit_collectives.py`, `fit_collectives_vllm.py` | AISimulate `comm` |
| `collective_*_all_reduce_fp16_{2,4,8}rank_{h100,h200}` (their 16-rank extrapolations follow, `extrapolate_collective_width.py`) | `relane_allreduce_vllm.py --check` (`$AISIMULATE_ALLREDUCE_DATA`, aisimulate `7980e95`) | AISimulate `comm/vllm/0.31.0` custom all-reduce, CUDA-graph lane |
| `collective_*` (new part) | `insert_collectives_part.py` | same, per operator per lane |
| `collective_*` (wide groups) | `insert_collectives_wide.py` | same, widths 8 and 16 |
| descriptors (`vendor_spec`) and A100's `moe_routing_imbalance_median` | `emit_primitives.py --check` (entry-level) | NVIDIA `systems/<sku>.yaml`; AISimulate `moe/sglang/0.5.10` for A100 |
| `host_link_bandwidth` (measured, six parts) | `fit_host_link.py --check` (`$COLLECTIVEX_DATA`) | InferenceX CollectiveX swap_blocks, contiguous, payload >= 64 MiB; h100 is the catalog `cpu_dram` fit's larger direction |
| `moe_a2a_*_deepep_low_latency_graph_8rank_{h100,h200}` | `fit_deepep_ll_graph.py --check` (`$COLLECTIVEX_DATA`) | InferenceX CollectiveX deepep-v2 low-latency, EP8 one node |
| `moe_a2a_*_deepep_{low_latency_graph,v2_graph}_{8,16}rank_gb200_nvl72`, `..._16rank_h200` | `fit_deepep_collectivex.py --check` (`$COLLECTIVEX_DATA`) | InferenceX CollectiveX deepep-v2 low-latency and normal; GB200 EP8/EP16 inside one NVL72, H200 EP16 over two nodes (scoped `fabric: [ib-400g]`) |
| `gemm_*_nvfp4_grouped` (PARKED, none committed) | `fit_nvfp4_grouped_ramp.py` (`$AISIMULATE_NVFP4_DATA` at aisimulate `8ae6b658`) | AISimulate `moe/vllm/0.30.0` TrtLlmNvFp4ExpertsMonolithic; its held-out error is no better than the dense `nvfp4` ramp's on any part, so the kernel's fallback stands (see the script's docstring) |
| `fabric_bandwidth_efficiency` (`cost-model-fabric`) | `emit_fabric.py --check` (`--verify DIR` re-hashes the cited pages) | NVIDIA/nccl-tests#309 for ib-400g (`literature`); roce-200g and ethernet-100gbe `assumed` with stated ranges |
| host overheads (`assumed`) | none — see that file's header | not measured by either dataset |
| `activation_buffer_count_*`, `activation_scratch_floor_bytes` | `emit_memory.py` | AISimulate `sdk/backends` source (`vendor_spec`) |
| `cudagraph_capture_bytes_*` | `emit_memory.py` | `docs/vllm-cudagraph-capture-samples.csv`, from `mine_vllm_cudagraph_memory.py` (`assumed`) |

**Derived entries carry their own reproduction too.** Two scripts write `method: assumed`
values, and both state the predictor, its holdout error and the parts it was derived
from, inside the entry:

| script | what it derives | predictor, and how it was chosen |
|---|---|---|
| `extrapolate_by_generation.py` | a family absent for one part | same-generation median, or a kernel-matched within-part ratio; both chosen by leave-one-part-out holdout against every part that *is* fitted |
| `extrapolate_collective_width.py` | a 16-rank collective triple | `floor(n) = a + b(n−1)` on that part's own {2,4,8}, holdout-validated against gb200-nvl72, the one part with a measured 16-rank floor |

Neither script will write a value it cannot defend: they refuse a part whose generation
has no fitted sibling, a prediction below the measured 8-rank floor, a set of fit widths
spanning two lanes, and a lane with no same-lane holdout. A refusal is the correct output
when the data does not support an estimate.

### Adding a part — the GB300 worked example

Every fitted family has a writer with `--check` (it re-runs its own fitter and exits
non-zero if the committed file would change) and `--insert CHIP` (it appends a part the
set does not carry yet, through the same renderer, so an inserted entry is byte-identical
to what a later re-run produces). The order matters in one place: prefill attention reads
the bf16 GEMM ramp **from the registry**, so the envelope must be committed first.

```bash
export AISIMULATE_DATA=<aisimulate>/python/aisimulate/src/aisimulate_core/systems/data
export BLIS_CATALOG=<path-to-blis-catalog>

# 1. descriptors (vendor_spec, from NVIDIA's own systems/gb300.yaml)
#    emit_primitives.py owns this family; its GEMM/MoE path is superseded — see below.
# 2. GEMM ramp, then MoE imbalance
python scripts/relane_gemm_envelope.py  --insert gb300
python scripts/relane_moe_imbalance.py  --insert gb300
# 3. attention: decode, then prefill (needs the ramp from step 2), then windowed
python scripts/relane_attention_decode.py  --insert gb300 gb300:gb300
python scripts/relane_attention_prefill.py --insert gb300 gb300:gb300
python scripts/relane_attention_swa.py     --insert gb300 gb300:gb300
# 4. recurrent
python scripts/relane_recurrent_kda.py     --insert gb300 gb300:gb300
# 5. collectives, per operator from the lane each belongs on
python scripts/insert_collectives_part.py --chip gb300 --sku gb300
```

Then assert idempotence, which is what makes the result auditable:

```bash
for s in relane_gemm_envelope relane_moe_imbalance relane_attention_decode \
         relane_attention_prefill relane_attention_swa relane_recurrent_kda; do
    python scripts/$s.py --check || echo "$s DRIFTED"
done
python -m pytest validator/ -q
```

**Do not run `emit_primitives.py` or `emit_collectives.py` to regenerate a whole set.**
Both predate lane decisions that moved families off them. `emit_primitives.py`'s
`COLLECTIONS` table still pins `trtllm/1.3.0rc20` while every committed `gemm_*` entry
cites a vLLM lane (5be809b), and regenerating `cost-model-collectives.yaml` from NCCL
gives 144 removals and 36 changes — all 36 being the `all_reduce` entries 2f71dd6 moved to
vLLM's custom kernel, so it would silently re-introduce a 3.4× mispricing. Use the
per-family writers; they are what `--check` guards.

## 4. Why the registry holds only what the kernel reads

`blis-latency-kernel` requests five sets — `cost-model-primitives`,
`cost-model-collectives`, `cost-model-host-overheads`, `cost-model-attention`,
`cost-model-recurrent` — and when the removals below were made `coefficients/` held exactly
those, 724 entries.

A sixth, `cost-model-memory`, is the one deliberate exception, and it is committed **ahead
of its reader** rather than without one: it supplies magnitudes the kernel was hardcoding
(inference-sim/blis-registry#33), and the kernel change that reads them,
inference-sim/blis-latency-kernel#22, is sequenced to land after it. Until then no scenario
lists it. The rule this section states still applies: if that reader does not land, the
set should go.

Four sets were removed because nothing in BLIS read them; they are recoverable from git history, where they last appear at 8d78ff8:

| set | what it held | why it went |
|---|---|---|
| `communication-coefficients.yaml` | 10 named TP/MoE collectives | the kernel prices collectives from `cost-model-collectives`; 9 of its 10 entries were `assumed`/`copied` placeholders, 7 of them the value `1.0` |
| `pd-transfer-estimates.yaml` | PD KV-transfer base latency, fabric overhead | `PDTransferTime` prices from catalog facts instead |
| `legacy-kv-transfer.yaml` | pre-#1590 CPU↔GPU transfer defaults | `TierTime` prices from catalog facts instead |
| `lora-adapter-costs.yaml` | Digital-Twin adapter terms | no consumer |

The kernel's `TierTime` and `PDTransferTime` take their numbers from blis-catalog hardware
facts — `StorageDevice` read/write bandwidths and base latency, `IntraNodeBwGBps`,
`InterNodeBwGBps` — using only `host_link_bandwidth` from `cost-model-primitives` as a
host-link ceiling. `inference-sim` does not read any of the four either; its own
`defaults.yaml` carries the values its backends use.

**The removal was verified to change nothing measurable.** Scored against the InferenceX
measured tier (`-config-tier measured -framework vllm -length-range-ratio 1.0`), the
before and after reports are identical after stripping timestamps — every summary table,
every GPU family, every model, the same `n`:

| registry | TPOT shape | TPOT mape | TTFT shape | TTFT mape |
|---|---|---|---|---|
| 747 entries, 9 sets | 11.60% | 13.52% | 15.32% | 50.68% |
| 724 entries, 5 sets | 11.60% | 13.52% | 15.32% | 50.68% |

Both runs used one binary, one catalog revision and one kernel revision, varying only the
registry — the figures move with the catalog, so a comparison across catalog revisions
proves nothing. `n` is 292/360/288/356.

Anyone extending the kernel to price offload tiers, PD transfer or LoRA should expect to
calibrate those terms rather than recover them from history: the removed values were
placeholders and transcriptions, not measurements.

## 5. Checking a re-derivation

```bash
python -m pytest validator/ -q        # derivation / value-preservation gates (schema validation lives in blis-schemas)
python scripts/validate_against_aisimulate_tables.py   # oracle check against AISimulate's own tables
```

A re-fit that lands on a different number is not automatically a correction. This registry
has recorded six independently verified per-primitive improvements that each made
end-to-end accuracy worse, because the composed model's accuracy rests on partially
cancelling errors. No per-primitive refit ships on the strength of its own fit quality
alone; see the standing rule in [`methodology.md`](methodology.md).

## 6. Reproducing the evaluation figures exactly

The accuracy numbers quoted for this registry are reproducible, but three details
decide whether a reader gets the same figures, and each has silently produced
wrong numbers during this work.

### 6.1 The scorer is not on `main`

> **Status (2026-10-30).** [inference-sim#1851](https://github.com/inference-sim/inference-sim/pull/1851)
> merged on 2026-10-13 (`0d092d4`), and the `kernel-exclusive` branch below no longer exists.
> The scorer now lives at `tools/metricscore` on `inference-sim/inference-sim` main.
> `scripts/compare_registries.py` still runs `go run ./cmd/metricscore` from
> `/private/tmp/blis-kernel-wt`, so it needs that path updated before it can run. The scorer
> also needs three corpora (`-corpus aisimulate_e2e.json`, `-engine-settings
> inferencex_engine_settings.json`, `-absolutes inferencex_absolutes.json`, or
> `BLIS_MEASUREMENTS`). blis-latency-kernel does not distribute them: its `testdata/README.md`
> says they are extracted from third-party publications by `scripts/extract_*`. Only
> `aisimulate_e2e.json` was available for the Muse Spark change (#60), so that change ships
> without an end-to-end score; see the change's pull request. Separately, no committed
> scenario lists `cost-model-collectives-inferencex` or `cost-model-fabric` (the scenarios list
> six sets, `cost-model-primitives` to `cost-model-memory`). An end-to-end score of entries in
> those two sets is therefore unchanged by construction until a scenario names them.

`cmd/metricscore` exists only on the `kernel-exclusive` branch of the
`sriumcp/inference-sim` fork, the head of the still-open
[inference-sim#1851](https://github.com/inference-sim/inference-sim/pull/1851).
`inference-sim/inference-sim` carries neither that branch nor the command, and a checkout
of its `main` fails with `stat cmd/metricscore: directory not found`.

```bash
git clone --branch kernel-exclusive https://github.com/sriumcp/inference-sim /tmp/blis-scorer
cd /tmp/blis-scorer
```

### 6.2 `-registry` defaults to the sibling checkout, not to your work

The flag's default is `/Users/sri/Documents/Projects/blis-registry` — the main
checkout. Scoring a branch or worktree without passing `-registry` explicitly
scores the *other* registry and reports figures that do not belong to the tree
under test. Always pass it:

```bash
go run ./cmd/metricscore \
  -config-tier measured -framework vllm -length-range-ratio 1.0 \
  -registry /path/to/the/registry/under/test
```

### 6.3 The three flags each change every figure

| flag | omitted | passed |
|---|---|---|
| `-config-tier measured` | mixes measured and resolved configurations in one average | only the 68 sweeps whose engine settings come from the run's own command line |
| `-framework vllm` | scores every framework, which only the published arms can do | the subset BLIS models |
| `-length-range-ratio 1.0` | AISimulate's one-sided `[0.8·len, len]` sampling | constant prompt lengths, which is what vLLM's client produces with no `--random-range-ratio` |

On the main-checkout kernel (`a9b60e1`) and registry (`ea08479`), the same
command with and without `-length-range-ratio 1.0` gives 11.62/12.05/15.23/52.94
and 11.22/12.19/16.22/57.13 respectively. Neither is wrong; they answer different
questions, and a figure quoted without its flags cannot be checked.

### 6.4 Which tier to weigh

`inferencex_engine_settings.json` records 68 sweeps with a captured engine-args
log and 136 without, each of the latter listed in its `incomplete` array with
`reason: "no engine-args log"`. For the 136 the engine configuration is inferred
from vLLM's defaults rather than observed, so an error on those points is as
likely to be a wrong assumed configuration as a wrong model. The measured tier is
the only one where the simulated deployment is ground truth, and it is the tier
on which a coefficient change can be attributed to the model.

### 6.5 Rebuild the kernel between runs

`go run` recompiles, but a pre-built binary does not. A stale binary has
invalidated a result during this work. When scoring repeatedly, build once per
kernel revision and name the binary after it.
