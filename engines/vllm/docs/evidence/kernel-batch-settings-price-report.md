# Price-change report: the kernel prices the batch settings Engine reports

Change: `compat/kernelmodel` copies the resolved `max_num_seqs` and `max_num_batched_tokens`
(`Model.Engine`) into the settings the latency kernel is built with, beside `block_size` and
`gpu_memory_utilization`. Before, both stayed 0 and the kernel fell back to its own vLLM v0.31.0
defaults, which skip the caps v0.32.0 applies (batched data-parallel 256,
`min(max_num_seqs, max_num_batched_tokens)`, `max_num_seqs x max_model_len`). Design X13.

## Method

Every v0.32.0 pool in the committed scenario fixtures (`kerneltests/testdata`,
`kerneltests/testdata/aisimulate`, `testdata/scenarios`, `vllmengine/testdata`: 133 pools) was
opened with `kernelmodel.OpenPool` and priced with the kernel's NoOverlap step time (microseconds)
before and after the change, at: decode batch 1 and batch 256 (context 1024, one token each),
prefill (one request, 2048 tokens), and decode batch 700 (context 1024), which is above the
capture ceiling the cap can change. Three runs: the fixtures as committed; the fixtures with
`max_num_seqs` and `max_num_batched_tokens` removed; and the same with
`all2all_backend: deepep_low_latency` added.

The figures are written by `compat/kernelmodel/pricereport_test.go` (`TestWritePriceReport`, with
`BLIS_PRICE_REPORT=<file>`; the third run is `BLIS_PRICE_VARIANT=nobatch-deepep`). It prints the step times
unrounded, in microseconds, at a decode context of 1023 computed tokens. Run on the code before this
change (the commit before it) and after, it reproduces the table below to the digit: all seven rows,
in every column. It finds 59 pools that move in that third run, not seven: the seven below are the
ones of the original table, and the other 52 are the `direct` fixtures and other pools of the same
shape (expert parallel, DP > 1), which that table did not list because its run covered only the
133 pools of the four directories it names. The cause and the size of the change are the same.

## Result

- Fixtures as committed: no price moved. Every v0.32.0 fixture states both settings, so the
  resolved value is the stated one, which the kernel already read.
- Both settings removed from every v0.32.0 fixture: no price moved. Without a batched data-parallel
  MoE no cap applies that the kernel's fallback lacks, so both resolve to the same values.
- Both removed and `deepep_low_latency` added (a hypothetical, no committed fixture states it):
  seven pools with expert parallel and DP > 1 moved. Decode at batch 1, at batch 256 and prefill
  did not move; decode at batch 700 did. Cause: Engine and the scheduler run `max_num_seqs=256`
  and 256 batched tokens, so the kernel now prices the 256-sequence engine (its capture ceiling follows the
  setting) instead of a 1024-sequence one.

| fixture | decode b1 | decode b256 | prefill 2048 | decode b700 before | decode b700 after |
|---|---|---|---|---|---|
| aisimulate/minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2 | 5504.252 | 15225.705 | 34254.726 | 23780.195 | 28995.195 |
| aisimulate/minimax-m2.5-b300-fp4-vllm-tp2-ep4-dp2 | 5401.607 | 15162.212 | 33330.111 | 23784.690 | 28999.690 |
| aisimulate/minimax-m3-b300-fp4-vllm-tp2-ep4-dp2 | 7284.498 | 21986.467 | 47303.722 | 32182.506 | 38342.506 |
| testdata/d079-032 | 5596.502 | 28437.611 | 49194.626 | 51445.337 | 56226.337 |
| testdata/d088-032 | 5491.657 | 28848.753 | 49100.276 | 52754.905 | 57535.905 |
| testdata/d103-032 | 5491.657 | 28848.753 | 49100.276 | 52754.905 | 57535.905 |
| testdata/d116-032 | 8586.343 | 31242.415 | 61445.481 | 56227.917 | 61967.917 |

A pool that moves in production is therefore a v0.32.0 pool with `all2all_backend`
`deepep_low_latency` or `nixl_ep`, expert parallel and DP > 1, that states neither batch setting.
Goldens under `oracle/` and `kerneltests` did not move (they price stated settings).
