# Price-change report: refused settings, the GLM-5 cache dtype and the block-size resolution

Change (design X13), four items of one review round:

1. a stated `kv_cache_memory_bytes` is refused on the v0.32.0 path (it was sized from 0.92);
3. an unstated `cache_dtype` of a GLM-5-family model on compute capability 10.x resolves to `fp8_e4m3`
   (`vllm/model_executor/models/config.py:43-55@5df63a72`);
6. the block-size resolution accepts `fp8_e4m3`, `bfloat16` and `float16` (the oracle now runs them);
7. the block-size resolution refuses speculation only when the method adds attention layers.

Item 1 refuses, and items 6 and 7 turn a refusal into a resolution; none of the three changes a price
that was computed. Item 3 changes the bytes of every KV block of a GLM-5-family
model whose deployment states no cache dtype, so it moves the KV block count and the step time.

## Method

`compat/kernelmodel/pricereport_test.go` (`TestWritePriceReport`, run with `BLIS_PRICE_REPORT=<file>`;
`BLIS_PRICE_VARIANT` picks the variant below)
opens every pool of every v0.32.0 scenario fixture in the repository (`kerneltests/testdata`, its
`aisimulate` and `direct` directories, `testdata/scenarios`, `vllmengine/testdata`: 724 pools) and
writes, for each, the engine's resolved block size and cache dtype, the per-rank KV block count and
bytes a block, and the kernel's NoOverlap step time, unrounded, in microseconds at decode batch 1, decode batch
256 (1,023 tokens computed and one scheduled, so a context of 1,024; one token each), prefill of one 2048-token request, and decode batch 700.
Three runs on the code before (the head of the pull request, 0cd33933) and after, each with a
variant set by `BLIS_PRICE_VARIANT`: the fixtures as committed (`""`), with `cache_dtype` removed
(`nocache`), and with `block_size` removed (`nobs`). The "before" run used a checkout of the
earlier head with only the generator file added.

## Result

- Fixtures as committed: no price moved (0 of 724 pools). Every committed fixture that states a cache
  dtype states it, and a stated dtype is never changed; no fixture states `kv_cache_memory_bytes`.
- `block_size` removed: no price moved (0 of 724). The rows that resolve a block size resolve the same
  one before and after: no committed fixture states `fp8_e4m3`, `bfloat16` or `float16`, or a
  speculative method that loads no model, so items 6 and 7 move nothing that is committed.
  The census and the recorded-run comparison (`blocksize_census_test.go`,
  `blocksize_recorded_test.go`) assert their counts exactly.
- `cache_dtype` removed: four pools moved, all GLM-5 on Blackwell. They are the only committed
  deployments of a GlmMoeDsaForCausalLM model on compute capability 10.x. Bytes a block fall from
  5,750,784 to 2,875,392 and the pool doubles; decode at batch 256 and 700 get cheaper (the cache
  read halves), decode at batch 1 moves by under 10 microseconds, prefill does not move.

| fixture | blocks before | blocks after | decode b1 | decode b256 | prefill 2048 | decode b700 |
|---|---|---|---|---|---|---|
| aisimulate/glm-5-b200-fp4-sglang-tp4 | 12,059 | 24,119 | 10968.184 to 10959.885 | 36304.058 to 34179.670 | 79433.977 (same) | 61769.952 to 55961.080 |
| aisimulate/glm-5-b200-fp8-sglang-tp8 | 14,122 | 28,244 | 11976.900 to 11968.601 | 37472.377 to 35347.990 | 93449.158 (same) | 66315.654 to 60506.781 |
| aisimulate/glm-5-b300-fp4-sglang-tp4 | 27,405 | 54,811 | 10778.427 to 10769.734 | 36070.274 to 33844.725 | 76435.315 (same) | 61459.508 to 55374.023 |
| aisimulate/glm-5-b300-fp8-sglang-tp8 | 29,468 | 58,936 | 11906.430 to 11897.736 | 37324.913 to 35099.364 | 90566.083 (same) | 66034.126 to 59948.640 |

A pool that moves in production is therefore a GLM-5, GLM-5.2, GLM-5.2-fp8 or GLM-5.3 pool on
B200, B300, GB200 or GB300 that states no `cache_dtype` (or states `auto`). Hopper pools, and a pool
that states `bfloat16` or `fp8`, do not move. The doubling is exactly the ratio of bytes a block.

Goldens under `oracle/` and the kernel tests did not move; none prices an unstated GLM-5 cache dtype
on Blackwell.

Item 1 as a number: before, `llama-3.1-70b-instruct-h200-fp8-vllm-tp2` with
`kv_cache_memory_bytes: 4294967296` in place of `gpu_memory_utilization` was sized at 76,727 blocks
from 0.92 (about 23 times what 4 GiB holds); now `engine.New` refuses it, citing blis-vllm#81.
