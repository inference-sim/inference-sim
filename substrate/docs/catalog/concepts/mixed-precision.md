# Mixed-precision checkpoints

Some checkpoints store different layers at different widths, and say so layer by layer. The
catalog places such a layout in the graph only when the checkpoint states it completely and
regularly; this page says how, for the one checkpoint that does so today,
`minimax-m3-nvfp4` (`nvidia/MiniMax-M3-NVFP4`).

## What the checkpoint stores

ModelOpt writes `quant_algo: MIXED_PRECISION` and a `quantized_layers` map from each quantized
module to its own algorithm. For this checkpoint the map has 22,422 entries:

| Modules | Algorithm | Count |
| :--- | :--- | ---: |
| each routed expert's `w1`, `w2`, `w3`, on the 57 routed layers, 128 experts each | NVFP4, group size 16 | 21,888 |
| `q_proj`, `k_proj`, `v_proj`, `o_proj` on all 60 layers | MXFP8 | 240 |
| `index_q_proj`, `index_k_proj` on the 57 block-sparse layers | MXFP8 | 114 |
| the dense MLP's `gate_proj`, `up_proj`, `down_proj` on the 3 dense layers | MXFP8 | 9 |
| the shared expert's `gate_proj`, `up_proj`, `down_proj` on the 57 routed layers | MXFP8 | 171 |

The router, the embedding, the head and the vision tower are not in the map, so vLLM builds
them unquantized. The 88 safetensors headers agree: expert weights are `U8` (two FP4 values to
a byte) with an `F8_E4M3` scale per 16 values, every listed linear is `F8_E4M3` with a `U8`
scale per 32 values, the router is `F32` and the head and embedding are `BF16`.

## What vLLM does with it

`ModelOptMixedPrecisionConfig` (`vllm/model_executor/layers/quantization/modelopt.py:1543-1892`
at v0.32.0) looks each module up in `quantized_layers`. A listed linear gets the method of its
algorithm; a linear that is not listed is built unquantized (`:1851-1854`); a routed-experts
module takes the algorithm of the first entry under its prefix (`:1754-1767`); a fused
projection takes the algorithm of its shards, so the fused `qkv_proj` that also holds the
indexer's two projections is decided by `q_proj`, `k_proj` and `v_proj` (`:1734-1752`,
`:1769-1795`). A stated `--quantization` must equal the method the checkpoint resolves to,
which is `modelopt_mixed` (`:1626-1632`); that resolution is vLLM's, made by the engine model.

## How the deriver places it

`minimax_m3_mixed_precision` in `catalog/scripts/derive_graph.py` builds, from the config's own
layer vectors (`sparse_attention_freq`, `moe_layer_freq`) and expert count, the exact set of
modules the layers contain, and requires the map to hold that set and no other. It then
requires one algorithm per class: MXFP8 for every linear, NVFP4 with group size 16 for every
routed expert. Anything else is refused with the module named, because one graph node prices
one width and a role stored two ways would be priced at one:

- a module missing from the map, or a listed module the layers do not build;
- an expert matrix or a linear at another algorithm or group size, on any layer;
- a quantized KV cache;
- an exclusion list (`ignore` in `config.json`, `exclude_modules` in `hf_quant_config.json`;
  vLLM v0.32.0 reads the first from `config.json`, `modelopt.py:320-406`) that names a listed
  module, since the two files would then place it differently.

The graph is the MXFP8 graph of `minimax-m3-mxfp8` with the routed experts at NVFP4 and the
shared expert as a node of its own at FP8 (the same split `minimax-m3.1-flash` uses). The
global width is FP8 with the microscaling scale, the router FP32, the head BF16.

The kernel does not recognise the spelling `modelopt_mixed`: it is vLLM's name, and the cost model
resolves no engine's setting. The engine model resolves it, refuses what vLLM refuses, and states
the layout it serves in `Settings.ServedWeights` (for this checkpoint: linear FP8 with one scale
byte per 32 weights, routed experts NVFP4, router FP32, head BF16, embedding BF16). The kernel
proves the graph stores exactly that, class by class (`checkServedWeights`); the
[refusals page](../../kernel/reference/refusals.md) lists what is refused.

## Reproduce

```sh
rev=901464083161bf8612a29ff7ad29914cd4ab4a85
curl -fsSL https://huggingface.co/nvidia/MiniMax-M3-NVFP4/resolve/$rev/config.json | shasum -a 256
# 440d22038ffa0bcb93fa01e8d2d555e3840d939167c08eace81b9cce386fb117, the file committed as
# catalog/models/minimax-m3-nvfp4/config.json and the digest graph.yaml records
python3 catalog/scripts/derive_graph.py --check --model minimax-m3-nvfp4
python3 -m pytest catalog/tests/test_derive_graph.py -k m3_nvfp4
```

The derivation reads one 2 MB JSON file and takes under a second. The tests include a table of
every tensor class in the 88 headers (89,091 language-model tensors, 248,357,099,520 bytes) and
check that the graph's priced weights and scale bytes equal it, layer by layer.

## What this does not show

- It does not measure anything. NVFP4 grouped matrix multiplies on Blackwell have no ramp of
  their own in the registry: they are priced on the dense NVFP4 ramp, which the kernel records
  as an assumption (`gemm_eps_max_nvfp4_grouped`, method `assumed`), so a point priced from
  this graph carries that label. No coefficient was added for it.
- The graph carries no declared quantization method. It cannot tell this checkpoint from
  MiniMax-M3.1-Flash, which stores the same widths under `quant_method: mxfp8`; telling them
  apart is the engine model's, which reads the checkpoint's own declaration.
- No MiniMax-M3 checkpoint's safetensors index (bf16, MXFP8, NVFP4) holds a tensor above layer
  59 or named for an MTP module. The graphs state an in-graph speculator because the config
  does (`num_mtp_modules`); whether the engine loads MTP from these repositories is not shown.
  A deployment that names a separate drafter replaces it in any case.
- Other mixed-precision checkpoints (Nemotron's and DeepSeek-V4's layouts are read by their own
  handlers) are not read by this function; a MiniMax-M3 config whose map differs from the
  regular layout above is refused.

## Where the data came from

`nvidia/MiniMax-M3-NVFP4`, revision `901464083161bf8612a29ff7ad29914cd4ab4a85`, retrieved
2026-11-11, under the MiniMax Community License: the repository's `LICENSE` is byte for byte the
text already on the allow-list (SHA-256 `b53f2fdd…`), so `config.json` is committed and the
licence index lists the model with that text.
