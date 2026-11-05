# Model graphs

A model's `config.json` describes the model in the vocabulary of whoever trained it. A cost model needs something else: the operations a forward pass launches on the GPU, each with the dimensions that determine its cost. A *model graph* is that description. The catalog commits one for every model, as `models/<name>/graph.yaml`, generated from the config by a script.

## Why the config is not enough

Configs disagree about how to say the same thing. Hidden size is `hidden_size` in one config, `d_model` in another and `n_embd` in a third. Head dimension is stated by some configs and omitted by others, because it equals the hidden size divided by the number of heads. One hybrid model in the catalog states no layer count at all: its depth is the length of a 108-entry list of layer types. A cost model that read configs directly would need a reading of every vocabulary, and each reading would be a place for the meaning of a field to diverge.

Configs also describe parameters, not work. A config field that rescales a softmax launches no GPU operation, and a collective that runs only under tensor parallelism appears in no config. The graph records what runs.

## What a graph contains

A graph is built from *primitives*, the operations a cost model knows how to price. The schema defines nine:

| Primitive | What it is |
| :--- | :--- |
| `GEMM` | a dense matrix multiply: a projection, the LM head |
| `GroupedGEMM` | the expert matrix multiplies of a mixture-of-experts (MoE) layer |
| `Attention` | attention over the batch. Its `kind` is `gqa`, `mla`, `sparse_mla` or `swa`. |
| `RecurrentUpdate` | a linear-attention or state-space layer, whose state does not grow with context. Its `recurrent_kind` is `mamba2`, `kda` or `gdn`. |
| `Elementwise` | normalization, activation, rotary embedding: work limited by memory bandwidth |
| `AllReduce`, `AllGather`, `ReduceScatter`, `All2All` | collectives that move data between GPUs |

The [glossary](glossary.md) defines each attention and recurrent kind. The models in this release use [[catalog.count.primitives_used]] of the nine: [[catalog.primitives_used]].

Primitives are grouped into *layer kinds*. A layer kind is a small acyclic graph of primitives, and a model has one for each distinct layer structure. A *stack* says how the kinds repeat. A uniform 40-layer model is one kind repeated 40 times. A hybrid that interleaves three recurrent layers with one attention layer is a four-entry pattern of two kinds, repeated. A model with no repeating structure lists its layers in full. The stack also allows an irregular run of layers before the repeating part and after it, because several models open with a few dense layers before their MoE layers begin.

Qwen3-14B's single layer kind, as committed:

```yaml
- id: dense
  nodes:
  - {op: Elementwise, role: input_norm}
  - {op: GEMM, role: qkv_proj, n: 7168, k: 5120}
  - {op: Attention, kind: gqa, n_q: 40, n_kv: 8, d_h: 128}
  - {op: GEMM, role: o_proj, n: 5120, k: 5120}
  - {op: AllReduce, role: attn_out, emit: tensor_parallel}
  - {op: Elementwise, role: post_attn_norm}
  - {op: GEMM, role: mlp_gate_up, n: 34816, k: 5120}
  - {op: GEMM, role: mlp_down, n: 5120, k: 17408}
  - {op: AllReduce, role: mlp_out, emit: tensor_parallel_unless_sp_moe}
  edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
```

The committed file writes each node and edge as a block. An edge `[i, j]` says node `i` precedes node `j`. The [model's page](../reference/models/qwen3-14b.md) shows the whole graph.

The example shows four general rules:

- **Dimensions are derived as the serving engine derives them.** `qkv_proj` has `n` = 7168 because 40 query heads and 2 × 8 key-value heads, each of dimension 128, give 5120 + 2048.
- **A collective says when it runs.** Whether a collective runs depends on the deployment's parallelism, not on the model, so each carries an `emit` condition, one of the three below. At tensor-parallel width 1, both reductions in the example disappear.
- **Weights have a width.** The graph's `weight_dtype` sets the bytes each parameter occupies. A node can override it where the checkpoint stores part of itself at a different width, as DeepSeek-V4-Pro does for its experts.
- **A node exists only if it launches GPU work.** No node corresponds to a config field that changes no operation.

`tensor_parallel`
:   Runs when the tensor-parallel width exceeds one.

`expert_parallel`
:   Runs when the expert-parallel width exceeds one.

`tensor_parallel_unless_sp_moe`
:   Runs when the tensor-parallel width exceeds one, unless the engine runs the MoE layer's input sequence-parallel; a reduce-scatter and all-gather pair then replaces it.

Two further fields cover models that are more than one decoder stack. A `speculator` describes a draft module, such as a multi-token-prediction (MTP) layer, as a stack of its own, because for an MoE model the draft pass is itself an MoE pass. `modality: text_decoder_of_multimodal` records that the vendor config also describes a vision or audio encoder that the graph does not price. A prediction from such a graph is valid for text requests and too low for requests that carry images or audio.

## Generated, not written

`scripts/derive_graph.py` (the *deriver*) reads a config and writes the graph. It chooses a handler by the config's `architectures` entry; each handler encodes how one family of architectures builds its layers in the serving engine. When the deriver meets something it cannot translate, such as an unknown architecture or a missing field it needs, it stops with an error rather than guessing. A guessed graph would price a different model, with nothing to show that it had.

## Committed, not computed on demand

The deriver could run every time a model is loaded. Its output is committed instead, for three reasons.

A reviewer can read the operations BLIS will price, in a file in the repository, rather than trusting the output of a script.

Drift is detected. Every graph records the digest of the config it came from and the version of the deriver that produced it:

```yaml
derived_from:
  format: hf_config_json
  path: config.json
  sha256: e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a
  deriver_version: 1
```

CI derives every graph again and fails if any differs from the committed file. That catches a hand-edited graph, and a config changed without its graph being derived again. The current deriver version is [[catalog.deriver_version]].

Loading a model needs only the committed graph: no network access, and no copy of the deriver.
