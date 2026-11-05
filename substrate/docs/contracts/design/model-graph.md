# Why a model is a graph

<p class="lede">The catalog already holds each model's vendor configuration, byte for byte.
blis-schemas does not use it as the model schema. It describes a model as the GPU work a
forward pass launches, and the catalog derives that description from the vendor file.</p>

## A vendor configuration describes the wrong thing

A Hugging Face `config.json` is written in one provider's dialect. The model width is
`hidden_size` in one family, `d_model` in another and `n_embd` in a third. A state-space
layer is described by `mamba_expand` in one repository and `ssm_cfg` in another. A schema
that adopted those names would be tied to whichever vendor defined a model first.

A configuration also says how to build the model, not what running it costs. A cost model
needs to know which matrix multiplies run at what shapes, which attention variant reads how
much cache, and which collectives cross which links. A `ModelGraph` states those, along with
the model's name and the provenance of its source.

## Two rules for deriving a graph

The catalog's deriver, `scripts/derive_graph.py`, follows two rules.

**A node exists when it launches GPU work.** A configuration field that launches nothing
adds no node. Granite's `attention_multiplier`, for instance, replaces the softmax scale
rather than adding an operation, so it changes no node.

**A shape is derived the way the engine derives it.** Many configurations never state the
head dimension; the engine computes it as the hidden size over the head count, and so does
the deriver. Nemotron-3-Ultra states no layer count at all. Its depth is the length of a
per-layer type vector, and a deriver that read only the usual key would have produced a
model with no layers.

## Ten primitives

--8<-- "docs/contracts/generated/enum/model.Op.md"

A new architecture is a new composition of these, not a new primitive. Some of the cost
model's terms are not nodes. Reading weights is implied by the matrix multiplies. Host
overhead, host-to-device transfer, storage reads and point-to-point transfers are priced by
kernel methods the simulator calls directly, because they do not happen inside a layer.

The keys of a node are a union across primitives: a GEMM has `n` and `k`, attention has head
counts, a recurrent update has state dimensions. The validator rejects the likely
mismatches, such as a head count on a GEMM, so that a key the node's primitive ignores does
not pass unnoticed.

## Collectives are conditional

Whether a collective runs depends on the deployment, not the model. At `tp 1` there is no
tensor-parallel reduction. Every collective node therefore carries an `emit` condition, and
the deployment's resolved layout decides which nodes exist. Two conditions turn on a
model's own sequence parallelism, which changes the layout of some non-collective nodes
too, so those two may also sit on a node the graph states once for each mode.

The condition is one of a fixed list, not an expression. An earlier draft allowed
predicates such as `"tp > 1 and not sp_moe"`, which would have required an expression parser
in every cost model and let graphs state conditions no cost model understood. Every model in
the catalog needs one of five conditions, each decided by the node's role and the resolved
layout, so the list loses nothing, and a cost model decides a node with a switch statement.

## Stacks compress, and some do not repeat

A 200-layer model is not 200 entries. A uniform model is one layer kind repeated; a hybrid
with a regular period is a short pattern repeated. That mirrors how the engine itself builds
a hybrid model, from a per-layer type vector that selects a decoder class.

Not every model is periodic from its first layer. DeepSeek-V3 opens with three dense layers
before 58 MoE layers, so its stack has a prologue. Nemotron-3-Ultra repeats a 25-layer
pattern four times and ends with eight more layers, 108 in all, so its stack has an
epilogue. A model with no repeating unit at all would put its whole layer sequence in the
prologue. With only a pattern and a repeat, DeepSeek-V3 would need a 61-entry pattern, and a
model with no period could not be compressed at all.

## The graph is committed, not computed on demand

The deriver could run every time a model is loaded. Instead its output is committed, with
the SHA-256 of the configuration it came from, for three reasons.

1. **It can be read.** A reviewer asking why BLIS prices a model a certain way can open
   `graph.yaml`. If the graph were computed on demand, the only answer would be to run the
   deriver and trust it.
2. **Drift is detectable.** The catalog's CI re-derives every graph and fails if one differs
   from what is committed. That catches a hand-edited graph, and a configuration changed
   without re-deriving.
3. **A run is self-contained.** It needs neither network access nor the deriver.

Deriving a graph involves judgment, and a committed file can be reviewed where a silent
recomputation cannot.

## What a graph says it leaves out

Several shipped configurations describe a vision or audio encoder beside the text decoder.
A graph derived from one prices the decoder only, and says so with `modality:
text_decoder_of_multimodal`. Without that key, a prediction for a request carrying an image
would be too low, and nothing would say why.

## When one dtype is not enough

`global.weight_dtype` covers most checkpoints. DeepSeek-V4-Pro stores its routed experts in
a 4-bit format and everything else in FP8. Pricing the experts in FP8 doubled their size,
from 720 GiB to 1,441 GiB across 384 experts, and made a deployment that has been run look
impossible on its hardware. A `weight_dtype` key on the expert nodes states the split
directly rather than inferring it from the quantization method.

Source: [`spec/model`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/model).
