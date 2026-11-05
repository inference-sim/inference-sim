#!/usr/bin/env python3
"""Derive a BLIS ModelGraph from a vendor config.json.

    python3 scripts/derive_graph.py [ROOT] [--check] [--model NAME]

A ModelGraph expresses a model as a DAG of cost primitives, which is what a cost
model prices. The vendor config speaks one provider's dialect; this script performs
the translation once, mechanically, and writes the result beside the config with a
digest of the source. Committing the output rather than recomputing it means a reader
can inspect what a cost model will price, and the digest makes a drift between the
two detectable.

The translation is not a field rename. Three things make it a derivation:

  * A shape may not be stated. Depth is usually num_hidden_layers, but one shipped
    hybrid states no layer count at all: its depth is the length of a per-layer type
    vector. Head dimension is absent from nine of the committed configs because it is
    hidden_size over head count.
  * A node exists when it launches GPU work, not when a config field exists. An
    architecture-specific scalar that replaces a softmax scale rather than adding an
    operation contributes no node.
  * Collective nodes carry a layout condition rather than being emitted
    unconditionally, because whether a collective runs depends on the parallelism a
    deployment chooses, not on the model.

With --check the script re-derives every graph and reports any that differs from what
is committed, without writing. That is the CI gate: a graph edited by hand, or a
config changed without re-deriving, fails.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

DERIVER_VERSION = 1

# --- Dialect aliases ------------------------------------------------------------
# Each concept the graph needs, and the keys the committed configs use for it, in
# priority order. A concept absent from every alias is either derived (head
# dimension) or genuinely absent (a dense model has no expert count).
ALIASES: dict[str, tuple[str, ...]] = {
    "hidden_size": ("hidden_size", "d_model", "n_embd"),
    "num_layers": ("num_hidden_layers", "n_layer", "num_layers"),
    "num_q_heads": ("num_attention_heads", "n_head"),
    "num_kv_heads": ("num_key_value_heads", "num_kv_heads", "multi_query_group_num"),
    "head_dim": ("head_dim", "kv_channels"),
    "ffn_size": ("intermediate_size", "ffn_hidden_size"),
    "vocab_size": ("vocab_size",),
    "num_experts": ("num_local_experts", "n_routed_experts", "num_experts"),
    "top_k": ("num_experts_per_tok", "num_experts_per_token", "moe_topk"),
    "moe_ffn_size": ("moe_intermediate_size", "expert_intermediate_size"),
    "num_shared_experts": ("n_shared_experts", "num_shared_experts"),
    "shared_ffn_size": (
        "shared_expert_intermediate_size",
        "moe_shared_expert_intermediate_size",
    ),
    "sliding_window": ("sliding_window", "sliding_window_size"),
    "kv_lora_rank": ("kv_lora_rank",),
    "qk_rope_head_dim": ("qk_rope_head_dim",),
    "qk_nope_head_dim": ("qk_nope_head_dim",),
    "v_head_dim": ("v_head_dim",),
    "num_spec_tokens": ("num_nextn_predict_layers", "mtp_num_hidden_layers"),
    # routed_expert_hidden_size is Kimi's spelling of the same concept: its
    # KimiSparseMoeBlock projects hidden -> this width before routing, runs each expert
    # at K = this width, and projects back after (modeling_kimi_linear.py:776-832,
    # gated on `use_latent_moe = routed_expert_hidden_size is not None`). Reading only
    # the DeepSeek/Nemotron spelling left every Kimi expert priced at the full hidden
    # size, doubling its K.
    "moe_latent_size": ("moe_latent_size", "routed_expert_hidden_size"),
    "index_topk": ("index_topk",),
    "dtype": ("torch_dtype", "dtype"),
}

# Vendor dtype spellings to the graph's vocabulary. A format that changes only
# rounding is not a member: the graph records what changes a byte count or a rate.
DTYPES = {
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float16": "fp16",
    "fp16": "fp16",
    "half": "fp16",
    "float32": "fp32",
    "fp32": "fp32",
    "float8_e4m3fn": "fp8",
    "fp8": "fp8",
    "nvfp4": "nvfp4",
    "mxfp4": "mxfp4",
    "int8": "int8",
}


class DeriveError(Exception):
    """A config this deriver cannot translate. Raised rather than guessed around: a
    graph derived from a misread config prices the wrong model, and a missing branch
    is a deriver gap to fix."""


def pick(cfg: dict[str, Any], concept: str) -> Any:
    """Return the first alias present for a concept, or None."""
    for key in ALIASES[concept]:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return None


def require(cfg: dict[str, Any], concept: str, model: str) -> Any:
    val = pick(cfg, concept)
    if val is None:
        raise DeriveError(
            f"{model}: no key for {concept!r} among {ALIASES[concept]}; "
            f"the deriver needs a branch for this dialect"
        )
    return val


def require_key(cfg: dict[str, Any], key: str, model: str) -> Any:
    """Return a config key by its literal name, or raise.

    Distinct from `require`, which resolves a CONCEPT through the alias table. A field
    that only one family declares -- a GDN's head geometry, say -- has no cross-dialect
    concept to alias, so adding it to ALIASES would imply a generality it does not have.
    Reading it by name keeps the alias table a statement about shared concepts.
    """
    val = cfg.get(key)
    if val is None:
        raise DeriveError(
            f"{model}: config declares no {key!r}, which this architecture needs"
        )
    return val


def text_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the sub-object holding the language-model shapes.

    A multimodal config nests them under text_config beside a vision or audio tower.
    The graph prices the decoder, and records that it does so."""
    return raw.get("text_config", raw)


def raw_quant(cfg: dict[str, Any], model: str) -> dict[str, Any]:
    """The quantization block as seen from a config slice, or an empty mapping."""
    quant = cfg.get("quantization_config") or {}
    return quant if isinstance(quant, dict) else {}


# A role's dtype class, for a checkpoint that states its mixed-precision layout
# explicitly. The map is keyed by the tensor-name fragment the vendor uses and valued by
# the graph role it prices, so a width read from the checkpoint lands on the right node.
QUANT_ROLE_CLASSES = {
    "experts.": "routed_experts",
    "shared_experts.": "shared_experts",
    "mixer.in_proj": "recurrent_in",
    "mixer.out_proj": "recurrent_out",
}

# Vendor quant-algo spellings to the graph's dtype vocabulary.
QUANT_ALGOS = {
    "NVFP4": "nvfp4",
    "MXFP4": "mxfp4",
    "FP8": "fp8",
    "INT8": "int8",
    "INT4": "int4",
    # A W{weights}A{activations} spelling names both widths; the graph prices WEIGHT
    # bytes, so the W half is what it reads. nemotron-3.5-lightning-nvfp4 ships
    # W4A16_NVFP4 -- four-bit NVFP4 weights against bf16 activations -- which stores
    # the same bytes per parameter as the plain NVFP4 spelling.
    "W4A16_NVFP4": "nvfp4",
    "W4A16_MXFP4": "mxfp4",
    "W8A8_FP8": "fp8",
    "W8A16_INT8": "int8",
}


# The graph roles that price an attention module's weights, i.e. the ones a
# checkpoint's `self_attn` exclusion covers. Listed explicitly rather than matched on a
# substring so that adding a role is a deliberate act: a role missing from here keeps the
# global width, which is the conservative direction for a width override.
ATTENTION_WEIGHT_ROLES = (
    "qkv_proj", "qkv_a_proj", "kv_a_proj", "q_proj", "q_b_proj", "kv_b_proj", "o_proj",
    "index_wq_b", "index_wk_weights_proj", "index_head_gate_proj",
    "index_kpool_compress_gate",
    # DeepSeek-V4 and MiniMax-M3 name their indexer projections differently, because
    # their handlers build them rather than lightning_indexer. index_qk_proj is
    # MiniMax-M3's block-sparse scorer projection, which is a different mechanism from
    # the DSA lightning indexer and keeps its own shape.
    "index_wq_proj", "index_weights_proj", "index_compressor_fused_wkv_wgate",
    "index_qk_proj",
    # A linear-attention mixer is the layer's self_attn too, not a separate module:
    # vLLM builds Kimi-K3's KDA as `self.self_attn` with prefix "{...}.self_attn"
    # (kimi_k3/nvidia/model.py:888-891), so an ignore list naming self_attn covers these
    # projections exactly as it covers the latent ones.
    "kda_q_proj", "kda_k_proj", "kda_v_proj", "kda_b_proj",
    "kda_f_a_proj", "kda_f_b_proj", "kda_g_a_proj", "kda_g_b_proj", "kda_g_proj",
    "kda_o_proj",
)


def compressed_tensors_skips_attention(cfg: dict[str, Any],
                                       raw: dict[str, Any]) -> bool:
    """Whether a compressed-tensors checkpoint leaves the attention modules alone.

    compressed-tensors states an `ignore` list of module names and `re:`-prefixed
    regexes, and both Kimi configs name `re:.*self_attn.*` -- so every MLA projection and
    the indexer stay at the declared base width while the routed experts carry the
    quantized one. kimi-k2.5's global width is int4 and kimi-k3's mxfp4, so pricing those
    projections globally charged 4 bits for tensors the runtime never quantizes.

    Deliberately narrow: this answers only "does the ignore list cover self_attn", which
    is the pattern both committed configs use. A checkpoint ignoring individual attention
    tensors would need its own reading rather than this one."""
    quant = raw.get("quantization_config") or cfg.get("quantization_config") or {}
    if not isinstance(quant, dict):
        return False
    if "compressed-tensors" not in str(quant.get("quant_method", "")).lower():
        return False
    for entry in quant.get("ignore") or ():
        text = str(entry)
        if not text.startswith("re:"):
            continue
        body = text[3:]
        # Only the blanket self_attn form is recognised. A narrower regex may cover some
        # attention tensors and not others, and guessing which would be worse than
        # leaving the global width in place.
        if body in (".*self_attn.*", r".*self_attn.*"):
            return True
    return False


def mhc_mixing_bytes(cfg: dict[str, Any], raw: dict[str, Any], model: str,
                     hidden: int, hc_mult: int) -> int:
    """Bytes per token the mHC mixing moves, which hidden_size alone does not give.

    bytes_per_token is the Elementwise escape hatch for exactly this case: the schema
    documents it as "bytes touched per token, when it is not derivable from HiddenSize
    alone" (blis-schemas spec/model/graph.go). An ordinary RMSNorm IS derivable -- one
    hidden-wide read and write -- which is why every other Elementwise node in this
    catalog omits the field. mHC is not: it moves the whole hc_mult-stream residual
    bundle, and nothing in a ModelGraph carries hc_mult, so an omitted value would price
    this node as the ordinary norm it replaced.

    Counted from the shapes vLLM's reference kernel documents
    (model_executor/kernels/mhc/torch.py:60-78) for one fused post+pre, which is what
    both sides run in steady state: the attention side fuses the PREVIOUS layer's
    deferred hc_post with this layer's attention hc_pre, and the FFN side fuses its own
    (common/model.py:555-603). MHCFusedPostPreOp is documented as "equivalent to applying
    MHCPostOp and then MHCPreOp", so the two are summed:

      post: read x (hidden) + residual (hc_mult x hidden) + post_mix (hc_mult)
            + comb_mix (hc_mult^2), write residual (hc_mult x hidden)
      pre:  read residual (hc_mult x hidden), write post_mix (hc_mult)
            + comb_mix (hc_mult^2) + layer_input (hidden)

    The residual streams and layer_input are bf16 (the kernel asserts it); post_mix and
    comb_mix are fp32 (likewise). For the family this was first written for (GLM-5.3,
    common/model.py), two corners are deliberately not modelled, because a layer kind
    prices the common case and neither is it: layer 0's attention side runs a
    standalone pre rather than a fused post+pre, and the final layer materializes a
    trailing post, each one layer of 45. DeepSeek-V4 and V4.1 price both corners with
    their own nodes and use this figure for the fused boundary alone.

    The hc_sinkhorn_iters rounds iterate over the [mix_hc] logits, which at hc_mult 4 is
    24 fp32 values against 32 KiB of residual traffic, so they are not counted."""
    act = DTYPE_WIDTHS.get(base_dtype(cfg, raw, model))
    if act is None:
        raise DeriveError(
            f"{model}: mHC mixing needs the activation width to size its residual "
            f"traffic, and the base dtype has no byte width here")
    fp32 = DTYPE_WIDTHS["fp32"]
    bundle = hc_mult * hidden * act
    mixes = hc_mult * fp32 + hc_mult * hc_mult * fp32
    post = hidden * act + bundle + mixes + bundle
    pre = bundle + mixes + hidden * act
    return post + pre


def fp8_unquantized_roles(cfg: dict[str, Any], raw: dict[str, Any], model: str,
                          roles: dict[str, str]) -> dict[str, str]:
    """Map node role -> base width, for roles an fp8 checkpoint leaves unquantized.

    `modules_to_not_convert` names the tensor sets a quantization method SKIPS. The
    mxfp4/nvfp4 path already honours it by keeping the global width at the unquantized
    base and carrying the narrow width on the expert node (weight_dtype(), gemm_moe()),
    because there the quantized scope is the minority. An fp8 checkpoint is the mirror
    case: the routed experts dominate -- 288 experts x 3 matrices x 2048x4096 is 213x
    the rest of a layer -- so fp8 is the right global width and the EXCEPTIONS are what
    need carrying.

    GLM-5.3-Flash is the case in hand. It declares 1,509 exclusions, and the pinned
    safetensors headers agree with them exactly: every KDA projection, every indexer
    tensor and `lm_head` are BF16 while `q_a`/`q_b`/`kv_a`/`o_proj` and the experts are
    F8_E4M3. Pricing the BF16 tensors at one byte understates the weight traffic of all
    34 KDA layers and the head.

    `roles` maps a checkpoint module suffix to the graph role that prices it, and a role
    is returned when the list names that suffix anywhere. A graph role prices a layer
    kind rather than one tensor, so a suffix must be chosen to be unambiguous across the
    layers a kind covers -- see GLM5_NEXT_UNQUANTIZED on why `self_attn.o_proj` is safe
    here and `kv_b_proj` is omitted. This function does not verify that; the caller's
    table is what carries it.

    Returns an empty mapping for any checkpoint that declares no fp8 exclusions, which
    leaves every other model's nodes untouched."""
    quant = raw.get("quantization_config") or cfg.get("quantization_config") or {}
    if not isinstance(quant, dict) or not quant:
        return {}
    if "fp8" not in str(quant.get("quant_method", "")).lower():
        return {}
    excluded = quant.get("modules_to_not_convert")
    if not excluded:
        return {}
    names = set(excluded)
    matched = {role for suffix, role in roles.items()
               if any(n == suffix or n.endswith("." + suffix) for n in names)}
    if not matched:
        return {}
    # Resolved only once a role actually matches. minimax-m2.5 declares an fp8 exclusion
    # list and NO dtype at all, so asking for the base width unconditionally would raise
    # for a model this function has nothing to say about.
    base = base_dtype(cfg, raw, model)
    return {role: base for role in matched}


def quantized_classes(cfg: dict[str, Any], model: str) -> dict[str, str]:
    """Map each role class to its stored width, from an explicit per-tensor map.

    A checkpoint that enumerates its quantized tensors states a LAYOUT, not a single
    width: nemotron-3-ultra-nvfp4 names 49,152 routed-expert matrices as NVFP4, 192
    Mamba and shared-expert matrices as FP8, and ignores 243 more -- attention, the
    latent projections, embeddings and the head -- which stay at the declared bf16 base.
    Collapsing that to one dominant width prices every ignored tensor at four bits.

    Returns an empty mapping where the config states no such map, which leaves the
    single-width path untouched for every other model."""
    quant = raw_quant(cfg, model)
    layers = quant.get("quantized_layers")
    if not isinstance(layers, dict) or not layers:
        return {}
    found: dict[str, set[str]] = {}
    for name, spec in layers.items():
        algo = spec.get("quant_algo") if isinstance(spec, dict) else spec
        width = QUANT_ALGOS.get(str(algo).upper())
        if width is None:
            raise DeriveError(
                f"{model}: quantized_layers names algo {algo!r}, which this deriver "
                f"does not map; add it to QUANT_ALGOS rather than letting the tensor "
                f"take a width the checkpoint does not state")
        # Longest fragment wins, so shared_experts does not match the experts rule.
        match = max((frag for frag in QUANT_ROLE_CLASSES if frag in name),
                    key=len, default=None)
        if match is None:
            continue
        found.setdefault(QUANT_ROLE_CLASSES[match], set()).add(width)
    out = {}
    for role, widths in found.items():
        if len(widths) > 1:
            raise DeriveError(
                f"{model}: role class {role!r} is quantized at more than one width "
                f"{sorted(widths)}; a single node cannot price both")
        out[role] = widths.pop()
    return out


# Architectures whose handler places each node's width under a ModelOpt quant_algo
# checkpoint, reading the exclusion list per module (see weight_dtype()).
MODELOPT_PER_NODE_ARCHS = frozenset({"MuseSparkForConditionalGeneration"})


def modelopt_exclusions(quant: dict[str, Any]) -> list[str]:
    """The module patterns a ModelOpt checkpoint leaves unquantized, as vLLM reads them.

    From config.json vLLM takes `ignore`, else `exclude_modules`, and never
    modules_to_not_convert (ModelOptQuantConfigBase.from_config,
    model_executor/layers/quantization/modelopt.py:358-359 at vllm-project/vllm 0f6b2619;
    v0.32.0 reads only `ignore`, :354-355). A pattern ending in `*` not preceded by `.`
    also names the module itself (apply_vllm_mapper, :263-281@v0.32.0). The model's
    WeightsMapper renames are not applied here; a handler that reads these patterns
    states why its entries are untouched by them."""
    pats = quant.get("ignore", quant.get("exclude_modules", [])) or []
    if not isinstance(pats, list) or not all(isinstance(p, str) for p in pats):
        raise DeriveError(f"ModelOpt exclusion list {pats!r} is not a list of names")
    out = []
    for p in pats:
        if len(p) >= 2 and p[-1] == "*" and p[-2] != ".":
            out += [p[:-1], p[:-1] + ".*"]
        else:
            out.append(p)
    return out


def modelopt_excluded(prefix: str, patterns: list[str]) -> bool:
    """Whether vLLM's ModelOpt config leaves the module at `prefix` unquantized: a
    pattern equal to it, a SUBSTRING of it, or a wildcard matching it
    (is_layer_excluded, modelopt.py:187-220@v0.32.0, the same at 0f6b2619). Substring,
    so "layers.1" excludes layers 10-19 as well as layer 1. The exact match through a
    fused module's shard names (is_layer_skipped) is not modelled."""
    return any(p == prefix or p in prefix or fnmatch.fnmatchcase(prefix, p)
               for p in patterns)


def weight_dtype(cfg: dict[str, Any], raw: dict[str, Any], model: str) -> str:
    """Return the graph's dtype name for the model's parameters.

    A quantization block overrides the base dtype: a checkpoint declaring bfloat16
    weights with an fp8 quantization config stores fp8, and pricing it at two bytes
    per parameter would overstate both occupancy and decode-time traffic."""
    quant = raw.get("quantization_config") or cfg.get("quantization_config") or {}
    if isinstance(quant, dict) and quant:
        # A checkpoint that enumerates its quantized tensors states a layout rather than
        # a width. The base dtype stays global and the per-tensor widths ride on the
        # nodes they apply to, so the tensors the map IGNORES keep the base.
        if quantized_classes(cfg, model):
            return base_dtype(cfg, raw, model)
        fmt = str(quant.get("quant_method", "")).lower()
        if "nvfp4" in fmt or "modelopt_fp4" in fmt:
            return "nvfp4"
        # ModelOpt states its format as quant_algo beside quant_method "modelopt" and
        # quantizes every module its exclusion list does not name (modelopt_exclusions).
        # Which modules those are is a question about the architecture's module tree, so
        # one global width is right only where a handler answers it node by node. Muse
        # Spark's does: its global width is the base, and it carries NVFP4 on the expert
        # nodes the list leaves quantized. Any other architecture is refused, as it was
        # before that handler existed, rather than priced at the base throughout.
        if fmt == "modelopt" and quant.get("quant_algo"):
            algo = QUANT_ALGOS.get(str(quant["quant_algo"]).upper())
            if algo is None:
                raise DeriveError(f"{model}: ModelOpt quant_algo {quant['quant_algo']!r} "
                                  f"is not one this deriver maps")
            arch = (raw.get("architectures") or [None])[0]
            if arch not in MODELOPT_PER_NODE_ARCHS:
                raise DeriveError(
                    f"{model}: a ModelOpt {algo} checkpoint quantizes every module its "
                    f"exclusion list does not name; only a handler that places each "
                    f"node's width prices one, and {arch} has none")
            return base_dtype(cfg, raw, model)
        if "fp8" in fmt:
            return "fp8"
        # A per-group block states the width per tensor set rather than once. The
        # dominant group decides the stored width: a checkpoint holding 49,152 tensors
        # at four bits and 192 at eight is a four-bit checkpoint, and pricing it at the
        # narrower group's minority width would overstate weight traffic.
        groups = quant.get("config_groups")
        if isinstance(groups, dict) and groups:
            widest = None
            for spec in groups.values():
                weights = (spec or {}).get("weights") or {}
                bits = weights.get("num_bits")
                count = len(spec.get("targets") or ())
                if bits is None:
                    continue
                if widest is None or count > widest[1]:
                    widest = (int(bits), count, weights.get("type"),
                              spec.get("format"), weights.get("group_size"))
            if widest is not None:
                bits, _, kind = widest[0], widest[1], widest[2]
                # Two 4-bit float formats ship in these checkpoints and they are not
                # interchangeable: NVFP4 shares an FP8 scale over 16 elements, MXFP4 an
                # E8M0 scale over 32. The format string distinguishes them; a 4-bit
                # float with no format named is ambiguous and raises.
                if bits == 4 and kind == "float":
                    fmt_name = str(widest[3] or quant.get("format") or "").lower()
                    if "mxfp4" in fmt_name:
                        return "mxfp4"
                    if "nvfp4" in fmt_name:
                        return "nvfp4"
                    if widest[4] == 16:
                        return "nvfp4"
                    if widest[4] == 32:
                        return "mxfp4"
                    raise DeriveError(
                        f"{model}: a 4-bit float weight group names no format and has "
                        f"group_size {widest[4]!r}; NVFP4 and MXFP4 cannot be told apart"
                    )
                if bits == 8 and kind == "float":
                    return "fp8"
                if bits == 8 and kind == "int":
                    return "int8"
                # A 4-bit INT group needs no format string to disambiguate, unlike the
                # 4-bit floats above: there is one integer grid, and the per-group scale
                # count follows group_size rather than being fixed by the format. Kimi-K2.5
                # ships this as num_bits 4, type "int", group_size 32.
                if bits == 4 and kind == "int":
                    return "int4"
                raise DeriveError(
                    f"{model}: a {bits}-bit {kind!r} weight group has no dtype in this "
                    f"schema's vocabulary"
                )
        weight_block = quant.get("weight_block_size")
        if fmt == "compressed-tensors" and weight_block:
            return "fp8"
        # A quant_method that names the format outright needs no group widths, because the
        # name fixes them: MXFP4 is 32-element groups with an E8M0 scale and NVFP4 is
        # 16-element with an FP8 scale, by definition of each format. gpt-oss declares
        # `quant_method: mxfp4` with no config_groups at all.
        #
        # Only these two are accepted by name. A method whose name does not fix a width --
        # "compressed-tensors" is the case in hand -- still needs its groups, which is why
        # this sits after that branch rather than before it.
        if fmt in ("mxfp4", "nvfp4"):
            # ...but only where it covers the whole checkpoint. modules_to_not_convert
            # names the tensor sets the method LEAVES ALONE, and gpt-oss excludes
            # self_attn, the router, the embeddings and lm_head -- so MXFP4 there is the
            # expert weights only, and returning it as the global width prices every
            # attention projection and the head at 4 bits instead of 16. The quantized
            # scope is carried on the expert node instead, the same way an expert_dtype
            # narrower than the global width is.
            if quant.get("modules_to_not_convert"):
                return base_dtype(cfg, raw, model)
            return fmt
        if fmt:
            raise DeriveError(
                f"{model}: quant_method {fmt!r} with no group widths; the deriver "
                f"cannot infer the stored width"
            )
    return base_dtype(cfg, raw, model)


# The width a partially-quantized checkpoint stores its UNQUANTIZED tensors in, for a
# config that states no dtype at all. gpt-oss is the case in hand: it declares neither
# torch_dtype nor dtype, and its quantization_config covers only the experts, so the
# non-expert width has to come from somewhere. OpenAI's model card and the InferenceX
# descriptor this catalog cites for the model both state BF16 for those tensors.
# Recorded per model rather than defaulted, because a default here would silently price
# every future dtype-less config at two bytes.
# Bytes per element, for the widths an activation tensor is stored in. Only the dtypes
# that can carry activations are listed: this sizes elementwise traffic, not weights, so
# the sub-byte quantized formats have no entry rather than a fractional one.
DTYPE_WIDTHS = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
}


UNQUANTIZED_BASE = {
    "gpt-oss-120b": "bf16",
    "gpt-oss-20b": "bf16",
}


def int_group_scale(cfg: dict[str, Any], raw: dict[str, Any],
                    model: str) -> dict[str, int] | None:
    """The resident scale of an integer group-quantized checkpoint, or None.

    An INT format fixes no scale layout (blis-schemas DType.IntrinsicScale), so a W4A16
    checkpoint's scales are resident only if the graph states them. compressed-tensors
    WNA16 allocates one scale per group_size weights in params_dtype, the model's BF16
    (w13_weight_scale / w2_weight_scale, compressed_tensors_moe_wna16.py:282-310 at
    v0.31.0). On Kimi-K2.5 at group 32 that is 1/16 B a parameter, 7.4 GiB a rank at TP8,
    which the KV pool otherwise absorbs."""
    quant = raw.get("quantization_config") or cfg.get("quantization_config") or {}
    if "compressed-tensors" not in str(quant.get("quant_method", "")).lower():
        return None
    groups = [(s or {}).get("weights") or {}
              for s in (quant.get("config_groups") or {}).values()]
    ints = [w for w in groups if w.get("type") == "int" and w.get("strategy") == "group"]
    if not ints:
        return None
    sizes = {int(w.get("group_size") or 0) for w in ints}
    if len(sizes) != 1 or min(sizes) < 1:
        raise DeriveError(f"{model}: integer weight groups state group sizes "
                          f"{sorted(sizes)}; one resident scale layout cannot hold them")
    return {"elements": sizes.pop(), "bytes": DTYPE_WIDTHS[base_dtype(cfg, raw, model)]}


def base_dtype(cfg: dict[str, Any], raw: dict[str, Any], model: str) -> str:
    """The declared parameter dtype, ignoring any quantization block."""
    declared = pick(cfg, "dtype") or pick(raw, "dtype")
    if declared is None:
        known = UNQUANTIZED_BASE.get(model)
        if known:
            return known
        raise DeriveError(f"{model}: no dtype declared")
    name = DTYPES.get(str(declared).lower())
    if name is None:
        raise DeriveError(f"{model}: unrecognized dtype {declared!r}")
    return name


def draft_aux_hidden_state_layers(hf: dict[str, Any], model: str, target_layers: int,
                                  states: int | None = None) -> list[int]:
    """The target layers a drafter reads, as the engine resolves them from its config.

    The ids are in the engine's capture convention, which the graph's
    speculator.aux_hidden_state_layers states: L is the state entering decoder layer L, 0
    the embedding output, the layer count the last layer's output. vLLM reads, in order,
    eagle_aux_hidden_state_layer_ids as-is; dflash_config.target_layer_ids plus one;
    dspark_target_layer_ids as-is for model_type deepseek_v41 and plus one otherwise; the
    dense-DSpark top-level target_layer_ids plus one; then dflash_config or eagle_config
    layer_ids as-is (get_eagle3_aux_layers_from_config,
    vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:100-137@v0.32.0, unchanged at
    vllm-project/vllm 0f6b2619). A config naming none of them falls back to a per-model
    default, which no graph can read from a config, so that is refused.

    The engine sorts and deduplicates the ids (_set_aux_hidden_state_layers,
    model_executor/models/interfaces.py:1577-1578@v0.32.0); a repeat would leave the
    drafter's input projection, sized for one state per listed layer, a state short, so
    a repeat is refused rather than collapsed. `hf` is the drafter's config as the engine
    sees it, flattened; `states`, where given, is how many states the drafter consumes.
    The upper bound is checked here because the schema cannot check it on a drafter
    graph, which does not hold its target's layer count."""
    def entries(key, ids):
        out = []
        for i in ids:
            if isinstance(i, bool) or not isinstance(i, int):
                raise DeriveError(f"{model}: {key} entry {i!r} is not a layer id")
            out.append(i)
        return out

    ids: list[int] | None = None
    if hf.get("eagle_aux_hidden_state_layer_ids"):
        ids = entries("eagle_aux_hidden_state_layer_ids",
                      hf["eagle_aux_hidden_state_layer_ids"])
    dflash = hf.get("dflash_config")
    if not ids and isinstance(dflash, dict) and dflash:
        ids = [i + 1 for i in entries("dflash_config.target_layer_ids",
                                      dflash.get("target_layer_ids") or ())]
    if not ids and hf.get("dspark_target_layer_ids"):
        ds = entries("dspark_target_layer_ids", hf["dspark_target_layer_ids"])
        ids = ds if hf.get("model_type") == "deepseek_v41" else [i + 1 for i in ds]
    if not ids and hf.get("target_layer_ids"):
        ids = [i + 1 for i in entries("target_layer_ids", hf["target_layer_ids"])]
    if not ids:
        for name in ("dflash_config", "eagle_config"):
            sub = hf.get(name)
            if isinstance(sub, dict) and sub.get("layer_ids"):
                ids = entries(f"{name}.layer_ids", sub["layer_ids"])
                break
    if not ids:
        raise DeriveError(f"{model}: the drafter names no target layers to read; the "
                          f"engine would fall back to a per-model default")
    ids = sorted(ids)
    if len(set(ids)) != len(ids):
        raise DeriveError(f"{model}: aux hidden-state layers {ids} repeat a capture point; "
                          f"the ids must be strictly increasing")
    if ids[0] < 0 or ids[-1] > target_layers:
        raise DeriveError(f"{model}: aux hidden-state layers {ids} reach past the target's "
                          f"{target_layers} layers")
    if states is not None and len(ids) != states:
        raise DeriveError(f"{model}: the engine captures {len(ids)} states {ids}, and the "
                          f"drafter's input projection takes {states}")
    return ids


def head_dim(cfg: dict[str, Any], model: str) -> int:
    """Return the attention head dimension.

    Stated where the config says so. Derived as hidden over head count otherwise,
    which is what an engine does and what nine committed configs require."""
    stated = pick(cfg, "head_dim")
    if stated:
        return int(stated)
    hidden = int(require(cfg, "hidden_size", model))
    heads = int(require(cfg, "num_q_heads", model))
    if hidden % heads:
        raise DeriveError(
            f"{model}: hidden_size {hidden} is not divisible by {heads} heads and no "
            f"head_dim is stated"
        )
    return hidden // heads


# --- Node builders --------------------------------------------------------------
# Each returns one primitive with the shape parameters that primitive prices, and
# nothing else. A parameter a node's op does not price is rejected by the schema, so
# the builders keep the unions honest rather than relying on review.


def norm(role: str) -> dict[str, Any]:
    return {"op": "Elementwise", "role": role}


def gemm(role: str, n: int, k: int) -> dict[str, Any]:
    return {"op": "GEMM", "role": role, "n": int(n), "k": int(k)}


def attention(cfg: dict[str, Any], model: str, *, kind: str | None = None,
              window: int | None = None, compress_ratio: int | None = None,
              index_topk: int | None = None,
              latent_width: int | None = None) -> dict[str, Any]:
    """Build the attention node, choosing the kind from the config's own evidence.

    `latent_width` is the per-token cache width for a family that states it directly
    rather than as kv_lora_rank + rope. `compress_ratio` and `index_topk` are the two
    bounds a sparse latent read can carry, and a layer may carry both: the positions it
    reads are the union of its sliding window and its compressed stream, so `window`
    is not an alternative to either."""
    nq = int(require(cfg, "num_q_heads", model))
    nkv = int(require(cfg, "num_kv_heads", model))
    lora = pick(cfg, "kv_lora_rank")

    if kind is None:
        if lora:
            # Every layer of a DSA model reads a top-k, including the ones that build no
            # indexer. vLLM constructs the shared MLA module with use_sparse=True on
            # every layer and passes the common topk_indices_buffer
            # (deepseek_v32/attention.py:202-218), which mla_attention.py:564-566 states
            # outright: "Sparse MLA reads top-k indices from a shared buffer. Pass it
            # explicitly so backbone 'skip' layers (indexer=None) still find it." So a
            # skipped layer omits the SCORING work, not the bounded read -- making it
            # plain mla would overprice its main KV read as unbounded.
            kind = "sparse_mla" if pick(cfg, "index_topk") else "mla"
        elif window:
            kind = "swa"
        else:
            kind = "gqa"

    node: dict[str, Any] = {"op": "Attention", "kind": kind, "n_q": nq}

    if latent_width is not None:
        # A family that states the per-token latent width inclusively, so there is no
        # lora rank to add a rope width to. DeepSeek-V4 is the case in hand: its
        # head_dim already contains qk_rope_head_dim, which is why the branch below
        # cannot size it.
        if kind not in ("mla", "sparse_mla"):
            raise DeriveError(
                f"{model}: a latent width was given for attention kind {kind!r}, which "
                f"stores no latent cache")
        node["n_kv"] = 1
        node["d_h"] = int(latent_width)
        node["qk_rope_head_dim"] = int(require(cfg, "qk_rope_head_dim", model))
        if window:
            node["window"] = int(window)
        if index_topk:
            node["index_topk"] = int(index_topk)
        if compress_ratio:
            node["compress_ratio"] = int(compress_ratio)
        # A window alone bounds a sparse read too: DeepSeek-V4.1's ratio-0 layers run
        # the sparse FlashMLA kernel over their window and nothing else
        # (deepseek_v41/nvidia/flashmla.py:218-246), which blis-schemas#55 admits.
        if kind == "sparse_mla" and not (index_topk or compress_ratio or window):
            raise DeriveError(
                f"{model}: a sparse latent read bounded by no top-k selection, "
                f"compressed stream or window describes no real layer")
        return node

    if kind in ("mla", "sparse_mla"):
        # A latent cache holds one vector per token, so the engine pins the KV head
        # count to one whatever the config's num_key_value_heads says. The per-token
        # width is the latent rank plus the RoPE dimension, which is what the cache
        # stores and therefore what a cost model reads.
        nope = int(pick(cfg, "qk_nope_head_dim") or 0)
        rope = int(pick(cfg, "qk_rope_head_dim") or 0)
        # A NoPE variant states it: GLM-5.3-Flash sets mla_use_nope with
        # qk_rope_head_dim 0, so its latent cache holds the rank alone and the width
        # below is correct at rope == 0. Absent that declaration a zero is a missing
        # field rather than a stated one, and accepting it would silently understate
        # every per-token cache read, so the two cases are kept apart.
        if not rope and not cfg.get("mla_use_nope"):
            raise DeriveError(f"{model}: latent attention with no qk_rope_head_dim")
        node["n_kv"] = 1
        node["d_h"] = int(lora) + rope
        node["kv_lora_rank"] = int(lora)
        node["qk_rope_head_dim"] = rope
        if kind == "sparse_mla":
            node["index_topk"] = int(pick(cfg, "index_topk"))
        # nope is the query-side width; it does not enter the cache, so it is not a
        # node parameter. Recorded here so a reader knows it was considered.
        del nope
    else:
        node["n_kv"] = nkv
        node["d_h"] = head_dim(cfg, model)
        if kind == "swa":
            if not window:
                raise DeriveError(f"{model}: sliding-window attention with no window")
            node["window"] = int(window)
    return node


def dense_mlp(cfg: dict[str, Any], model: str, hidden: int) -> list[dict[str, Any]]:
    """Two GEMMs for a gated MLP: the fused gate-and-up projection, then the down.

    The gate and up projections are one kernel in every engine that matters, so they
    are one node at twice the inner width rather than two nodes."""
    ffn = int(require(cfg, "ffn_size", model))
    return [
        gemm("mlp_gate_up", 2 * ffn, hidden),
        gemm("mlp_down", hidden, ffn),
    ]


# A checkpoint storing its routed experts at a different width from the rest states it in
# expert_dtype, and vLLM resolves that NAME rather than reading it as a dtype:
# models/deepseek_v4/quant_config.py documents "fp4" as MXFP4 experts with ue8m0 scales
# and "fp8" as FP8-block experts, the linear and attention layers being FP8 block either
# way. So this mapping is vLLM's, not an inference from the string.
EXPERT_DTYPES = {
    "fp4": "mxfp4",
    "fp8": "fp8",
}


def moe(cfg: dict[str, Any], model: str, hidden: int) -> dict[str, Any]:
    """Build the grouped-GEMM node for a routed expert layer."""
    experts = int(require(cfg, "num_experts", model))
    top_k = int(require(cfg, "top_k", model))
    inner = pick(cfg, "moe_ffn_size") or require(cfg, "ffn_size", model)
    node: dict[str, Any] = {
        "op": "GroupedGEMM",
        "role": "experts",
        "n": int(inner),
        "k": hidden,
        "experts": experts,
        "top_k": top_k,
    }
    shared = pick(cfg, "num_shared_experts")
    shared_inner = pick(cfg, "shared_ffn_size")
    # A config that states a shared-expert WIDTH but no count has one shared expert. The
    # Qwen3.5 configs are the case in hand: they declare
    # shared_expert_intermediate_size and no count, and vLLM's own loader for that family
    # passes n_shared_experts=1 (`models/qwen3_5.py`). Reading the width as "no shared
    # expert" would drop dense work every token pays.
    if shared is None and shared_inner:
        shared = 1
    if shared:
        node["shared_experts"] = int(shared)
        if shared_inner and int(shared_inner) != int(inner):
            node["shared_intermediate_size"] = int(shared_inner)
    # The expert width where the checkpoint states one, which is not always the global
    # weight dtype. DeepSeek-V4-Pro declares expert_dtype fp4 beside an fp8
    # quantization_config: pricing its 384 experts at the global width doubles them from
    # 720 GiB to 1,441 GiB, which puts 180 GiB per rank on a 141 GiB H200 and makes a
    # deployment InferenceX ran at tp=8 on 8 GPUs look impossible. An unrecognized value
    # raises rather than falling back to the global width: falling back is the bug.
    # A method that quantizes only part of the checkpoint leaves the global width at the
    # unquantized base, so the experts -- which ARE quantized -- carry the narrow width
    # here. The mirror of the expert_dtype case below: there the experts are narrower
    # than a global that is itself quantized; here they are the only narrow thing.
    quant = raw_quant(cfg, model)
    if quant:
        fmt = str(quant.get("quant_method", "")).lower()
        if fmt in ("mxfp4", "nvfp4") and quant.get("modules_to_not_convert"):
            node["weight_dtype"] = fmt
    classes = quantized_classes(cfg, model)
    if classes.get("routed_experts"):
        node["weight_dtype"] = classes["routed_experts"]

    expert_dtype = cfg.get("expert_dtype")
    if expert_dtype is not None:
        mapped = EXPERT_DTYPES.get(str(expert_dtype))
        if mapped is None:
            raise DeriveError(
                f"{model}: expert_dtype {expert_dtype!r} is not one this deriver maps; "
                f"add it to EXPERT_DTYPES with the width vLLM resolves it to rather "
                f"than letting the global dtype stand in"
            )
        if mapped != DTYPES.get(str(pick(cfg, "dtype") or "").lower()):
            node["weight_dtype"] = mapped

    latent = pick(cfg, "moe_latent_size")
    if latent:
        # A projection to a narrower space runs before the routed experts, so the
        # expert matmul's K is the latent width rather than the model's hidden size.
        # It changes the expert FLOP count and the dispatch volume together.
        node["latent_size"] = int(latent)
    return node


def moe_block(cfg: dict[str, Any], model: str, hidden: int) -> list[dict[str, Any]]:
    """The routed expert layer: its latent projections, where it has them, and the
    grouped GEMM.

    A latent MoE is three pieces of work, not one. Kimi-K3's KimiSparseMoeBlock
    projects hidden -> latent, routes and runs the experts at the latent width, then
    projects latent -> hidden (modeling_kimi_linear.py:776-832); DeepSeek and Nemotron
    state the same structure as moe_latent_size. The two projections are dense GEMMs
    every token pays, so they are nodes rather than an implication of latent_size.

    The routing gate and the shared experts both run on the PRE-projection hidden
    states -- vLLM gates before the down-projection and adds
    `self.shared_experts(identity)` after the up-projection -- so neither is affected by
    the latent width, which is why shared_intermediate_size stays as moe() states it."""
    node = moe(cfg, model, hidden)
    nodes = [node]

    # A shared expert stored at a DIFFERENT width from the routed ones cannot ride on
    # the routed node: shared_experts/shared_intermediate_size describe work priced at
    # the node's own weight_dtype. nemotron-3-ultra-nvfp4 is the case in hand -- routed
    # experts NVFP4, shared experts FP8 -- so the shared half becomes its own node and
    # the routed node stops claiming it.
    classes = quantized_classes(cfg, model)
    shared_width = classes.get("shared_experts")
    if shared_width and shared_width != node.get("weight_dtype") and node.get(
            "shared_experts"):
        inner = int(node.get("shared_intermediate_size") or node["n"])
        shared = {
            "op": "GroupedGEMM",
            "role": "shared_experts",
            "n": inner,
            "k": node["k"],
            "experts": int(node["shared_experts"]),
            "top_k": int(node["shared_experts"]),
            "weight_dtype": shared_width,
        }
        if node.get("latent_size"):
            shared["latent_size"] = node["latent_size"]
        node.pop("shared_experts", None)
        node.pop("shared_intermediate_size", None)
        nodes.append(shared)

    latent = node.get("latent_size")
    if not latent:
        return nodes
    return [
        gemm("routed_expert_down_proj", int(latent), hidden),
        *nodes,
        gemm("routed_expert_up_proj", hidden, int(latent)),
    ]


# Emit conditions are a closed set naming WHEN a conditional node is part of the graph.
# They are enum members rather than predicate strings: every condition the catalog's models
# need is decided by the collective's role, so a resolver evaluates them with a switch and
# needs no expression parser. A graph cannot express a condition no resolver knows.
EMIT_TENSOR_PARALLEL = "tensor_parallel"
EMIT_EXPERT_PARALLEL = "expert_parallel"
EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE = "tensor_parallel_unless_sp_moe"
# Decided by the model's own sequence parallelism, for the architecture
# global.architecture names (blis-schemas#55); DeepSeek-V4.1 is the family that needs it.
EMIT_SEQUENCE_PARALLEL = "sequence_parallel"
EMIT_TENSOR_PARALLEL_UNLESS_SEQUENCE_PARALLEL = "tensor_parallel_unless_sequence_parallel"
# An MoE that reduces a different tensor in each mode states one collective per mode
# (blis-schemas emit conditions): the TP-sharded experts' own reduction, and the combine
# of expert-parallel partials when every rank holds every token. Muse Spark is the family
# that needs them (handler_muse_spark).
EMIT_TENSOR_PARALLEL_WITHOUT_EXPERT_PARALLEL = "tensor_parallel_without_expert_parallel"
EMIT_EXPERT_PARALLEL_WITHOUT_DISPATCH = "expert_parallel_without_dispatch"


def collective(op: str, role: str, emit: str) -> dict[str, Any]:
    return {"op": op, "role": role, "emit": emit}


# The layer-kind id a handler uses for a draft module it prices itself. Reserved rather
# than conventional: derive() prefers it over mirroring a target layer, so a handler that
# defines it is stating that no target layer has the draft module's structure.
DRAFT_KIND_ID = "mtp_moe"


# Architectures whose head width this deriver reads from the fp8 exclusion list. Keyed
# by architecture so adding a family is a deliberate act: five other fp8 configs exclude
# lm_head too (glm-5, glm-5.2-fp8, glm-5.3, minimax-m2.5, minimax-m2.7) and are correct
# to add here, but doing so re-derives their graphs, so they are not swept in as a side
# effect of this entry. Until they are, each understates its head by vocab x hidden bytes.
HEAD_UNQUANTIZED = {
    "Glm5NextForConditionalGeneration": {"lm_head": "lm_head"},
}


# GLM-5.3-Flash: checkpoint module suffix -> the graph role that prices it, for the
# tensors modules_to_not_convert names. Every entry was checked against the pinned
# safetensors headers (revision eb9eb208): the named tensors are BF16, and the ones
# deliberately absent -- q_a_proj, q_b_proj, kv_a_proj_with_mqa, the MLA o_proj, the
# routed and shared experts -- are F8_E4M3 and correctly take the global fp8 width.
#
# `self_attn.o_proj` maps to kda_o_proj rather than the MLA o_proj because the config
# excludes it on exactly the 34 linear_attention layers and on no sparse-attention layer;
# the two have distinct roles, so the suffix cannot cross over.
#
# kv_b_proj now has a node of its own, so the BF16 width the checkpoint states for it is
# finally expressible: the latent up-projection is BF16 while q_a/q_b/kv_a beside it are
# F8_E4M3, which a single fused qkv_proj could not represent at either width. The
# indexer's three projections are BF16 too, and vLLM passes quant_config=None for the
# fused wk_weights_proj outright.
# Projections a MODEL-SPECIFIC LOADER dequantizes at load time, so the resident
# parameter is the base width even though the checkpoint ships FP8 with scales.
#
# This is a different mechanism from modules_to_not_convert, and the distinction matters:
# those tensors are never quantized, while these are stored quantized and converted on
# the way in. glm5next's load_weights routes every weight and scale through
# _try_load_fp8_attn_proj before the normal loader (common/model.py:876-913), whose map
# covers q_a_proj, kv_a_proj_with_mqa, q_b_proj and o_proj (:1369-1374). Once both the
# FP8 weight and its weight_scale_inv have arrived it calls _dequant_fp8_block, produces
# BF16, and loads that into the unquantized parameter (:1415-1441). Its own docstring
# states the intent: "Dequantize FP8 q_a_proj / kv_a_proj_with_mqa / o_proj to BF16 on
# load ... the model holds them in BF16".
#
# q_a and kv_a both feed the fused A-projection, so the graph's single qkv_a_proj node
# takes the base width when EITHER is dequantized -- here both are.
#
# The guard the loader itself applies is `if target_s in params_dict: return False` --
# it declines where the model really did keep an FP8 parameter. So the rule is the
# resident parameter's width, and for this family the loader decides it.
GLM5_NEXT_LOADER_DEQUANTIZED = (
    "qkv_a_proj",
    "q_b_proj",
    "o_proj",
)


GLM5_NEXT_UNQUANTIZED = {
    "self_attn.q_proj": "kda_q_proj",
    "self_attn.k_proj": "kda_k_proj",
    "self_attn.v_proj": "kda_v_proj",
    "self_attn.b_proj": "kda_b_proj",
    "self_attn.f_a_proj": "kda_f_a_proj",
    "self_attn.f_b_proj": "kda_f_b_proj",
    "self_attn.g_a_proj": "kda_g_a_proj",
    "self_attn.g_b_proj": "kda_g_b_proj",
    "self_attn.o_proj": "kda_o_proj",
    "self_attn.kv_b_proj": "kv_b_proj",
    "self_attn.indexer.wq_b": "index_wq_b",
    "self_attn.indexer.wk": "index_wk_weights_proj",
    "self_attn.indexer.index_kpool_compress_gate": "index_kpool_compress_gate",
    "eh_proj": "mtp_eh_proj",
    "lm_head": "lm_head",
}


def compress(sequence: list[str]) -> dict[str, Any]:
    """Express a layer sequence as the smallest prologue, repeated pattern and epilogue.

    A literal 108-entry list and a pattern repeated four times describe the same stack, so
    the choice is about storage rather than meaning. Compression is worth doing for two
    reasons: a reader can see a hybrid's period at a glance where a literal list hides it,
    and a diff after a config change stays legible.

    It is NOT what makes step-time computation fast. A kernel collapses the stack to a
    per-kind layer count once at construction, so its per-step work is proportional to the
    number of distinct layer kinds -- three for the widest model here -- whichever way the
    sequence was spelled. Compression saves file size and reading effort, not cycles.

    The search prefers the fewest stored entries, and declines to store a prologue or
    epilogue longer than the pattern it surrounds: past that point the literal form is
    clearer than a split that technically compresses."""
    n = len(sequence)
    if n == 0:
        return {}
    best: tuple[int, int, list[str], int, list[str]] | None = None
    for head in range(0, n + 1):
        rest = sequence[head:]
        if not rest:
            continue
        for period in range(1, len(rest) // 2 + 1):
            repeats = len(rest) // period
            if repeats < 2:
                continue
            body = rest[:period]
            if body * repeats != rest[:period * repeats]:
                continue
            tail = rest[period * repeats:]
            stored = head + period + len(tail)
            # Fewest stored entries wins; the SHORTEST period breaks a tie, so a stack
            # whose true period is 1 is not spelled as a longer multiple of it.
            if best is None or (stored, period) < (best[0], len(best[2])):
                best = (stored, head, body, repeats, tail)
    if best is None or best[0] >= n:
        # No split stores less than the literal sequence, so state it literally. This is
        # also what enforces the docstring's readability rule: a prologue or epilogue so
        # long that the split stores no less than the sequence it describes is not a
        # compression worth reading, and the literal form is clearer.
        return {"prologue": list(sequence)}
    _, head, body, repeats, tail = best
    out: dict[str, Any] = {}
    if head:
        out["prologue"] = sequence[:head]
    out["pattern"] = body
    out["repeat"] = repeats
    if tail:
        out["epilogue"] = tail
    return out


def chain(nodes: list[dict[str, Any]]) -> list[list[int]]:
    """Return edges for a linear chain, which every layer in these architectures is."""
    return [[i, i + 1] for i in range(len(nodes) - 1)]


# How a family lays out its MLA latent stages, because vLLM does not do it one way and
# the differences are launches rather than labels. Keyed by the architecture string, and
# resolved through vLLM's own registry rather than by family resemblance:
#
#   "DeepseekV2ForCausalLM"      -> deepseek_v2.DeepseekV2Attention
#   "DeepseekV3ForCausalLM"      -> deepseek_v2.DeepseekV3ForCausalLM (same attention)
#   "GlmMoeDsaForCausalLM"       -> vllm.models.deepseek_v32  (registry.py:118)
#   "KimiK3ForConditionalGeneration" -> vllm.models.kimi_k3
#   "Glm5NextForConditionalGeneration" -> vllm.models.glm5next
#
# The attention CLASS matters, not the module. deepseek_v2.py defines two, and
# DeepseekV2DecoderLayer picks DeepseekV2MLAAttention whenever model_config.use_mla is
# true, falling back to DeepseekV2Attention otherwise (deepseek_v2.py:1286-1291). Every
# latent-cache model in this catalog takes the MLA class -- which the graphs themselves
# assert, since they price kind=mla with n_kv=1 -- so the MLA class is what to read.
#
# fused_a: one A-projection GEMM over [q_lora_rank, kv_lora_rank + rope].
#   DeepseekV2MLAAttention builds DeepSeekV2FusedQkvAProjLinear over exactly that
#   (deepseek_v2.py:1050-1057), as do KimiK3MergedQKVGateLinear and the deepseek_v32 and
#   glm5next paths. Only the non-MLA DeepseekV2Attention splits them (:493, :605), and
#   no model here takes that class.
# fused_norm: one norm over BOTH latents. kimi_k3 calls fused_q_kv_rmsnorm (mla.py:520)
#   and glm5next opts in with fuse_qkv_rmsnorm=True (glm5next/common/attention.py:604).
#   DeepseekV2MLAAttention does NOT opt in: it builds q_a_layernorm and kv_a_layernorm
#   separately (deepseek_v2.py:1081, 1097) and the wrapper never sets that flag, so the
#   DeepSeek and Kimi-K2.5 graphs keep two norm launches with one fused projection.
#
# An architecture absent here takes {fused_a: True, fused_norm: False}, which is what
# DeepseekV2MLAAttention does -- the shape every MLA family shares unless it opts into
# the norm fusion.
MLA_LAYOUTS = {
    "DeepseekV2ForCausalLM": {"fused_a": True, "fused_norm": False},
    "DeepseekV3ForCausalLM": {"fused_a": True, "fused_norm": False},
    # fused_index_norm: deepseek_v32 calls fused_norm_rope ONCE per layer,
    # unconditionally on the shared code path, passing the q norm, the kv norm, the rope
    # cache AND the indexer's k-norm weight/bias/eps into that single launch
    # (deepseek_v32/attention.py:378-392). glm5next does not: it fuses the two latent
    # norms (fuse_qkv_rmsnorm=True, glm5next/common/attention.py:604) and runs
    # _fused_indexer_k_norm as its own call (:352). So this family launches one norm
    # kernel where glm5next launches two.
    "GlmMoeDsaForCausalLM": {"fused_a": True, "fused_norm": True,
                             "fused_index_norm": True},
    "DeepseekV4ForCausalLM": {"fused_a": True, "fused_norm": True},
    "KimiK3ForConditionalGeneration": {"fused_a": True, "fused_norm": True},
    # kimi-k3's text_config names KimiLinearForCausalLM while the outer config names
    # KimiK3ForConditionalGeneration; both resolve to vllm.models.kimi_k3, so both map
    # here rather than relying on which view the lookup happens to read.
    "KimiLinearForCausalLM": {"fused_a": True, "fused_norm": True},
    "Glm5NextForConditionalGeneration": {"fused_a": True, "fused_norm": True},
    # kimi-k2.5 takes the DeepseekV2MLAAttention shape: its text_config declares
    # DeepseekV3ForCausalLM, the registry sends that to deepseek_v2 (registry.py:90),
    # and KimiK25ForConditionalGeneration initializes its text decoder through
    # DeepseekV2ForCausalLM -- whose decoder layer selects the MLA class because its
    # kimi_k2 text type keeps use_mla true. So: fused projection, separate norms, by the
    # DeepseekV3ForCausalLM entry above.
}


# Architectures whose indexer recomputes the head gate as its own FP32 matmul, rather
# than slicing it out of the wk_weights_proj result. glm5next alone does this
# (common/attention.py:341-350); deepseek_v32 slices and reuses
# (deepseek_v32/attention.py:309-315).
# Architectures that build their whole MLA module -- latent projections and indexer
# alike -- with quant_config=None, so every one of those weights serves at the base
# width whatever the checkpoint's global. glm5next states it in the call:
# `quant_config=None,  # MLA projections are BF16 in checkpoint`
# (glm5next/common/model.py:394). deepseek_v32 passes the real quant_config to its
# indexer's wq_b and only forces None on wk_weights_proj (attention.py:71-86), so it is
# NOT in this set and keeps the narrower override.
MLA_MODULE_UNQUANTIZED = frozenset({
    "Glm5NextForConditionalGeneration",
})


def mla_module_unquantized(raw: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Whether this architecture builds its entire MLA module unquantized."""
    for arches in (raw.get("architectures"), cfg.get("architectures")):
        for arch in arches or ():
            if arch in MLA_MODULE_UNQUANTIZED:
                return True
    return False


HEAD_GATE_RECOMPUTED = frozenset({
    "Glm5NextForConditionalGeneration",
})


def recomputes_head_gate(raw: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Whether this architecture runs a separate FP32 head-gate matmul."""
    for arches in (raw.get("architectures"), cfg.get("architectures")):
        for arch in arches or ():
            if arch in HEAD_GATE_RECOMPUTED:
                return True
    return False


def mla_layout(raw: dict[str, Any], cfg: dict[str, Any]) -> dict[str, bool]:
    """The latent-stage layout for this checkpoint's architecture."""
    # The OUTER architecture wins where the two differ: it is what vLLM's registry is
    # keyed on, and it is the module that builds the attention. kimi-k3 declares
    # KimiK3ForConditionalGeneration outside and KimiLinearForCausalLM in its
    # text_config; kimi-k2.5 declares KimiK25ForConditionalGeneration outside and
    # DeepseekV3ForCausalLM inside, and there the INNER one is the text decoder vLLM
    # actually instantiates -- so both views are consulted and an entry for either
    # spelling resolves the same way.
    #
    # Every flag defaults off against the base shape, so an entry states only what it
    # differs in and a new flag cannot silently change a family that never declared it.
    base = {"fused_a": True, "fused_norm": False, "fused_index_norm": False}
    for arches in (raw.get("architectures"), cfg.get("architectures")):
        for arch in arches or ():
            if arch in MLA_LAYOUTS:
                return {**base, **MLA_LAYOUTS[arch]}
    # The DeepseekV2MLAAttention shape, which is what an MLA family does unless it opts
    # into the norm fusion. A latent layer reaching this function already has a latent
    # cache, so the non-MLA class is not the right default for it.
    return base


def attention_block(cfg: dict[str, Any], model: str, hidden: int, *,
                    raw: dict[str, Any] | None = None,
                    kind: str | None = None,
                    window: int | None = None,
                    compress_ratio: int | None = None,
                    index_topk: int | None = None,
                    latent_width: int | None = None,
                    indexed: bool = True) -> list[dict[str, Any]]:
    """The nodes common to every attention layer: norm, QKV, attention, output, reduce."""
    nq = int(require(cfg, "num_q_heads", model))
    nkv = int(require(cfg, "num_kv_heads", model))
    attn = attention(cfg, model, kind=kind, window=window,
                     compress_ratio=compress_ratio, index_topk=index_topk,
                     latent_width=latent_width)
    if latent_width is not None:
        # This family states one inclusive per-token width and no separate nope/v
        # widths, so the projections are sized from that width: a per-head query at the
        # full width, plus the one shared latent KV vector the cache stores. The rope
        # width is NOT added -- latent_width already contains it, which is the whole
        # point of its being inclusive, and adding it again would overstate the
        # projection by an eighth exactly as it would the cache.
        qkv_out = (nq + 1) * int(latent_width)
        o_in = nq * int(latent_width)
    elif attn["kind"] in ("mla", "sparse_mla"):
        # The latent path is a sequence of low-rank stages, not one fused projection,
        # and the config states every width it needs. Each stage below is a GEMM an
        # engine launches and a norm between the two it separates, so the whole chain is
        # returned rather than collapsed into a query-width stand-in.
        nope = int(pick(cfg, "qk_nope_head_dim") or 0)
        rope = int(pick(cfg, "qk_rope_head_dim") or 0)
        v = int(pick(cfg, "v_head_dim") or nope)
        kv_lora = int(pick(cfg, "kv_lora_rank") or 0)
        # q_lora_rank is read by its literal name: it is not a cross-dialect concept in
        # ALIASES, and its ABSENCE is meaningful here -- it selects the direct-q_proj
        # branch rather than defaulting a width.
        q_lora = int(cfg.get("q_lora_rank") or 0)
        qk_head = nope + rope
        if qk_head == 0 or v == 0:
            raise DeriveError(f"{model}: latent attention with no head widths to size "
                              f"its projections")
        if kv_lora == 0:
            raise DeriveError(f"{model}: latent attention with no kv_lora_rank to size "
                              f"its compressed KV stage")
        # A norm's traffic is derivable from hidden_size only when it runs at hidden
        # width. These do not: the q norm is q_lora_rank wide and the kv norm
        # kv_lora_rank, so each states its own traffic or it prices as a hidden-width
        # pass. RMSNorm accumulates in fp32 over a read and a write.
        # raw is needed for the activation width: two committed configs state their
        # dtype nowhere a text_config view can see, so guessing a default here would
        # silently halve or double the traffic on the stages below.
        if raw is None:
            raise DeriveError(
                f"{model}: the latent stages size their norm traffic from the declared "
                f"activation width, so this call must pass raw")
        act = DTYPE_WIDTHS[base_dtype(cfg, raw, model)]

        def latent_norm(role: str, width: int) -> dict[str, Any]:
            return dict(norm(role), bytes_per_token=2 * width * act)

        # Where the checkpoint's quantization skips the attention modules, these stages
        # store at the base width and the global one would misprice them. Both Kimi
        # configs do this: kimi-k2.5 is globally int4 and kimi-k3 mxfp4, and each names
        # `re:.*self_attn.*` in its compressed-tensors ignore list, so every projection
        # below stays unquantized at runtime.
        attn_width = (base_dtype(cfg, raw, model)
                      if compressed_tensors_skips_attention(cfg, raw) else None)

        def staged(node: dict[str, Any]) -> dict[str, Any]:
            if attn_width and node.get("role") in ATTENTION_WEIGHT_ROLES:
                return dict(node, weight_dtype=attn_width)
            return node

        layout = mla_layout(raw, cfg)
        stages = []
        if q_lora:
            # A declared q_lora_rank makes the query low-rank too, and vLLM fuses the
            # two A-stages into one GEMM: DeepSeekV2FusedQkvAProjLinear over
            # [q_lora_rank, kv_lora_rank + qk_rope_head_dim] (glm5next attention.py
            # 464-469, the DeepSeek MLA path it shares). Priced as that one launch,
            # which is what a deployment runs; the checkpoint stores the halves
            # separately, and the weight bytes are identical either way.
            # An output gate rides in the SAME launch where the config declares one.
            # (Only a fused-A family can carry it: it is a shard of that projection.)
            # Kimi-K3 sets mla_use_output_gate, and vLLM then builds
            # KimiK3MergedQKVGateLinear in place of the plain fused A-projection, with
            # the gate as a third shard (kimi_k3/nvidia/mla.py:218-228). The forward
            # splits [qkv_a_rows, num_local_heads * v_head_dim] off the one result and
            # sigmoid-multiplies the attention output by it (mla.py:574-576, 621-622).
            # Those rows are part of this GEMM, so omitting them dropped both the
            # parameters and the work: for kimi-k3 that is 96 * 128 = 12,288 rows, which
            # makes the real launch (14,400, 7,168) rather than (2,112, 7,168).
            gate_rows = nq * v if cfg.get("mla_use_output_gate") else 0
            if layout["fused_a"]:
                # One A-projection over both latents, as DeepSeekV2FusedQkvAProjLinear
                # and KimiK3MergedQKVGateLinear build it.
                stages.append(gemm("qkv_a_proj",
                                   q_lora + kv_lora + rope + gate_rows, hidden))
            else:
                # The generic DeepseekV2Attention builds these separately
                # (deepseek_v2.py:493 q_a_proj, :605 kv_a_proj_with_mqa), which is two
                # launches rather than one. An output gate is a shard of a fused
                # projection, so a family without one cannot be carrying it.
                if gate_rows:
                    raise DeriveError(
                        f"{model}: mla_use_output_gate is set, but this architecture "
                        f"builds its A-projections separately, so there is no fused "
                        f"launch for the gate to be a shard of")
                stages.append(gemm("q_a_proj", q_lora, hidden))
                stages.append(gemm("kv_a_proj", kv_lora + rope, hidden))
            if layout["fused_norm"]:
                # One launch, not two. Kimi-K3 calls fused_q_kv_rmsnorm over both
                # latents (kimi_k3/nvidia/mla.py:520) and glm5next asks the shared MLA
                # module for the same with fuse_qkv_rmsnorm=True
                # (glm5next common/attention.py:604).
                #
                # Where the family ALSO folds the indexer's k-norm into that same call,
                # its width joins this node's traffic. deepseek_v32 passes index_k and
                # the indexer's norm weight/bias/eps into the one fused_norm_rope
                # (deepseek_v32/attention.py:378-392), and that kernel still reads,
                # normalises and writes index_k -- fusion removes a LAUNCH, not the HBM
                # traffic. Suppressing the separate index_k_norm node without adding its
                # width here would delete bytes the kernel really moves.
                # The folded width belongs only to layers that actually RUN the
                # indexer. A skipped layer takes the else at
                # deepseek_v32/attention.py:339-360 with has_indexer = False, and
                # fused_norm_rope substitutes dummies and skips the indexer program
                # under HAS_INDEXER -- so its norm moves the two latents alone. Gated on
                # the per-layer state rather than the architecture, because
                # attention_block() builds the skipped kinds too.
                norm_width = q_lora + kv_lora
                if layout["fused_index_norm"] and indexed:
                    norm_width += int(require_key(cfg, "index_head_dim", model))
                stages.append(latent_norm("qkv_a_layernorm", norm_width))
            else:
                # Two calls, in the order the generic forward makes them: q_a_layernorm
                # right after q_a_proj (deepseek_v2.py:598) and kv_a_layernorm after the
                # KV A-projection (:608).
                stages.append(latent_norm("q_a_layernorm", q_lora))
                stages.append(latent_norm("kv_a_layernorm", kv_lora))
            stages.append(gemm("q_b_proj", nq * qk_head, q_lora))
        else:
            # No q_lora_rank: vLLM takes the else branch and builds a direct q_proj at
            # full width, with only the KV side compressed (deepseek-v2-lite).
            stages.append(gemm("kv_a_proj", kv_lora + rope, hidden))
            # Only the KV side is compressed here, so there is one latent norm and
            # nothing to fuse it with.
            stages.append(latent_norm("kv_a_layernorm", kv_lora))
            stages.append(gemm("q_proj", nq * qk_head, hidden))
        # The up-projection of the cached latent into per-head nope + value widths. Read
        # on every decode step for every cached token, so its width is the one a fused
        # node hid most consequentially.
        stages.append(gemm("kv_b_proj", nq * (nope + v), kv_lora))
        post_attn = []
        if q_lora and cfg.get("mla_use_output_gate"):
            # The sigmoid-multiply itself, which runs on the attention output before
            # o_proj: read the output and the gate, write the product.
            post_attn.append(dict(norm("mla_output_gate"),
                                  bytes_per_token=3 * nq * v * act))
        return [
            norm("input_norm"),
            *[staged(n) for n in stages],
            attn,
            *post_attn,
            staged(gemm("o_proj", hidden, nq * v)),
            collective("AllReduce", "attn_out", EMIT_TENSOR_PARALLEL),
        ]
    else:
        dh = attn["d_h"]
        qkv_out = (nq + 2 * nkv) * dh
        o_in = nq * dh
    return [
        norm("input_norm"),
        gemm("qkv_proj", qkv_out, hidden),
        attn,
        gemm("o_proj", hidden, o_in),
        collective("AllReduce", "attn_out", EMIT_TENSOR_PARALLEL),
    ]


def indexed_layers(cfg: dict[str, Any], model: str, layers: int) -> list[bool] | None:
    """Which layers actually build an indexer, or None where the family has no indexer.

    A DSA config does not necessarily index every layer. vLLM computes a per-layer
    `skip_topk` and constructs the indexer only when it is false
    (models/deepseek_v32/attention.py:166-200, where GlmMoeDsaForCausalLM is served):

        if index_topk_pattern is None:
            skip_topk = max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
        elif 0 <= layer_id < len(index_topk_pattern):
            skip_topk = index_topk_pattern[layer_id] == "S"
        else:
            skip_topk = False

    The defaults matter: freq 1 and offset 2 make every layer indexed, which is why the
    families that declare neither field are unaffected. glm-5.2, glm-5.2-fp8 and glm-5.3
    declare freq 4 with offset 3 over 78 layers, which indexes 21 of them -- so charging
    all 78 for the scorer and its projections overprices that work by 78/21.

    Returns a per-layer list so a handler can key its layer kinds off it rather than
    collapsing indexing to one model-wide flag."""
    if not (pick(cfg, "index_topk") and cfg.get("index_n_heads") is not None):
        return None
    pattern = cfg.get("index_topk_pattern")
    if pattern is not None:
        if len(pattern) < layers:
            raise DeriveError(
                f"{model}: index_topk_pattern has {len(pattern)} entries for {layers} "
                f"layers; a layer past the end would silently take the indexed branch")
        unknown = set(pattern[:layers]) - {"S", "I"}
        if unknown:
            raise DeriveError(
                f"{model}: index_topk_pattern names {sorted(unknown)}; this deriver "
                f"reads 'S' as skipped and anything else as indexed, so an unexpected "
                f"code would be priced as indexed on a guess")
        return [pattern[i] != "S" for i in range(layers)]
    freq = int(cfg.get("index_topk_freq") or 1)
    offset = cfg.get("index_skip_topk_offset")
    offset = 2 if offset is None else int(offset)
    if freq < 1:
        raise DeriveError(
            f"{model}: index_topk_freq is {freq}; the modulus must be at least 1")
    return [max(i - offset + 1, 0) % freq == 0 for i in range(layers)]


def lightning_indexer(cfg: dict[str, Any], model: str, hidden: int,
                      nodes: list[dict[str, Any]],
                      raw: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Splice a DSA lightning indexer in front of the attention node.

    The indexer is what makes a sparse latent read sparse: it scores every cached
    position to select the top-k the attention then reads. That makes it the ONLY
    context-proportional term in such a layer -- measured at 3.53x over a 12x context
    increase where the attention it feeds moves 1.18x -- so it is its own node, with no
    window, rather than folded into a bounded attention.

    A graph recording index_topk without this node asserts a selection it does not
    charge for, which understates the dominant long-context cost. Shared by every family
    that ships the indexer rather than reimplemented per handler, which is how the
    GLM-5 family came to declare the full indexer spec and emit none of it."""
    index_heads = int(require_key(cfg, "index_n_heads", model))
    index_dim = int(require_key(cfg, "index_head_dim", model))
    kpool = int(cfg.get("index_kpool") or 0)
    q_lora = int(cfg.get("q_lora_rank") or 0)
    at = next(i for i, n in enumerate(nodes) if n.get("op") == "Attention")

    # One shape for every family that reaches this helper, because vLLM builds one:
    # GlmMoeDsaForCausalLM is served by the deepseek_v32 module (its __init__ aliases
    # GlmMoeDsaForCausalLM = DeepseekV32ForCausalLM), and that indexer
    # (models/deepseek_v32/attention.py:71-85) constructs exactly what glm5next's does
    # (models/glm5next/common/attention.py:265-282):
    #
    #   wq_b              q_lora_rank -> n_head * head_dim      (ReplicatedLinear)
    #   wk_weights_proj   hidden -> [head_dim, n_head]           (fused, ONE GEMM)
    #
    # The query stage reads the LATENT rank, not hidden. An earlier revision of this
    # helper emitted a single hidden-wide GEMM of n_head*head_dim + head_dim, which
    # overstated the projection work -- 2.3x on GLM-5.3-Flash -- and had the wrong k.
    if not q_lora:
        raise DeriveError(
            f"{model}: a lightning indexer projects its query from the latent rank, "
            f"and the config declares no q_lora_rank to size wq_b from")
    # wk_weights_proj is constructed with quant_config=None in both implementations
    # (deepseek_v32/attention.py:78-86, glm5next/common/attention.py:274-281), so it is
    # served unquantized whatever the checkpoint's global width. The fp8 GLM DSA graphs
    # would otherwise inherit fp8 for a matrix the runtime keeps at the base width.
    base = base_dtype(cfg, raw, model)
    # wk_weights_proj is built quant_config=None in BOTH implementations
    # (deepseek_v32/attention.py:78-86, glm5next/common/attention.py:274-281), so it
    # always takes the base width. wq_b takes it only where the whole module is
    # unquantized: deepseek_v32 hands wq_b the real quant_config.
    module_base = mla_module_unquantized(raw, cfg)
    wq_b = gemm("index_wq_b", index_heads * index_dim, q_lora)
    if module_base:
        wq_b = dict(wq_b, weight_dtype=base)
    projections = [
        wq_b,
        dict(gemm("index_wk_weights_proj", index_dim + index_heads, hidden),
             weight_dtype=base),
    ]
    # The FP32 head gate is glm5next's ALONE. Its forward keeps the first head_dim rows
    # of wk_weights_proj for K, then caches the remaining n_head rows transposed to FP32
    # and runs its own matmul against fp32 activations -- `weights =
    # torch.mm(hidden_states.float(), self._wp_fp32)` (glm5next/common/attention.py
    # 341-350) -- which is a second launch and a resident FP32 copy.
    #
    # deepseek_v32 does NOT: it slices index_weights straight out of the same GEMM's
    # result and hands it to fused_q (deepseek_v32/attention.py:309-315, 433-446), with
    # no FP32 copy and no second matmul. Emitting this node for that family would invent
    # an FP32 launch on every indexed glm-5* layer, so it is gated on the architecture
    # rather than on the presence of an indexer.
    if recomputes_head_gate(raw, cfg):
        projections.append(dict(gemm("index_head_gate_proj", index_heads, hidden),
                                weight_dtype="fp32"))
    if kpool > 1:
        if raw is None:
            raise DeriveError(
                f"{model}: the kpool compression pass sizes its traffic from the "
                f"declared activation width, so this call must pass raw")
        # The compression gate is BF16 in the checkpoint, and glm5next -- the only
        # family with a pooled indexer -- builds the module unquantized anyway.
        projections.append(dict(gemm("index_kpool_compress_gate", index_dim, hidden),
                                weight_dtype=base))
        # The compression kernel runs ONE PROGRAM PER POOL, not per token
        # (nvidia/ops/kpool_compress.py:165-245), so its figure is per-pool traffic
        # amortized over the index_kpool tokens that fill one pool -- bytes_per_token is
        # charged once per model token by a cost model.
        #
        # Per pool, read off the kernel: slot_score is loaded in BOTH passes (the
        # per-dim max for softmax stability, then the weighted sum), the FP32 APE row
        # likewise, slot_k once, and the result is one Hadamard-rotated fp8 vector plus
        # its fp32 scale.
        # Each tensor at its STORED width, not the width the kernel accumulates in.
        # slot_score is bf16: gate_score comes from F.linear over bf16 hidden states and
        # the bf16 compression gate, and the tail cache documents its own copy as the
        # "bf16 gate score" (common/attention.py:172). The Triton load casts to fp32 in
        # registers, which costs no HBM. Only the APE is genuinely fp32 in memory.
        act = DTYPE_WIDTHS[base_dtype(cfg, raw, model)]
        f32 = DTYPE_WIDTHS["fp32"]
        per_pool = (2 * kpool * index_dim * act       # slot_score, bf16, both passes
                    + 2 * kpool * index_dim * f32     # APE rows, fp32, both passes
                    + kpool * index_dim * act         # slot_k, bf16
                    + index_dim + f32)                # fp8 vector + its fp32 scale
        projections.append({
            "op": "Elementwise",
            "role": "index_kpool_compress",
            "bytes_per_token": per_pool // kpool,
        })
        # The APE's read traffic is already inside that figure. It is NOT emitted as a
        # node of its own: the only primitive that could carry its bytes is a GEMM, and
        # a GEMM means 2*n*k FLOPs per token plus a launch in the consuming kernel
        # (blis-latency-kernel internal/price/plan.go:266-269). The APE is a storage-only
        # [index_kpool, index_head_dim] fp32 table consumed inside the pooling kernel, so
        # a GEMM would invent 1,024 FLOPs/token and a launch per indexed layer -- a worse
        # error than the residency it would record. Representing resident parameter bytes
        # that carry no compute needs a schema primitive the ModelGraph does not have;
        # tracked rather than faked.
    # The indexer's k-norm is a node only where it is its OWN launch. deepseek_v32 folds
    # it into the same fused_norm_rope call as the two latent norms
    # (deepseek_v32/attention.py:378-392), so emitting it there would charge a
    # normalisation kernel that family does not launch -- the traffic is already in the
    # fused latent norm. glm5next runs _fused_indexer_k_norm separately (:352), so it
    # keeps its node.
    #
    # Where it is emitted, it reads and writes at the ACTIVATION width:
    # _fused_indexer_k_norm is `F.layer_norm(x.float(), ...).type_as(x)`
    # (glm5next/common/attention.py:55-59), so the fp32 cast is a temporary inside the
    # fused kernel and x is the bf16 output of the unquantized wk_weights_proj. Sizing
    # this from the accumulation dtype would double its HBM traffic.
    if not mla_layout(raw or {}, cfg).get("fused_index_norm"):
        projections.append(dict(norm("index_k_norm"),
                                bytes_per_token=2 * index_dim
                                * DTYPE_WIDTHS[base_dtype(cfg, raw, model)]))

    scorer = {
        "op": "Attention",
        "role": "block_index_scores",
        "kind": "gqa",
        "n_q": index_heads,
        "n_kv": 1,
        "d_h": index_dim,
    }
    if kpool > 1:
        # The scan is over POOLED candidates: one K state per index_kpool tokens, which
        # vLLM expresses as the cache spec's tokens_per_state and sizes as
        # max_model_len // index_kpool (common/attention.py:128-131, 314).
        #
        # index_topk is deliberately NOT set here. The scorer must score EVERY pooled
        # candidate in order to discover the top-k; the selection bounds the main MLA
        # read, which already carries index_topk. Setting it here would make the scorer
        # read only the answers it exists to find, and would cap the one
        # context-proportional term in the layer at a constant. DeepSeek-V4's
        # representation draws the same line: block_index_scores is unbounded and only
        # the downstream sparse MLA node carries the top-k.
        scorer["kind"] = "sparse_mla"
        scorer["compress_ratio"] = kpool
    return nodes[:at] + projections + [scorer] + nodes[at:]


# --- Architecture handlers ------------------------------------------------------
# One handler per family. A family whose layer sequence is uniform returns one layer
# kind and a repeat; a hybrid returns several kinds and the sequence that orders them.
# An architecture with no handler raises rather than falling back on a default, which
# is what keeps a misread config from becoming a plausible-looking graph.


def handler_dense(cfg, raw, model):
    """Llama, Mistral, Qwen2/3, Yi, CodeLlama: GQA attention and a dense MLP.

    A sliding window, where the config declares one, changes the attention kind: the
    per-token read is bounded by the window rather than by context length."""
    hidden = int(require(cfg, "hidden_size", model))
    window = pick(cfg, "sliding_window")
    # A declared window is inert unless the config also switches it on. Qwen2.5 ships
    # use_sliding_window: false with a window value present, and pricing it as
    # windowed would understate every layer's KV read.
    if window and cfg.get("use_sliding_window") is False:
        window = None
    nodes = attention_block(cfg, model, hidden, raw=raw, window=window)
    nodes += [norm("post_attn_norm")]
    nodes += dense_mlp(cfg, model, hidden)
    nodes += [collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
    kind = {"id": "dense", "nodes": nodes, "edges": chain(nodes)}
    layers = int(require(cfg, "num_layers", model))
    return [kind], {"pattern": ["dense"], "repeat": layers}


def handler_moe(cfg, raw, model):
    """Mixtral, Qwen3-MoE, Llama-4, Granite-MoE, Inkling, DeepSeek-V2, GLM: attention
    plus a routed expert layer.

    Where a config alternates dense and sparse MLP layers, the dense ones become a
    prologue: the sequence has no repeating unit that includes them.

    KNOWN GAP (Llama-4): this family uses chunked local attention bounded by
    attention_chunk_size on most layers, with periodic full-attention NoPE layers, and
    this handler prices every layer as unbounded GQA. The split is NOT derivable from the
    committed Scout checkpoint: transformers reads it from config.layer_types
    (modeling_llama4.py dispatches create_chunked_causal_mask per layer from that list),
    and this config declares no layer_types, no moe_layers, and an EMPTY no_rope_layers.
    Deriving a pattern would mean inventing one, which is what this deriver's handler
    registry exists to prevent.

    The cost of the gap is not uniform across the catalog's workloads, and is larger than
    "chunk size exceeds every context" would suggest. At attention_chunk_size 8192,
    chatbot (512) and contentgen (2048) sit well inside one chunk, where a bounded and an
    unbounded read coincide exactly. summarization reaches 8704 only at its maximum. But
    multidoc's TYPICAL context is 11776 and its maximum 22016, so on that workload this
    handler overstates the per-token read on the chunked layers by roughly the ratio of
    context to chunk -- about 1.4x typical, 2.7x at the tail.

    Recorded rather than silently accepted: a reader comparing a Llama-4 multidoc
    prediction against a measurement should suspect this first. Closing it needs the
    per-layer split, which needs either a checkpoint that declares layer_types or the
    vendor pattern confirmed against an implementation.
    """
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))

    window = pick(cfg, "sliding_window")
    if window and cfg.get("use_sliding_window") is False:
        window = None

    # A family in this group may ship a DSA lightning indexer: the GLM-5 generations
    # declare model_type glm_moe_dsa with a full index_n_heads/index_head_dim/index_topk
    # spec, and attention() reads the topk into kind=sparse_mla. The selection work is
    # a node of its own, and omitting it leaves the graph asserting a top-k it never
    # charges for -- which is what happened here across 78 layers per model.
    # ...but not necessarily on every layer. indexed_layers() expands the config's own
    # skip_topk rule, which for glm-5.2/5.2-fp8/5.3 indexes 21 of 78 layers. Keyed per
    # layer rather than per model: a single flag charged all 78 for a scorer that 57 of
    # them never build.
    index_map = indexed_layers(cfg, model, layers)
    indexed_any = index_map is not None and any(index_map)
    indexed_all = index_map is not None and all(index_map)
    # A layer kind must be uniform in whether it indexes, so a partially-indexed model
    # needs the distinction carried in the kind id.
    split_index = indexed_any and not indexed_all

    def attn_nodes(win=None, index=None):
        use = indexed_any if index is None else index
        n = attention_block(cfg, model, hidden, raw=raw, window=win, indexed=use)
        return lightning_indexer(cfg, model, hidden, n, raw) if use else n

    def sparse_nodes(win=None, index=None):
        n = attn_nodes(win, index)
        n += [norm("post_attn_norm"), *moe_block(cfg, model, hidden),
              collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    def dense_nodes(win=None, index=None):
        n = attn_nodes(win, index)
        n += [norm("post_attn_norm")] + dense_mlp(cfg, model, hidden)
        n += [collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    def suffix(index: bool) -> str:
        """The kind-id suffix that keeps an indexed layer distinct from a skipped one."""
        if not split_index:
            return ""
        return "_indexed" if index else "_noindex"

    # The MTP module always builds an indexer, whatever the base skip rule says. vLLM
    # evaluates the draft at layer_id == num_hidden_layers and then constructs under
    # `if not skip_topk or is_mtp_layer` (deepseek_v32/attention.py:179-200), so the
    # is_mtp_layer term forces it even on a frequency-4 config whose layer 78 would skip.
    # Without a declared draft kind, derive()'s fallback mirrors the last DECLARED kind,
    # which on these configs is the unindexed one -- underpricing the draft indexer on
    # every speculative step.
    def declare_draft(kinds: list[dict[str, Any]]) -> None:
        if not split_index:
            return
        if not int(pick(cfg, "num_spec_tokens") or 0):
            return
        nodes = sparse_nodes(window, True)
        kinds.append({"id": DRAFT_KIND_ID, "nodes": nodes, "edges": chain(nodes)})


    # A per-layer ATTENTION-kind vector, which is a different thing from the per-layer
    # MLP-type vector handled below: this one says which layers see a bounded window
    # while every layer keeps the same MLP. gpt-oss declares it as layer_types over
    # "sliding_attention" and "full_attention".
    #
    # Checked against the layer count rather than trusted: a vector of the wrong length
    # would silently price a different depth than the config states.
    attn_types = cfg.get("layer_types")
    if attn_types and window:
        if len(attn_types) != layers:
            raise DeriveError(
                f"{model}: layer_types has {len(attn_types)} entries for {layers} layers"
            )
        unknown = set(attn_types) - {"sliding_attention", "full_attention"}
        if unknown:
            raise DeriveError(
                f"{model}: layer_types names {sorted(unknown)}, which this handler cannot "
                f"price; add the kind rather than defaulting it to full attention"
            )
        if split_index:
            raise DeriveError(
                f"{model}: a per-layer attention-kind vector AND a partial indexer "
                f"pattern would make the layer kind a three-way product; no committed "
                f"config does both, so this handler refuses rather than pricing one "
                f"dimension and dropping the other")
        swa = sparse_nodes(win=window)
        full = sparse_nodes(win=None)
        kinds = [
            {"id": "swa_moe", "nodes": swa, "edges": chain(swa)},
            {"id": "full_moe", "nodes": full, "edges": chain(full)},
        ]
        sequence = ["swa_moe" if t == "sliding_attention" else "full_moe"
                    for t in attn_types]
        return kinds, compress(sequence)

    # Inkling interleaves sliding-window and full-attention layers on a declared
    # index list; every other member of this family is uniform in attention kind.
    local_ids = cfg.get("local_layer_ids")
    if local_ids and window:
        if split_index:
            raise DeriveError(
                f"{model}: a declared local-layer list AND a partial indexer pattern "
                f"would make the layer kind a three-way product; no committed config "
                f"does both, so this handler refuses rather than dropping one")
        swa = sparse_nodes(win=window)
        full = sparse_nodes(win=None)
        kinds = [
            {"id": "swa_moe", "nodes": swa, "edges": chain(swa)},
            {"id": "full_moe", "nodes": full, "edges": chain(full)},
        ]
        local = set(int(i) for i in local_ids)
        sequence = ["swa_moe" if i in local else "full_moe" for i in range(layers)]
        if raw.get("architectures", [None])[0] != "InklingForConditionalGeneration":
            return kinds, compress(sequence)
        # Inkling's stage boundary carries hidden_states alone at the model dtype, not
        # vLLM's standard hidden_states + residual pair: the pending short-conv residual
        # add is applied before the send (make_empty_intermediate_tensors_factory(
        # ["hidden_states"]), inkling/nvidia/model.py:294-296, 354-359@v0.32.0; add_rmsnorm
        # allocates its output like the residual, nvidia/ops/norm.py:160-161).
        boundary = hidden * DTYPE_WIDTHS[base_dtype(cfg, raw, model)]
        return kinds, compress(sequence), {
            "global": {"pipeline_boundary_bytes_per_token": boundary}}

    # A per-layer MLP type vector marks which layers are sparse.
    mlp_types = cfg.get("mlp_layer_types")
    if mlp_types:
        if len(mlp_types) != layers:
            raise DeriveError(
                f"{model}: mlp_layer_types has {len(mlp_types)} entries for "
                f"{layers} layers"
            )
        # The kind of a layer is the pair (mlp type, does it index), because a skipped
        # layer builds no indexer at all. Only the combinations the sequence actually
        # uses are emitted.
        sequence = []
        for i, t in enumerate(mlp_types):
            idx = True if index_map is None else index_map[i]
            sequence.append(("attn_moe" if t == "sparse" else "attn_dense") + suffix(idx))
        kinds = []
        for mlp_t, builder in (("dense", dense_nodes), ("moe", sparse_nodes)):
            for idx in (True, False):
                kid = ("attn_moe" if mlp_t == "moe" else "attn_dense") + suffix(idx)
                if kid in {k["id"] for k in kinds} or kid not in set(sequence):
                    continue
                nodes = builder(window, idx)
                kinds.append({"id": kid, "nodes": nodes, "edges": chain(nodes)})
        declare_draft(kinds)
        # compress() finds the trailing uniform run itself, which is the common shape
        # here: a short dense prologue then sparse throughout.
        return kinds, compress(sequence)

    # A first_k_dense_replace count does the same job as a type vector.
    first_dense = int(cfg.get("first_k_dense_replace") or 0)
    if split_index:
        # A partially-indexed stack has no single repeated layer, so the sequence is
        # expanded per layer and compress() finds whatever period it really has.
        sequence = []
        for i in range(layers):
            base = "attn_dense" if i < first_dense else "attn_moe"
            sequence.append(base + suffix(index_map[i]))
        kinds = []
        for kid in dict.fromkeys(sequence):
            idx = kid.endswith("_indexed")
            builder = dense_nodes if kid.startswith("attn_dense") else sparse_nodes
            nodes = builder(window, idx)
            kinds.append({"id": kid, "nodes": nodes, "edges": chain(nodes)})
        declare_draft(kinds)
        return kinds, compress(sequence)
    sparse = sparse_nodes(window)
    kinds = [{"id": "attn_moe", "nodes": sparse, "edges": chain(sparse)}]
    if first_dense:
        dense = dense_nodes(window)
        kinds.insert(0, {"id": "attn_dense", "nodes": dense, "edges": chain(dense)})
        return kinds, {
            "prologue": ["attn_dense"] * first_dense,
            "pattern": ["attn_moe"],
            "repeat": layers - first_dense,
        }
    return kinds, {"pattern": ["attn_moe"], "repeat": layers}


def handler_minimax_m2(cfg, raw, model):
    """MiniMax-M2: GQA attention plus a routed expert layer on every layer.

    The config carries attn_type_list, a per-layer vector, and on the published M2.5
    weights every entry is the same. So the stack is uniform and handler_moe prices it
    exactly -- but the vector is checked rather than ignored, because a later variant that
    mixed attention kinds would otherwise be priced as though it did not.
    """
    types = cfg.get("attn_type_list")
    if types:
        layers = int(require(cfg, "num_layers", model))
        if len(types) != layers:
            raise DeriveError(
                f"{model}: attn_type_list has {len(types)} entries for {layers} layers"
            )
        if len(set(types)) != 1:
            raise DeriveError(
                f"{model}: attn_type_list mixes {sorted(set(types))}. This handler prices "
                f"a uniform stack; a mixed one needs its kinds named and priced rather "
                f"than collapsed to one"
            )
    return handler_moe(cfg, raw, model)


# DeepSeek-V4's KV record, the same on SM90 and SM100: 448 FP8 NoPE dims, 64 BF16 RoPE
# dims and eight bytes of UE8M0 scales, one per 64 NoPE dims plus a pad
# (deepseek_v4/attention.py:876-878; compressor.py:299-301). vLLM forces the cache
# dtype to fp8_ds_mla (attention.py:106-122), whose pages pad to FlashMLA's 576 B stride
# (attention.py:873; v1/attention/backends/mla/sparse_swa.py:84).
DSV4_KV_RECORD_BYTES = 584
DSV4_PAGE_ALIGNMENT = 576
# The compressed and indexer caches take 256-token pages: FlashMLA and the V4 indexer
# accept exactly that (sparse_mla.py:60-61; v1/attention/backends/mla/indexer.py:
# 260-262), and the engine sizes its block from the backends' common size
# (platforms/interface.py:616-697), so a page holds 64 entries at ratio 4 and 2 at 128.
DSV4_SPARSE_BLOCK_TOKENS = 256


def handler_deepseek_v4(cfg, raw, model):
    """DeepSeek-V4: compressed sparse MLA at two ratios, each compressed layer with its
    own compressor and caches, an indexer on the ratio-4 layers, mHC on every
    sublayer, hash routing on the first num_hash_layers layers, and one MTP draft layer.

    Priced as vLLM v0.31.0 serves it on CUDA with default backends (FlashMLA, no
    mega-MoE). The attention class, the 584 B record and every resident width are the
    same on SM90 and SM100 (nvidia/model.py:1146-1177; attention.py:856-879), so the
    graph has no architecture overlay. One launch's figure differs: on SM100 the
    inverse-RoPE quant packs its scales as E8M0 rather than SM90's FP32 per 128
    (nvidia/ops/o_proj.py:19-32), under 1% of that launch. The kernels that do differ -- Marlin W4A16 experts on SM90,
    FlashInfer or DeepGEMM MXFP4 on SM100 (fused_moe/oracle/mxfp4.py:367-403) -- are
    the cost model's to price.

    Every layer, per vLLM's own reading of compress_ratios (attention.py:229-235):

      * every layer keeps its own 128-token sliding window of 584 B records in
        64-token pages (attention.py:345-352; sparse_swa.py:82-84);
      * ratio 4 (C4A): an overlapping compressor (2 x 512 wide, compressor.py:239-240)
        owns a compressed stream of one record per 4 tokens, and an
        indexer scores its own FP8 key stream -- one 132 B record per 4 tokens -- to
        select at most index_topk of them (attention.py:298-299, 1005-1031). The main
        read is the window plus that top-k (nvidia/flashmla.py:193-203);
      * ratio 128 (C128A): a compressor, no indexer, and a main read of the window plus
        EVERY compressed entry, (pos + 1) // 128 of them, which grows with context
        (sparse_mla.py:403-437);
      * the MTP layer, at id num_hidden_layers, takes ratio 1: its window alone, with
        no compressor, indexer or compressed cache (attention.py:232-235, 857-860).

    No cache is shared between layers: each compressed layer builds its own
    compressor, state cache and compressed cache under its own prefix
    (attention.py:356-374; compressor.py:153-154).

    What the graph cannot carry, recorded so a reader does not have to rediscover it:
      * FlashMLA pads each rank's heads to 64, or 128 above 64 (nvidia/flashmla.py:
        77-85): 16 heads per rank at TP 8 run as 64. The main read states no tp and
        the consumer applies it (inference-sim/blis-latency-kernel#39). The Q RoPE +
        insert reads each rank's heads and writes the padded set, and every rank reads
        the whole KV row and writes the whole record. It is stated sharded, which is
        exact at TP 1, exact for the Q half at TP 2, and nearer than replicated above;
      * the indexer's short-context bypass: on a step that is not captured in a CUDA
        graph, while every request in the batch has at most index_topk compressed
        entries, vLLM skips the indexer's wq_b, its query quantization and the scoring
        and selects every entry (attention.py:1085-1107). Captured decode always runs
        the full path, which is what the graph prices;
      * the compressor state caches are sliding windows of coff x ratio tokens, which
        the schema's window kind cannot state on a node whose own window differs, so
        each is stated as the fixed rows a decoding request holds. That count is this
        deriver's reading, not a number vLLM states: vLLM bounds a sliding-window
        spec at cdiv(window - 1 + tokens in flight, block) + 1 pages
        (v1/kv_cache_interface.py:840-866) and reserves for its engine-wide
        max_in_flight_tokens (config/vllm.py:604-613), while a decoding request's own
        in-flight tokens are 1 + num_spec_tokens a step, twice that under async
        scheduling -- 2 or 4 at MTP's one draft token, which give the same pages for
        these windows. Their pages' 576 B padding, and the larger hold during chunked
        prefill, are not carried (inference-sim/blis-schemas#59);
      * h_proj runs on all hc_mult streams of the MTP input, hc_mult rows per token
        (nvidia/mtp.py:185-187); a GEMM node states one (inference-sim/blis-schemas#58);
      * the shared expert's TP padding: without sequence parallelism its intermediate
        rounds up to whole 128-blocks per rank (nvidia/model.py:150-158), which is no
        padding at TP 8 (24 blocks) but 4,096 for 3,072 at TP 16. It depends on the TP
        width, so the consumer applies it;
      * the resident block scales of the FP8 linears: FP32 per 128 x 128 block on the
        DeepGEMM path (quantization/utils/fp8_utils.py:1085-1108), at most 0.025% of
        their bytes, and unverified on the FlashInfer path, so weight_scale is not
        stated;
      * resident tensors no GEMM or Gather prices: layer 0 keeps its (24, 28672)
        hc_attn_fn beside the summed broadcast matrix it multiplies by
        (nvidia/model.py:1229-1236, 1333-1348, 1821-1834), 2.75 MB, and the norms,
        biases, attention sinks, compressor ape tables and mHC scales, about 11 MB --
        some 0.002% of the weights together;
      * that the MTP layer's first pass each step runs over every token the target
        scheduled, not one row per request (v1/worker/gpu/spec_decode/autoregressive/
        speculator.py:311-327, 453-493): an autoregressive stack states only that it
        runs once per drafted token (inference-sim/blis-schemas#60);
      * wo_a's input read: the grouped einsum reads all o_groups x 4,096 FP8 values a
        token (nvidia/ops/o_proj.py:89-95), where one GEMM node of k 4,096 states one
        group's; the FLOPs and weight bytes are exact;
      * work the target runs only with MTP enabled: the copy of the whole hc stream
        into the drafter's buffer (nvidia/model.py:1637-1639)
        (inference-sim/blis-schemas#51);
      * prefill's gather of each request's window records into the BF16 workspace,
        about 1.6 KB per prefill token (nvidia/flashmla.py:340-349), and launches
        below any node's resolution: the decode top-k translation, the hash-route
        index cast and the compressor's RoPE table reads.
    """
    def whole(value, what):
        if type(value) is not int:
            raise DeriveError(f"{model}: {what} is {value!r}, not a whole number")
        return value

    def positive(key, *, concept=False, least=1):
        value = whole(require(cfg, key, model) if concept else require_key(cfg, key, model),
                      key)
        if value < least:
            raise DeriveError(f"{model}: {key} is {value}; this architecture needs at "
                              f"least {least}")
        return value

    hidden = positive("hidden_size", concept=True)
    layers = positive("num_layers", concept=True)
    base = base_dtype(cfg, raw, model)
    if base != "bf16":
        # The unquantized tensors are BF16 in this checkpoint, and the mHC kernels
        # assert BF16 residual streams (kernels/mhc/tilelang.py:855).
        raise DeriveError(f"{model}: V4's unquantized tensors and mHC residual are "
                          f"BF16; this config declares {base}")
    act = DTYPE_WIDTHS[base]
    fp32 = DTYPE_WIDTHS["fp32"]

    ratios = [whole(r, "a compress_ratios entry")
              for r in require_key(cfg, "compress_ratios", model)]
    spec_layers = whole(require(cfg, "num_spec_tokens", model), "num_nextn_predict_layers")
    if spec_layers not in (0, 1):
        # One MTP module, reused for every draft token, spec_step_idx % num_mtp_layers
        # (nvidia/mtp.py:267, 281; config/speculative.py:1487-1495).
        raise DeriveError(f"{model}: num_nextn_predict_layers is {spec_layers}; V4's "
                          f"MTP builds one draft layer")
    if len(ratios) not in (layers, layers + spec_layers):
        raise DeriveError(
            f"{model}: compress_ratios has {len(ratios)} entries for {layers} layers and "
            f"{spec_layers} draft layers")
    # The draft layer's entry is never read: any layer id at or past num_hidden_layers
    # takes ratio 1 (attention.py:232-235).
    ratios = ratios[:layers]
    unknown = sorted(set(ratios) - {4, 128})
    if unknown:
        # A ratio of 0 or 1 would be a window-only layer, which vLLM builds as max(1, r)
        # with no compressor; the published config has none, so one is a different
        # layout rather than a value to price.
        raise DeriveError(
            f"{model}: compress_ratios names {unknown}; V4's compressor accepts 4 and "
            f"128 (compressor.py:158)")
    hash_layers = whole(require_key(cfg, "num_hash_layers", model), "num_hash_layers")
    if not 0 <= hash_layers <= layers:
        raise DeriveError(f"{model}: num_hash_layers {hash_layers} is outside "
                          f"[0, {layers}]")

    q_heads = positive("num_q_heads", concept=True)
    total_dim = positive("head_dim", concept=True)
    rope = positive("qk_rope_head_dim", concept=True)
    q_lora = positive("q_lora_rank")
    o_lora = positive("o_lora_rank")
    o_groups = positive("o_groups")
    window = positive("sliding_window", concept=True)
    topk = positive("index_topk")
    index_heads = positive("index_n_heads")
    index_dim = positive("index_head_dim")
    hc_mult = positive("hc_mult")
    if (q_heads * total_dim) % o_groups:
        raise DeriveError(
            f"{model}: o_groups {o_groups} does not divide {q_heads * total_dim}")
    if total_dim != 512 or rope != 64:
        # The 584 B record is laid out for exactly this (compressor.py:299-301).
        raise DeriveError(
            f"{model}: V4's KV record assumes head_dim 512 with rope 64; this config "
            f"states {total_dim} and {rope}")
    if index_dim != 128:
        # The indexer's compressor quantizes in one 128-block with one FP32 scale
        # (compressor.py:308-310).
        raise DeriveError(f"{model}: index_head_dim {index_dim}; V4's indexer cache "
                          f"is laid out for 128")

    record = DSV4_KV_RECORD_BYTES
    index_record = index_dim + index_dim // 128 * fp32      # attention.py:1010-1014
    bundle = hc_mult * hidden * act
    mixes = hc_mult * fp32 + hc_mult * hc_mult * fp32
    mix_hc = (2 + hc_mult) * hc_mult

    def fp32_gemm(role, n, k):
        node = gemm(role, n, k)
        node["weight_dtype"] = "fp32"
        return node

    def base_gemm(role, n, k):
        node = gemm(role, n, k)
        node["weight_dtype"] = base
        return node

    def with_tp(node, layout):
        node["tp"] = layout
        return node

    def token_sharded(*nodes):
        # Work on the residual stream. Under V4's own sequence parallelism that stream
        # is sharded by token before the first layer (sp_shard, nvidia/model.py:
        # 1571-1578), and only attention runs on the gathered rows, between its
        # all-gather and its reduce-scatter (:1385-1390).
        for node in nodes:
            node["under_sequence_parallel"] = "token_sharded"
        return list(nodes)

    def traffic(role, bytes_per_token, *, tp=None):
        if bytes_per_token != int(bytes_per_token):
            raise DeriveError(f"{model}: {role} moves {bytes_per_token} B/token, which "
                              f"is not a whole number of bytes")
        node = norm(role)
        node["bytes_per_token"] = int(bytes_per_token)
        if tp:
            node["tp"] = tp
        return node

    def per_entry(role, bytes_per_run, ratio, cache=None):
        # A launch that does its work once per compressed entry, on the token that
        # closes a group of `ratio` (compressor.py:413-420; fused_compress_quant_cache.py:
        # 201-202). With `cache` it writes one entry of it, whose record the consumer
        # adds from the cache.
        node = traffic(role, bytes_per_run, tp="replicated")
        if cache:
            node["writes_cache"] = cache
        node["per_tokens"] = ratio
        return node

    def context_work(role, *, per, bytes_per_entry, reads_cache=None, tokens_per_entry=1,
                     phase=None):
        node = with_tp(norm(role), "replicated")
        work: dict[str, Any] = {}
        if reads_cache:
            work["reads_cache"] = reads_cache
        work["bytes_per_entry"] = int(bytes_per_entry)
        if tokens_per_entry > 1 and not reads_cache:
            work["tokens_per_entry"] = tokens_per_entry
        work["per"] = per
        if phase:
            work["phase"] = phase
        node["context"] = work
        return node

    # mHC: FP32 parameters (nvidia/model.py:1227-1271; checkpoint F32 [24, 28672]) and the
    # TileLang kernels, with the sublayer's RMSNorm fused into the pre's epilogue through
    # norm_weight (:1346, 1362, 1381, 1409), so no attn_norm / ffn_norm node exists.
    # Traffic is counted from the tensors each wrapper reads and writes
    # (kernels/mhc/tilelang.py): the residual streams and layer input BF16, the mixes
    # FP32. The projection onto hc_*_fn is its own GEMM node, which reads the bundle.
    mhc_post_bytes = hidden * act + bundle + mixes + bundle     # tilelang.py:791-811
    mhc_pre_bytes = bundle + mixes + hidden * act               # tilelang.py:542-652
    fused_mhc = mhc_mixing_bytes(cfg, raw, model, hidden, hc_mult)
    if fused_mhc != mhc_post_bytes + mhc_pre_bytes:
        raise DeriveError(f"{model}: mHC post + pre disagree with the fused figure")

    def mhc_boundary(side):
        # A post fused with the next pre (tilelang.py:814-982).
        return token_sharded(
            traffic(f"mhc_{side}_post_pre", fused_mhc, tp="replicated"),
            with_tp(fp32_gemm(f"hc_{side}_fn", mix_hc, hc_mult * hidden), "replicated"))

    def broadcast_mhc():
        # Layer 0 on the first PP rank: the (T, hidden) embedding is read, and the
        # bundle, the mixes and the layer input written (tilelang.py:698-789), with the
        # projection over hc_attn_fn summed across its copies, (mix_hc, hidden) FP32
        # (nvidia/model.py:1821-1834).
        return token_sharded(
            traffic("mhc_attn_broadcast_pre", hidden * act + bundle + mixes + hidden * act,
                    tp="replicated"),
            with_tp(fp32_gemm("hc_attn_fn_broadcast", mix_hc, hidden), "replicated"))

    def mhc_head(prefix):
        # hc_head_fused_kernel_tilelang (tilelang.py:1041-1071), the TileLang kernel
        # hc_head_fuse_tilelang (tilelang_kernels.py:1021-1114): pass 1 projects
        # the bundle onto hc_head_fn (mixes kept in registers), pass 2 reads the bundle
        # again for the gated sum and writes one hidden row. The final RMSNorm is a
        # separate launch (nvidia/model.py:1641-1649).
        return [with_tp(fp32_gemm(f"{prefix}hc_head_fn", hc_mult, hc_mult * hidden),
                        "replicated"),
                traffic(f"{prefix}hc_head_mix", bundle + hidden * act, tp="replicated")]

    def projections_in():
        return [
            # Under V4's sequence parallelism the attention input is all-gathered from
            # the token shards (nvidia/model.py:1385-1386).
            collective("AllGather", "attn_in", EMIT_SEQUENCE_PARALLEL),
            # wq_a and wkv merged into one replicated FP8 GEMM (attention.py:248-255;
            # checkpoint F8 [1536, 7168] and [512, 7168]), then one launch norms both
            # latents (attention.py:528-535).
            with_tp(gemm("fused_wqa_wkv", q_lora + total_dim, hidden), "replicated"),
            traffic("qkv_a_layernorm", 2 * (q_lora + total_dim) * act, tp="replicated"),
            with_tp(gemm("wq_b", q_heads * total_dim, q_lora), "column"),
            # One launch: per-head RMSNorm and GPT-J RoPE over Q into the padded head
            # set, and RoPE, UE8M0 FP8 quantization and insertion of the token's KV
            # into the layer's window cache (attention.py:793-810).
            {**traffic("q_rope_kv_insert",
                       2 * q_heads * total_dim * act + total_dim * act, tp="sharded"),
             "writes_cache": "swa"},
        ]

    def projections_out():
        return [
            # Inverse RoPE and FP8 quantization of the attention output in groups of
            # 128, with an FP32 scale per group on SM90 (common/ops/
            # fused_inv_rope_fp8_quant.py:352-470; nvidia/ops/o_proj.py:19-32, 61-76).
            traffic("o_inv_rope_quant",
                    q_heads * total_dim * act + q_heads * total_dim
                    + q_heads * total_dim // 128 * fp32, tp="sharded"),
            # wo_a stays FP8: a grouped DeepGEMM fp8_einsum over o_groups
            # (nvidia/ops/o_proj.py:61-95; attention.py:267-276, is_bmm), unlike
            # V4.1's MXFP8 wo_a, which SM90 dequantizes.
            with_tp(gemm("wo_a", o_groups * o_lora, q_heads * total_dim // o_groups),
                    "column"),
            with_tp(gemm("wo_b", hidden, o_groups * o_lora), "row"),
            # wo_b reduces unless V4's sequence parallelism is on, which turns the
            # reduction into a reduce-scatter (nvidia/model.py:1212-1213, 1389-1390).
            collective("AllReduce", "attn_out",
                       EMIT_TENSOR_PARALLEL_UNLESS_SEQUENCE_PARALLEL),
            collective("ReduceScatter", "attn_out", EMIT_SEQUENCE_PARALLEL),
        ]

    def state_rows(state_window, block):
        # The rows a decoding request holds of a sliding-window state: cdiv(window - 1
        # + tokens in flight, block) + 1 pages (v1/kv_cache_interface.py:840-866), with
        # 1 + num_spec_tokens tokens in flight.
        in_flight = 1 + spec_layers
        return (-(-(state_window - 1 + in_flight) // block) + 1) * block

    def compressor(ratio):
        # DeepseekCompressor (compressor.py:200-499): overlapping at ratio 4, coff 2.
        coff = 2 if ratio == 4 else 1
        width = coff * total_dim
        row = 2 * width * fp32                  # kv and score state, FP32
        state = {"name": "compressor_state", "kind": "state", "owned": True,
                 "bytes_per_entry": row,
                 # sliding window coff x ratio, blocks of 4 at ratio 4 and 8 at 128
                 # (compressor.py:157-173)
                 "entries": state_rows(coff * ratio, 4 if ratio == 4 else 8)}
        # Each closing token gathers coff x ratio state rows, kv and score at head_dim
        # each -- read from the Triton reference (fused_compress_quant_cache.py:
        # 205-247); CUDA runs the CuTeDSL kernels, which pool the same rows -- softmax-
        # pools them, and on CUDA
        # norms, ropes, quantizes and stores one record in one CuTeDSL launch at ratio 4
        # and two at ratio 128, through a head_dim FP32 latent
        # (nvidia/ops/sparse_attn_compress_cutedsl.py:2398-2485, 1425-1450).
        gathered = coff * ratio * 2 * total_dim * fp32
        nodes = [
            # fused wkv + wgate, BF16 and replicated (compressor.py:271-279; checkpoint
            # BF16 [coff x 512, 7168] each), with an FP32 output (attention.py:694-701)
            with_tp(base_gemm("compressor_fused_wkv_wgate", 2 * width, hidden),
                    "replicated"),
            # reads the FP32 kv and score and the ape row, writes one state row
            # (compressor.py:399-410)
            {**traffic("compressor_save_state", row + width * fp32, tp="replicated"),
             "writes_cache": "compressor_state"},
        ]
        if ratio == 4:
            nodes.append(per_entry("compressor_compress_store", gathered, ratio,
                                   cache="compressed"))
        else:
            nodes += [per_entry("compressor_compress", gathered + total_dim * fp32, ratio),
                      per_entry("compressor_store", total_dim * fp32, ratio,
                                cache="compressed")]
        return nodes, state

    def indexer():
        # The C4A indexer (attention.py:974-1130): its own overlapping compressor at
        # index_head_dim, writing 132 B FP8 keys, a replicated FP8 wq_b from the q
        # latent, a BF16 weights projection, and FP8 MQA logits over every key
        # (sparse_attn_indexer.py:726-736).
        width = 2 * index_dim
        row = 2 * width * fp32
        state = {"name": "index_compressor_state", "kind": "state", "owned": True,
                 "bytes_per_entry": row, "entries": state_rows(2 * 4, 4)}
        nodes = [
            with_tp(gemm("index_wq_b", index_heads * index_dim, q_lora), "replicated"),
            with_tp(base_gemm("index_weights_proj", index_heads, hidden), "replicated"),
            with_tp(base_gemm("index_compressor_fused_wkv_wgate", 2 * width, hidden),
                    "replicated"),
            {**traffic("index_compressor_save_state", row + width * fp32,
                       tp="replicated"),
             "writes_cache": "index_compressor_state"},
            per_entry("index_compressor_compress_store",
                      2 * 4 * 2 * index_dim * fp32, 4, cache="index"),
            # BF16 q and weights in; FP8 q and FP32 weights out, the q scale folded into
            # the weights (common/ops/fused_indexer_q.py:547-680)
            traffic("index_q_rope_quant",
                    index_heads * index_dim * act + index_heads * act
                    + index_heads * index_dim + index_heads * fp32, tp="replicated"),
            {"op": "Attention", "role": "block_index_scores", "kind": "sparse_mla",
             "n_q": index_heads, "n_kv": 1, "d_h": index_dim, "compress_ratio": 4,
             "tp": "replicated",
             "caches": [{"name": "index", "kind": "index", "owned": True,
                         "bytes_per_entry": index_record, "tokens_per_entry": 4,
                         "block_size": DSV4_SPARSE_BLOCK_TOKENS,
                         "page_alignment": DSV4_PAGE_ALIGNMENT}, state]},
            # The logits land as an FP32 row per query token over the scanned entries,
            # read back by the top-k (sparse_attn_indexer.py:583-626, 726-785).
            context_work("index_logits_topk", per="query_token", bytes_per_entry=2 * fp32,
                         tokens_per_entry=4),
            # Each prefill chunk gathers the request's whole key stream into a
            # workspace, one 132 B key written per entry read
            # (sparse_attn_indexer.py:515-528).
            context_work("index_prefill_k_gather", per="request",
                         bytes_per_entry=index_record, reads_cache="index",
                         phase="prefill"),
        ]
        return nodes

    def main_attention(ratio, extra_caches):
        node = attention(
            cfg, model,
            kind="sparse_mla",
            latent_width=total_dim,
            window=window,
            compress_ratio=ratio if ratio > 1 else None,
            index_topk=topk if ratio == 4 else None,
        )
        node["caches"] = [{"name": "swa", "kind": "window", "owned": True,
                           "bytes_per_entry": record, "window": window,
                           "block_size": 64, "page_alignment": DSV4_PAGE_ALIGNMENT}]
        if ratio > 1:
            node["caches"].append({"name": "compressed", "kind": "kv", "owned": True,
                                   "bytes_per_entry": record, "tokens_per_entry": ratio,
                                   "block_size": DSV4_SPARSE_BLOCK_TOKENS,
                                   "page_alignment": DSV4_PAGE_ALIGNMENT})
        node["caches"] += extra_caches
        return node

    experts = positive("num_experts", concept=True)
    top_k = positive("top_k", concept=True)
    if top_k > experts:
        raise DeriveError(f"{model}: top_k {top_k} exceeds {experts} experts")
    shared = positive("num_shared_experts", concept=True)
    expert_width = EXPERT_DTYPES.get(str(cfg.get("expert_dtype")))
    if expert_width is None:
        raise DeriveError(f"{model}: expert_dtype {cfg.get('expert_dtype')!r} is not one "
                          f"this deriver maps")
    vocab = positive("vocab_size", concept=True)

    def ffn(hashed):
        nodes = [
            *mhc_boundary("ffn"),
            # The gate GEMM runs on every layer, hash layers included: the hash table
            # replaces only the expert choice, and the weights still come from the
            # logits (fused_moe/runner/moe_runner.py:903-910; fused_topk_bias_router.py:
            # 87-100). GateLinear is unquantized with FP32 logits (nvidia/model.py:
            # 850-856; checkpoint BF16 [384, 7168]).
            *token_sharded(with_tp(base_gemm("router", experts, hidden), "replicated")),
        ]
        if hashed:
            # tid2eid[input_ids]: top_k INT32 expert ids per token on the default path
            # (nvidia/model.py:866-880; the checkpoint stores I64 [129280, 6]).
            nodes += token_sharded({"op": "Gather", "role": "hash_route", "rows": vocab,
                                    "row_bytes": top_k * 4, "rows_per_token": 1,
                                    "out_bytes_per_token": top_k * 4})
        nodes += [
            # Mxfp4MoEMethod (deepseek_v4/quant_config.py:181-191; checkpoint packed FP4
            # with an E8M0 scale per 32)
            *token_sharded({"op": "GroupedGEMM", "role": "experts",
                            "n": positive("moe_ffn_size", concept=True), "k": hidden,
                            "experts": experts, "top_k": top_k,
                            "weight_dtype": expert_width},
                           # The shared expert is a dense FP8 MLP, DeepseekV4MLP: a
                           # merged gate/up column-parallel linear and a row-parallel
                           # down, TP-split normally and whole on every rank over the
                           # token shard under V4's sequence parallelism
                           # (nvidia/model.py:132-175, 900-914). Expert parallelism
                           # does not place it; its output joins the routed experts'
                           # all-reduce (reduce_results False, :911).
                           with_tp(gemm("shared_gate_up",
                                        2 * positive("moe_ffn_size", concept=True) * shared,
                                        hidden), "column"),
                           with_tp(gemm("shared_down", hidden,
                                        positive("moe_ffn_size", concept=True) * shared),
                                   "row")),
            collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
            # FusedMoE all-reduces shared + routed once unless sequence parallelism is
            # on (fused_moe/runner/moe_runner.py:506-512); V4 passes its own flag and
            # never reads the engine-wide use_sequence_parallel_moe.
            collective("AllReduce", "mlp_out",
                       EMIT_TENSOR_PARALLEL_UNLESS_SEQUENCE_PARALLEL),
        ]
        return nodes

    def attention_block(ratio):
        nodes = projections_in()
        extra = []
        if ratio > 1:
            comp, state = compressor(ratio)
            nodes += comp
            extra.append(state)
        if ratio == 4:
            nodes += indexer()
        nodes.append(main_attention(ratio, extra))
        if ratio > 1:
            # Prefill dequantizes the request's whole compressed stream into a BF16
            # workspace, head_dim values per entry, once per chunk (nvidia/flashmla.py:
            # 324-337).
            nodes.append(context_work("prefill_dequant_gather", per="request",
                                      bytes_per_entry=total_dim * act,
                                      reads_cache="compressed", phase="prefill"))
        return nodes + projections_out()

    def layer_nodes(i):
        nodes = broadcast_mhc() if i == 0 else mhc_boundary("attn")
        nodes += attention_block(ratios[i])
        return nodes + ffn(i < hash_layers)

    def kind_id(i):
        name = f"csa{ratios[i]}"
        if i < hash_layers:
            name += "_hash"
        if i == 0:
            name += "_first"
        return name + "_moe"

    kinds, built, sequence = [], {}, []
    for i in range(layers):
        kid = kind_id(i)
        sequence.append(kid)
        nodes = layer_nodes(i)
        if kid in built:
            if nodes != built[kid]:
                raise DeriveError(f"{model}: layer {i} shares kind {kid} with an earlier "
                                  f"layer but builds different work")
            continue
        built[kid] = nodes
        kinds.append({"id": kid, "nodes": nodes, "edges": chain(nodes)})

    # BF16 embedding: the FP8 config gives an embedding no quant method
    # (quantization/fp8.py:177-208), so it is vocab-sharded and all-reduced before the
    # stream is sharded (nvidia/model.py:1455-1462, 1558-1578).
    embed = {"op": "Gather", "role": "embed_tokens", "rows": vocab,
             "row_bytes": hidden * act, "rows_per_token": 1,
             "out_bytes_per_token": hidden * act, "shard": "rows"}
    lm_head = with_tp(base_gemm("lm_head", vocab, hidden), "column")
    head = [
        *token_sharded(traffic("mhc_final_post", mhc_post_bytes, tp="replicated")),
        # V4 gathers the whole hc stream before its learned hc_head, so the mixer and
        # the final norm run on every token on every rank (nvidia/model.py:1622-1649).
        {**collective("AllGather", "head_in", EMIT_SEQUENCE_PARALLEL),
         "bytes_per_token": bundle},
        *mhc_head(""),
        norm("final_norm"),
        # ParallelLMHead with no quant_config (nvidia/model.py:1956-1960; checkpoint
        # head.weight BF16).
        lm_head,
    ]

    extras: dict[str, Any] = {
        "global": {
            "architecture": "DeepseekV4ForCausalLM",
            "context_window": positive("max_position_embeddings"),
            # A stage hands the next the hc_mult-stream bundle alone, (tokens, hc_mult,
            # hidden) at the activation dtype, and no residual: the streams ARE the
            # residual (make_empty_intermediate_tensors, nvidia/model.py:1406-1424@v0.32.0;
            # the send, :1501-1507). Sequence parallelism is off under PP
            # (_use_sequence_parallel, :1047-1058), so every boundary moves this much.
            "pipeline_boundary_bytes_per_token": bundle,
        },
        "embedding": [embed, collective("AllReduce", "embed_out", EMIT_TENSOR_PARALLEL)],
        "head": head,
    }

    if spec_layers:
        # DeepSeekV4MTP (nvidia/mtp.py:305-, built from the layer and predictor at
        # :83-302): the target's embedding and lm_head
        # (shared, v1/worker/gpu/spec_decode/eagle/utils.py:15-18, 38-118), separate FP8
        # e_proj and h_proj, its own hc_head, and one DeepseekV4DecoderLayer at
        # ratio 1 with a learned gate. Its first pass each step runs over every token
        # the target scheduled -- two per request in pure decode at one draft token --
        # and only its logits over one row per request
        # (spec_decode/autoregressive/speculator.py:277-278, 311-327, 453-493).
        draft = [
            # mask, enorm over the embedding and hnorm over the hc streams, before the
            # shard (nvidia/mtp.py:168-184; common/ops/fused_mtp_input_rmsnorm.py)
            traffic("mtp_input_norm", 2 * (hidden * act + bundle), tp="replicated"),
            *token_sharded(
                with_tp(gemm("mtp_e_proj", hidden, hidden), "replicated"),
                with_tp(gemm("mtp_h_proj", hidden, hidden), "replicated"),
                # h_proj(previous) + e_proj(embeds).unsqueeze(-2) (nvidia/mtp.py:185-187)
                traffic("mtp_input_add", 2 * bundle + hidden * act, tp="replicated"),
                # a plain pre on the 3-D stream (nvidia/model.py:1350-1364)
                traffic("mhc_attn_pre", mhc_pre_bytes, tp="replicated"),
                with_tp(fp32_gemm("hc_attn_fn", mix_hc, hc_mult * hidden), "replicated")),
            *attention_block(1),
            *ffn(False),
            *token_sharded(traffic("mtp_mhc_post", mhc_post_bytes, tp="replicated")),
            # gathered back before the head under sequence parallelism (nvidia/mtp.py:
            # 206-209)
            {**collective("AllGather", "mtp_out", EMIT_SEQUENCE_PARALLEL),
             "bytes_per_token": bundle},
        ]
        kinds.append({"id": "mtp", "nodes": draft, "edges": chain(draft)})
        extras["speculator"] = {
            # vLLM v0.31.0 rewrites every MTP-family name to "mtp"
            # (config/speculative.py:1135-1139); one draft token by default, from
            # n_predict = num_nextn_predict_layers (:696-707, 1483-1486).
            "method": "mtp",
            "num_spec_tokens": spec_layers,
            "embedding": [{**embed, "role": "mtp_embed_tokens",
                           "aliases": "embedding.embed_tokens"},
                          collective("AllReduce", "mtp_embed_out", EMIT_TENSOR_PARALLEL)],
            "stack": {"pattern": ["mtp"], "repeat": 1},
            # compute_logits over the sampled rows: hc_head, the shared head's norm, and
            # the target's lm_head (nvidia/mtp.py:276-302).
            "head": [*mhc_head("mtp_"), norm("mtp_final_norm"),
                     {**with_tp(gemm("mtp_lm_head", vocab, hidden), "column"),
                      "aliases": "head.lm_head"}],
        }
    return kinds, compress(sequence), extras


# DeepSeek-V4.1's SM90 KV record, in bytes per cached entry: 448 FP8 NoPE dims, 64 BF16
# RoPE dims, seven UE8M0 scales (one per 64 NoPE dims) and a pad byte
# (deepseek_v41/common/ops/fused_compress_quant_cache.py:303-351). attention.py:503-508
# sizes both the sliding-window and the compressed record at 584 by default on every
# part but the SM100 family, where _use_v41_mxfp8_kv_record() (:119-126) switches to the
# MXFP8 record; an explicit nvfp4_ds_mla cache makes the compressed record 288 B
# (:506-508), and FlashMLA serves that only on SM100 (:513-517).
DSV41_SM90_KV_RECORD_BYTES = 584
# The indexer K cache entry is index_head_dim FP8 values plus one FP32 scale per 128
# of them: `index_head_dim + index_head_dim // 128 * 4` (attention.py:89-97). The
# MXFP4 layout is opt-in and raises on anything but SM100
# (v1/attention/backends/mla/indexer.py:58-73).
# Every packed V4.1 page is padded to the SM90 decode kernel's 576 B TMA stride
# (attention.py:512, 1174-1176); SM100's 512 is the architecture overlay's.
DSV41_SM90_PAGE_ALIGNMENT = 576
# The compressed and indexer caches take the engine's block, which V4.1's backends fix:
# its sparse MLA and indexer backends accept only 64-token pages on SM90 and 128 on
# SM100 (deepseek_v41/sparse_mla.py:89-90; v1/attention/backends/mla/indexer.py:277-278),
# and the engine picks the size every backend accepts (platforms/interface.py:616-697).
DSV41_SM90_SPARSE_BLOCK_TOKENS = 64
# MXFP8 weights' resident scale: the checkpoint's 32 x 32 UE8M0 blocks repeated to one
# byte per 32 elements of a row at load (model_executor/layers/quantization/modelopt.py:
# 2200-2248).
DSV41_MXFP8_RESIDENT_SCALE = {"elements": 32, "bytes": 1}


def dsv41_index_k_record_bytes(index_dim: int) -> int:
    return index_dim + index_dim // 128 * 4


def handler_deepseek_v41(cfg, raw, model):
    """DeepSeek-V4.1: V4's compressed sparse MLA, with the compressed cache and the
    indexer SHARED across layers, two engram n-gram layers, mHC on every sublayer, and
    a three-layer DSpark draft module.

    Priced as vLLM v0.31.0 serves it on SM90 (H200, the part the serving campaign in
    blis-catalog#29 ran on), with no attention backend forced, as the base of
    capability family 9; `arch_overrides` states what family 10 (SM100) changes:

      * wo_a is MXFP8 in the checkpoint and BF16 resident on SM90. It is a batched
        matmul, whose only kernels are DeepGemmMxfp8BmmLinearKernel and the emulation
        kernel (model_executor/kernels/linear/__init__.py:895-900). The first is
        Blackwell-only (mxfp8/deep_gemm.py:34-35); the second dequantizes at load
        (mxfp8/emulation.py:31-50, with VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD defaulting
        True). On SM100 it stays FP8.
      * the KV record is 584 B on SM90 (DSV41_SM90_KV_RECORD_BYTES) in 576 B-aligned
        pages. SM100 writes a 528 B MXFP8 record in 512 B pages, and its default mega
        attention resolves an unstated or "fp8" cache dtype to nvfp4_ds_mla
        (attention.py:140-149; nvidia/flash_mla_mega_attn.py:193-197), whose
        compressed record is 288 B NVFP4 while the window keeps 528 B
        (attention.py:500-508). The overlay states both records; which cache dtype
        runs is the consumer's to resolve. The 132 B indexer record is the default on
        both.
      * FlashMLA is the SM90 attention class (nvidia/model.py:190-236); SM100 defaults
        to mega attention, which fuses Q RoPE, the attention and the inverse RoPE into
        one launch. The overlay changes widths and records, not launches, so SM100's
        fused launches are priced as SM90's separate ones.

    Every layer index below comes from the config's own vectors, which vLLM reads the
    same way (attention.py:294-338):

      * compress_ratios[i] is 0 (sliding window only), 1 (a full-length compressed
        stream, no pooling) or 2 (pairs pooled by a learned gate). vLLM rejects any
        other value (attention.py:300-311, compressor.py:192-197).
      * kv_source_layer_ids name the only layers that build a compressor and OWN a
        compressed cache. Every other compressed layer reads the cache of the nearest
        source at or below it, `max(s for s in kv_source_layers if s <= layer_id)`
        (attention.py:333-335), while still projecting its own 512-wide KV into its own
        sliding-window cache (attention.py:547-558). The checkpoint agrees: compressor
        tensors exist on layers 2, 8, 14 and 20 only.
      * index_source_layer_ids name the layers that build an indexer (attention.py:
        421-472). Those that are also KV sources own an indexer K cache and its wk /
        k_norm; the rest score their own queries against the nearest KV source's keys
        (attention.py:437-449), which the checkpoint confirms by carrying wk and k_norm
        on 2, 8, 14 and 20 alone. Every other compressed layer reads the top-k the last
        indexer wrote into the shared buffer (nvidia/flashmla.py:185-189).

    Each layer states the caches its main read and its scorer hold: its own window,
    and on compressed layers the stream (and the keys) it owns or borrows, owners with
    the page alignment and readers with the owner's record. A ratio-2 owner also holds
    the compressor's FP32 state ring. The main read states compress_ratio to name the
    stream its top-k positions come from, which its kv cache must match; it reads its
    window plus at most index_topk of them whatever the ratio (nvidia/flashmla.py:
    230-246; inference-sim/blis-latency-kernel#32). The scorer scans ctx / ratio.

    The collectives follow V4.1's own sequence parallelism, `_use_sequence_parallel`
    (nvidia/model.py:239-247: PP 1, EP on, TP > 1, and a mega-MoE backend or DP > 1),
    never the engine-wide use_sequence_parallel_moe (config/parallel.py:719-736).
    Under it `wo_b` stops reducing and attention is all-gathered and reduce-scattered
    (nvidia/model.py:318-319, 579-586), FusedMoE skips its final all-reduce (fused_moe/
    runner/moe_runner.py:506-512), and the engram rows are gathered along tokens. The
    draft layers take the same flag (dspark.py:85). Work on the residual stream -- mHC,
    the engram injection, the MoE and the head before its gather -- states
    `under_sequence_parallel: token_sharded`, since under the mode it runs on the rank's
    ceil(tokens / tp) rows; attention, its compressor and its indexer run on all of them.

    What the graph cannot carry, recorded so a reader does not have to rediscover it:
      * FlashMLA pads each rank's query heads to 64 (nvidia/flashmla.py:66-74). When TP
        splits the heads (attention.py:276-284), a rank at TP 8 computes 64 heads where
        n_q / tp gives 8. That depends on the backend and the TP width rather than on
        the model, so the main read states no tp and the consumer applies it
        (inference-sim/blis-latency-kernel#39);
      * SM100's fused all-reduce + mHC: at TP 4 without expert parallelism, where
        NVLink multicast is available, wo_b's and the MoE's reductions fold into the
        mHC boundary kernel (nvidia/model.py:318-328, 645-653; nvidia/ops/mhc.py:
        44-72), which an overlay cannot restate, so the graph keeps them as
        all-reduces;
      * the shared expert's TP padding: its intermediate rounds up to whole 32-wide
        blocks per rank (deepseek_v4/nvidia/model.py:150-158, under weight_block_size
        [32, 32]), none at TP 8 but 2,560 for 2,304 at TP 16, so the consumer applies
        it;
      * resident tensors no GEMM or Gather prices: the norms, attention sinks, gate
        biases and the engram gate's q and k weights, a few MB;
      * the SM100 page size of the compressed and indexer caches, 128 tokens where
        SM90's is 64: an architecture overlay restates a cache's record and alignment
        but not its block_size (inference-sim/blis-schemas#62);
      * the SM100 mega-attention launches (above), and the draft context's window
        insert, which writes a draft layer's cache from another kind and so states the
        SM90 record in its traffic rather than following the overlay;
      * the 68 B MXFP4 indexer record, opt-in through indexer_kv_dtype, which no
        deployment key carries (inference-sim/blis-schemas#57);
      * work the target runs only when DSpark is enabled: the copy of the whole hc
        stream into the drafter's hidden buffer (nvidia/model.py:773-784, 975-980),
        the auxiliary writes at the dspark_target_layer_ids, and, under sequence
        parallelism, the head's gather of the whole hc stream and its pre-mix ahead of
        the collapse and norm, in place of the collapsed state after them (:976-990),
        and the all-gathers of the auxiliary states it reads (:944-946, 958-959). No
        emit condition names speculation (inference-sim/blis-schemas#51);
      * prefill's gather of each layer's own window records into the BF16 workspace,
        query_len + min(prefix, window - 1) entries per request (nvidia/flashmla.py:
        315-325; sparse_swa.py:366-371), about 1.6 KB per prefill token a layer: it
        grows with the chunk, not the context, and no context field states that (the
        compressed part of the gather is prefill_dequant_gather). A per-request
        `context` read of `swa`, for the window part of the gather, is charged over
        the whole context (computed + scheduled), which on the last 8K chunk of a
        100K-token prompt states 12x the window-bounded gather
        (inference-sim/blis-catalog#33);
      * Engram's DP-sharded path: with DP > 1 and /dev/shm too small for the ~189 GiB
        of tables, they shard over TP x DP and the hashes and rows are exchanged over
        the engram DP group (nvidia/engram.py:104-159, 408-424, 490-496;
        distributed/parallel_state.py:2104-2128). The graph states the shared-memory
        case, one copy per node;
      * on SM100, qkv_a_layernorm writes an FP8 query with its scales when the query
        quantization fuses (attention.py:744-756), which the overlay cannot restate;
      * layer 0's (24, 20480) hc_attn_fn, which stays resident beside the summed
        broadcast matrix it multiplies by (nvidia/model.py:402-409, 1178-1191), 1.9 MB;
      * small per-token launches below any node's resolution: the decode top-k index
        translation (nvidia/flashmla.py:181-193), the top-k buffer clear
        (sparse_attn_indexer.py:479-480) and the engram n-gram hash (nvidia/model.py:
        842-879).
    """
    # Every count below is read as a whole number and checked against what it counts.
    # The config is the only source for these shapes, so a float, a bool, a zero or an
    # id outside the stack is a misread config rather than a value to round or drop.
    def whole(value, what):
        if type(value) is not int:
            raise DeriveError(f"{model}: {what} is {value!r}, not a whole number")
        return value

    def positive(key, *, concept=False, least=1):
        value = whole(require(cfg, key, model) if concept else require_key(cfg, key, model),
                      key)
        if value < least:
            raise DeriveError(f"{model}: {key} is {value}; this architecture needs at "
                              f"least {least}")
        return value

    def layer_ids(key, lo, hi, *, nonempty=False):
        ids = [whole(i, f"an entry of {key}") for i in require_key(cfg, key, model)]
        if len(set(ids)) != len(ids):
            raise DeriveError(f"{model}: {key} repeats a layer: {ids}")
        if any(not lo <= i < hi for i in ids):
            raise DeriveError(f"{model}: {key} names a layer outside [{lo}, {hi}): {ids}")
        if nonempty and not ids:
            raise DeriveError(f"{model}: {key} is empty")
        return sorted(ids)

    hidden = positive("hidden_size", concept=True)
    layers = positive("num_layers", concept=True)
    base = base_dtype(cfg, raw, model)
    if base != "bf16":
        # The tensors vLLM keeps unquantized are BF16 in this checkpoint, and the mHC
        # kernels assert BF16 residual streams (kernels/mhc/tilelang.py:342-358), so a
        # different declared dtype is a different model rather than a different width.
        raise DeriveError(f"{model}: V4.1's unquantized tensors and mHC residual are "
                          f"BF16; this config declares {base}")
    act = DTYPE_WIDTHS[base]
    fp32 = DTYPE_WIDTHS["fp32"]

    ratios = [whole(r, "a compress_ratios entry")
              for r in require_key(cfg, "compress_ratios", model)]
    # Zero draft layers is a checkpoint without DSpark: no speculator, and no dspark_*
    # field is read.
    spec_layers = whole(require(cfg, "num_spec_tokens", model), "num_nextn_predict_layers")
    if spec_layers < 0:
        raise DeriveError(f"{model}: num_nextn_predict_layers is {spec_layers}")
    if len(ratios) not in (layers, layers + spec_layers):
        raise DeriveError(
            f"{model}: compress_ratios has {len(ratios)} entries for {layers} layers and "
            f"{spec_layers} draft layers")
    # vLLM indexes the vector by layer id and gives any id past its end ratio 0
    # (attention.py:300-305), so the draft layers 40-42 read their own entries here.
    draft_ratios = [ratios[layers + i] if layers + i < len(ratios) else 0
                    for i in range(spec_layers)]
    ratios = ratios[:layers]
    unknown = sorted(set(ratios + draft_ratios) - {0, 1, 2})
    if unknown:
        raise DeriveError(
            f"{model}: compress_ratios names {unknown}; vLLM's V4.1 compressor accepts "
            f"0, 1 and 2 only (deepseek_v41/compressor.py:192-197)")
    if any(draft_ratios):
        raise DeriveError(
            f"{model}: a draft layer declares a compressed read, but a DSpark layer is "
            f"never a KV or index source (attention.py:322-324), so it has no stream to "
            f"read")

    kv_sources = layer_ids("kv_source_layer_ids", 0, layers)
    index_sources = layer_ids("index_source_layer_ids", 0, layers)
    # Layer 0 takes the broadcast branch, and the engram injection is the elif after
    # it (nvidia/model.py:485-522), so vLLM would silently never run an engram there;
    # and no draft layer is given an engram layout (dspark.py:122-132).
    engram_layers = set(layer_ids("engram_layer_ids", 1, layers))
    # vLLM finds a layer's table by its position in the config's own list,
    # engram_layout.layer_ids.index(layer_id) over tuple(config.engram_layer_ids)
    # (nvidia/model.py:296-305; common/engram.py:176-178), which need not be sorted.
    engram_order = [whole(i, "an entry of engram_layer_ids")
                    for i in require_key(cfg, "engram_layer_ids", model)]

    def kv_source_of(i):
        return max((s for s in kv_sources if s <= i), default=None)

    for s in kv_sources + index_sources:
        if ratios[s] == 0:
            raise DeriveError(
                f"{model}: source layer {s} has no compressed stream to own or index")
    for i, r in enumerate(ratios):
        if r == 0:
            continue
        src = kv_source_of(i)
        # vLLM does not check this; a reader layer whose source pools differently would
        # read a stream at a ratio it does not declare, which no node can price.
        if src is None or ratios[src] != r:
            raise DeriveError(
                f"{model}: layer {i} (ratio {r}) reads the compressed cache of layer "
                f"{src}, which declares ratio {None if src is None else ratios[src]}")
    for i, r in enumerate(ratios):
        # vLLM takes max() over the index sources at or below every compressed layer,
        # and max() over an empty set raises (attention.py:333-338).
        if r and not any(s <= i for s in index_sources):
            raise DeriveError(
                f"{model}: compressed layer {i} has no index source at or below it, "
                f"which vLLM rejects")
    for s in index_sources:
        # An index source that is not a KV source scores against its KV source's
        # indexer K cache, which exists only if that layer indexes too
        # (attention.py:437-449, NotImplementedError otherwise).
        if s not in kv_sources and kv_source_of(s) not in index_sources:
            raise DeriveError(
                f"{model}: index source {s} needs the keys of layer {kv_source_of(s)}, "
                f"which builds no indexer")

    q_heads = positive("num_q_heads", concept=True)
    total_dim = positive("head_dim", concept=True)
    rope = positive("qk_rope_head_dim", concept=True)
    q_lora = positive("q_lora_rank")
    o_lora = positive("o_lora_rank")
    o_groups = positive("o_groups")
    window = positive("sliding_window", concept=True)
    topk = positive("index_topk")
    index_heads = positive("index_n_heads")
    index_dim = positive("index_head_dim")
    hc_mult = positive("hc_mult")
    if (q_heads * total_dim) % o_groups:
        raise DeriveError(
            f"{model}: o_groups {o_groups} does not divide {q_heads * total_dim}")
    if total_dim != 512 or rope != 64:
        # compressor.py:210 asserts exactly this, and the 584 B record is laid out for
        # it, so another width is a different kernel rather than a different value.
        raise DeriveError(
            f"{model}: V4.1's compressor and KV record assume head_dim 512 with rope "
            f"64; this config states {total_dim} and {rope}")
    if index_dim % 128:
        # The FP8 indexer cache carries one FP32 scale per 128 values (attention.py:
        # 89-97); another width would leave a partial group no record describes.
        raise DeriveError(
            f"{model}: index_head_dim {index_dim} is not a multiple of 128")

    # The candidate source publishes the blocks later indexers mask to
    # (nvidia/model.py:662-675, attention.py:450-471); it must be an index source.
    # vLLM reads both fields with getattr defaults of -1 and 0 and turns candidate
    # selection off unless the source is a layer and the block count is positive
    # (nvidia/model.py:666-668).
    candidate_source = cfg.get("candidate_source_layer_id")
    if candidate_source is not None:
        candidate_source = whole(candidate_source, "candidate_source_layer_id")
        blocks = whole(cfg.get("candidate_topk_blocks", 0), "candidate_topk_blocks")
        if candidate_source < 0 or blocks <= 0:
            candidate_source = None
        elif candidate_source not in index_sources:
            raise DeriveError(f"{model}: candidate_source_layer_id {candidate_source} "
                              f"is not an index source")
    # One Engram table per engram layer, in the order engram_layer_ids lists them
    # (common/engram.py:167-202).
    engram_tables = []
    if engram_layers:
        n_cols = (positive("engram_max_ngram_size", least=2) - 1) * positive(
            "engram_n_heads")
        row_dim = positive("engram_head_dim")
        engram_tables = [whole(n, "an entry of engram_num_embeddings")
                         for n in require_key(cfg, "engram_num_embeddings", model)]
        if len(engram_tables) != len(engram_layers) or min(engram_tables) < 1:
            raise DeriveError(f"{model}: engram_num_embeddings {engram_tables} does not "
                              f"give one table per engram layer")
    # Read before any layer is built: the compressor ring is sized by it.
    block = positive("dspark_block_size") if spec_layers else 0

    record = DSV41_SM90_KV_RECORD_BYTES
    index_record = dsv41_index_k_record_bytes(index_dim)
    bundle = hc_mult * hidden * act
    mixes = hc_mult * fp32 + hc_mult * hc_mult * fp32
    mix_hc = (2 + hc_mult) * hc_mult

    def fp32_gemm(role, n, k):
        node = gemm(role, n, k)
        node["weight_dtype"] = "fp32"
        return node

    def base_gemm(role, n, k):
        node = gemm(role, n, k)
        node["weight_dtype"] = base
        return node

    def with_tp(node, layout):
        node["tp"] = layout
        return node

    def token_sharded(*nodes):
        # Work on the residual stream. Under V4.1's own sequence parallelism that stream
        # is sharded by token (sp_shard, nvidia/model.py:898-905) and only attention runs
        # on the gathered rows, between its all-gather and its reduce-scatter (:579-586).
        for node in nodes:
            node["under_sequence_parallel"] = "token_sharded"
        return list(nodes)

    def traffic(role, bytes_per_token, *, tp=None):
        # Amortized figures divide by the pool ratio; a remainder would be a fraction of
        # a byte the schema's integer field cannot hold, so it raises rather than rounds.
        if bytes_per_token != int(bytes_per_token):
            raise DeriveError(f"{model}: {role} moves {bytes_per_token} B/token, which "
                              f"is not a whole number of bytes")
        node = norm(role)
        node["bytes_per_token"] = int(bytes_per_token)
        if tp:
            node["tp"] = tp
        return node

    def store(role, bytes_per_run, cache, ratio=1):
        # A launch that writes one entry of its layer's `cache` each time it runs: the
        # record's own bytes are added by the consumer from the cache (and follow the
        # architecture overlay and the cache dtype), so bytes_per_token is the rest.
        node = traffic(role, bytes_per_run, tp="replicated")
        node["writes_cache"] = cache
        if ratio > 1:
            node["per_tokens"] = ratio
        return node

    def context_work(role, *, per, bytes_per_entry, reads_cache=None, tokens_per_entry=1,
                     phase=None, decode_extent=None):
        node = with_tp(norm(role), "replicated")
        work: dict[str, Any] = {}
        if reads_cache:
            work["reads_cache"] = reads_cache
        work["bytes_per_entry"] = int(bytes_per_entry)
        if tokens_per_entry > 1 and not reads_cache:
            work["tokens_per_entry"] = tokens_per_entry
        work["per"] = per
        if phase:
            work["phase"] = phase
        if decode_extent:
            work["decode_extent"] = decode_extent
        node["context"] = work
        return node

    # Caches, from what each layer allocates (attention.py:503-512, 547-558, 1100-1134,
    # 1166-1195; compressor.py:146-163). A reader states its owner's record exactly and
    # finds the owner by name, as the nearest earlier layer that owns one -- vLLM's own
    # `max(s for s in kv_source_layers if s <= layer_id)` (attention.py:333-338).
    # Pages: the window ring takes 32-token blocks (attention.py:554); every packed
    # record pads its page to the decode kernel's 576 B TMA stride on SM90 (:512, and
    # :1174-1176 for the index cache, which shares the block).
    def window_cache():
        return {"name": "swa", "kind": "window", "owned": True,
                "bytes_per_entry": record, "window": window, "block_size": 32,
                "page_alignment": DSV41_SM90_PAGE_ALIGNMENT}

    def compressed_cache(ratio, owned):
        cache = {"name": "compressed", "kind": "kv", "owned": owned,
                 "bytes_per_entry": record}
        if ratio > 1:
            cache["tokens_per_entry"] = ratio
        if owned:
            cache["block_size"] = DSV41_SM90_SPARSE_BLOCK_TOKENS
            cache["page_alignment"] = DSV41_SM90_PAGE_ALIGNMENT
        return cache

    def index_cache(ratio, owned):
        cache = {"name": "index", "kind": "index", "owned": owned,
                 "bytes_per_entry": index_record}
        if ratio > 1:
            cache["tokens_per_entry"] = ratio
        if owned:
            cache["block_size"] = DSV41_SM90_SPARSE_BLOCK_TOKENS
            cache["page_alignment"] = DSV41_SM90_PAGE_ALIGNMENT
        return cache

    def ring_cache():
        # A ratio-2 compressor's FP32 state ring: kv and score rows of head_dim each, in
        # blocks of max(8, the next power of two at or above num_speculative_tokens + 2)
        # rows (compressor.py:146-163, 230-238). At DSpark's 5 tokens and without
        # speculation alike that is 8.
        rows = max(8, 1 << ((block + 2 if spec_layers else 2) - 1).bit_length())
        return {"name": "compressor_ring", "kind": "state", "owned": True,
                "bytes_per_entry": 2 * total_dim * fp32, "entries": rows}

    # mHC. The parameters are FP32 (nvidia/model.py:400-444; the checkpoint agrees, F32
    # [24, 20480]) and on SM90 every boundary is TileLang's delayed fused post + pre
    # (nvidia/ops/mhc.py:305-322), because the overlap stream and Mega-mHC are
    # SM100-only (nvidia/ops/mhc.py:44-56, nvidia/ops/mega_mhc.py:17-28). The
    # sublayer's RMSNorm is fused into the epilogue -- norm_weight is passed in
    # (nvidia/model.py:503-604) -- so no attn_norm / ffn_norm node exists. Traffic is
    # mhc_mixing_bytes(): the post and the pre each as one pass, read against the
    # TileLang contract (kernels/mhc/tilelang.py:296-358: residual bf16 (T, hc, H),
    # post and comb mixes fp32; the 4 FP32 pre-mix values, 32 B a token, are not
    # counted). The pre's re-read of the updated residual happens on both paths the
    # kernel takes (tilelang.py:445-457). Within its fused token range (32 or fewer per
    # rank, tilelang_kernels.py:64-95) the projection is computed inside the post
    # kernel, so the hc_*_fn GEMM node's own bundle read does not happen there; above
    # it a separate GEMM reads the bundle as the node states. Neither has been
    # measured.
    #
    # Every rank holds the whole residual stream under plain TP, so mHC is replicated.
    # Every mHC launch and projection runs on the residual stream: replicated on each
    # rank, and on the rank's token shard under V4.1's sequence parallelism.
    fused_mhc = mhc_mixing_bytes(cfg, raw, model, hidden, hc_mult)
    mhc_post_bytes = hidden * act + bundle + mixes + bundle
    mhc_pre_bytes = bundle + mixes + hidden * act

    def mhc_boundary(side):
        return token_sharded(
            traffic(f"mhc_{side}_post_pre", fused_mhc, tp="replicated"),
            with_tp(fp32_gemm(f"hc_{side}_fn", mix_hc, hc_mult * hidden), "replicated"))

    def final_mhc():
        # The final mHC post and the hc collapse run before the final norm
        # (nvidia/model.py:947-987); V4.1 has no learned hc_head (:982-986).
        return token_sharded(
            traffic("mhc_final_post", mhc_post_bytes, tp="replicated"),
            traffic("hc_collapse", bundle + hc_mult * fp32 + hidden * act,
                    tp="replicated"))

    def broadcast_mhc(fn_width, fn_role):
        # The embedding is broadcast to hc_mult copies by a materializing expand (read
        # hidden, write the bundle), then a standalone pre. Target layer 0 projects with
        # hc_attn_fn summed over its copies, a (mix_hc, hidden) matrix
        # (nvidia/model.py:485-505, 1178-1191); the drafter repeats the copies itself
        # (dspark.py:226) and its first layer takes the full hc_attn_fn on the 3-D
        # stream (nvidia/model.py:506-521).
        return token_sharded(
            traffic("mhc_attn_broadcast_pre", hidden * act + bundle + mhc_pre_bytes,
                    tp="replicated"),
            with_tp(fp32_gemm(fn_role, mix_hc, fn_width), "replicated"))

    engram_rows = dict(zip(engram_order, engram_tables))

    def engram_block(i):
        # Engram replaces the fused boundary before its attention with a separate post,
        # the injection, and a separate pre (nvidia/model.py:522-552).
        #
        # The lookup gathers (max_ngram_size - 1) * engram_n_heads rows of engram_head_dim
        # FP8 values plus one UE8M0 scale per 32 from each layer's own table and writes
        # them to HBM as BF16 (common/engram.py:588-642, 959-966). The tables are pinned
        # host memory by default (config/engram.py:38-56, `cpu_offload: bool = True`),
        # split across TP by hash head (common/engram.py:663-676). wkv is an FP8
        # column-parallel GEMM from the flattened rows to (hc_mult + 1) hidden-wide
        # outputs (:928-937; checkpoint F8_E4M3 [25600, 6144]). The fused gate then reads
        # the bundle, the hc_mult keys and the value, and writes the bundle (:755-843).
        #
        # Without sequence parallelism each rank looks up its heads for every token, the
        # rows are all-gathered along heads, and wkv's output is all-gathered
        # (gather_output=True). With it the lookup still covers every token, before the
        # stream is sharded (nvidia/model.py:890-895), the rows are all-gathered along
        # tokens and cut to the rank's shard, and wkv is replicated over that shard with
        # no gather (common/engram.py:928-937, 977-1016), as is the gate.
        out = n_cols * row_dim * act
        return [
            *token_sharded(traffic("mhc_attn_post", mhc_post_bytes, tp="replicated")),
            {"op": "Gather", "role": "engram_lookup", "rows": engram_rows[i],
             "row_bytes": row_dim + row_dim // 32, "rows_per_token": n_cols,
             "out_bytes_per_token": out, "residency": "host",
             "residency_control": "engram_cpu_offload", "shard": "heads"},
            {**collective("AllGather", "engram_rows",
                          EMIT_TENSOR_PARALLEL_UNLESS_SEQUENCE_PARALLEL),
             "bytes_per_token": out},
            {**collective("AllGather", "engram_rows_by_token", EMIT_SEQUENCE_PARALLEL),
             "bytes_per_token": out},
            *token_sharded(with_tp(gemm("engram_wkv", (hc_mult + 1) * hidden,
                                        n_cols * row_dim), "column")),
            {**collective("AllGather", "engram_wkv_out",
                          EMIT_TENSOR_PARALLEL_UNLESS_SEQUENCE_PARALLEL),
             "bytes_per_token": (hc_mult + 1) * hidden * act},
            *token_sharded(
                traffic("engram_gate", bundle + bundle + hidden * act + bundle,
                        tp="replicated"),
                traffic("mhc_attn_pre", mhc_pre_bytes, tp="replicated"),
                with_tp(fp32_gemm("hc_attn_fn", mix_hc, hc_mult * hidden), "replicated")),
        ]

    def projections_in():
        # attention.py:354-370. fused_wqa_wkv merges wq_a and wkv into one replicated
        # GEMM (`disable_tp=True`; the checkpoint ships them separately: F8 [1280, 5120]
        # and [512, 5120]); one fused_q_kv_rmsnorm launch norms both halves (:757-763),
        # at q_lora_rank and head_dim wide, on the replicated latents.
        return [
            # Under V4.1's own sequence parallelism the attention input is all-gathered
            # from the token shards (nvidia/model.py:579-580).
            collective("AllGather", "attn_in", EMIT_SEQUENCE_PARALLEL),
            with_tp(gemm("fused_wqa_wkv", q_lora + total_dim, hidden), "replicated"),
            traffic("qkv_a_layernorm", 2 * (q_lora + total_dim) * act, tp="replicated"),
            with_tp(gemm("wq_b", q_heads * total_dim, q_lora), "column"),
            # One launch: GPT-J RoPE over Q and its pad to padded_heads, then RoPE, UE8M0
            # FP8 quantization and insertion of this token's KV into the layer's own
            # window cache (attention.py:1034-1048). Q is read and written at q_heads x
            # head_dim and the KV row read at head_dim; the record it writes is the
            # cache's. FlashMLA pads each rank's Q to 64 heads (nvidia/flashmla.py:
            # 66-74), so the Q write does not shrink with TP; replicated is the nearer
            # layout (inference-sim/blis-latency-kernel#39).
            store("q_rope_kv_insert", 2 * q_heads * total_dim * act + total_dim * act,
                  "swa"),
        ]

    def projections_out():
        return [
            # The inverse RoPE over the attention output, unquantized on SM90. FlashMLA's
            # _o_proj is dsv41_o_proj, whose fused WO-A path is SM100-only, so it falls
            # through to deep_gemm_fp8_o_proj (deepseek_v41/nvidia/ops/o_proj.py:30-45,
            # 63-95), which runs fused_inv_rope_fp8_quant with quantize=False once wo_a is
            # BF16 (deepseek_v4/nvidia/ops/o_proj.py:61-77), over each rank's own heads.
            traffic("o_inv_rope", 2 * q_heads * total_dim * act, tp="sharded"),
            # Grouped: o_groups batched matmuls of (o_lora, heads*head_dim/o_groups),
            # split by group across TP. BF16 resident on SM90 (see the docstring), with
            # its E8M0 scales kept "for the dtype/ndim asserts" (mxfp8/emulation.py:
            # 36-50); FP8 on SM100, by the architecture overlay.
            {**with_tp(base_gemm("wo_a", o_groups * o_lora,
                                 q_heads * total_dim // o_groups), "column"),
             "weight_scale": dict(DSV41_MXFP8_RESIDENT_SCALE)},
            with_tp(gemm("wo_b", hidden, o_groups * o_lora), "row"),
            # wo_b reduces unless V4.1's own sequence parallelism is on, which turns its
            # reduction into a reduce-scatter (nvidia/model.py:318-319, 585-586).
            collective("AllReduce", "attn_out",
                       EMIT_TENSOR_PARALLEL_UNLESS_SEQUENCE_PARALLEL),
            collective("ReduceScatter", "attn_out", EMIT_SEQUENCE_PARALLEL),
        ]

    def compressor(ratio):
        # Owned by KV sources only (attention.py:565-575). A BF16 replicated GEMM with an
        # FP32 output (compressor.py:216-227, attention.py:916-921), wkv and wgate merged
        # where the ratio pools (no gate at ratio 1: the checkpoint has no wgate on layer
        # 20). fused_save_compress_norm (fused_compress_quant_cache.py:118-221) pools
        # each closed group into one BF16 latent, which rope_quant_insert then reads and
        # writes as one record of the compressed cache (:304-351), once per entry.
        #
        # The pool's traffic is the DECODE figure. At ratio 2 the request program also
        # rewrites the chunk's last rows into the FP32 state ring every step (:180-193,
        # ring capacity >= 8 rows, compressor.py:146-150), and a decode chunk is one
        # token, so every token reads its raw row and writes it to the ring, and a
        # group-closing token reads the previous row from the ring and the current one
        # raw before writing the latent. In prefill only a chunk's tail reaches the
        # ring, and the figure falls to one raw row plus the amortized latent. Ratio 1
        # keeps no ring (compressor.py:230-238).
        width = 2 * total_dim if ratio > 1 else total_dim
        raw_row = width * fp32
        if ratio > 1:
            pool = 2 * raw_row + (2 * raw_row + total_dim * act) / ratio
        else:
            pool = raw_row + total_dim * act
        return [
            with_tp(base_gemm("compressor_fused_wkv_wgate", width, hidden), "replicated"),
            traffic("compressor_pool_norm", pool, tp="replicated"),
            store("compressor_rope_quant_insert", total_dim * act, "compressed", ratio),
        ]

    def indexer(i, ratio, owns_k):
        nodes = [
            # Replicated FP8 query projection from the q-LoRA latent, and the BF16
            # per-head weight projection from hidden (attention.py:1258-1271; checkpoint
            # F8 [4096, 1280] and BF16 [32, 5120]).
            with_tp(gemm("index_wq_b", index_heads * index_dim, q_lora), "replicated"),
            with_tp(base_gemm("index_weights_proj", index_heads, hidden), "replicated"),
            # RoPE and FP8 quantization of the index query. The FP8 path stores no q
            # scale: it is folded into the FP32 head weights it writes
            # (attention.py:1398-1407; deepseek_v4/common/ops/fused_indexer_q.py:
            # 669-727). Reads the BF16 query and weights, writes FP8 q and FP32 weights.
            traffic("index_q_rope_quant",
                    index_heads * index_dim * act + index_heads * act
                    + index_heads * index_dim + index_heads * fp32, tp="replicated"),
        ]
        if owns_k:
            # The key comes from the compressor's latent, not from hidden: a BF16
            # (index_head_dim, head_dim) GEMM, then one launch for k_norm, RoPE, FP8
            # quantization and the paged store (attention.py:1288-1298;
            # common/ops/indexer_k_store.py). The GEMM runs over every token's latent
            # row; only the store keeps the group-boundary rows (attention.py:
            # 1365-1372), so only the store runs once per entry.
            nodes += [
                with_tp(base_gemm("index_wk", index_dim, total_dim), "replicated"),
                store("index_k_norm_rope_store", index_dim * act, "index", ratio),
            ]
        nodes.append({
            # The scoring pass scans every compressed position of its KV source to
            # discover the top-k, so it carries the compression bound and no top-k
            # (sparse_attn_indexer.py:585-626). Its heads are replicated.
            "op": "Attention", "role": "block_index_scores", "kind": "sparse_mla",
            "n_q": index_heads, "n_kv": 1, "d_h": index_dim, "compress_ratio": ratio,
            "tp": "replicated", "caches": [index_cache(ratio, owns_k)],
        })
        # The scores land as an FP32 row per query token, written over the scanned
        # positions only (clean_logits=False) and read back by the top-k
        # (sparse_attn_indexer.py:583-626, 726-785).
        nodes.append(context_work("index_logits_topk", per="query_token",
                                  bytes_per_entry=2 * fp32, tokens_per_entry=ratio))
        # Each prefill chunk gathers the request's whole key stream out of the cache
        # into a workspace, one 132 B key written per entry it reads
        # (sparse_attn_indexer.py:515-528).
        nodes.append(context_work("index_prefill_k_gather", per="request",
                                  bytes_per_entry=index_record, reads_cache="index",
                                  phase="prefill"))
        if i == candidate_source:
            # The candidate source scores blocks of candidate_block_size positions from
            # its logits row and keeps the top candidate_topk_blocks
            # (kernels/attention/dsa/candidate_blocks.py:20-58, 392-445).
            nodes.append(context_work("candidate_select", per="query_token",
                                      bytes_per_entry=fp32, tokens_per_entry=ratio))
            # It writes one FP32 score per candidate block across the logits row's
            # whole width, max_model_len / ratio in decode, and top-k reads them all
            # back (candidate_blocks.py:400-445; sparse_attn_indexer.py:748-756).
            nodes.append(context_work(
                "candidate_block_scores", per="query_token", bytes_per_entry=2 * fp32,
                tokens_per_entry=ratio * positive("candidate_block_size"),
                decode_extent="max_model_len"))
        elif candidate_source is not None and i > candidate_source:
            # Later indexers mask their rows to those blocks with -inf stores across the
            # row's whole width, which in decode is max_model_len
            # (candidate_blocks.py:114-145, 448-480; sparse_attn_indexer.py:726-765).
            nodes.append(context_work("candidate_mask", per="query_token",
                                      bytes_per_entry=fp32, tokens_per_entry=ratio,
                                      decode_extent="max_model_len"))
        return nodes

    def main_attention(ratio, owns_stream, *, read_window=None):
        # Every layer reads its own last sliding_window positions; a compressed layer
        # adds at most index_topk positions of its source's compressed stream
        # (nvidia/flashmla.py:181-246) and a ratio-0 layer passes no extra cache (:242).
        # compress_ratio states the stream the top-k positions come from, which its kv
        # cache must match; the read itself is window + index_topk positions
        # (inference-sim/blis-latency-kernel#32).
        node = attention(
            cfg, model,
            kind="sparse_mla",
            latent_width=total_dim,
            window=read_window or window,
            compress_ratio=ratio or None,
            index_topk=topk if ratio else None,
        )
        node["caches"] = [window_cache()]
        if ratio:
            node["caches"].append(compressed_cache(ratio, owns_stream))
            if owns_stream and ratio > 1:
                node["caches"].append(ring_cache())
        return node

    def routed_experts(experts, top_k):
        quant = raw.get("quantization_config") or {}
        mapped = EXPERT_DTYPES.get(str(quant.get("expert_dtype")))
        if mapped is None:
            raise DeriveError(
                f"{model}: quantization_config.expert_dtype "
                f"{quant.get('expert_dtype')!r} is not one this deriver maps")
        # Mxfp4MoEMethod (deepseek_v41/quant_config.py:211-228); gate and up fused, so
        # the node's N is the inner width and the consumer prices three matrices.
        return {"op": "GroupedGEMM", "role": "experts",
                "n": positive("moe_ffn_size", concept=True), "k": hidden,
                "experts": experts, "top_k": top_k,
                "weight_dtype": mapped}

    def ffn(experts, top_k):
        shared = positive("num_shared_experts", concept=True)
        routed_node = routed_experts(experts, top_k)
        return [
            *mhc_boundary("ffn"),
            # The whole MoE runs on the residual stream, so on the rank's token shard
            # under V4.1's sequence parallelism.
            # The router is an unquantized GateLinear with FP32 logits
            # (deepseek_v4/nvidia/model.py:850-856; checkpoint BF16 [384, 5120]).
            *token_sharded(with_tp(base_gemm("router", experts, hidden), "replicated"),
                           routed_node),
            # The shared expert is MXFP8, the routed ones MXFP4
            # (deepseek_v4/nvidia/model.py:900-914 under
            # deepseek_v41/quant_config.py:188-210), so it is its own node at the global
            # width rather than a field of the routed one. It is a dense MLP, a merged
            # gate/up column-parallel linear and a row-parallel down, TP-split normally
            # and whole on every rank over the token shard under V4.1's sequence
            # parallelism (`disable_tp=is_sequence_parallel`, :164, :173); expert
            # parallelism does not place it.
            *token_sharded(
                with_tp(gemm("shared_gate_up", 2 * routed_node["n"] * shared, hidden),
                        "column"),
                with_tp(gemm("shared_down", hidden, routed_node["n"] * shared), "row")),
            collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
            # V4.1 passes its own sequence-parallel flag to the MoE (nvidia/model.py:
            # 324-327), and FusedMoE then skips its final reduction
            # (fused_moe/runner/moe_runner.py:506-512); it never reads the engine-wide
            # use_sequence_parallel_moe.
            collective("AllReduce", "mlp_out",
                       EMIT_TENSOR_PARALLEL_UNLESS_SEQUENCE_PARALLEL),
        ]

    routed = positive("num_experts", concept=True)
    routed_top_k = positive("top_k", concept=True)
    if routed_top_k > routed:
        raise DeriveError(f"{model}: top_k {routed_top_k} exceeds {routed} experts")

    def layer_nodes(i):
        ratio = ratios[i]
        if i == 0:
            nodes = broadcast_mhc(hidden, "hc_attn_fn_broadcast")
        elif i in engram_layers:
            nodes = engram_block(i)
        else:
            nodes = mhc_boundary("attn")
        nodes += projections_in()
        if i in kv_sources:
            nodes += compressor(ratio)
        if i in index_sources:
            nodes += indexer(i, ratio, owns_k=i in kv_sources)
        nodes.append(main_attention(ratio, i in kv_sources))
        if ratio:
            # Prefill dequantizes the whole compressed stream it reads into a BF16
            # workspace before the sparse kernel, once per request chunk: each record
            # read, 512 BF16 values written (nvidia/flashmla.py:305-315). Readers run it
            # too, over their source's stream.
            nodes.append(context_work("prefill_dequant_gather", per="request",
                                      bytes_per_entry=total_dim * act,
                                      reads_cache="compressed", phase="prefill"))
        nodes += projections_out()
        nodes += ffn(routed, routed_top_k)
        return nodes

    def kind_id(i):
        ratio = ratios[i]
        name = f"csa{ratio}" if ratio else "swa"
        # Both roles are named independently: a KV source need not index, and an id
        # that folded the two would give such a layer an indexing source's nodes.
        if i in kv_sources:
            name += "_kvsrc"
        if i in index_sources:
            name += "_idx"
        if i == candidate_source:
            name += "_cand"
        elif candidate_source is not None and i in index_sources and i > candidate_source:
            name += "_mask"
        if i in engram_layers:
            # Each engram layer gathers from its own table, of its own size.
            name += f"_engram{i}"
        if i == 0:
            name += "_first"
        return name + "_moe"

    kinds, built, sequence = [], {}, []
    for i in range(layers):
        kid = kind_id(i)
        sequence.append(kid)
        nodes = layer_nodes(i)
        if kid in built:
            # A layer that reuses a kind must build exactly its nodes; an id that misses
            # a per-layer difference would otherwise price this layer as the first one.
            if nodes != built[kid]:
                raise DeriveError(f"{model}: layer {i} shares kind {kid} with an earlier "
                                  f"layer but builds different work")
            continue
        built[kid] = nodes
        kinds.append({"id": kid, "nodes": nodes, "edges": chain(nodes)})

    vocab = positive("vocab_size", concept=True)
    embed = {"op": "Gather", "role": "embed_tokens", "rows": vocab,
             "row_bytes": hidden * act, "rows_per_token": 1,
             "out_bytes_per_token": hidden * act, "shard": "rows"}
    lm_head = with_tp(base_gemm("lm_head", vocab, hidden), "column")

    # The DSpark draft module (nvidia/dspark.py): num_nextn_predict_layers decoder layers
    # at ids num_hidden_layers + i, each sliding-window only, with mHC and an MoE of
    # dspark_n_routed_experts at dspark_num_experts_per_tok (nvidia/model.py:118-145), and
    # no engram (no engram_layout is passed, dspark.py:122-132). It drafts
    # dspark_block_size tokens in ONE pass over that many positions per request
    # (v1/worker/gpu/spec_decode/dspark/speculator.py:1-23, 250-268), and each draft
    # query reads its window plus its whole block, non-causally (sparse_swa.py:
    # 1253-1258).
    #
    # Before that pass the drafter projects the hidden states of the
    # dspark_target_layer_ids through main_proj and main_norm (dspark.py:100-108,
    # 159-164), and each draft layer derives its context KV from that through its own
    # fused_wqa_wkv, kv_norm and a fused RoPE + insert (dspark.py:166-197, 252-275), over
    # the target's scheduled tokens: the context stack.
    #
    # After the pass: mHC post, hc collapse, the final norm and the target's lm_head over
    # every block position (aliased, dspark.py:279-282, 341-347). Then the Markov head
    # once per draft position, in order, over one row per request: an embedding of the
    # previous token and a (vocab, markov_rank) BF16 projection (speculator.py:156-199;
    # model_executor/models/qwen3_dspark.py:48-93), on its full path because
    # dspark_draft_topk is unset and defaults to None (config/speculative.py:1503-1515).
    # The bias add and sampling over each vocab-wide row are not priced. The confidence
    # head is built but consulted only under adaptive verification
    # (speculator.py:106-113), which is off by default (config/speculative.py:552).
    draft_kinds = []
    if spec_layers:
        # main_proj's input is one hidden-wide state per target layer, so the ids are
        # checked as layers of this stack, not just counted.
        targets = layer_ids("dspark_target_layer_ids", 0, layers, nonempty=True)
        # The capture points. The draft config is the target's, with model_type set to
        # deepseek_v41 where the target's is (config/speculative.py:1439-1447@v0.32.0),
        # and for that model_type vLLM takes the ids as-is: V4.1 reads the stream
        # entering each named layer (eagle3_utils.py:112-122). Each captured state is the
        # mean over the hc_mult bf16 streams, (tokens, hidden) -- one hidden-width
        # activation, which is the schema's default width, so none is stated
        # (mhc_shifted_post_pre's aux, nvidia/ops/mhc.py:260-264; tilelang.py:313-323;
        # previous_post.mean(dim=1), nvidia/model.py:541-542; _collapse, :1153-1157).
        # V4.1's decoder does not relay aux states over pipeline stages (it sets no
        # supports_aux_hidden_states_over_pp, and its boundary carries none,
        # nvidia/model.py:1052-1055), so DSpark runs without PP.
        spec_type = "deepseek_v41" if raw.get("model_type") == "deepseek_v41" else "deepseek_v4"
        aux_layers = draft_aux_hidden_state_layers({**cfg, "model_type": spec_type}, model,
                                                   layers, states=len(targets))
        draft_experts = positive("dspark_n_routed_experts")
        draft_top_k = positive("dspark_num_experts_per_tok")
        if draft_top_k > draft_experts:
            raise DeriveError(f"{model}: dspark_num_experts_per_tok {draft_top_k} "
                              f"exceeds {draft_experts} draft experts")
        markov = positive("dspark_markov_rank")
    for d in range(spec_layers):
        nodes = []
        if d == 0:
            nodes += broadcast_mhc(hc_mult * hidden, "hc_attn_fn")
        else:
            nodes += mhc_boundary("attn")
        nodes += projections_in()
        nodes.append(main_attention(draft_ratios[d], False, read_window=window + block))
        nodes += projections_out()
        nodes += ffn(draft_experts, draft_top_k)
        draft_kinds.append({"id": f"dspark_{d}", "nodes": nodes, "edges": chain(nodes)})
    kinds += draft_kinds
    if spec_layers:
        context = [with_tp(gemm("dspark_main_proj", hidden, hidden * len(targets)),
                           "replicated"),
                   with_tp(norm("dspark_main_norm"), "replicated")]
        for d in range(spec_layers):
            context += [
                {**with_tp(gemm(f"dspark_context_fused_wqa_wkv_{d}", q_lora + total_dim,
                                hidden), "replicated"),
                 "aliases": f"dspark_{d}.fused_wqa_wkv"},
                traffic(f"dspark_context_kv_norm_{d}", 2 * total_dim * act,
                        tp="replicated"),
                # It writes draft layer d's window cache, which a node of this kind
                # cannot name, so the record is stated in its traffic.
                traffic(f"dspark_context_kv_rope_insert_{d}", total_dim * act + record,
                        tp="replicated"),
            ]
        kinds.append({"id": "dspark_context", "nodes": context, "edges": chain(context)})

    head = [*final_mhc(), *token_sharded(norm("final_norm")),
            # Under V4.1's own sequence parallelism, and without a drafter's hidden
            # buffer, the collapsed and normalized state is all-gathered before the head
            # (nvidia/model.py:986-990).
            collective("AllGather", "head_in", EMIT_SEQUENCE_PARALLEL),
            # ParallelLMHead is built with no quant_config (nvidia/model.py:1338-1342;
            # checkpoint head.weight BF16), so it is not the global FP8.
            lm_head]

    extras: dict[str, Any] = {
        "global": {
            "architecture": raw["architectures"][0],
            # The checkpoint's 32 x 32 UE8M0 blocks are expanded to 1 x 32 at load
            # (modelopt.py:2200-2248), so every MXFP8 weight holds one byte per 32.
            "weight_scale": dict(DSV41_MXFP8_RESIDENT_SCALE),
            # vLLM derives max_model_len from max_position_embeddings and applies no
            # further factor for rope_type yarn (config/model.py:2471-2570); the
            # tokenizer's model_max_length takes part through a min()
            # (config/model.py:2496-2501), and at 1,048,576 at the pinned revision it
            # equals max_position_embeddings, so it does not lower it.
            "context_window": positive("max_position_embeddings"),
            # A stage hands the next the hc_mult-stream bundle at the activation dtype and
            # the (tokens, hc_mult) fp32 pre-mix the next stage's first layer collapses
            # with, and no separate residual (make_empty_intermediate_tensors,
            # nvidia/model.py:885-910@v0.32.0; the send, :1052-1055). Sequence parallelism
            # is off under PP (_use_sequence_parallel, :246-256), so every boundary moves
            # this much.
            "pipeline_boundary_bytes_per_token": bundle + hc_mult * fp32,
        },
        # BF16, not FP8: vLLM passes the quant config, but it returns no method for an
        # embedding (quantization/fp8.py:177-209), so the table falls back to
        # UnquantizedEmbeddingMethod (vocab_parallel_embedding.py:314-317). Each rank
        # looks up its vocabulary shard and the partials are all-reduced
        # (vocab_parallel_embedding.py:518-563), before sp_shard (nvidia/model.py:828,
        # 904).
        "embedding": [embed, collective("AllReduce", "embed_out", EMIT_TENSOR_PARALLEL)],
        "head": head,
        "base_capability_families": [9],
        "arch_overrides": [{
            # SM100 and SM103 run the MXFP8 BMM natively (mxfp8/deep_gemm.py:34-35), and
            # store the 528 B MXFP8 record with 512 B pages (attention.py:125-126,
            # 500-512); under nvfp4_ds_mla the compressed record is 288 B NVFP4
            # (:506-508), which mega attention resolves an unstated cache dtype to
            # (nvidia/flash_mla_mega_attn.py:193-197). The index record keeps 132 B and
            # takes 512 B pages (:1174-1176).
            "capability_family": 10,
            "nodes": [{"role": "wo_a", "weight_dtype": "fp8"}],
            "caches": [
                {"name": "swa", "bytes_per_entry": 528, "page_alignment": 512},
                {"name": "compressed", "bytes_per_entry": 528, "page_alignment": 512,
                 "bytes_by_cache_dtype": {"nvfp4_ds_mla": 288}},
                {"name": "index", "page_alignment": 512},
            ],
        }],
    }
    if spec_layers:
        extras["speculator"] = {
            # vLLM v0.31.0 names this method "dspark" and rejects "mtp" for deepseek_v41
            # (config/speculative.py:1358-1369). Its default draft length is
            # dspark_block_size, not num_nextn_predict_layers, which counts draft LAYERS
            # (:1475-1482; nvidia/dspark.py:87-91).
            "method": "dspark",
            "num_spec_tokens": block,
            "drafting": "parallel_block",
            # The drafter's own VocabParallelEmbedding, whose weights are the target's
            # (dspark.py:93-98), looks up its block's tokens and all-reduces them across TP
            # before the stream is sharded (dspark.py:205-215;
            # vocab_parallel_embedding.py:560-563), so the reduction runs in both modes.
            "embedding": [{**embed, "role": "dspark_embed_tokens",
                           "aliases": "embedding.embed_tokens"},
                          collective("AllReduce", "dspark_embed_out", EMIT_TENSOR_PARALLEL)],
            "stack": {"pattern": [k["id"] for k in draft_kinds], "repeat": 1},
            "context_stack": {"pattern": ["dspark_context"], "repeat": 1},
            "head": [
                *final_mhc(),
                # Under its sequence parallelism the drafter all-gathers its collapsed
                # state (dspark.py:247-248).
                collective("AllGather", "dspark_head_in", EMIT_SEQUENCE_PARALLEL),
                norm("dspark_final_norm"),
                {**with_tp(gemm("dspark_lm_head", vocab, hidden), "column"),
                 "aliases": "head.lm_head"},
            ],
            "sequential_head": [
                {"op": "Gather", "role": "markov_w1", "rows": vocab,
                 "row_bytes": markov * act, "rows_per_token": 1,
                 "out_bytes_per_token": markov * act},
                with_tp(base_gemm("markov_w2", vocab, markov), "replicated"),
            ],
            "aux_hidden_state_layers": aux_layers,
        }
    # With no draft layers there is no speculator to state: derive()'s fallback reads
    # the same num_nextn_predict_layers and emits none at zero.
    return kinds, compress(sequence), extras


def handler_minimax_m3(cfg, raw, model):
    """MiniMax-M3: block-sparse attention with a learned indexer, over a MoE stack whose
    first three layers are dense.

    Two layer kinds, ordered by the config's own per-layer vectors. `moe_layer_freq`
    and `sparse_attention_config.sparse_attention_freq` are checked to be the same
    vector rather than assumed to be: they are two statements of one fact on the
    published weights, and a variant that decoupled them would need its kinds named
    rather than collapsed.

    The attention kind is `swa`. What the schema's SWA means is a per-token read
    bounded by something other than context length, and M3's bound is
    `sparse_topk_blocks * sparse_block_size` tokens. The bound is the part a KV-byte
    cost model reads, and the arithmetic is the same for a contiguous window as for
    scattered blocks. What is NOT the same is which positions those are: SWA reads the
    most recent window, M3 reads blocks chosen per query. That does not change the bytes
    read, so it does not change this node; it shows up as the indexer below.

    The indexer is its own Attention node rather than a term folded into the main one.
    It has its own projections on the published weights -- index_q_proj [512, 6144] and
    index_k_proj [128, 6144], i.e. `sparse_num_index_heads` heads of `sparse_index_dim`
    for the query and one such head for the key -- so it maintains a second, narrow KV
    cache and scores every block to pick the top k. Its read is bounded by context, not
    by the top-k, which is why it cannot ride inside a node whose read is bounded.

    The dense layers' MLP width is `dense_intermediate_size`, not `intermediate_size`.
    Both are declared and they differ by 4x; the published weights settle which is which
    (`layers.0.mlp.gate_proj` is [12288, 6144] where a routed expert's w1 is
    [3072, 6144]).
    """
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))

    sparse = cfg.get("sparse_attention_config")
    if not sparse:
        raise DeriveError(
            f"{model}: no sparse_attention_config. This family's attention read is "
            f"bounded by a block top-k, and pricing it as dense GQA would overstate "
            f"every sparse layer's KV read"
        )
    if sparse.get("use_sparse_attention") is not True:
        raise DeriveError(
            f"{model}: sparse_attention_config present but use_sparse_attention is "
            f"{sparse.get('use_sparse_attention')!r}; a declared-but-off bound is a "
            f"different model and this handler will not guess which"
        )

    attn_freq = sparse.get("sparse_attention_freq")
    moe_freq = cfg.get("moe_layer_freq")
    if not attn_freq or not moe_freq:
        raise DeriveError(
            f"{model}: needs both sparse_attention_freq and moe_layer_freq to order "
            f"its layer kinds"
        )
    pairs = (("sparse_attention_freq", attn_freq),
             ("moe_layer_freq", moe_freq))
    for name, vec in pairs:
        if len(vec) != layers:
            raise DeriveError(
                f"{model}: {name} has {len(vec)} entries for {layers} layers"
            )
    if list(attn_freq) != list(moe_freq):
        raise DeriveError(
            f"{model}: sparse_attention_freq and moe_layer_freq differ. This handler "
            f"prices one dense prologue and one sparse-plus-routed body; a stack that "
            f"mixed them independently needs its kinds named rather than paired"
        )
    unknown = set(attn_freq) - {0, 1}
    if unknown:
        raise DeriveError(
            f"{model}: sparse_attention_freq names {sorted(unknown)}, which this "
            f"handler cannot price; add the kind rather than defaulting it"
        )

    # A stated dense-prologue length and the vector state the same fact. Read by
    # literal name: only this family declares it, so it has no cross-dialect concept.
    first_dense = cfg.get("first_k_dense_replace")
    if first_dense is not None:
        expected = [0] * int(first_dense) + [1] * (layers - int(first_dense))
        if expected != list(attn_freq):
            raise DeriveError(
                f"{model}: first_k_dense_replace {first_dense} disagrees with the "
                f"per-layer vector; the two state one fact and this handler will not "
                f"choose between them"
            )

    topk_blocks = int(require_key(sparse, "sparse_topk_blocks", model))
    block = int(require_key(sparse, "sparse_block_size", model))
    index_heads = int(require_key(sparse, "sparse_num_index_heads", model))
    index_dim = int(require_key(sparse, "sparse_index_dim", model))
    local = int(sparse.get("sparse_local_block") or 0)
    init = int(sparse.get("sparse_init_block") or 0)
    # Every block the kernel is guaranteed to read: the top-k plus the always-resident
    # local and initial blocks. Counted in tokens because that is what a KV read costs.
    bound = (topk_blocks + local + init) * block

    dense_attn = attention_block(cfg, model, hidden, raw=raw)
    dense_ffn = int(require_key(cfg, "dense_intermediate_size", model))
    dense_nodes = dense_attn + [
        norm("post_attn_norm"),
        gemm("mlp_gate_up", 2 * dense_ffn, hidden),
        gemm("mlp_down", hidden, dense_ffn),
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE),
    ]

    sparse_attn = attention_block(cfg, model, hidden, raw=raw, kind="swa", window=bound)
    # The indexer scores the whole context to choose blocks, so its read is bounded by
    # context rather than by the top-k. It is placed BEFORE the bounded attention node,
    # which is the order it runs in: its scores pick the blocks that node then reads.
    #
    # The indexer's keys are a cache of their own, resident beside the main K/V and
    # growing with context: MiniMaxM3IndexerCache registers a key-only MLAAttentionSpec
    # of one head of sparse_index_dim per token (models/minimax_m3/common/indexer.py:
    # 102-153 at v0.31.0). Its dtype is attention_config.indexer_kv_dtype, which resolves
    # "auto" to bf16 whatever --kv-cache-dtype says (nvidia/model.py:501-505;
    # config/attention.py:83-87, 146-150), so the record is index_dim BF16 values under
    # every cache dtype. Left unstated, a consumer infers no cache for this node and the
    # pool comes out 1.5x too large at TP8 (45,312 B a token in the server's own
    # "Available KV cache memory" over "GPU KV cache size", against 30,720 of main K/V).
    #
    # The main K/V is NOT bounded either: vLLM registers a FullAttentionSpec for it
    # (nvidia/model.py:559-567), so only its READ is bounded by the top-k. It is left
    # inferred, as a GQA cache whose heads split over TP, which a stated entry (one
    # whole-layer record, never divided by TP) cannot express.
    indexer = {
        "op": "Attention",
        "role": "block_index_scores",
        "kind": "gqa",
        "n_q": index_heads,
        "n_kv": 1,
        "d_h": index_dim,
        "caches": [{"name": "index", "kind": "index", "owned": True,
                    "bytes_per_entry": index_dim * DTYPE_WIDTHS["bf16"]}],
    }
    at = next(i for i, n in enumerate(sparse_attn) if n.get("op") == "Attention")
    sparse_nodes = sparse_attn[:at] + [
        gemm("index_qk_proj", index_heads * index_dim + index_dim, hidden),
        indexer,
    ] + sparse_attn[at:]
    sparse_nodes += [
        norm("post_attn_norm"),
        *moe_block(cfg, model, hidden),
        collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE),
    ]

    kinds = [
        {"id": "dense", "nodes": dense_nodes, "edges": chain(dense_nodes)},
        {"id": "sparse_attn_moe", "nodes": sparse_nodes, "edges": chain(sparse_nodes)},
    ]
    sequence = ["sparse_attn_moe" if f else "dense" for f in attn_freq]
    return kinds, compress(sequence)


def handler_qwen3_5_moe(cfg, raw, model):
    """Qwen3.5-MoE: a Gated DeltaNet linear-attention layer on most layers, full attention
    on every fourth, and a routed expert MLP with a shared expert on all of them.

    The full-attention positions come from the declared layer_types vector rather than
    from full_attention_interval. Both are present and they agree, but the vector is the
    ground truth: an interval is a generator for the common case and the vector is what the
    model was built with, so a variant that broke the period would still be priced right.

    The GDN state shape follows vLLM's own calculator
    (`MambaStateShapeCalculator.gated_delta_net_state_shape`): a convolutional state of
    width `2*k_heads*k_dim + v_heads*v_dim` over `conv_kernel - 1` positions, and a
    temporal state of `v_heads * v_dim * k_dim`. The graph records the temporal state's
    per-head geometry, which is what a cost model reads; the convolution's width rides in
    intermediate_size.
    """
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))

    types = cfg.get("layer_types")
    if not types:
        raise DeriveError(
            f"{model}: no layer_types vector. This family mixes linear and full attention "
            f"per layer, and full_attention_interval alone would be a guess at which"
        )
    if len(types) != layers:
        raise DeriveError(
            f"{model}: layer_types has {len(types)} entries for {layers} layers"
        )
    unknown = set(types) - {"linear_attention", "full_attention"}
    if unknown:
        raise DeriveError(
            f"{model}: layer_types names {sorted(unknown)}, which this handler cannot "
            f"price; add the kind rather than defaulting it"
        )

    # The vector and the stated interval must agree. They are two statements of the same
    # fact, and a config where they disagree is one this handler should not guess at.
    interval = int(cfg.get("full_attention_interval") or 0)
    if interval:
        expected = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                    for i in range(layers)]
        if expected != list(types):
            raise DeriveError(
                f"{model}: layer_types disagrees with full_attention_interval "
                f"{interval}; the two state the same fact and this handler will not "
                f"choose between them"
            )

    k_heads = int(require_key(cfg, "linear_num_key_heads", model))
    v_heads = int(require_key(cfg, "linear_num_value_heads", model))
    k_dim = int(require_key(cfg, "linear_key_head_dim", model))
    v_dim = int(require_key(cfg, "linear_value_head_dim", model))
    conv = int(require_key(cfg, "linear_conv_kernel_dim", model))

    def with_moe(mixer):
        n = list(mixer)
        n += [norm("post_attn_norm"), *moe_block(cfg, model, hidden),
              collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    gdn_mixer = [
        norm("input_norm"),
        {
            "op": "RecurrentUpdate",
            "recurrent_kind": "gdn",
            "n_heads": v_heads,
            "state_size": k_dim,
            "conv_kernel": conv,
            # The convolution's width, which is what the conv state spans.
            "intermediate_size": 2 * k_heads * k_dim + v_heads * v_dim,
            "state_dtype": DTYPES.get(
                str(cfg.get("mamba_ssm_dtype", "float32")).lower(), "fp32"),
        },
        gemm("mixer_out", hidden, v_heads * v_dim),
        collective("AllReduce", "mixer_out", EMIT_TENSOR_PARALLEL),
    ]
    gdn = with_moe(gdn_mixer)
    full = with_moe(attention_block(cfg, model, hidden, raw=raw))
    kinds = [
        {"id": "gdn_moe", "nodes": gdn, "edges": chain(gdn)},
        {"id": "attn_moe", "nodes": full, "edges": chain(full)},
    ]
    sequence = ["attn_moe" if t == "full_attention" else "gdn_moe" for t in types]
    return kinds, compress(sequence)


def handler_nemotron_h(cfg, raw, model):
    """NemotronH: a declared per-layer vector over state-space, MoE and attention
    layers, with no repeating unit and -- on one variant -- no stated layer count.

    Depth is the vector's length. The MoE entries are MLP layers rather than separate
    transformer blocks, which is why the vector is longer than a reader expecting
    num_hidden_layers would guess."""
    hidden = int(require(cfg, "hidden_size", model))
    vector = cfg.get("layers_block_type") or cfg.get("hybrid_override_pattern")
    if not vector:
        raise DeriveError(f"{model}: no layers_block_type or hybrid_override_pattern")
    if isinstance(vector, str):
        # The character form: M state-space, * attention, - MLP, E MoE.
        chars = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}
        try:
            vector = [chars[c] for c in vector]
        except KeyError as exc:
            raise DeriveError(f"{model}: unknown layer character {exc}") from exc

    stated = pick(cfg, "num_layers")
    if stated is not None and int(stated) != len(vector):
        raise DeriveError(
            f"{model}: num_hidden_layers {stated} disagrees with a {len(vector)}-entry "
            f"layer vector"
        )

    mamba = [
        norm("input_norm"),
        {
            "op": "RecurrentUpdate",
            "recurrent_kind": "mamba2",
            "n_heads": int(cfg["mamba_num_heads"]),
            "state_size": int(cfg["ssm_state_size"]),
            "n_groups": int(cfg.get("n_groups") or cfg.get("mamba_n_groups") or 1),
            "conv_kernel": int(cfg["conv_kernel"]),
            "intermediate_size": int(cfg["mamba_num_heads"]) * int(cfg["mamba_head_dim"]),
            "state_dtype": DTYPES.get(
                str(cfg.get("mamba_ssm_cache_dtype", "float32")).lower(), "fp32"),
        },
        gemm("mixer_out", hidden,
             int(cfg["mamba_num_heads"]) * int(cfg["mamba_head_dim"])),
        collective("AllReduce", "mixer_out", EMIT_TENSOR_PARALLEL),
    ]
    # The recurrent mixer's projections where the checkpoint states their width. The
    # nvfp4 variant stores these at FP8 while its routed experts are NVFP4, so they
    # cannot take the global width either way round.
    classes = quantized_classes(cfg, model)
    if classes.get("recurrent_out"):
        for n in mamba:
            # Scoped to the projection: the AllReduce that follows shares the role name
            # and holds no parameters, so a dtype on it describes nothing.
            if n.get("role") == "mixer_out" and n["op"] == "GEMM":
                n["weight_dtype"] = classes["recurrent_out"]
    moe_nodes = [
        norm("input_norm"),
        *moe_block(cfg, model, hidden),
        collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE),
    ]
    attn = attention_block(cfg, model, hidden, raw=raw)
    mlp = [norm("input_norm")] + dense_mlp(cfg, model, hidden) + [
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]

    catalogue = {
        "mamba": {"id": "mamba", "nodes": mamba, "edges": chain(mamba)},
        "moe": {"id": "moe", "nodes": moe_nodes, "edges": chain(moe_nodes)},
        "attention": {"id": "attention", "nodes": attn, "edges": chain(attn)},
        "mlp": {"id": "mlp", "nodes": mlp, "edges": chain(mlp)},
    }
    used = []
    for name in vector:
        if name not in catalogue:
            raise DeriveError(f"{model}: unknown layer type {name!r}")
        if name not in used:
            used.append(name)
    return [catalogue[n] for n in used], compress(list(vector))


def handler_kimi_k3(cfg, raw, model):
    """Kimi-K3: linear attention on most layers, latent attention on a declared list,
    and a dense MLP on the leading layers.

    The full-attention layer indices are given explicitly rather than by a period, so
    the sequence is built from that list.

    first_k_dense_replace makes the leading layers dense: KimiDecoderLayer builds a
    KimiSparseMoeBlock only when `layer_idx >= first_k_dense_replace`, and otherwise a
    KimiMLP at the config's own intermediate_size rather than moe_intermediate_size
    (modeling_kimi_linear.py:893-900). Pricing layer 0 as routed charges 896 experts
    for a layer that runs one dense MLP."""
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))
    linear = cfg.get("linear_attn_config") or {}
    full = linear.get("full_attn_layers")
    if not full:
        raise DeriveError(f"{model}: no linear_attn_config.full_attn_layers")

    def with_moe(mixer_nodes):
        n = list(mixer_nodes)
        n += [norm("post_attn_norm"), *moe_block(cfg, model, hidden),
              collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    def with_dense(mixer_nodes):
        n = list(mixer_nodes)
        n += [norm("post_attn_norm"), *dense_mlp(cfg, model, hidden),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    # The KDA mixer's dense projections. RecurrentUpdate carries no N or K, so it
    # cannot account for these weights: without them the layer had no producer GEMM at
    # all for the reduction that follows it, dropping 443,580,416 parameters per layer
    # over 69 KDA layers (about 30.6B) and their FLOPs with them.
    #
    # Shapes from the pinned KimiDeltaAttention (modeling_kimi_linear.py:495-541). Q, K
    # and V are three separate Linears rather than one fused projection, each followed
    # by its own ShortConvolution; the decay path bottlenecks through head_dim; beta is
    # one scalar per head; and the output gate is a single dense g_proj here because
    # this config sets use_full_rank_gate (the low-rank g_a/g_b pair is the default).
    kda_heads = int(linear.get("num_heads") or require(cfg, "num_q_heads", model))
    kda_dim = int(linear.get("head_dim") or head_dim(cfg, model))
    kda_inner = kda_heads * kda_dim
    kda_proj = [
        gemm("kda_q_proj", kda_inner, hidden),
        gemm("kda_k_proj", kda_inner, hidden),
        gemm("kda_v_proj", kda_inner, hidden),
        # The decay gate, low-rank through head_dim.
        gemm("kda_f_a_proj", kda_dim, hidden),
        gemm("kda_f_b_proj", kda_inner, kda_dim),
        # One beta scalar per head.
        gemm("kda_b_proj", kda_heads, hidden),
    ]
    if linear.get("use_full_rank_gate"):
        kda_proj.append(gemm("kda_g_proj", kda_inner, hidden))
    else:
        kda_proj += [
            gemm("kda_g_a_proj", kda_dim, hidden),
            gemm("kda_g_b_proj", kda_inner, kda_dim),
        ]
    kda_mixer = [
        norm("input_norm"),
        *kda_proj,
        {
            "op": "RecurrentUpdate",
            "recurrent_kind": "kda",
            "n_heads": kda_heads,
            "state_size": kda_dim,
            "conv_kernel": int(linear.get("short_conv_kernel_size") or 0) or None,
            "state_dtype": "fp32",
        },
        gemm("kda_o_proj", hidden, kda_inner),
        collective("AllReduce", "mixer_out", EMIT_TENSOR_PARALLEL),
    ]
    # conv_kernel is omitted rather than zeroed where the config states none.
    kda_mixer = [n if n.get("conv_kernel") is not None or n.get("op") != "RecurrentUpdate"
                 else {k: v for k, v in n.items() if k != "conv_kernel"}
                 for n in kda_mixer]
    # The KDA mixer IS this layer's self_attn (vllm kimi_k3/nvidia/model.py:888-891), so
    # a compressed-tensors ignore list naming self_attn leaves these projections at the
    # base width just as it does the latent ones. kimi-k3's global width is mxfp4, so
    # without this every KDA projection prices at four bits for tensors the runtime
    # never quantizes.
    if compressed_tensors_skips_attention(cfg, raw):
        base = base_dtype(cfg, raw, model)
        kda_mixer = [dict(n, weight_dtype=base)
                     if n.get("role") in ATTENTION_WEIGHT_ROLES else n
                     for n in kda_mixer]
    kda = with_moe(kda_mixer)
    mla = with_moe(attention_block(cfg, model, hidden, raw=raw))
    kinds = [
        {"id": "kda_moe", "nodes": kda, "edges": chain(kda)},
        {"id": "mla_moe", "nodes": mla, "edges": chain(mla)},
    ]
    first_dense = int(cfg.get("first_k_dense_replace") or 0)
    if first_dense:
        kda_dense = with_dense(kda_mixer)
        mla_dense = with_dense(attention_block(cfg, model, hidden, raw=raw))
        kinds += [
            {"id": "kda_dense", "nodes": kda_dense, "edges": chain(kda_dense)},
            {"id": "mla_dense", "nodes": mla_dense, "edges": chain(mla_dense)},
        ]
    full_set = set(int(i) for i in full)
    # The published list is 1-based against a 1-indexed layer numbering; a 0-based
    # reading would shift every attention layer by one. Detected rather than assumed:
    # a 0-based list would contain 0.
    offset = 1 if 0 not in full_set else 0
    # The mixer and the MLP vary independently: which layers are attention comes from
    # the declared list, and which are dense from first_k_dense_replace.
    sequence = []
    for i in range(layers):
        mixer = "mla" if (i + offset) in full_set else "kda"
        sequence.append(f"{mixer}_dense" if i < first_dense else f"{mixer}_moe")
    # A kind no layer uses would be a node a reader has to account for and no step runs.
    kinds = [k for k in kinds if k["id"] in set(sequence)]
    return kinds, compress(sequence)


def handler_glm5_next(cfg, raw, model):
    """GLM-5.3-Flash: KDA on most layers, sparse latent attention on the rest, and a
    dense MLP prologue, with the composition stated by two per-layer vectors.

    Three things differ from handler_kimi_k3, which this otherwise mirrors.

    The layer sequence comes from `layer_types` and `mlp_layer_types` rather than from
    `full_attn_layers` plus `first_k_dense_replace`. Both vectors are length
    num_hidden_layers and they vary independently, so the kind of a layer is the pair:
    34 `linear_attention` against 11 `deepseek_sparse_attention`, and 3 `dense` MLPs
    before 42 `sparse` ones. Reading either vector as a period would misplace layers --
    the attention layers sit at every fourth index, which no single period expresses
    once the dense prologue is also in play.

    The attention layers are `deepseek_sparse_attention`, so they carry a lightning
    indexer. The config declares the full spec (`index_n_heads` 32, `index_head_dim`
    128, `index_topk` 2048) and `lightning_indexer()` emits it. Without that node the
    graph would record a top-k selection it never charges for, which is the omission
    that helper's docstring records against this family.

    `num_nextn_predict_layers` is 1 rather than kimi-k3's 0, so the model carries one
    MTP module. That reaches the graph through the `num_spec_tokens` alias, as it does
    for the DeepSeek-V3 family, rather than through a node here: an MTP module runs as
    its own forward pass, not as part of a base layer.

    That draft module is sparse-MLA plus MoE -- layer 45 carries `self_attn.indexer.*`,
    `kv_a_layernorm` and `kv_b_proj` and none of layer 44's KDA projections (`A_log`,
    `k_conv1d`, `dt_bias`) -- but it is NOT a copy of any target layer, so this handler
    prices it as DRAFT_KIND_ID rather than letting derive()'s fallback mirror one. It
    runs its own embedding-mixing prefix (`enorm`, `hnorm`, an `eh_proj` over the
    concatenated pair) and the `shared_head` norm, and it carries no mHC.

    Leaving it to the fallback was the earlier behaviour and was wrong twice over. The
    fallback mirrors the last DECLARED kind -- insertion order, not sequence position --
    which lands on `mla_moe` here only by the order the kinds below happen to be built
    in; this model's last base LAYER is `kda_moe`, so the positional reading of that
    fallback would have priced the draft pass as KDA. And even landing on `mla_moe`, a
    mirrored target layer pays none of the prefix above, which is 33,554,432 parameters
    of `eh_proj` on every speculative step.

    The published repo is multimodal and ships a `vision_config`. Only the text path is
    priced, as for the other multimodal entries: the vision tower runs once per image,
    not per decode step, and the deployments this catalog serves are text."""
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))

    mixer_types = cfg.get("layer_types")
    if not mixer_types:
        raise DeriveError(f"{model}: no layer_types vector")
    if len(mixer_types) != layers:
        raise DeriveError(
            f"{model}: layer_types has {len(mixer_types)} entries for {layers} layers"
        )
    mlp_types = cfg.get("mlp_layer_types")
    if not mlp_types:
        raise DeriveError(f"{model}: no mlp_layer_types vector")
    if len(mlp_types) != layers:
        raise DeriveError(
            f"{model}: mlp_layer_types has {len(mlp_types)} entries for {layers} layers"
        )

    linear = cfg.get("linear_attn_config") or {}
    if not linear:
        raise DeriveError(f"{model}: no linear_attn_config")

    # Manifold hyper-connections. The config states `mhc: true` with hc_mult 4, and vLLM
    # builds, per base layer, a learned mixing around BOTH the attention and the FFN
    # (common/model.py:439-493): two [mix_hc, hc_mult*hidden] matrices where
    # mix_hc = (2 + n) * n, plus a pre/post/comb elementwise pass over the n residual
    # streams with hc_sinkhorn_iters normalisation rounds. The pinned headers confirm
    # the shapes exactly: hc_attn_fn and hc_ffn_fn are both [24, 16384] at n = 4.
    #
    # Gated on the `mhc` flag, NOT on hc_mult alone. deepseek-v4-pro declares hc_mult 4
    # and hc_sinkhorn_iters 20 but no `mhc` key, and vLLM reads `self.mhc` to decide --
    # so keying off hc_mult would invent mixing nodes for a model whose config never
    # switches it on.
    use_mhc = bool(cfg.get("mhc"))
    hc_mult = int(cfg.get("hc_mult") or 0)
    sinkhorn = int(cfg.get("hc_sinkhorn_iters") or 0)
    if use_mhc and hc_mult < 2:
        raise DeriveError(
            f"{model}: mhc is set but hc_mult is {hc_mult or 'absent'}; the mixing is "
            f"over hc_mult residual streams and needs at least two")
    if use_mhc and sinkhorn < 1:
        raise DeriveError(
            f"{model}: mhc is set but hc_sinkhorn_iters is {sinkhorn or 'absent'}; the "
            f"normalisation rounds are per-token work and a count is required")
    mhc_mix = (2 + hc_mult) * hc_mult if use_mhc else 0
    mhc_width = hc_mult * hidden if use_mhc else 0
    # Resolved only when the mixing exists, so a future config in this family that
    # declares no dtype cannot fail here for a width it never uses.
    # The mixing matrices are FP32 AT RUNTIME, whatever the checkpoint stores. vLLM
    # allocates both as `nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))`
    # (glm5next/common/model.py:454-463) and the kernel asserts it --
    # `assert fn.dtype == torch.float32` (model_executor/kernels/mhc/torch.py:82) -- so
    # a load-time upcast must occur for the graph to run at all. The pinned checkpoint
    # stores them BF16, which is why this is an override rather than the base width:
    # pricing the stored width halves the bytes the kernel actually reads, across both
    # matrices of all 45 layers.
    #
    # This is the one place in this handler where the resident width and the stored width
    # differ. Everywhere else the checkpoint is authoritative, because an absent
    # quant_config cannot dequantize a tensor that ships with weight_scale_inv.
    mhc_weight_dtype = "fp32" if use_mhc else None
    mhc_bytes = mhc_mixing_bytes(cfg, raw, model, hidden, hc_mult) if use_mhc else 0

    def mhc(nodes):
        """Replace each of a base layer's two norms with the mHC mixing that subsumes it.

        vLLM runs one mixing op per side, and each one FUSES the norm it replaces: the
        attention side calls hc_pre (or hc_fused_post_pre) with
        `norm_weight=self.input_layernorm.weight`, and the FFN side calls
        hc_fused_post_pre with `norm_weight=self.post_attention_layernorm.weight`
        (common/model.py:556-603). So the mixing stands IN PLACE OF each norm rather
        than in addition to it -- emitting both would charge the layer two normalisation
        passes it does not run.

        Each side is a [mix_hc, hc_mult*hidden] projection plus the mixing pass over the
        hc_mult residual streams. The projections are BF16 in the checkpoint, which the
        global fp8 width would otherwise understate.

        Each mixing node carries bytes_per_token, which every other Elementwise node in
        this catalog omits. Those are ordinary norms, whose traffic a cost model derives
        from hidden_size; this one moves the hc_mult-stream residual bundle, and no field
        of a ModelGraph carries hc_mult -- so an omitted value would price the mixing as
        the single-stream norm it replaced. See mhc_mixing_bytes().

        Returns the nodes untouched for any config that does not declare mhc, which is
        every other model in the catalog."""
        if not use_mhc:
            return nodes
        out = []
        replaced = set()
        for n in nodes:
            role = n.get("role")
            if role == "input_norm":
                out.append(dict(gemm("hc_attn_fn", mhc_mix, mhc_width),
                                weight_dtype=mhc_weight_dtype))
                out.append(dict(norm("hc_attn_mix"), bytes_per_token=mhc_bytes))
                replaced.add(role)
            elif role == "post_attn_norm":
                out.append(dict(gemm("hc_ffn_fn", mhc_mix, mhc_width),
                                weight_dtype=mhc_weight_dtype))
                out.append(dict(norm("hc_ffn_mix"), bytes_per_token=mhc_bytes))
                replaced.add(role)
            else:
                out.append(n)
        if replaced != {"input_norm", "post_attn_norm"}:
            raise DeriveError(
                f"{model}: mHC fuses the norm on each side, and this layer does not "
                f"have both to replace (found {sorted(replaced)})")
        return out

    def with_moe(mixer_nodes):
        n = list(mixer_nodes)
        n += [norm("post_attn_norm"), *moe_block(cfg, model, hidden),
              collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    def with_dense(mixer_nodes):
        n = list(mixer_nodes)
        n += [norm("post_attn_norm"), *dense_mlp(cfg, model, hidden),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    # The KDA mixer's dense projections, as for kimi-k3: RecurrentUpdate carries no N
    # or K, so without these the layer has no producer GEMM for the reduction after it.
    # This config states no use_full_rank_gate, so the low-rank g_a/g_b pair applies.
    kda_heads = int(linear.get("num_heads") or require(cfg, "num_q_heads", model))
    kda_dim = int(linear.get("head_dim") or head_dim(cfg, model))
    kda_inner = kda_heads * kda_dim
    kda_proj = [
        gemm("kda_q_proj", kda_inner, hidden),
        gemm("kda_k_proj", kda_inner, hidden),
        gemm("kda_v_proj", kda_inner, hidden),
        gemm("kda_f_a_proj", kda_dim, hidden),
        gemm("kda_f_b_proj", kda_inner, kda_dim),
        gemm("kda_b_proj", kda_heads, hidden),
    ]
    if linear.get("use_full_rank_gate"):
        kda_proj.append(gemm("kda_g_proj", kda_inner, hidden))
    else:
        kda_proj += [
            gemm("kda_g_a_proj", kda_dim, hidden),
            gemm("kda_g_b_proj", kda_inner, kda_dim),
        ]
    conv = int(linear.get("short_conv_kernel_size") or 0)
    recurrent = {
        "op": "RecurrentUpdate",
        "recurrent_kind": "kda",
        "n_heads": kda_heads,
        "state_size": kda_dim,
        "state_dtype": "fp32",
    }
    if conv:
        recurrent["conv_kernel"] = conv
    kda_mixer = [
        norm("input_norm"),
        *kda_proj,
        recurrent,
        gemm("kda_o_proj", hidden, kda_inner),
        collective("AllReduce", "mixer_out", EMIT_TENSOR_PARALLEL),
    ]
    # The sparse-attention layers are latent attention fronted by the indexer.
    mla_mixer = lightning_indexer(cfg, model, hidden,
                                  attention_block(cfg, model, hidden, raw=raw), raw)

    builders = {
        ("kda", "moe"): lambda: with_moe(kda_mixer),
        ("kda", "dense"): lambda: with_dense(kda_mixer),
        ("mla", "moe"): lambda: with_moe(mla_mixer),
        ("mla", "dense"): lambda: with_dense(mla_mixer),
    }
    sequence = []
    for mixer_t, mlp_t in zip(mixer_types, mlp_types):
        if mixer_t == "linear_attention":
            mixer = "kda"
        elif mixer_t == "deepseek_sparse_attention":
            mixer = "mla"
        else:
            raise DeriveError(f"{model}: unknown layer_types entry {mixer_t!r}")
        if mlp_t not in ("sparse", "dense"):
            raise DeriveError(f"{model}: unknown mlp_layer_types entry {mlp_t!r}")
        mlp = "moe" if mlp_t == "sparse" else "dense"
        sequence.append(f"{mixer}_{mlp}")

    # The widths this fp8 checkpoint leaves unquantized, per graph role. Read from
    # modules_to_not_convert rather than assumed: the config names every KDA projection
    # and every indexer tensor, and the pinned safetensors headers agree.
    unquant = fp8_unquantized_roles(cfg, raw, model, GLM5_NEXT_UNQUANTIZED)

    # Two separate reasons a GEMM here holds base-width parameters, and they are not the
    # same mechanism. modules_to_not_convert names tensors the quantizer SKIPS, so they
    # were never quantized. GLM5_NEXT_LOADER_DEQUANTIZED names tensors the checkpoint
    # ships quantized and this model's own loader converts on the way in. Both end at
    # the base width in HBM, which is what the graph prices; keeping them distinct is
    # what keeps each one's evidence checkable.
    loader_base = base_dtype(cfg, raw, model)

    def priced(nodes):
        """Stamp the base width on the GEMMs whose resident parameters are not quantized.

        Only a GEMM or GroupedGEMM may carry a per-node weight_dtype -- a norm holds no
        parameters -- so a role naming anything else would be a mistake here."""
        for n in nodes:
            role = n.get("role")
            width = unquant.get(role)
            if width is None and role in GLM5_NEXT_LOADER_DEQUANTIZED:
                width = loader_base
            if width is None:
                continue
            if n["op"] not in ("GEMM", "GroupedGEMM"):
                raise DeriveError(
                    f"{model}: role {role!r} holds base-width parameters but is a "
                    f"{n['op']}, which holds no parameters to width")
            n["weight_dtype"] = width
        return nodes

    kinds = []
    for key, build in builders.items():
        kid = f"{key[0]}_{key[1]}"
        if kid not in set(sequence):
            continue
        nodes = mhc(priced(build()))
        kinds.append({"id": kid, "nodes": nodes, "edges": chain(nodes)})

    # The MTP module. vLLM builds it as a sparse-MLA + MoE decoder layer (the layer_idx
    # past num_hidden_layers selects the DSA branch, common/mtp.py:78-88) fronted by its
    # own embedding-mixing prefix, and the pinned checkpoint's layer 45 matches: an
    # `eh_proj` of [4096, 8192], `enorm`, `hnorm`, and the `shared_head.norm` before
    # logits. It carries NO hc_* tensors, which vLLM states as `if self.mhc and not
    # is_mtp_layer` (common/model.py:439) -- so the draft layer gets no mHC nodes.
    #
    # Declared as DRAFT_KIND_ID rather than left to derive()'s fallback: that fallback
    # mirrors a TARGET layer, and no target layer carries this prefix. Before this the
    # draft stack reused mla_moe and so paid neither the eh_proj nor the three norms on
    # every speculative step.
    if int(pick(cfg, "num_spec_tokens") or 0) > 0:
        if ("mla", "moe") not in builders:
            raise DeriveError(
                f"{model}: an MTP module is declared but no sparse-MLA MoE layer was "
                f"built for it to mirror")
        draft = priced([
            norm("mtp_enorm"),
            norm("mtp_hnorm"),
            # Concatenated [previous hidden, token embedding] -> hidden. BF16 in the
            # checkpoint, which modules_to_not_convert also names.
            gemm("mtp_eh_proj", hidden, 2 * hidden),
            *builders[("mla", "moe")](),
            norm("mtp_shared_head_norm"),
        ])
        kinds.append({"id": DRAFT_KIND_ID, "nodes": draft, "edges": chain(draft)})

    return kinds, compress(sequence)


# --- Muse Spark -------------------------------------------------------------------
# Every fact below is read from vLLM's Muse Spark model code as merged in
# vllm-project/vllm#61879 (merge commit 0f6b26191064384aa35060c87f4ed570c7875e66, cited
# as @0f6b2619), the code the vLLM recipe serves the model with. Muse Spark support
# landed after vLLM v0.32.0, so these are model-code facts at that commit, not v0.32.0
# behaviour.
MUSE_SPARK_SOURCE = "vllm/models/muse_spark/nvidia/model.py@0f6b2619"

# When Muse Spark splits its replicated MoE pooler and unpooler over TP: on a device of
# capability 10.0 or later (has_device_capability(100), tep_default, model.py:516-520) and
# a step of at most _SHARDED_PROJECTION_MAX_TOKENS = 256 tokens (model.py:151). Whether the
# model's other conditions hold -- tp > 1, no DP, no mega-MoE or sequence parallelism
# (replicated_tokens, :505-512) -- is the consumer's to read from the layout.
MUSE_SHARDED_PROJECTION = {"capability_family": 10, "max_tokens": 256}

# Per token, a fused residual boundary reads the branch (activation width) and the fp32
# residual stream, writes the stream back and the normalized activation for the next
# block: post-norm, residual add and pre-norm in one pass (norm_residual_then_norm,
# model.py:1425-1500). The stream is carried in fp32 (residual_dtype, model.py:1396).
def _muse_boundary_bytes(hidden: int, act: int) -> int:
    fp32 = DTYPE_WIDTHS["fp32"]
    return hidden * act + 2 * hidden * fp32 + hidden * act


def handler_muse_spark(cfg, raw, model):
    """Muse Spark: GQA with a hybrid sliding-window/full pattern, an MoE on every layer
    whose experts run at a narrower latent width, and an fp32 residual stream.

    The layer (MuseSpark1xDecoderLayer and its attention and MoE, model.py@0f6b2619):

      * attention: a fused [Q, K, V, O-gate] projection of 2 (nq + nkv) d_h rows
        (MuseSpark1xQKVOGateParallelLinear, common/layers.py:339-369), QK-norm and RoPE
        fused with the KV-cache insert (model.py:1167-1222), GQA over a 2048-token window
        on the sliding layers and the whole context on the full ones (layer_types,
        model.py:1101-1110), the per-head output norm fused with its sigmoid gate
        (model.py:1240-1262), the row-parallel o_proj and its TP all-reduce;
      * the MoE (MuseSpark1xMoE, model.py:373-940): an fp32 router over the hidden state
        before the squeeze (GateLinear with fp32 params and fp32 input, :413-428), the
        replicated pooler hidden -> moe_hidden_size and its norm, 256 experts at top-8
        whose GEMMs run at the latent width (latent_size), the post-expert finalize
        (a per-route RMSNorm, then the fp32 weighted sum, muse_moe_runner.py:160-330),
        and the replicated unpooler back to hidden. No shared expert.
      * the reductions an MoE layer runs depend on the mode, so each states its own
        condition. Under tensor parallelism without expert parallelism the experts'
        GEMM2 rows are partial sums and RMS is not linear in them, so the top_k
        per-route rows are all-reduced BEFORE the norm (needs_pre_norm_all_reduce,
        common/moe.py:114-120; muse_moe_runner.py:288-290). Under expert parallelism
        with every token on every rank (no dispatch: dp 1, no sequence parallelism),
        each rank's combined latent output is partial over its experts and is
        all-reduced once (_create_all_reduce, muse_moe_runner.py:200-215). With a
        dispatch (data parallelism), the all-to-all moves it. The unpooler's output is
        replicated, so nothing reduces at hidden width after the MoE.

    The NVFP4 checkpoint (ModelOpt, quant_algo NVFP4, group 16) leaves the attention,
    lm_head and the experts of layers 0 and 61 at BF16 by its exclude_modules, the list
    vLLM reads; the pooler, unpooler and router are built unquantized. Layers 0 and 61
    are kinds of their own.

    The pooler and unpooler state sharded_at_decode: on SM100 with TP, dp 1 and at most
    256 tokens a step they shard their output columns over TP and all-gather
    (shard_pooler/shard_unpooler, model.py:502-541, 638-660), which the consumer prices.

    KNOWN GAPS, recorded rather than priced:
      * Under PP the residual stream is not fused across layers (fuse_across_layers is
        pp == 1, model.py:1533): every layer's output runs norm_then_gated_residual
        (model.py:1711-1712) and the next layer's input norm runs on its own, one extra
        fp32 stream pass, hidden x 4 B a token a layer, against the fused boundary priced
        here. No emit condition names pipeline parallelism, so it is not priced (about
        +9% of a layer's elementwise traffic under PP).
      * The token-embedding norm (tok_embeddings_norm, model.py:1662-1664) reads the bf16
        lookup and writes the fp32 stream, hidden x (2 + 4) B a token, and layer 0's input
        norm then runs on its own (model.py:1431-1435), hidden x (4 + 2) B, where the
        layer kind prices the fused boundary, hidden x (2 + 8 + 2) B. The two offset
        exactly in bytes; one launch is unpriced.
      * The fp32 router input for 32 < tokens <= 128 (wants_fp32_router_input,
        model.py:670-677, 1451-1455) is on only where torch allows TF32 matmuls
        (model.py:413-416), which VLLM_FLOAT32_MATMUL_PRECISION's default "highest" does
        not; no node can scope bytes to a token range, so it is not priced.
      * Under EP the finalize reads only the rank's local routes (fused_finalize.py
        masks remote experts, muse_moe_runner.py:340-342), where post_expert_norm charges
        every rank all top_k rows; no node scales its bytes by the expert-parallel width.
      * The native finalize (_finalize_chunk, muse_moe_runner.py:433-439) materialises a
        (tokens x top_k x moe_hidden_size) fp32 intermediate; it runs only on Blackwell at
        TP > 1 above 8192 tokens or where FlashInfer's TP finalize is unavailable.
      * The final logit soft-cap (final_logit_softcapping) runs on the sampled rows'
        vocabulary-wide logits, which no node of a head can scope to sampled rows, so it
        is not priced.
      * Under sequence-parallel MoE the gathered unpooler output (model.py:936-938) is
        not stated; Muse runs data parallelism only on Blackwell's TRT-LLM experts.
    """
    def whole(v, key):
        if isinstance(v, bool) or int(v) != v:
            raise DeriveError(f"{model}: {key} is {v!r}, not a whole number")
        return int(v)

    hidden = whole(require(cfg, "hidden_size", model), "hidden_size")
    layers = whole(require(cfg, "num_layers", model), "num_hidden_layers")
    nq = whole(require(cfg, "num_q_heads", model), "num_attention_heads")
    nkv = whole(require(cfg, "num_kv_heads", model), "num_key_value_heads")
    dh = head_dim(cfg, model)
    vocab = whole(require(cfg, "vocab_size", model), "vocab_size")
    experts = whole(require(cfg, "num_experts", model), "num_local_experts")
    top_k = whole(require(cfg, "top_k", model), "num_experts_per_tok")
    inner = whole(require_key(cfg, "moe_intermediate_size", model), "moe_intermediate_size")
    latent = whole(require_key(cfg, "moe_hidden_size", model), "moe_hidden_size")
    window = pick(cfg, "sliding_window")
    if top_k > experts:
        raise DeriveError(f"{model}: top-{top_k} routing over {experts} experts")
    if int(pick(cfg, "num_shared_experts") or 0):
        raise DeriveError(f"{model}: a shared expert, which this architecture's MoE never "
                          f"builds (model.py:373-470)")
    if not 0 < latent <= hidden:
        raise DeriveError(f"{model}: moe_hidden_size {latent} is not a narrowing of "
                          f"hidden_size {hidden}")
    attn_types = require_key(cfg, "layer_types", model)
    mlp_types = require_key(cfg, "mlp_layer_types", model)
    for key, vec in (("layer_types", attn_types), ("mlp_layer_types", mlp_types)):
        if len(vec) != layers:
            raise DeriveError(f"{model}: {key} has {len(vec)} entries for {layers} layers")
    unknown = set(attn_types) - {"sliding_attention", "full_attention"}
    if unknown:
        raise DeriveError(f"{model}: layer_types names {sorted(unknown)}")
    if set(mlp_types) != {"sparse"}:
        raise DeriveError(f"{model}: mlp_layer_types {sorted(set(mlp_types))}; every Muse "
                          f"Spark layer is an MoE, and a dense one is not priced here")
    if "sliding_attention" in attn_types and not window:
        raise DeriveError(f"{model}: sliding layers with no sliding_window")

    base = base_dtype(cfg, raw, model)
    act = DTYPE_WIDTHS[base]
    fp32 = DTYPE_WIDTHS["fp32"]

    # Which layers' experts the checkpoint quantizes. A layer the exclusion list names
    # keeps the base width; the rest carry the method's.
    quant = raw.get("quantization_config") or {}
    expert_fmt = None
    bf16_experts: set[int] = set()
    if quant:
        if str(quant.get("quant_method", "")).lower() != "modelopt":
            raise DeriveError(f"{model}: quant_method {quant.get('quant_method')!r}; this "
                              f"handler reads ModelOpt's layout")
        expert_fmt = QUANT_ALGOS.get(str(quant.get("quant_algo", "")).upper())
        if expert_fmt is None:
            raise DeriveError(f"{model}: quant_algo {quant.get('quant_algo')!r}")
        # The list vLLM reads (exclude_modules here), matched as vLLM matches it, against
        # the module names vLLM builds: MuseSpark1xForCausalLM's "model.layers.{i}" with
        # the MoE at ".feed_forward" (model.py:1352-1357) and its FusedMoE at ".experts"
        # (:597-605). The checkpoint's legacy modules_to_not_convert is not read. The
        # Omni WeightsMapper, which vLLM applies to the list (model_loader/utils.py:379),
        # rewrites only "model.language_model." and "model.layers.N.mlp..." spellings
        # (common/omni.py:48-80@0f6b2619), which the committed list does not use.
        excluded = modelopt_exclusions(quant)
        bf16_experts = {i for i in range(layers) if modelopt_excluded(
            f"model.layers.{i}.feed_forward.experts", excluded)}
        # The attention projections and lm_head take the quant config and are quantized
        # unless the list names them (model.py:1062-1080; LlamaForCausalLM's
        # ParallelLMHead), and this handler prices them at the base width, so a list that
        # leaves them quantized is a layout it does not price. The router and the
        # pooler/unpooler are built without a quant config (model.py:421-428, 450-463).
        for what, where in (("attention", "model.layers.{}.self_attn.qkv_proj"),
                            ("attention", "model.layers.{}.self_attn.o_proj")):
            left = [i for i in range(layers)
                    if not modelopt_excluded(where.format(i), excluded)]
            if left:
                raise DeriveError(f"{model}: the ModelOpt list leaves {what} quantized on "
                                  f"layers {left}; this handler prices it at the base")
        if not modelopt_excluded("lm_head", excluded):
            raise DeriveError(f"{model}: the ModelOpt list leaves lm_head quantized; this "
                              f"handler prices it at the base")

    def gemm_at(role, n, k, tp=None, dtype=None):
        node = gemm(role, n, k)
        if tp:
            node["tp"] = tp
        if dtype:
            node["weight_dtype"] = dtype
        return node

    def traffic(role, nbytes, tp=None):
        node = dict(norm(role), bytes_per_token=int(nbytes))
        if tp:
            node["tp"] = tp
        return node

    def attention_nodes(win):
        # The target always projects Q, K, V and the output gate in one fused linear that
        # slices by head group and cannot replicate a KV head (use_o_gate is True off the
        # drafter, model.py:977; MuseSpark1xQKVOGateParallelLinear asserts
        # total_num_kv_heads % tp_size == 0, common/layers.py:368-371@0f6b2619).
        attn = {"op": "Attention", "kind": "swa" if win else "gqa", "n_q": nq,
                "n_kv": nkv, "d_h": dh, "kv_heads_not_replicated": True}
        if win:
            attn["window"] = int(win)
        # Unquantized in both checkpoints: ModelOpt excludes self_attn, so the attention
        # GEMMs carry the base width explicitly where the global width is the base anyway.
        return [
            traffic("input_norm", _muse_boundary_bytes(hidden, act), tp="replicated"),
            gemm_at("qkv_gate_proj", 2 * (nq + nkv) * dh, hidden),
            # QK-norm and RoPE read and write Q and K once, fused with the cache insert,
            # whose record the attention's cache accounts for.
            traffic("qk_norm_rope", 2 * (nq + nkv) * dh * act),
            attn,
            # The per-head output norm and its sigmoid gate: read the output and the gate,
            # write the gated output.
            traffic("o_norm_gate", 3 * nq * dh * act),
            gemm_at("o_proj", hidden, nq * dh),
            collective("AllReduce", "attn_out", EMIT_TENSOR_PARALLEL),
        ]

    def moe_nodes(expert_dtype):
        node = {"op": "GroupedGEMM", "role": "experts", "n": inner, "k": hidden,
                "experts": experts, "top_k": top_k, "latent_size": latent}
        if expert_dtype and expert_dtype != base:
            node["weight_dtype"] = expert_dtype
        rows = top_k * latent * act
        return [
            traffic("post_attn_norm", _muse_boundary_bytes(hidden, act), tp="replicated"),
            gemm_at("router", experts, hidden, tp="replicated", dtype="fp32"),
            # The pooler: the catalog's name for the projection into a latent MoE. On SM100
            # and later with TP and no DP, a step of at most 256 tokens splits its output
            # columns over TP and all-gathers them (shard_pooler, shards_pooler,
            # model.py:151, 505-523, 635-636, 658-664@0f6b2619).
            dict(gemm_at("routed_expert_down_proj", latent, hidden, tp="replicated"),
                 sharded_at_decode=MUSE_SHARDED_PROJECTION),
            traffic("pre_expert_norm", 2 * latent * act, tp="replicated"),
            node,
            collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
            dict(collective("AllReduce", "moe_route_rows",
                            EMIT_TENSOR_PARALLEL_WITHOUT_EXPERT_PARALLEL),
                 bytes_per_token=rows),
            # The finalize: read each route's row, norm and weight it, and write the sum.
            # It accumulates in fp32 but stores its output buffer's dtype
            # (fused_finalize.py:142-143@0f6b2619), and that buffer is allocated like the
            # GEMM2 rows, at the activation width (_run_post_gemm2_hook,
            # model_executor/layers/fused_moe/modular_kernel.py:543-548@0f6b2619).
            traffic("post_expert_norm", rows + latent * act, tp="replicated"),
            dict(collective("AllReduce", "moe_combine", EMIT_EXPERT_PARALLEL_WITHOUT_DISPATCH),
                 bytes_per_token=latent * act),
            # The unpooler, back out of it; split the same way, from 32 tokens where the TP
            # group spans nodes (shards_unpooler, min_sharded_unpooler_tokens,
            # model.py:154, 535-539, 638-644, 932-936).
            dict(gemm_at("routed_expert_up_proj", hidden, latent, tp="replicated"),
                 sharded_at_decode=dict(MUSE_SHARDED_PROJECTION, min_tokens_across_nodes=32)),
        ]

    kinds, sequence, seen = [], [], set()
    for i, t in enumerate(attn_types):
        win = window if t == "sliding_attention" else None
        dtype = base if i in bf16_experts else expert_fmt
        kid = ("swa" if win else "full") + "_moe" + (
            "_" + base if expert_fmt and i in bf16_experts else "")
        sequence.append(kid)
        if kid in seen:
            continue
        seen.add(kid)
        nodes = attention_nodes(win) + moe_nodes(dtype)
        kinds.append({"id": kid, "nodes": nodes, "edges": chain(nodes)})

    embed = {"op": "Gather", "role": "embed_tokens", "rows": vocab,
             "row_bytes": hidden * act, "rows_per_token": 1,
             "out_bytes_per_token": hidden * act, "shard": "rows"}
    head = [traffic("final_norm", _muse_boundary_bytes(hidden, act), tp="replicated"),
            gemm_at("lm_head", vocab, hidden, tp="column")]
    extras = {
        "global": {
            "architecture": raw["architectures"][0],
            # max_position_embeddings with no RoPE scaling factor (rope_type default), so
            # vLLM's derived max_model_len is that value.
            "context_window": whole(require_key(cfg, "max_position_embeddings", model),
                                    "max_position_embeddings"),
            # What a pipeline stage hands the next: the fp32 residual stream alone, with
            # the residual folded into it and no separate residual tensor
            # (_make_empty_intermediate_tensors, model.py:1639-1650; the send carries
            # hidden_states and any captured aux states only, :1724-1731).
            "pipeline_boundary_bytes_per_token": hidden * fp32,
        },
        # A vocabulary-parallel lookup whose partials are all-reduced.
        "embedding": [embed, collective("AllReduce", "embed_out", EMIT_TENSOR_PARALLEL)],
        "head": head,
    }
    return kinds, compress(sequence), extras


def handler_muse_spark_dspark(cfg, raw, model):
    """Muse Spark's DSpark drafter, a checkpoint of its own (MuseSparkDSparkModel,
    vllm/models/muse_spark/common/dspark.py@0f6b2619), which a deployment names as its
    speculative draft_model.

    The graph is the drafter, and states itself as the speculator a kernel prices beside
    the target: num_hidden_layers dense Muse layers, each sliding-window only
    (global_attn_cfg, model.py:1101-1108), that draft block_size tokens in ONE pass;
    before it, the context stack over the target's scheduled tokens (the fc projection of
    the target's hidden states at target_layer_ids and its norm, then each layer's
    pre-norm and K/V projection and cache insert, _project_muse_context_kv and
    precompute_and_store_context_kv, dspark.py:205-250); after it, the final norm and the
    drafter's own lm_head (has_own_lm_head, dspark.py:281, 305-309); then the Markov head once
    per draft position (DSparkMarkovHead, qwen3_dspark.py:48-93), both of whose weights
    are replicated.

    NOT PRICED, each small: the drafter's token embedding, which it does not load and so
    reads from the target's table (load_weights drops model.embed_tokens, dspark.py:
    360-368); the confidence head, which adaptive verification alone consults; the
    output soft-cap. The context K/V projection reads a slice of each layer's
    qkv_gate_proj rows, which a separate node cannot alias at a different shape, so those
    rows' bytes are resident twice: 2 nkv d_h hidden per layer, 21 MiB a rank at TP8."""
    def whole(v, key):
        if isinstance(v, bool) or int(v) != v:
            raise DeriveError(f"{model}: {key} is {v!r}, not a whole number")
        return int(v)

    hidden = whole(require(cfg, "hidden_size", model), "hidden_size")
    layers = whole(require(cfg, "num_layers", model), "num_hidden_layers")
    nq = whole(require(cfg, "num_q_heads", model), "num_attention_heads")
    nkv = whole(require(cfg, "num_kv_heads", model), "num_key_value_heads")
    dh = head_dim(cfg, model)
    vocab = whole(require(cfg, "vocab_size", model), "vocab_size")
    # The draft MLP's width is intermediate_size_mlp, which vLLM's config computes from
    # the multiplier and does not read from the file (int(ffn_dim_multiplier *
    # hidden_size), configs/muse_spark.py:247@0f6b2619, default multiplier 1.0 at :116;
    # LlamaMLP(intermediate_size=config.intermediate_size_mlp), model.py:1360-1366).
    mult = cfg.get("ffn_dim_multiplier", 1.0)
    if isinstance(mult, bool) or not isinstance(mult, (int, float)) or mult <= 0:
        raise DeriveError(f"{model}: ffn_dim_multiplier is {mult!r}")
    ffn = int(mult * hidden)
    window = whole(require(cfg, "sliding_window", model), "sliding_window")
    block = whole(require_key(cfg, "block_size", model), "block_size")
    markov = whole(require_key(cfg, "markov_rank", model), "markov_rank")
    targets = require_key(cfg, "target_layer_ids", model)
    target_hidden = whole(require_key(cfg, "target_hidden_size", model), "target_hidden_size")
    target_layers = whole(require_key(cfg, "target_num_hidden_layers", model),
                          "target_num_hidden_layers")
    if (cfg.get("moe") or {}).get("is_enabled"):
        raise DeriveError(f"{model}: an MoE drafter; this handler prices the dense one")
    if not targets or any(not 0 <= int(t) < target_layers for t in targets):
        raise DeriveError(f"{model}: target_layer_ids {targets} outside the target's "
                          f"{target_layers} layers")
    # The capture points the engine resolves for this drafter. Its config states the ids
    # twice, in dflash_config and at the top level, and vLLM reads dflash_config first and
    # adds one: target_layer_ids 13, 29, 45, 57, 61 are captured as 14, 30, 46, 58, 62, the
    # last being the final layer's output (eagle3_utils.py:107-111@v0.32.0). The drafter
    # concatenates one state per top-level id into its fc (dspark.py:106, 125@0f6b2619), so
    # the engine must capture exactly that many.
    aux_layers = draft_aux_hidden_state_layers(cfg, model, target_layers,
                                               states=len(targets))
    if set(cfg.get("layer_types") or ["sliding_attention"]) != {"sliding_attention"}:
        raise DeriveError(f"{model}: a full-attention draft layer")
    base = base_dtype(cfg, raw, model)
    act = DTYPE_WIDTHS[base]
    fp32 = DTYPE_WIDTHS["fp32"]

    def traffic(role, nbytes, tp=None):
        node = dict(norm(role), bytes_per_token=int(nbytes))
        if tp:
            node["tp"] = tp
        return node

    def with_tp(node, tp):
        node["tp"] = tp
        return node

    # A draft layer's attention projects through the fused [Q, K, V, O-gate] linear when its
    # configuration turns the output gate on (use_o_gate = config.use_attn_o_gate on the
    # drafter, model.py:977@0f6b2619), which cannot replicate a KV head across ranks
    # (MuseSpark1xQKVOGateParallelLinear, common/layers.py:368-371); the qkv_gate_proj and
    # o_norm_gate nodes below price that gate, so the handler requires it.
    if not cfg.get("use_attn_o_gate", False):
        raise DeriveError(f"{model}: use_attn_o_gate is off, so the draft layers run the plain "
                          f"QKV projection and no output gate, which this handler does not price")
    attention = {"op": "Attention", "kind": "swa", "n_q": nq, "n_kv": nkv, "d_h": dh,
                 "window": window, "kv_heads_not_replicated": True}

    layer = [
        traffic("input_norm", _muse_boundary_bytes(hidden, act), tp="replicated"),
        gemm("qkv_gate_proj", 2 * (nq + nkv) * dh, hidden),
        traffic("qk_norm_rope", 2 * (nq + nkv) * dh * act),
        attention,
        traffic("o_norm_gate", 3 * nq * dh * act),
        gemm("o_proj", hidden, nq * dh),
        collective("AllReduce", "attn_out", EMIT_TENSOR_PARALLEL),
        traffic("post_attn_norm", _muse_boundary_bytes(hidden, act), tp="replicated"),
        gemm("mlp_gate_up", 2 * ffn, hidden),
        gemm("mlp_down", hidden, ffn),
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL),
    ]
    context = [
        with_tp(gemm("dspark_fc", hidden, target_hidden * len(targets)), "replicated"),
        traffic("dspark_hidden_norm", 2 * hidden * act, tp="replicated"),
    ]
    for d in range(layers):
        context += [
            traffic(f"dspark_context_norm_{d}", 2 * hidden * act, tp="replicated"),
            gemm(f"dspark_context_kv_{d}", 2 * nkv * dh, hidden),
            # The K norm, RoPE and the cache insert over the projected K and V.
            traffic(f"dspark_context_kv_insert_{d}", 2 * 2 * nkv * dh * act),
        ]
    kinds = [{"id": "dspark_layer", "nodes": layer, "edges": chain(layer)},
             {"id": "dspark_context", "nodes": context, "edges": chain(context)}]
    head = [traffic("final_norm", hidden * fp32 + hidden * act, tp="replicated"),
            with_tp(gemm("lm_head", vocab, hidden), "column")]
    extras = {
        "global": {"architecture": raw["architectures"][0]},
        "head": head,
        "speculator": {
            "method": "dspark",
            "num_spec_tokens": block,
            "drafting": "parallel_block",
            "stack": {"pattern": ["dspark_layer"], "repeat": layers},
            "context_stack": {"pattern": ["dspark_context"], "repeat": 1},
            "head": [traffic("dspark_final_norm", hidden * fp32 + hidden * act,
                             tp="replicated"),
                     {**with_tp(gemm("dspark_lm_head", vocab, hidden), "column"),
                      "aliases": "head.lm_head"}],
            "sequential_head": [
                {"op": "Gather", "role": "markov_w1", "rows": vocab,
                 "row_bytes": markov * act, "rows_per_token": 1,
                 "out_bytes_per_token": markov * act},
                with_tp(gemm("markov_w2", vocab, markov), "replicated"),
            ],
            "aux_hidden_state_layers": aux_layers,
            # Each captured state is the target's fp32 residual stream, not one bf16
            # activation: the target keeps the stream in fp32 (residual_dtype,
            # model.py:1531@0f6b2619) and, at a capture point, appends either the stream
            # itself or norm_then_gated_residual of it (model.py:1713-1722), which returns
            # the stream's dtype (common/layers.py:276-286, 299-322; the fused kernel
            # allocates its output like the stream, common/ops/fused_rmsnorm.py:295-296).
            "aux_hidden_state_bytes_per_token": target_hidden * fp32,
        },
    }
    return kinds, {"pattern": ["dspark_layer"], "repeat": layers}, extras


HANDLERS = {
    "LlamaForCausalLM": handler_dense,
    "MistralForCausalLM": handler_dense,
    "Qwen2ForCausalLM": handler_dense,
    "Qwen3ForCausalLM": handler_dense,
    "MixtralForCausalLM": handler_moe,
    "Qwen3MoeForCausalLM": handler_moe,
    "Llama4ForConditionalGeneration": handler_moe,
    "GraniteMoeForCausalLM": handler_moe,
    "DeepseekV2ForCausalLM": handler_moe,
    # V3 is the same shape family as V2: MLA attention keyed by kv_lora_rank, a routed
    # expert layer, and a dense prologue named by first_k_dense_replace. Every difference
    # between the committed V2-Lite config and V3 is a VALUE -- depth, hidden size, expert
    # count, scoring function -- not a structure, so the handler prices it with no new node
    # kinds. Registered explicitly rather than by a prefix match, because a default is what
    # this registry exists to prevent.
    #
    # DeepseekV4ForCausalLM has its own handler rather than this one. It declares no
    # kv_lora_rank and states an inclusive head_dim, so the MLA branch here cannot
    # reconstruct its per-token width; and its sliding_window field is 128 where the
    # read is actually bounded by index_topk over a compressed stream, so this handler
    # would derive kind=swa window=128 and misprice every layer. See
    # handler_deepseek_v4, which reads the bound from compress_ratios and index_topk and
    # gives the indexer its own node.
    "DeepseekV3ForCausalLM": handler_moe,
    "DeepseekV4ForCausalLM": handler_deepseek_v4,
    # Its own handler rather than V4's: V4.1 shares the compressed cache and the indexer
    # across layers, pools at ratios 1 and 2 where V4 asserts 4 or 128, and adds engram
    # layers and a DSpark draft module. See handler_deepseek_v41.
    "DeepseekV41ForCausalLM": handler_deepseek_v41,
    "GlmMoeDsaForCausalLM": handler_moe,
    "InklingForConditionalGeneration": handler_moe,
    "GptOssForCausalLM": handler_moe,
    "MiniMaxM2ForCausalLM": handler_minimax_m2,
    "MiniMaxM3SparseForCausalLM": handler_minimax_m3,
    "MiniMaxM3SparseForConditionalGeneration": handler_minimax_m3,
    "NemotronHForCausalLM": handler_nemotron_h,
    "Qwen3_5MoeForConditionalGeneration": handler_qwen3_5_moe,
    "KimiK3ForConditionalGeneration": handler_kimi_k3,
    "Glm5NextForConditionalGeneration": handler_glm5_next,
    # Muse Spark's language model is the text decoder of a multimodal checkpoint; the
    # vision and audio towers are not priced (modality text_decoder_of_multimodal).
    "MuseSparkForConditionalGeneration": handler_muse_spark,
    "MuseSparkDSparkModel": handler_muse_spark_dspark,
    # Kimi-K2.5 is a vision-language model whose TEXT decoder vLLM instantiates as
    # DeepseekV2ForCausalLM over config.text_config
    # (model_executor/models/kimi_k25.py:371-376, architectures=["DeepseekV2ForCausalLM"]),
    # and whose text_config names DeepseekV3ForCausalLM itself. Either way the shape is the
    # DeepSeek family handler_moe already prices: MLA with kv_lora_rank 512, routed experts,
    # a dense prologue from first_k_dense_replace. The vision tower is not priced, which the
    # graph records as modality text_decoder_of_multimodal; the corpus's workloads are all
    # text (1024:1024, 1024:8192, 8192:1024), so the tower never runs in these measurements.
    "KimiK25ForConditionalGeneration": handler_moe,
}


# --- Driver ---------------------------------------------------------------------

# What a handler's optional third result may carry; see derive().
EXTRAS_KEYS = frozenset({"global", "embedding", "head", "speculator",
                         "base_capability_families", "arch_overrides"})
# The global shapes a handler may add; the rest derive() computes itself.
EXTRA_GLOBAL_KEYS = frozenset({"architecture", "weight_scale", "context_window",
                               "default_context_window",
                               "pipeline_boundary_bytes_per_token"})


def derive(config_path: Path, model: str) -> dict[str, Any]:
    """Derive one graph from one vendor config."""
    source = config_path.read_bytes()
    raw = json.loads(source)
    cfg = text_config(raw)

    arches = raw.get("architectures") or cfg.get("architectures") or []
    if not arches:
        raise DeriveError(f"{model}: config declares no architectures")
    arch = arches[0]
    handler = HANDLERS.get(arch)
    if handler is None:
        raise DeriveError(
            f"{model}: no handler for architecture {arch!r}; add one rather than "
            f"letting a default misprice the model"
        )

    # A handler returns (kinds, stack), or (kinds, stack, extras) where its family needs
    # what derive() cannot work out from a config alone: global shapes beyond the ones
    # below, its own embedding or head, a speculator whose structure no fallback below
    # reproduces, or the facts that differ by GPU architecture.
    result = handler(cfg, raw, model)
    if len(result) not in (2, 3):
        raise DeriveError(f"{model}: handler returned {len(result)} results, not 2 or 3")
    kinds, stack = result[0], result[1]
    extras = result[2] if len(result) == 3 else {}
    # A misspelled key would be ignored and the default it exists to replace would
    # stand, which is the mispricing the key was added to prevent.
    unknown = set(extras) - EXTRAS_KEYS
    if unknown:
        raise DeriveError(f"{model}: handler extras name {sorted(unknown)}, which "
                          f"derive() does not read")

    graph: dict[str, Any] = {
        "kind": "ModelGraph",
        "name": model,
        "derived_from": {
            "format": "hf_config_json",
            "path": "config.json",
            "sha256": hashlib.sha256(source).hexdigest(),
            "deriver_version": DERIVER_VERSION,
        },
        "global": {
            "hidden_size": int(require(cfg, "hidden_size", model)),
            "vocab_size": int(require(cfg, "vocab_size", model)),
            "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
            "weight_dtype": weight_dtype(cfg, raw, model),
        },
        "layer_kinds": kinds,
        "stack": stack,
    }

    # A config that nests its text shapes beside another tower describes more than
    # this graph prices, and the graph says so rather than leaving a reader to find out
    # from a prediction that is low.
    if "text_config" in raw:
        graph["modality"] = "text_decoder_of_multimodal"

    extra_global = extras.get("global") or {}
    clash = set(extra_global) - EXTRA_GLOBAL_KEYS
    if clash:
        raise DeriveError(f"{model}: handler restates global {sorted(clash)}, which "
                          f"derive() computes itself")
    graph["global"].update(extra_global)
    scale = int_group_scale(cfg, raw, model)
    if scale and graph["global"]["weight_dtype"] in ("int4", "int8"):
        if graph["global"].get("weight_scale") not in (None, scale):
            raise DeriveError(f"{model}: the handler and the quantization config state "
                              f"different weight scales")
        graph["global"]["weight_scale"] = scale
    if "embedding" in extras:
        graph["embedding"] = extras["embedding"]

    hidden = graph["global"]["hidden_size"]
    lm_head = gemm("lm_head", graph["global"]["vocab_size"], hidden)
    # An fp8 checkpoint that excludes lm_head from quantization stores it at the base
    # width, and six committed configs do (glm-5, glm-5.2-fp8, glm-5.3, glm-5.3-flash,
    # minimax-m2.5, minimax-m2.7). Applied here only for the handler that asks for it,
    # via its unquantized-role map, rather than globally: widening it for all six is a
    # correct change but re-derives five other graphs, which belongs in the pass that
    # takes the whole exclusion list rather than in this model's entry. Each of those
    # five understates the head by vocab x hidden bytes until then.
    head_widths = fp8_unquantized_roles(cfg, raw, model,
                                        HEAD_UNQUANTIZED.get(arch, {}))
    if head_widths.get("lm_head"):
        lm_head["weight_dtype"] = head_widths["lm_head"]
    if "head" in extras:
        # A handler that states its own head still prices one LM head of vocab x
        # hidden, and at the width the exclusion list gives where it gives one: two
        # sources for one width must agree rather than the later silently winning.
        heads = [n for n in extras["head"] if n.get("role") == "lm_head"]
        if len(heads) != 1 or (heads[0].get("n"), heads[0].get("k")) != (
                graph["global"]["vocab_size"], hidden):
            raise DeriveError(f"{model}: handler head does not state one "
                              f"{graph['global']['vocab_size']} x {hidden} lm_head")
        if head_widths.get("lm_head") not in (None, heads[0].get("weight_dtype")):
            raise DeriveError(
                f"{model}: the fp8 exclusion list gives lm_head {head_widths['lm_head']} "
                f"and the handler {heads[0].get('weight_dtype')}")
        graph["head"] = extras["head"]
    else:
        graph["head"] = [norm("final_norm"), lm_head]

    if ("arch_overrides" in extras) != ("base_capability_families" in extras):
        raise DeriveError(f"{model}: arch_overrides and base_capability_families are "
                          f"stated together or not at all")
    for key in ("base_capability_families", "arch_overrides"):
        if key in extras:
            graph[key] = extras[key]

    if "speculator" in extras:
        # The handler priced the draft module itself, so the fallbacks below do not
        # apply. It must still describe one: tokens to draft, through layer kinds the
        # handler defined.
        spec_extra = extras["speculator"]
        defined = {k["id"] for k in kinds}
        pattern = spec_extra["stack"].get("pattern") or []
        if (int(spec_extra["num_spec_tokens"]) < 1 or not pattern
                or not set(pattern) <= defined):
            raise DeriveError(
                f"{model}: handler speculator drafts {spec_extra['num_spec_tokens']} "
                f"tokens through {pattern}, which is not a draft module")
        graph["speculator"] = spec_extra
        return graph

    spec = pick(cfg, "num_spec_tokens")
    # num_mtp_modules is NOT in the alias table: it is not a cross-family spelling of
    # "draft layers". vLLM reads it for n_predict only on the families whose speculative
    # method consumes it (config/speculative.py:958-984 for MiniMax-M3), and
    # MiniMax-M2.5 declares num_mtp_modules 3 while vLLM registers no minimax_m2_mtp
    # method at all -- its only draft path is a separate Eagle3 checkpoint. Reading the
    # field globally would invent a draft stack for every such config.
    if arch in MTP_MODULE_ARCHS:
        modules = cfg.get("num_mtp_modules")
        if modules is None:
            raise DeriveError(
                f"{model}: {arch} takes its draft length from num_mtp_modules and the "
                f"config declares none"
            )
        spec = modules
    if spec and int(spec) > 0:
        # A draft module is its own stack: for an MoE target the draft pass is a second
        # MoE, which a scalar draft length would hide. Its layer composition comes from
        # three sources, in order of how much the config actually says.
        defined = {k["id"] for k in kinds}
        mtp_vector = cfg.get("mtp_layers_block_type")
        if DRAFT_KIND_ID in defined:
            # A handler that priced the draft module itself, because the family gives it
            # a structure no target layer shares. GLM-5.3-Flash is a case in hand
            # (handler_glm5_next): its draft layer is sparse-MLA with its own
            # embedding-mixing prefix, a copy of no target layer. DeepSeek-V4 and V4.1
            # state their whole speculator in the handler's extras instead.
            pattern = [DRAFT_KIND_ID]
        elif mtp_vector:
            pattern = [n if n in defined else kinds[-1]["id"] for n in mtp_vector]
        else:
            # The fallback: mirror the last kind the handler DECLARED. That is insertion
            # order, not position in the layer sequence -- the two coincide for most
            # families, but not all, and where they differ this reads the declaration.
            # GLM-5.3-Flash is the case in hand: its last declared kind is mla_moe and
            # its last base layer is kda_moe, and the draft module really is sparse-MLA,
            # so the right answer here is reached by the order the handler happens to
            # declare its kinds in rather than by anything about the module.
            #
            # Sound only where the draft module really does repeat a declared layer's
            # structure, which is why a handler whose family differs declares
            # DRAFT_KIND_ID instead. A handler that cannot say that of its last
            # declaration should declare DRAFT_KIND_ID rather than rely on this.
            pattern = [kinds[-1]["id"]]
        graph["speculator"] = {
            "method": speculative_method(arch, model),
            "num_spec_tokens": int(spec),
            "stack": {"pattern": pattern, "repeat": 1},
        }
    return graph


# The engine's speculative method name per architecture. It is a vLLM vocabulary
# rather than a config field, so it is recorded here per family rather than guessed
# from the model name.
SPEC_METHODS = {
    # glm_moe_dsa is rewritten to deepseek_mtp, NOT to glm4_moe_mtp. vLLM lists it
    # alongside deepseek_v3 and deepseek_v32 in one branch -- `if hf_config.model_type in
    # ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"): hf_config.model_type =
    # "deepseek_mtp"` (config/speculative.py:678-685) -- and all four committed GLM DSA
    # configs declare model_type glm_moe_dsa. glm4_moe_mtp is set on a different branch
    # (:780) for a different family, so naming it here priced the wrong draft method.
    # The comment below already stated the correct mapping while this entry contradicted
    # it, which is how the error survived.
    "GlmMoeDsaForCausalLM": "deepseek_mtp",
    "KimiK3ForConditionalGeneration": "kimi_k3_mtp",
    "NemotronHForCausalLM": "nemotron_h_mtp",
    "InklingForConditionalGeneration": "inkling_mtp",
    "DeepseekV2ForCausalLM": "deepseek_mtp",
    # Verified in vLLM's own resolution rather than inferred from the family name:
    # config/speculative.py rewrites model_type deepseek_v3 to deepseek_mtp alongside
    # deepseek_v32 and glm_moe_dsa (architectures DeepSeekMTPModel). The method names
    # "deepseek_mtp" here and above are v0.31.0's deprecated aliases, which it accepts
    # and rewrites to "mtp" (:1135-1139); DeepSeek-V4 states "mtp" itself.
    "DeepseekV3ForCausalLM": "deepseek_mtp",
    # vLLM v0.31.0 rewrites the draft config's model_type deepseek_v4 to deepseek_mtp
    # and reads n_predict from num_nextn_predict_layers, NOT from num_mtp_modules the
    # way MiniMax-M3 does (config/speculative.py:696-707, architectures
    # DeepSeekV4MTPModel), so this architecture needs no entry in MTP_MODULE_ARCHS. The
    # METHOD it serves is "mtp", which every MTP-family name is rewritten to
    # (:1135-1139). handler_deepseek_v4 states its own speculator; this entry is the
    # fallback's.
    "DeepseekV4ForCausalLM": "mtp",
    "Qwen3_5MoeForConditionalGeneration": "qwen3_5_mtp",
    # vLLM rewrites model_type glm5_next to glm5_next_mtp and reads n_predict from
    # get_text_config().num_nextn_predict_layers, architectures ["Glm5NextMTPModel"]
    # (config/speculative.py:1060-1065 on vllm-project/vllm main, with
    # "glm5_next_mtp" in its valid-method list at line 66). Read from upstream rather
    # than inferred from the glm4_moe_mtp entry above: the two are distinct methods and
    # this family is not the GLM-4 one. The alias table already resolves
    # num_nextn_predict_layers, so this needs no MTP_MODULE_ARCHS entry.
    "Glm5NextForConditionalGeneration": "glm5_next_mtp",
    # vLLM resolves both the VL wrapper and the text decoder to one method, reading
    # n_predict from num_mtp_modules rather than num_nextn_predict_layers
    # (config/speculative.py:958-974, and "minimax_m3_mtp" in its valid-method list).
    "MiniMaxM3SparseForConditionalGeneration": "minimax_m3_mtp",
    "MiniMaxM3SparseForCausalLM": "minimax_m3_mtp",
}


# Architectures whose speculative method takes its draft length from num_mtp_modules
# rather than from num_nextn_predict_layers. Named rather than pattern-matched: the two
# fields disagree on MiniMax-M3 (7 against 1), so which one is read changes the graph.
MTP_MODULE_ARCHS = frozenset({
    "MiniMaxM3SparseForConditionalGeneration",
    "MiniMaxM3SparseForCausalLM",
})


def speculative_method(arch: str, model: str) -> str:
    method = SPEC_METHODS.get(arch)
    if method is None:
        raise DeriveError(
            f"{model}: {arch} declares draft layers but no speculative method is "
            f"recorded for it"
        )
    return method


def dump(graph: dict[str, Any]) -> str:
    """Render a graph deterministically, so a re-derivation diffs cleanly."""
    return yaml.safe_dump(graph, sort_keys=False, default_flow_style=False, width=100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=None)
    ap.add_argument("--check", action="store_true",
                    help="re-derive and report differences without writing")
    ap.add_argument("--model", default=None, help="derive one model only")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    if not models_dir.is_dir():
        print(f"{models_dir}: not a directory", file=sys.stderr)
        return 2

    written, unchanged, stale, failed = [], [], [], []
    for entry in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if args.model and entry.name != args.model:
            continue
        cfg = entry / "config.json"
        # Each entry is the full line to print. A DeriveError already names the model,
        # because derive() is also called directly and its message has to stand alone
        # there; the other failures do not, so only they get the prefix.
        if not cfg.is_file():
            failed.append(f"{entry.name}: no config.json")
            continue
        try:
            graph = derive(cfg, entry.name)
        except DeriveError as exc:
            why = str(exc)
            # Every message in this file starts with the model name; one that does not
            # still gets it, so no failure line is ever anonymous.
            failed.append(why if why.startswith(f"{entry.name}:")
                          else f"{entry.name}: {why}")
            continue
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            # Any other exception is a deriver bug or a config it did not anticipate.
            # It is recorded against this model like any failure, so one bad entry
            # neither hides the failures already collected nor goes unattributed; the
            # run still exits non-zero.
            failed.append(f"{entry.name}: {type(exc).__name__}: {exc}")
            continue

        text = dump(graph)
        out = entry / "graph.yaml"
        if args.check:
            if not out.is_file():
                stale.append((entry.name, "graph.yaml is missing"))
            elif out.read_text() != text:
                stale.append((entry.name, "graph.yaml differs from a re-derivation"))
            else:
                unchanged.append(entry.name)
            continue
        if out.is_file() and out.read_text() == text:
            unchanged.append(entry.name)
        else:
            out.write_text(text)
            written.append(entry.name)

    for line in failed:
        print(line, file=sys.stderr)
    for name, why in stale:
        print(f"{name}: {why}", file=sys.stderr)

    if args.check:
        print(f"checked {len(unchanged) + len(stale)} model(s): "
              f"{len(unchanged)} current, {len(stale)} stale, {len(failed)} failed")
    else:
        print(f"derived {len(written) + len(unchanged)} graph(s): "
              f"{len(written)} written, {len(unchanged)} unchanged, {len(failed)} failed")
    return 1 if (failed or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
