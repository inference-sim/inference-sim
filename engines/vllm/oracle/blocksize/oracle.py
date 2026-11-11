# Original to blis-vllm.
"""Block-size oracle: what vLLM v0.32.0 resolves block_size to when a deployment states none.

For each combination of model, GPU compute capability and KV dtype, this builds the inputs each
attention layer hands vLLM's selector (the values come from the model's code, cited in LAYERS),
and runs vLLM's own selection and block-size code on a CPU:

  CudaPlatform.get_valid_backends       platforms/cuda.py:375-415   (priority list :83-180)
  Platform._preferred_block_size_for_backends
                                        platforms/interface.py:699-745
  update_block_size_for_backend phase 1 platforms/interface.py:757-781

Only the GPU, the compiled wheels and the facts a CPU cannot discover are replaced (stubs.py,
design FID-1). Each combination is run under every setting of those facts. It RESOLVES if the
size is the same under all of them, and is REFUSED, naming the fact, if it differs or if vLLM
finds no valid backend. Where ASSUMED_FACTS states a fact for a compute capability, with its
evidence, the size must be the same only over the settings that agree with it, and the
combination records the assumption it rests on (assumed_facts).

The model's hybrid and heterogeneous-dtype phases (interface.py:783-790 and :791-796) do not
apply to the models here: none is a hybrid model, and none sets kv_cache_dtype_skip_layers.

Usage:
    python3 -I oracle/blocksize/oracle.py <vllm checkout> table     print the table
    python3 -I oracle/blocksize/oracle.py <vllm checkout> json      print it as JSON (the Go test's input)
    python3 -I oracle/blocksize/oracle.py <vllm checkout> check <resolve.json> [TABLE.tsv]
                                                                  fail unless the committed table, and the readable
                                                                  TABLE.tsv beside it when named, equal what it prints
"""
import itertools
import json
import os
import sys
import types
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# A layer of a model: what get_attn_backend receives (v1/attention/selector.py:105-120).
#   head_size, num_heads, has_sink, has_sliding_window, use_mla, use_sparse
# cited at the model code that builds the layer, at 5df63a7203.
LAYERS = {
    # vllm/model_executor/models/gpt_oss.py: Attention(num_local_attention_heads, head_dim, ...,
    # per_layer_sliding_window=config.sliding_window on even layers, sinks=self.sinks) :176-188.
    # head_dim 64 and 64 query heads: config.json (catalog/models/gpt-oss-120b).
    "gpt-oss-120b": [
        dict(head_size=64, num_heads=64, num_kv_heads=8, has_sink=True, has_sliding_window=True),
        dict(head_size=64, num_heads=64, num_kv_heads=8, has_sink=True, has_sliding_window=False),
    ],
    # vllm/model_executor/models/minimax_m2.py:204-213 Attention(num_heads, head_dim, ...,
    # per_layer_sliding_window=attn_window_size); the layer is built at :270-282 with no window.
    "minimax-m2.5": [
        dict(head_size=128, num_heads=48, num_kv_heads=8, has_sink=False, has_sliding_window=False),
    ],
    # MiniMax-M2.5-NVFP4 (nvidia/MiniMax-M2.5-NVFP4) is the same class, MiniMaxM2ForCausalLM, with
    # the same attention shape: its config.json, fetched at the catalog's pinned revision and
    # checked against the catalog's SHA-256 (modelconfig.py), states 48 heads, 8 KV heads, head_dim
    # 128 and no sliding_window, layer_types or attn_window_size, as minimax-m2.5's does. Its
    # quantization_config excludes every model.layers.N.self_attn* module and declares
    # kv_cache_scheme {type: float, num_bits: 8}, which vLLM reads as kv_cache_quant_algo FP8
    # (modelopt.py:344-352), so the KV cache may be fp8 where the deployment states auto. The
    # selector's inputs are the layer's shape, bfloat16 and the KV dtype name, and the table runs
    # every KV dtype name: it is the same size under auto and fp8, so the checkpoint's KV scheme
    # changes no size.
    "minimax-m2.5-nvfp4": [
        dict(head_size=128, num_heads=48, num_kv_heads=8, has_sink=False, has_sliding_window=False),
    ],
    # MiniMax-M2.7 is the same class with the same attention shape (config.json: 48 heads, 8 KV
    # heads, head_dim 128, no sliding window).
    "minimax-m2.7": [
        dict(head_size=128, num_heads=48, num_kv_heads=8, has_sink=False, has_sliding_window=False),
    ],
    # vllm/model_executor/models/llama.py:158-219: head_dim = hidden_size // heads (8192 // 64 =
    # 128); config.json has no layer_types, so sliding_window stays None (:185-204).
    "llama-3.1-70b-instruct": [
        dict(head_size=128, num_heads=64, num_kv_heads=8, has_sink=False, has_sliding_window=False),
    ],
    # vllm/model_executor/layers/attention/mla_attention.py:479, 513-520:
    # head_size = kv_lora_rank + qk_rope_head_dim; get_attn_backend(..., use_mla=True,
    # use_sparse=False, num_heads=num_heads). DeepseekV3: 128 heads, kimi-k2.5: 64 (config.json).
    "deepseek-v3": [dict(head_size=576, num_heads=128, num_kv_heads=1, use_mla=True)],
    "kimi-k2.5": [dict(head_size=576, num_heads=64, num_kv_heads=1, use_mla=True)],
}

# Models not resolved, with the reason. A fixture of one of these keeps its block_size refused.
NOT_RESOLVED = {
    # vllm/models/minimax_m3/nvidia/model.py:392-567: layers listed in sparse_attention_config
    # select MiniMaxM3SparseBackend by their own get_attn_backend (:559-567), not the selector;
    # the dense layers use Attention (:356). Two backends share one block pool, and the sparse
    # backend's block size is 128 (common/sparse_attention.py:163-165). The oracle runs the
    # dense layers' selection only, so it cannot say which layers are sparse.
    "minimax-m3": "its sparse-attention layers pick MiniMaxM3SparseBackend directly "
                  "(vllm/models/minimax_m3/nvidia/model.py:559), outside the selector",
    "deepseek-v4-pro": "DeepSeek-V4 layers use their own sparse MLA backends and block sizes "
                       "(vllm/models/deepseek_v4), not the selector alone",
    "glm-5": "its sparse MLA layers need the indexer backends' sizes (use_sparse); not modelled here",
    "qwen3.5-397b-a17b": "a hybrid model: its block size is aligned to the recurrent page "
                         "(platforms/interface.py:783-790, _align_hybrid_block_size)",
}

# The catalog's model directories, whose config.json files are the models' own. Set by
# BLIS_ORACLE_CATALOG. A derived-only model (its licence is not redistributable) has no config.json
# there: its file is fetched from the source and revision its model.yaml pins and checked against
# the SHA-256 recorded there (modelconfig.py), so that model needs network access to huggingface.co.
CATALOG = os.environ.get("BLIS_ORACLE_CATALOG", "")

# catalog/hardware/<gpu>.yaml ComputeCapability.
COMPUTE_CAPABILITY = {"h100": (9, 0), "h200": (9, 0), "b200": (10, 0), "b300": (10, 3),
                      "gb200-nvl72": (10, 0), "gb300": (10, 3)}
# fp8_e4m3, bfloat16 and float16 are names a deployment may state for the same KV cache as fp8 and
# auto (CacheDType, vllm/config/cache.py:39-59). Backends test the name against their own
# supported_kv_cache_dtypes (v1/attention/backend.py:110-116), and a few list one name of a pair and
# not the other, so the oracle runs each name and the table says which resolve alike, rather than Go
# assuming an alias (blis-vllm#68).
KV_DTYPES = ["auto", "fp8", "fp8_e4m3", "bfloat16", "float16"]
# Facts the oracle assumes for a compute capability, with the evidence for each. A CPU cannot
# discover a fact, and a combination whose size depends on one is refused; an assumption lets it
# resolve over the settings that agree with the assumption. Every combination is still run under all
# 16 settings (by_facts), so the table shows what each setting gives, and a combination resolves by
# assumption only where the settings that agree with it all give one size. The assumption is not
# derived from the pinned tree; whoever runs vLLM from a build that lacks it is resolved wrongly.
# {(major, minor): {fact: (value, evidence)}}.
ASSUMED_FACTS = {
    (9, 0): {
        "fa3_available": (True, (
            "vLLM's official CUDA release images ship FlashAttention 3 on SM9x: the pinned docker/Dockerfile:273 builds "
            "for arch 9.0 (list '7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0') with CUDA 13.0.3 (:25), CMakeLists.txt:1576-1585 "
            "includes the vllm_flash_attn project on every CUDA build, and FA3_AVAILABLE is true when _vllm_fa3_C imports "
            "(vllm_flash_attn/flash_attn_interface.py:20-28, :67-75). The condition that builds _vllm_fa3_C is in the "
            "external flash-attention repository, which is not in the pinned tree, so this is an assumption about the "
            "images, not derived from the tree. The one recorded Kimi-K2.5 h200 run (vllm/vllm-openai:v0.22.0, KV auto, "
            "block_size 16; compat/kernelmodel/testdata/recorded_block_sizes) is consistent with it, and rules out only "
            "fa3_available=0 with flashmla_built=1 (size 64)")),
    },
}
# The tensor-parallel widths each combination is run at. The per-rank head counts shrink with the
# width (num_heads // tp), and a backend may accept a head count at one width and not at another,
# so a combination resolves only if the size is the same at all of them. The table is valid only
# for a width up to the largest of these; core/vllm.MaxBlockSizeTP is that bound, and
# ResolveBlockSize refuses a wider pool.
TP_WIDTHS = list(range(1, 17))


def load_vllm(vllm_dir):
    warnings.simplefilter("ignore")
    sys.path.insert(0, vllm_dir)
    import stubs

    stubs.install_early()
    import torch
    import vllm.platforms as P
    from vllm.platforms.cuda import CudaPlatform
    from vllm.platforms.interface import DeviceCapability

    class Sim(CudaPlatform):
        """vLLM's CUDA platform with the device capability supplied (platforms/cuda.py)."""

        cap = (9, 0)

        @classmethod
        def get_device_capability(cls, device_id=0):
            return DeviceCapability(*cls.cap)

    P._current_platform = Sim()
    stubs.install_late()
    stubs.patch_trtllm()
    from vllm.config import VllmConfig, set_current_vllm_config
    import vllm.v1.attention.backends.flashinfer as fb
    import vllm.utils.flashinfer as fi

    fb.can_use_trtllm_attention = fi.can_use_trtllm_attention
    fb.supports_trtllm_attention = fi.supports_trtllm_attention
    stubs.patch_flashmla()
    return types.SimpleNamespace(torch=torch, Sim=Sim, DeviceCapability=DeviceCapability, VllmConfig=VllmConfig,
                                 set_current_vllm_config=set_current_vllm_config, stubs=stubs)


def hf_text_config(model):
    """The model's text config, as the backends' supports_combination read it (for example
    mla/flashinfer_mla.py:240-248, mla/tokenspeed_mla.py:157-166). It is the model's own
    config.json, from the catalog or, for a derived-only model, fetched at the pinned revision and
    verified by hash; an attribute the file does not state is absent, as it is
    in vLLM's PretrainedConfig."""
    import modelconfig
    c = json.loads(modelconfig.config_bytes(model, os.path.join(CATALOG, model)))
    return types.SimpleNamespace(**c.get("text_config", c))


def resolve_once(v, model, cap, kv_dtype, facts, tp=1):
    """vLLM's block size for one model on one capability under one setting of the facts, or
    (None, reason) when vLLM finds no valid backend for a layer."""
    from vllm.v1.attention.selector import AttentionSelectorConfig

    v.stubs.FACTS.update(facts)
    v.Sim.cap = cap
    vc = v.VllmConfig()
    # Each layer is built with the per-rank head counts (gpt_oss.py:176-188 num_local_attention_heads,
    # mla_attention.py:513-520 num_local_heads); a KV head is replicated when tp exceeds them.
    layers = [dict(L, num_heads=max(1, L["num_heads"] // tp), num_kv_heads=max(1, L["num_kv_heads"] // tp))
              for L in LAYERS[model]]
    # What the layer's backend consults from the config (flashinfer.py:465-473): the head counts.
    vc.model_config = types.SimpleNamespace(
        get_num_attention_heads=lambda pc: layers[0]["num_heads"],
        get_num_kv_heads=lambda pc: layers[0]["num_kv_heads"],
        get_head_size=lambda: layers[0]["head_size"],
        hf_text_config=hf_text_config(model),
        is_mm_prefix_lm=False, is_diffusion=False, is_hybrid=False,
    )
    # Phase 1 runs only when block_size is not user-specified (interface.py:757-781).
    backends = []
    names = []
    with v.set_current_vllm_config(vc):
        for L in layers:
            cfg = AttentionSelectorConfig(
                head_size=L["head_size"], dtype=v.torch.bfloat16, kv_cache_dtype=kv_dtype, block_size=None,
                use_mla=L.get("use_mla", False), has_sink=L.get("has_sink", False),
                use_sparse=L.get("use_sparse", False), has_sliding_window=L.get("has_sliding_window", False))
            valid, invalid = v.Sim.get_valid_backends(v.DeviceCapability(*cap), cfg, num_heads=L["num_heads"])
            if not valid:
                reasons = {k.name: r[1] for k, r in invalid.items()}
                return None, f"no valid backend for a layer ({reasons})"
            sel = min(valid, key=lambda c: c.priority)
            if sel.backend_class not in backends:
                backends.append(sel.backend_class)
                names.append(sel.backend.name)
        size = v.Sim._preferred_block_size_for_backends(backends, 16, vc)
    return (size, "/".join(names)), None


def run(vllm_dir):
    v = load_vllm(vllm_dir)
    fact_names = sorted(v.stubs.FACTS)
    rows = []
    caps = sorted(set(COMPUTE_CAPABILITY.values()))
    for model, cap, kv in itertools.product(sorted(LAYERS), caps, KV_DTYPES):
        gpu = "/".join(sorted(g for g, c in COMPUTE_CAPABILITY.items() if c == cap))
        results = {}
        for vals in itertools.product([False, True], repeat=len(fact_names)):
            facts = dict(zip(fact_names, vals))
            # The size must also be the same at every tensor-parallel width; a width that
            # changes it makes the combination depend on tp, and it is refused.
            per_tp = {tp: resolve_once(v, model, cap, kv, facts, tp) for tp in TP_WIDTHS}
            outs = {tp: r[0] for tp, r in per_tp.items()}
            if any(o is None for o in outs.values()):
                results[tuple(vals)] = next((None, r[1]) for r in per_tp.values() if r[1])
            elif len({o[0] for o in outs.values()}) > 1:
                results[tuple(vals)] = (None, "the size depends on the tensor-parallel width: " +
                                        ", ".join(f"tp{t}={o[0]}" for t, o in sorted(outs.items())))
            else:
                results[tuple(vals)] = (outs[TP_WIDTHS[0]], None)
        # The settings that agree with what is assumed for this capability; all 16 when nothing is.
        assumed = ASSUMED_FACTS.get(cap, {})

        def agrees(vals):
            return all(dict(zip(fact_names, vals))[f] == want for f, (want, _) in assumed.items())

        kept = {k: r for k, r in results.items() if agrees(k)}
        sizes = {r[0][0] if r[0] else None for r in kept.values()}
        backends = sorted({r[0][1] for r in kept.values() if r[0]})
        # A combination rests on an assumption when the size differs over all 16 settings.
        every = {r[0][0] if r[0] else None for r in results.values()}
        assumed_facts = {f: ev for f, (_, ev) in assumed.items()} if len(sizes) == 1 and None not in sizes and len(every) > 1 else {}
        if None in sizes:
            first_err = next(r[1] for r in kept.values() if r[1])
            status, size = "refused", None
            note = first_err
        elif len(sizes) == 1:
            status, size, note = "resolved", sizes.pop(), ("the same under every setting of " + ", ".join(fact_names) +
                                                     (" that agrees with " + ", ".join(f"{f}={int(w)}" for f, (w, _) in sorted(assumed.items()))
                                                      if assumed_facts else "") +
                                                     " and at every tensor-parallel width in " + str(TP_WIDTHS) + " (valid up to tp %d only)" % max(TP_WIDTHS))
        else:
            status, size = "refused", None
            differs = []
            for i, f in enumerate(fact_names):
                for vals in itertools.product([False, True], repeat=len(fact_names)):
                    if vals[i]:
                        continue
                    other = list(vals)
                    other[i] = True
                    if not (agrees(vals) and agrees(other)):
                        continue
                    a, b = results[tuple(vals)][0], results[tuple(other)][0]
                    if a and b and a[0] != b[0]:
                        differs.append(f)
                        break
            note = "depends on " + ", ".join(sorted(set(differs)) or fact_names)
        rows.append(dict(model=model, gpu=gpu, capability=f"{cap[0]}.{cap[1]}", kv_cache_dtype=kv, status=status,
                         block_size=size, backends=backends,
                         assumed_facts=assumed_facts,
                         by_facts={",".join(f"{n}={int(x)}" for n, x in zip(fact_names, k)): (r[0][0] if r[0] else None)
                                   for k, r in results.items()},
                         note=note))
    return v, fact_names, rows


def render_table(v, fact_names, rows):
    out = []
    out.append("# Block-size resolution oracle: vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be)")
    out.append("#")
    out.append("# Runs vLLM's own CudaPlatform.get_valid_backends and _preferred_block_size_for_backends on a CPU.")
    out.append("# Facts the CPU cannot discover. Each combination is run under every setting of them; it resolves")
    out.append("# only if the size is the same under all:")
    for f in fact_names:
        out.append(f"#   {f}: {v.stubs.FACT_DOC[f]}")
    out.append("# Assumed facts. A combination whose size depends on one is resolved over the settings that agree with it,")
    out.append("# and its assumed_facts (JSON) and its Source say so; the columns below still show all 16 settings.")
    for cap, facts in sorted(ASSUMED_FACTS.items()):
        for f, (want, ev) in sorted(facts.items()):
            out.append(f"#   compute capability {cap[0]}.{cap[1]}: {f}={int(want)}: {ev}")
    out.append("# Platform replacements (design FID-1). None decides a block size:")
    for what, why in v.stubs.REPLACEMENTS:
        out.append(f"#   {what}: {why}")
    out.append("# Tensor parallelism: every combination is run at tp in " + str(TP_WIDTHS) + " and resolves only if the size")
    out.append("# is the same at all of them. The table is valid for tp up to %d; a wider pool is refused (core/vllm.MaxBlockSizeTP)." % max(TP_WIDTHS))
    out.append("# Not resolved, whatever the GPU and dtype:")
    for m, why in sorted(NOT_RESOLVED.items()):
        out.append(f"#   {m}: {why}")
    out.append("#")
    hdr = ["model", "gpu", "cc", "kv_dtype", "backend(s)", "status", "size"] + [
        "[" + ",".join(f"{n}={int(x)}" for n, x in zip(fact_names, k)) + "]" for k in itertools.product([False, True], repeat=len(fact_names))]
    out.append("\t".join(hdr))
    for r in rows:
        sizes = [str(r["by_facts"][",".join(f"{n}={int(x)}" for n, x in zip(fact_names, k))]) for k in itertools.product([False, True], repeat=len(fact_names))]
        out.append("\t".join([r["model"], r["gpu"], r["capability"], r["kv_cache_dtype"], "/".join(r["backends"]) or "-",
                              r["status"], str(r["block_size"]) if r["block_size"] else "-"] + sizes))
    for r in rows:
        if r["status"] == "refused":
            out.append(f"# refused: {r['model']} {r['gpu']} {r['kv_cache_dtype']}: {r['note']}")
    return "\n".join(out) + "\n"


def as_json(v, fact_names, rows):
    return json.dumps({
        "vllm": "5df63a7203cdb9328d824a27b682e77c7b9928be",
        "facts": {f: v.stubs.FACT_DOC[f] for f in fact_names},
        "replacements": [{"what": a, "why": b} for a, b in v.stubs.REPLACEMENTS],
        "assumptions": {f"{cap[0]}.{cap[1]}": {f: {"value": want, "evidence": ev} for f, (want, ev) in facts.items()}
                        for cap, facts in ASSUMED_FACTS.items()},
        "not_resolved": NOT_RESOLVED,
        "combinations": [{k: r[k] for k in ("model", "gpu", "capability", "kv_cache_dtype", "status", "block_size", "backends", "assumed_facts", "by_facts", "note")} for r in rows],
    }, indent=1, sort_keys=True) + "\n"


def main(argv):
    if len(argv) < 3 or argv[2] not in ("table", "json", "check"):
        print(__doc__)
        return 2
    v, fact_names, rows = run(argv[1])
    if argv[2] == "table":
        sys.stdout.write(render_table(v, fact_names, rows))
    elif argv[2] == "json":
        sys.stdout.write(as_json(v, fact_names, rows))
    else:
        for path, want in [(argv[3], as_json(v, fact_names, rows))] + (
                [(argv[4], render_table(v, fact_names, rows))] if len(argv) > 4 else []):
            if want != open(path).read():
                print(f"{path} differs from what the oracle prints; regenerate it")
                return 1
            print(f"{path}: current")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
