# Original to blis-vllm.
"""Quantization oracle: what vLLM v0.32.0 resolves `quantization` to on the catalog checkpoints a deployment names.

A deployment states `quantization` (or none), and vLLM's ModelConfig replaces it, accepts it or
refuses it by the checkpoint's own quantization_config:

  ModelConfig._verify_quantization        vllm/config/model.py:1370-1449
  DeepseekV4FP8Config.override_quantization_method
                                          vllm/models/deepseek_v4/quant_config.py:140-157
  ModelOpt*Config.override_quantization_method
                                          vllm/model_executor/layers/quantization/modelopt.py:449-455,
                                          772-778, 1205-1211, 1627-1633

The override reads the checkpoint's quant_method and quant_algo, so this runs vLLM's own ModelConfig
over each checkpoint's catalog config.json, once for each stated value, and records the method each
resolves to, or the text of the error vLLM raises. Only the GPU is absent: a CPU falls back from
bfloat16 to float16, which no quantization decision reads, and nothing is replaced (design FID-1).
The config.json files are the catalog's (BLIS_ORACLE_CATALOG), or fetched at the pinned revision and
checked against their SHA-256 (oracle/blocksize/modelconfig.py).

For a MiniMax-M3 checkpoint it also records the weight format each class of layers is served in. The
class of a layer is the one vLLM's model builds it as (vllm/models/minimax_m3/nvidia/model.py:
MiniMaxM3Attention/MiniMaxM3MLP linear layers, MiniMaxM3MoE experts, ParallelLMHead), its prefix is
the one the model gives it, and the answer is vLLM's own: the quantization config built by
`from_config` over the checkpoint's quantization_config, asked `is_layer_excluded` and, for a
mixed-precision checkpoint, `_resolve_quant_algo` (the per-layer mapping) for every such prefix.

Usage:
    python3 -I oracle/quantization/oracle.py <vllm checkout> json              print the resolutions as JSON
    python3 -I oracle/quantization/oracle.py <vllm checkout> check <file>      fail unless <file> equals what it prints

compat/quantization/resolved.json is the committed output; the package embeds it, and
compat/kernelmodel/quantization_v4_test.go and quantization_m3_test.go hold the kernel's plans to it.
"""
import hashlib
import json
import os
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "blocksize"))

COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
CATALOG = os.environ.get("BLIS_ORACLE_CATALOG", "")
# None is a deployment that states no quantization. deepseek_v4_fp8 is the value the dashboard's
# points state; the rest are values a user could name for the same checkpoint.
V4_STATED = [None, "deepseek_v4_fp8", "fp8", "awq", "nvfp4"]
# A string no vLLM release names: vLLM's answer to it is its answer to every value it does not list.
UNLISTED = "not_a_quantization_method"
# The checkpoints and, for each, whether the answer also records the served layout of a MiniMax-M3
# model. deepseek-v4-pro keeps the five values and the control it was recorded with.
MODELS = [
    ("deepseek-v4-pro", False),
    ("minimax-m3-mxfp8", True),
    ("minimax-m3-nvfp4", True),
]
# vllm/models/minimax_m3/nvidia/model.py:1045-1048,1124-1127 (packed_modules_mapping of both classes).
M3_PACKED = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}
UNQUANTIZED = "unquantized"


def _vllm_text(e):
    """The message vLLM's own check raised. ModelConfig is a pydantic dataclass, which wraps a ValueError
    raised in its post-init in a ValidationError (its str adds the pydantic version, the input and a
    URL); the refusal is the wrapped message. An error that is not a quantization decision (the
    architecture could not be inspected, so the oracle's environment is wrong) stops the run."""
    errs = e.errors() if hasattr(e, "errors") else [{"msg": str(e)}]
    msg = errs[0]["msg"]
    msg = msg.removeprefix("Value error, ")
    if "failed to be inspected" in msg or len(errs) != 1:
        raise RuntimeError("the oracle's environment cannot build a ModelConfig: %s" % e)
    return msg


def _resolve(cfg_json, stated, mc_cls):
    """ModelConfig over one config.json for each stated value: [{stated, resolved} | {stated, error}]."""
    rows = []
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "wb") as f:
            f.write(cfg_json)
        for q in stated:
            try:
                mc = mc_cls(model=d, quantization=q, skip_tokenizer_init=True, dtype="auto", seed=0,
                            enforce_eager=True)
            except ValueError as e:
                rows.append({"stated": q, "error": _vllm_text(e)})
                continue
            rows.append({"stated": q, "resolved": mc.quantization})
    return rows


def _m3_prefixes(text_cfg):
    """The prefix of each layer vLLM's MiniMax-M3 builds with the quantization config, by class.

    vllm/models/minimax_m3/nvidia/model.py: a decoder layer holds self_attn (qkv_proj, o_proj: :321-339,
    :454-465), then block_sparse_moe on the layers moe_layer_freq marks (:738-745,
    shared_experts gate_up_proj/down_proj: :161-174, :241-250; experts: :258-272) or mlp otherwise
    (:748-754; gate_up_proj, down_proj); lm_head is a ParallelLMHead (:1060-1064). The router
    `gate` is a GateLinear built with no quantization config and float32 weights (:231-238), and
    the token embedding is not quantized by any ModelOpt config (get_quant_method returns None for it,
    modelopt.py:244-257, :1810-1860), so neither is probed.
    """
    n = text_cfg["num_hidden_layers"]
    freq = text_cfg.get("moe_layer_freq")
    linear, experts = [], []
    for i in range(n):
        base = "language_model.model.layers.%d" % i
        linear += [base + ".self_attn.qkv_proj", base + ".self_attn.o_proj"]
        if freq is None or freq[i] != 0:
            linear += [base + ".block_sparse_moe.shared_experts.gate_up_proj",
                       base + ".block_sparse_moe.shared_experts.down_proj"]
            experts.append(base + ".block_sparse_moe.experts")
        else:
            linear += [base + ".mlp.gate_up_proj", base + ".mlp.down_proj"]
    return {"linear": linear, "routed_experts": experts, "head": ["language_model.lm_head"]}


def _m3_layout(quant_cfg, method):
    """The algorithm each class of M3 layers is served in, from vLLM's own quantization config."""
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import MXFP8_BLOCK_SIZE

    cls = get_quantization_config(method)
    qc = cls.from_config(quant_cfg)
    qc.packed_modules_mapping = M3_PACKED
    mixed = hasattr(qc, "_resolve_quant_algo")

    def algo_of(prefix, is_experts):
        if qc.is_layer_excluded(prefix):
            return UNQUANTIZED
        if mixed:
            a = qc._resolve_quant_algo(prefix)
            return UNQUANTIZED if a is None else a
        return qc.quant_method

    return qc, cls.__name__, MXFP8_BLOCK_SIZE, algo_of


def _layout(data, cfg, method):
    qc, cls_name, block, algo_of = _m3_layout(cfg["quantization_config"], method)
    text = cfg["text_config"]
    classes = {}
    for cname, prefixes in _m3_prefixes(text).items():
        counts = {}
        for p in prefixes:
            a = algo_of(p, cname == "routed_experts")
            counts[a] = counts.get(a, 0) + 1
        classes[cname] = counts
    return {
        "config_class": cls_name,
        "mxfp8_block_size": block,
        "unquantized_dtype": text["dtype"],
        "classes": classes,
    }


def run(vllm_dir):
    warnings.simplefilter("ignore")
    import modelconfig  # pyright: ignore[reportMissingImports]

    sys.path.insert(0, vllm_dir)
    # ModelConfig inspects the architecture in a subprocess (vllm/model_executor/models/registry.py),
    # which starts without this interpreter's sys.path; it needs the checkout to import vLLM.
    os.environ["PYTHONPATH"] = os.pathsep.join([os.path.abspath(vllm_dir)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    from vllm.config import ModelConfig
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

    out = {}
    for model, with_layout in MODELS:
        data = modelconfig.config_bytes(model, os.path.join(CATALOG, model))
        cfg = json.loads(data)
        entry = {
            "config_sha256": hashlib.sha256(data).hexdigest(),
            "model_type": cfg["model_type"],
            "checkpoint_quant_method": cfg["quantization_config"]["quant_method"],
        }
        if with_layout:
            stated = [None] + sorted(QUANTIZATION_METHODS) + [UNLISTED]
            entry["cases"] = _resolve(data, stated, ModelConfig)
            own = entry["cases"][0]["resolved"]
            entry["resolved_without_quantization"] = own
            entry["layout"] = _layout(data, cfg, own)
        else:
            entry["cases"] = _resolve(data, V4_STATED, ModelConfig)
            # The same config without its quantization_config (an unquantized checkpoint): the
            # override does not apply, so a stated value stays as stated. It shows the answers above
            # come from the checkpoint's quant_method, not from the oracle.
            bare = {k: v for k, v in cfg.items() if k != "quantization_config"}
            entry["control_without_quantization_config"] = _resolve(json.dumps(bare).encode(), [None, "fp8"],
                                                                    ModelConfig)
        out[model] = entry
    return {"vllm": COMMIT, "checkpoints": out}


def main(argv):
    if len(argv) < 3 or argv[2] not in ("json", "check") or (argv[2] == "check" and len(argv) < 4):
        print(__doc__)
        return 2
    out = json.dumps(run(argv[1]), indent=1, sort_keys=True) + "\n"
    if argv[2] == "json":
        sys.stdout.write(out)
        return 0
    with open(argv[3], encoding="utf-8") as f:
        if f.read() != out:
            print(f"{argv[3]} differs from what the oracle prints; regenerate it")
            return 1
    print(f"{argv[3]}: current")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
