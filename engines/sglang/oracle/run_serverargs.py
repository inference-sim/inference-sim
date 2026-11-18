# Original to blis-sglang.
"""Tier 1, default resolution: run each case of oracle/serverargs/cases.json through SGLang's own
ServerArgs(...).resolve_once() (python/sglang/srt/server_args.py:244, the pipeline of
arg_groups/pipeline.py) and record, for each case, whether SGLang accepts it and every field's
resolved value, with the step that last declared it.

The CPU oracle restores the CUDA platform of the case's chip before SGLang reads it (FID-1, each
replacement listed in oracle/mocks.json):

  platform   override_platform(...) (python/sglang/srt/runtime_context.py:1950, SGLang's own seam
             for this) states the chip's facts: CUDA, its compute capability, and flashinfer
             installed, as SGLang's CUDA image installs it (docker/Dockerfile).
             sglang.srt.platforms._current_platform is CudaSRTPlatform().
             torch.cuda.get_device_capability and get_device_name report the chip's, and
             utils.common's get_device, is_cuda, is_sm90_supported, is_sm100_supported and
             get_device_capability, get_device_sm and get_device_memory_capacity report the
             chip's, the last as the MiB nvidia-smi reports (memory.total), each value from the
             chip's entry in cases.json with its source. torch.version.cuda is the CUDA toolkit
             of SGLang's image, 13.0 (docker/Dockerfile:1).

Nothing in arg_groups/ is replaced. A model is read as configuration only (no weights), at the
revision cases.json pins. Paths are recorded as "<model>" so the answer does not depend on where
the model was fetched; random_seed is recorded as "<random>" when SGLang draws it.

Usage: python -I oracle/run_serverargs.py <sglang checkout> <cases.json> <models dir> > answers.json
<models dir>/<name>/ holds each model's configuration files (oracle/serverargs/fetch_models.py).
"""
import hashlib
import json
import os
import subprocess
import sys
import traceback

CUDA_VERSION = "13.0"
# The CUDA-only kernel packages SGLang imports once the CUDA platform is restored. They are not
# installed on the CPU runner (oracle/requirements.txt); each is stubbed as stubs.py stubs a
# module, and listed in oracle/mocks.json. Resolution reads none of their values.
CUDA_STUBBED = ("sgl_kernel", "flashinfer")
HERE = os.path.dirname(os.path.abspath(__file__))


def jsonable(x):
    """A resolved value as JSON: records, enums and tuples as their plain forms."""
    import enum

    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, enum.Enum):
        return jsonable(x.value)
    if isinstance(x, (list, tuple, set, frozenset)):
        vals = [jsonable(v) for v in x]
        return sorted(vals, key=json.dumps) if isinstance(x, (set, frozenset)) else vals
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if hasattr(x, "to_dict"):
        return jsonable(x.to_dict())
    if hasattr(x, "__struct_fields__"):
        return {f: jsonable(getattr(x, f)) for f in x.__struct_fields__}
    if hasattr(x, "__dataclass_fields__"):
        return {f: jsonable(getattr(x, f)) for f in x.__dataclass_fields__}
    if hasattr(x, "_asdict"):
        return jsonable(x._asdict())
    return {"<unrecorded type>": type(x).__name__}


def restore_cuda(chip: dict) -> None:
    """The chip's CUDA platform, stated through SGLang's own override and the probes it reads."""
    maj, mi = chip["capability"]
    sm = maj * 10 + mi
    from sglang.srt.runtime_context import override_platform

    override_platform(
        is_cuda=True, is_hip=False, is_npu=False, is_xpu=False, is_musa=False, is_mps=False,
        is_sm90=sm == 90, is_sm100=maj == 10, is_sm100_or_sm110=maj in (10, 11), is_sm110=maj == 11,
        is_sm120=maj == 12, is_blackwell=maj >= 10, is_hopper_with_cuda_12_3=sm == 90,
        has_amx=False, has_flashinfer=True, device_sm=sm, device_capability=(maj, mi),
    ).install()
    import sglang.srt.platforms as platforms
    from sglang.srt.platforms.cuda import CudaSRTPlatform

    platforms._current_platform = CudaSRTPlatform()
    import torch

    torch.cuda.get_device_capability = lambda device=None: (maj, mi)
    torch.cuda.get_device_name = lambda device=None: chip["device_name"]
    # The CUDA toolkit of SGLang's own image (docker/Dockerfile:1 CUDA_VERSION=13.0.3).
    torch.version.cuda = CUDA_VERSION
    import sglang.srt.utils.common as common

    mib = float(chip["nvidia_smi_memory_total_mib"])
    common.get_device_memory_capacity = lambda device=None: mib
    common.get_nvgpu_memory_capacity = lambda: mib
    common.get_device = lambda device_id=None: "cuda"
    common.is_cuda = lambda: True
    common.is_sm90_supported = lambda device=None: sm == 90
    common.is_sm100_supported = lambda device=None: maj == 10
    common.get_device_capability = lambda device_id=0: (maj, mi)
    common.get_device_sm = lambda: sm
    # SGLang's CUDA image installs flashinfer (docker/Dockerfile); the CPU runner does not, and
    # is_flashinfer_available (utils/common.py:388-395) looks the package up.
    common.is_flashinfer_available = lambda: True
    # sglang.srt.utils re-exports utils.common; a module that imports a probe from it binds the
    # re-exported name, so each is restored there too.
    import sglang.srt.utils as srt_utils

    srt_utils.get_device_memory_capacity = common.get_device_memory_capacity
    srt_utils.get_nvgpu_memory_capacity = common.get_nvgpu_memory_capacity
    srt_utils.get_device = common.get_device
    srt_utils.is_cuda = common.is_cuda
    srt_utils.is_sm90_supported = common.is_sm90_supported
    srt_utils.is_sm100_supported = common.is_sm100_supported
    srt_utils.get_device_capability = common.get_device_capability
    srt_utils.get_device_sm = common.get_device_sm
    srt_utils.is_flashinfer_available = common.is_flashinfer_available


def model_facts(sa) -> dict:
    """The model facts the Go resolution reads, as SGLang's own ModelConfig derives them from the
    model's configuration (configs/model_config.py). config/ takes them as an input, so the
    answers record them with each case."""
    from sglang.srt.arg_groups.model_override_base import model_config_of
    from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_spec
    from sglang.srt.configs.model_config import (
        get_dsa_index_kpool,
        is_deepseek_dsa,
        uses_kda_attention,
    )

    mc = model_config_of(sa)
    hf = mc.hf_config
    quant = getattr(hf, "quantization_config", None) or {}
    dsa = is_deepseek_dsa(hf)
    from sglang.srt.arg_groups.model_overrides.qwen3_moe import _mixed_precision_moe_quant_algos

    vision = getattr(hf, "vision_config", None)
    return {
        "architecture": hf.architectures[0],
        "model_type": getattr(hf, "model_type", None),
        "text_model_type": getattr(mc.hf_text_config, "model_type", None),
        "attention_arch": mc.attention_arch.name,
        "is_multimodal": bool(mc.is_multimodal),
        "is_multimodal_breakable_cuda_graph_supported": bool(mc.is_multimodal_breakable_cuda_graph_supported),
        "has_vision_config": getattr(hf, "vision_config", None) is not None,
        "is_deepseek_dsa": bool(dsa),
        "dsa_index_kpool": get_dsa_index_kpool(hf) if dsa else None,
        "learnable_sink": bool(getattr(hf, "learnable_sink", False)),
        "has_attention_sinks": bool(mc.has_attention_sinks),
        "has_asymmetric_kv": bool(mc.has_asymmetric_kv),
        "uses_kda_attention": bool(uses_kda_attention(hf)),
        "has_linear_attn_spec": get_linear_attn_spec(hf) is not None,
        "quant_method": quant.get("quant_method"),
        "quant_weight_block_size": quant.get("weight_block_size"),
        "quant_scale_fmt": quant.get("scale_fmt"),
        "hidden_size": getattr(hf, "hidden_size", None),
        "is_fp4_experts": bool(getattr(mc, "is_fp4_experts", False)),
        "has_nvfp4_moe_meta": getattr(mc, "nvfp4_moe_meta", None) is not None,
        # adjust_mem_fraction_for_vlm reads these two (arg_groups/memory_hook.py:440-441).
        "vision_num_hidden_layers": getattr(vision, "num_hidden_layers", 24) if vision is not None else None,
        "vision_hidden_size": getattr(vision, "hidden_size", 1024) if vision is not None else None,
        # _qwen3_moe_family_overrides reads the quant_algo of the expert layers.
        "mixed_precision_moe_quant_algos": sorted(_mixed_precision_moe_quant_algos(hf)),
    }


def run_case(case: dict, chip: dict, models: str) -> dict:
    """One case in a fresh process state: the caller forks before calling."""
    model_path = os.path.join(models, case["model"])
    restore_cuda(chip)
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs(model_path=model_path, **case.get("args", {}))
    out = {"name": case["name"]}
    try:
        sa.resolve_once()
    except Exception as e:  # SGLang's own refusal is the answer
        msg = str(e).replace(model_path, "<model>")
        out["refused"] = {"type": type(e).__name__, "message": msg.splitlines()[0] if msg else ""}
        out["refused_at"] = refusal_site(e)
        out["model_facts"] = model_facts(sa)
        return out
    out["model_facts"] = model_facts(sa)
    view = resolved_view(sa)
    raw = sa._raw_input
    by = {}
    for src, declared in sa._resolved_overrides:
        for k in declared:
            by[k] = src
    resolved = {}
    for name in sorted(raw):
        v = getattr(view, name)
        if name == "random_seed" and raw[name] is None:
            v = "<random>"
        j = json.loads(json.dumps(jsonable(v)).replace(json.dumps(model_path)[1:-1], "<model>"))
        resolved[name] = j
    out["resolved"] = resolved
    out["declared_by"] = {k: by[k] for k in sorted(by) if k in raw and jsonable(raw[k]) != resolved[k]}
    return out


def refusal_site(e: BaseException) -> str:
    """The SGLang file and function the refusal was raised in."""
    for fr in reversed(traceback.extract_tb(e.__traceback__)):
        if "/sglang/srt/" in fr.filename:
            return fr.filename.split("/python/", 1)[-1] + ":" + fr.name
    return ""


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    src, cases_path, models = argv
    source = os.path.realpath(os.path.join(src, "python"))
    sys.path.insert(0, source)
    sys.path.insert(0, HERE)
    with open(cases_path, "rb") as f:
        blob = f.read()
    spec = json.loads(blob)
    # SGLang reads its SGLANG_* (and the deprecated SGL_*) variables during resolution; none is
    # an input of a case, so none of the caller's may reach it (SGLang's defaults apply).
    for k in sorted(os.environ):
        if k.startswith(("SGLANG_", "SGL_")):
            del os.environ[k]
    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    results = []
    for case in spec["cases"]:
        chip = spec["chips"][case["chip"]]
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                import stubs

                stubs.STUBBED = tuple(stubs.STUBBED) + CUDA_STUBBED
                stubs.install()
                res = run_case(case, chip, models)
            except BaseException as e:  # a harness failure, not a refusal
                res = {"name": case["name"], "harness_error": "%s: %s" % (type(e).__name__, e)}
            with os.fdopen(w, "w") as fw:
                json.dump(res, fw)
            os._exit(0)
        os.close(w)
        with os.fdopen(r) as fr:
            data = fr.read()
        os.waitpid(pid, 0)
        res = json.loads(data) if data else {"name": case["name"], "harness_error": "no answer"}
        # The case's inputs travel with its answer, so the Go test reads one file.
        res["args"] = case.get("args", {})
        res["chip"] = {k: chip[k] for k in ("capability", "nvidia_smi_memory_total_mib")}
        if "harness_error" in res:
            print("%s: %s" % (case["name"], res["harness_error"]), file=sys.stderr)
            return 1
        results.append(res)
    doc = {
        "provenance": {"sglang_commit": commit, "cases_sha256": hashlib.sha256(blob).hexdigest(),
                       "models": {k: v["revision"] for k, v in sorted(spec["models"].items())}},
        "cases": results,
    }
    sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
