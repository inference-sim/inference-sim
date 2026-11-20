"""Runs vLLM's own CUDA graph capture resolver on stated inputs, without importing vllm.

usage: harness.py VLLM_SOURCE < cases.json > results.json

The functions it runs are cut out of the checkout's source by AST, not retyped:
VllmConfig._set_cudagraph_sizes and _split_cudagraph_capture_sizes
(vllm/config/vllm.py), the grid and shape helpers and CUDAGraphMode
(vllm/config/compilation.py), and the platform defaults
(vllm/platforms/interface.py, vllm/platforms/cuda.py). Only the object they read is
faked: a config holding the scheduler limits, the mode and the speculative method.

Each case: {"seqs", "mnbt", "spec", "family100", "mode", "method", "npredict"}. Each result:
{"sizes", "max", "prefill", "decode"}.
"""
import __future__
import ast
import bisect
import dataclasses
import enum
import json
import math
import sys
import types
from collections.abc import Callable, Iterable, Mapping
from types import SimpleNamespace

root = sys.argv[1]


def parse(path):
    src = open(f"{root}/{path}").read()
    return ast.parse(src)


def pick(tree, names, cls=None):
    body = tree.body
    if cls:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    found = [n for n in body if getattr(n, "name", None) in names]
    missing = set(names) - {n.name for n in found}
    assert not missing, f"not found in source: {missing}"
    return found


def run(nodes, ns):
    mod = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "<vllm>", "exec", flags=__future__.annotations.compiler_flag), ns)


comp = parse("vllm/config/compilation.py")
cns = {
    "bisect": bisect, "math": math, "enum": enum, "dataclass": dataclasses.dataclass,
    "Callable": Callable, "Iterable": Iterable, "Mapping": Mapping,
}
run(
    pick(comp, {
        "decode_cudagraph_capture_shapes", "_walk_grid", "prefill_cudagraph_capture_grid",
        "decode_cudagraph_capture_grid", "CUDAGraphCaptureDefaults",
        "validate_cudagraph_capture_grid", "CUDAGraphMode",
    }),
    cns,
)
fake = types.ModuleType("vllm.config.compilation")
fake.__dict__.update(cns)
sys.modules.setdefault("vllm", types.ModuleType("vllm"))
sys.modules.setdefault("vllm.config", types.ModuleType("vllm.config"))
sys.modules["vllm.config.compilation"] = fake

iface = parse("vllm/platforms/interface.py")
cuda = parse("vllm/platforms/cuda.py")
base_fns = pick(iface, {
    "get_cudagraph_capture_defaults", "get_prefill_cudagraph_capture_grid",
    "get_decode_cudagraph_capture_grid",
}, cls="Platform")
cuda_fns = pick(cuda, {"get_cudagraph_capture_defaults"}, cls="CudaPlatformBase")
pns = {"VllmConfig": object, "CUDAGraphCaptureDefaults": cns["CUDAGraphCaptureDefaults"]}
base_cls = ast.ClassDef(name="Platform", bases=[], keywords=[], body=base_fns, decorator_list=[])
cuda_cls = ast.ClassDef(name="Cuda", bases=[ast.Name("Platform", ast.Load())], keywords=[],
                        body=cuda_fns, decorator_list=[])
run([base_cls, cuda_cls], pns)
Base, Cuda = pns["Platform"], pns["Cuda"]


class Cuda100(Cuda):
    def is_device_capability_family(self, n):
        return n == 100


class CudaOther(Cuda):
    def is_device_capability_family(self, n):
        return False


vtree = parse("vllm/config/vllm.py")
vfns = pick(vtree, {
    "_set_cudagraph_sizes", "_split_cudagraph_capture_sizes",
    "get_decode_cudagraph_query_lens", "get_draft_decode_cudagraph_request_sizes",
}, cls="VllmConfig")
vns = {
    "CUDAGraphMode": cns["CUDAGraphMode"],
    "CUDAGraphCaptureDefaults": cns["CUDAGraphCaptureDefaults"],
    "decode_cudagraph_capture_shapes": cns["decode_cudagraph_capture_shapes"],
    "validate_cudagraph_capture_grid": cns["validate_cudagraph_capture_grid"],
    "cdiv": lambda a, b: -(a // -b),
    "logger": SimpleNamespace(warning=lambda *a, **k: None, warning_once=lambda *a, **k: None,
                              info_once=lambda *a, **k: None),
    "current_platform": None,
}
run([ast.ClassDef(name="Cfg", bases=[], keywords=[], body=vfns, decorator_list=[])], vns)
Cfg = vns["Cfg"]


def resolve(c):
    vns["current_platform"] = Cuda100() if c["family100"] else CudaOther()
    mode = cns["CUDAGraphMode"][c["mode"]]
    spec = c["spec"]
    cfg = Cfg()
    cfg.model_config = SimpleNamespace(enforce_eager=False)
    cfg.compilation_config = SimpleNamespace(
        cudagraph_mode=mode, max_cudagraph_capture_size=None, cudagraph_capture_sizes=None,
        prefill_cudagraph_capture_token_sizes=None, max_prefill_cudagraph_capture_tokens=None,
        decode_cudagraph_capture_request_sizes=None, max_decode_cudagraph_capture_requests=None,
        max_decode_cudagraph_capture_tokens=None, post_init_cudagraph_sizes=lambda: None,
    )
    cfg.scheduler_config = SimpleNamespace(max_num_seqs=c["seqs"], max_num_batched_tokens=c["mnbt"])
    cfg.performance_mode = "balanced"
    cfg.num_speculative_tokens = spec
    cfg.uniform_decode_query_len = 1 + spec
    cfg.speculative_config = None
    if spec:
        cfg.speculative_config = SimpleNamespace(
            method=c["method"], uses_dynamic_speculative_decoding=lambda: False,
            use_multi_module_mtp=lambda: c["method"] == "mtp" and min(c.get("npredict", 1), spec) > 1, use_gemma4_mtp=lambda: False,
            enable_adaptive_verification=False,
        )
    cfg._set_cudagraph_sizes()
    cc = cfg.compilation_config
    return {
        "sizes": cc.cudagraph_capture_sizes, "max": cc.max_cudagraph_capture_size,
        "prefill": cc.prefill_cudagraph_capture_token_sizes,
        "decode": cc.decode_cudagraph_capture_request_sizes,
    }


json.dump([resolve(c) for c in json.load(sys.stdin)], sys.stdout)
