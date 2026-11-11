# Original to blis-vllm.
"""Oracle for the lowering in core/lower: vLLM v0.32.0's own functions, run unmodified.

The functions that decide a CUDA-graph capture, a padded launch and a drafter's context rows
are plain Python over small objects, but the modules that hold them import torch and CUDA
wheels. This script parses each module with `ast`, takes the named definitions out as written,
and runs them in a namespace that holds only what they read. Nothing is rewritten or copied
(FID-8): the definitions are the pinned release's own text.

Replacements (design FID-1) are listed in oracle/lower/replacements.json. They stand in for
configuration objects and for the data-parallel all-reduce, never for a decision.

Usage:
    python -I oracle/lower/extract.py <vllm source root> print            the cases and vLLM answers, as JSON
    python -I oracle/lower/extract.py <vllm source root> check <file>     fail unless <file> is what print gives

<vllm source root> is a directory holding vllm/ at commit 5df63a7203cdb9328d824a27b682e77c7b9928be, run with a
Python that has torch installed (the oracle environment, oracle/requirements.lock).
"""
import ast
import dataclasses
import enum
import itertools
import json
import os
import sys
import types
from collections import defaultdict
from itertools import groupby, product

COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"


def parse(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return ast.parse(f.read(), rel)


def pick(tree, *names):
    """The top-level definitions called names, in file order, as written."""
    out = [n for n in tree.body if getattr(n, "name", None) in names]
    got = {n.name for n in out}
    if got != set(names):
        raise SystemExit(f"definitions not found: {sorted(set(names) - got)}")
    return out


def method(tree, cls, *names):
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    out = [n for n in c.body if getattr(n, "name", None) in names]
    got = {n.name for n in out}
    if got != set(names):
        raise SystemExit(f"{cls}: methods not found: {sorted(set(names) - got)}")
    return out


def run(nodes, ns, rel):
    mod = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, rel, "exec"), ns)


class Quiet:
    def __getattr__(self, _):
        return lambda *a, **k: None


def ns_base():
    return {"__builtins__": __builtins__, "dataclass": dataclasses.dataclass, "replace": dataclasses.replace,
            "enum": enum, "Self": object, "product": product, "groupby": groupby, "defaultdict": defaultdict,
            "logger": Quiet(), "Any": object, "annotations": None}


def load(root):
    import torch
    import torch.distributed as dist
    from contextlib import nullcontext
    ns = ns_base()
    ns.update(torch=torch, dist=dist, nullcontext=nullcontext)
    comp = parse(root, "vllm/config/compilation.py")
    # `from __future__ import annotations` is not needed: the pieces below use no forward reference.
    run(pick(comp, "CUDAGraphMode"), ns, "vllm/config/compilation.py")
    mm = parse(root, "vllm/utils/math_utils.py")
    run(pick(mm, "round_up", "cdiv"), ns, "vllm/utils/math_utils.py")
    cg = parse(root, "vllm/v1/worker/gpu/cudagraph_utils.py")
    run(pick(cg, "BatchExecutionDescriptor", "_is_compatible"), ns, "cudagraph_utils.py")
    ns["get_num_ubatches"] = ns["check_ubatch_thresholds"] = _no_ubatch
    ns["build_dynamic_sd_schedule_lookup"] = _no_dynamic
    cls = ast.ClassDef(name="Manager", bases=[], keywords=[], decorator_list=[], type_params=[],
                       body=method(cg, "CudaGraphManager", "_build_lora_dispatch_map", "_resolve_effective_loras",
                                   "_maybe_ubatch_twin", "_init_candidates", "dispatch"))
    run([cls], ns, "cudagraph_utils.py")
    be = parse(root, "vllm/v1/attention/backend.py")
    run(pick(be, "max_decode_query_len"), ns, "backend.py")
    vc = parse(root, "vllm/config/vllm.py")
    sizes = method(vc, "VllmConfig", "_set_cudagraph_sizes")[0]
    sizes.decorator_list = []
    run([sizes], ns, "vllm/config/vllm.py")
    br = [n for n in vc.body if isinstance(n, ast.Assign)
          and any(getattr(t, "id", "") == "DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES" for t in n.targets)]
    assert len(br) == 1
    run(br, ns, "vllm/config/vllm.py")
    eg = parse(root, "vllm/config/engram.py")
    for name in ("resolve_dp_shared_memory", "verify_parallel_config", "get_parallel_size", "_validate_shared_memory"):
        fn = method(eg, "EngramConfig", name)[0]
        fn.decorator_list = []  # model_validator, a pydantic registration; the body is run as written
        run([fn], ns, "vllm/config/engram.py")
        ns["engram_" + name] = ns.pop(name)
    en = parse(root, "vllm/models/deepseek_v41/common/engram.py")
    fn = method(en, "ParallelEngramEmbedding", "_get_shard_info")[0]
    fn.decorator_list = []
    run([fn], ns, "engram.py")
    ns["_get_shard_info"] = ns.pop("_get_shard_info")
    run(pick(en, "engram_head_shard_rank"), ns, "engram.py")
    # The two group lookups engram_head_shard_rank makes: this process's tensor-parallel rank and
    # the node-local Engram data-parallel group, both answered from the case.
    ns["get_tensor_model_parallel_rank"] = lambda: ns["_cur"]["tp_rank"]
    ns["get_engram_dp_group"] = lambda: (None if ns["_cur"]["edp"] == 1 else types.SimpleNamespace(
        world_size=ns["_cur"]["edp"], rank_in_group=ns["_cur"]["dp_rank"]))
    pc = parse(root, "vllm/config/parallel.py")
    ns["DCPCommBackend"] = ns["ParallelConfig"] = str  # annotations, evaluated when the def runs
    for name in ("set_dcp_defaults", "use_ubatching", "num_ubatches"):
        fn = method(pc, "ParallelConfig", name)[0]
        fn.decorator_list = []
        run([fn], ns, "vllm/config/parallel.py")
        ns["pc_" + name] = ns.pop(name)
    ub = parse(root, "vllm/v1/worker/ubatch_utils.py")
    run(pick(ub, "check_ubatch_thresholds", "is_last_ubatch_empty"), ns, "vllm/v1/worker/ubatch_utils.py")
    ns["ub_check_ubatch_thresholds"] = ns.pop("check_ubatch_thresholds")
    ns["ub_is_last_ubatch_empty"] = ns.pop("is_last_ubatch_empty")
    dp = parse(root, "vllm/v1/worker/gpu/dp_utils.py")
    # Type names the signatures mention; annotations are evaluated when the def runs.
    ns.update(CudaGraphManager=object, ParallelConfig=object)
    run(pick(dp, "DPSyncState", "sync_cudagraph_and_dp_padding", "dispatch_cg_and_sync_dp"), ns, "vllm/v1/worker/gpu/dp_utils.py")
    ns["get_dp_group"] = lambda: types.SimpleNamespace(cpu_group=None)
    ns["should_skip_dp_coordination"] = lambda: False
    df = parse(root, "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py")
    ctx = method(df, "DFlashSpeculator", "_num_graph_context_tokens")[0]
    run([ctx], ns, "dflash/speculator.py")
    sp = parse(root, "vllm/v1/worker/gpu/spec_decode/speculator.py")
    run(method(sp, "DraftModelSpeculator", "_build_uniform_batch_dp_sync"), ns, "spec_decode/speculator.py")
    return ns


def _no_ubatch(*a, **k):
    raise AssertionError("a case reached microbatching, which is not modelled")


def _no_dynamic(*a, **k):
    raise AssertionError("a case reached dynamic speculative decoding, which is not modelled")


class Platform:
    def __init__(self, family100):
        self.f = family100

    def is_device_capability_family(self, n):
        assert n == 100
        return self.f


def set_cudagraph_sizes(ns, c):
    """VllmConfig._set_cudagraph_sizes over case c; returns the resolved mode, max and sizes."""
    comp = types.SimpleNamespace(
        cudagraph_mode=ns["CUDAGraphMode"][c["mode"]],
        max_cudagraph_capture_size=c.get("max_capture"),
        cudagraph_capture_sizes=list(c["sizes"]) if c.get("sizes") is not None else None,
        post_init_cudagraph_sizes=lambda: None,
    )
    spec = None
    if c.get("spec_tokens"):
        spec = types.SimpleNamespace(uses_dynamic_speculative_decoding=lambda: False)
    self_ = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=bool(c.get("enforce_eager"))),
        compilation_config=comp,
        scheduler_config=types.SimpleNamespace(max_num_seqs=c["max_num_seqs"],
                                               max_num_batched_tokens=c["max_num_batched_tokens"]),
        performance_mode=c.get("performance_mode", "balanced"),
        uniform_decode_query_len=1 + c.get("spec_tokens", 0),
        speculative_config=spec,
    )
    sys.modules["vllm"] = types.ModuleType("vllm")
    m = types.ModuleType("vllm.platforms")
    m.current_platform = Platform(bool(c.get("blackwell")))
    sys.modules["vllm.platforms"] = m
    try:
        ns["_set_cudagraph_sizes"](self_)
    except (ValueError, AssertionError) as e:
        # The refusals the function makes at start-up: a stated maximum that disagrees with the
        # stated sizes, no sizes, a ceiling below 1.
        return {"max": None, "sizes": None, "error": f"{type(e).__name__}: {e}"}
    return {"max": comp.max_cudagraph_capture_size, "sizes": list(comp.cudagraph_capture_sizes), "error": None}


def manager(ns, c, resolved):
    M = ns["Manager"]
    m = M.__new__(M)
    m.cudagraph_mode = ns["CUDAGraphMode"][c["graph_mode"]]
    m.max_num_reqs = c["max_num_seqs"]
    m.decode_query_len = 1 + c.get("spec_tokens", 0)
    m.varlen_decode = bool(c.get("varlen"))
    m.lora_capture_cases = [0]
    m._lora_dispatch_map, m._max_lora_case = {}, 0
    m.ubatch_runner = None
    m._graphs_captured = True
    m._candidates, m._capture_descs = {}, {}
    m.compilation_config = types.SimpleNamespace(cudagraph_capture_sizes=resolved["sizes"],
                                                 max_cudagraph_capture_size=resolved["max"])
    m.vllm_config = types.SimpleNamespace(
        speculative_config=None, num_speculative_tokens=c.get("spec_tokens", 0),
        cache_config=types.SimpleNamespace(use_kda_recoverssm=bool(c.get("kda"))))
    m._init_candidates()
    return m


def desc_json(d):
    return {"mode": d.cg_mode.name, "tokens": d.num_tokens, "reqs": d.num_reqs,
            "uniform": d.uniform_token_count, "max_query_len": d.max_query_len}


# The configurations the lowering is held to. Each states what a deployment states; everything
# else is vLLM's. "graph_mode" is the mode the runner gets after the attention backend's override
# (resolve_cudagraph_mode_and_sizes), which the engine states, because that override reads the
# attention backend the engine picks.
CAPTURE_CASES = [
    dict(name="default-decode", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=256, max_num_batched_tokens=8192),
    dict(name="default-blackwell", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=1024, max_num_batched_tokens=8192, blackwell=True),
    dict(name="tiny-seqs-ceiling-8", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=4, max_num_batched_tokens=2048),
    dict(name="batched-tokens-off-stride", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=256, max_num_batched_tokens=300),
    dict(name="seqs-off-stride", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=100, max_num_batched_tokens=8192),
    dict(name="small-seqs", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=16, max_num_batched_tokens=2048),
    dict(name="piecewise", mode="PIECEWISE", graph_mode="PIECEWISE", max_num_seqs=256, max_num_batched_tokens=8192),
    dict(name="full-decode-only", mode="FULL_DECODE_ONLY", graph_mode="FULL_DECODE_ONLY", max_num_seqs=128, max_num_batched_tokens=4096),
    dict(name="none", mode="NONE", graph_mode="NONE", max_num_seqs=256, max_num_batched_tokens=8192),
    dict(name="enforce-eager", mode="FULL_AND_PIECEWISE", graph_mode="NONE", max_num_seqs=256, max_num_batched_tokens=8192, enforce_eager=True),
    dict(name="stated-max", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=256, max_num_batched_tokens=8192, max_capture=300),
    dict(name="stated-max-above-batched", mode="PIECEWISE", graph_mode="PIECEWISE", max_num_seqs=256, max_num_batched_tokens=512, max_capture=1024),
    dict(name="stated-sizes", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=256, max_num_batched_tokens=8192, sizes=[64, 8, 8, 16, 100000]),
    dict(name="mtp3", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=128, max_num_batched_tokens=8192, spec_tokens=3),
    dict(name="mtp1-small", mode="FULL_DECODE_ONLY", graph_mode="FULL_DECODE_ONLY", max_num_seqs=32, max_num_batched_tokens=4096, spec_tokens=1),
    dict(name="mtp2-rounds-past-the-ceiling", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=256, max_num_batched_tokens=8192, spec_tokens=2),
    dict(name="mtp6-more-seqs-than-fit", mode="FULL_DECODE_ONLY", graph_mode="FULL_DECODE_ONLY", max_num_seqs=256, max_num_batched_tokens=8192, spec_tokens=6),
    dict(name="mtp5-varlen", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=64, max_num_batched_tokens=8192, spec_tokens=5, varlen=True),
    dict(name="full-mode", mode="FULL", graph_mode="FULL", max_num_seqs=128, max_num_batched_tokens=4096),
    dict(name="full-mode-spec", mode="FULL", graph_mode="FULL", max_num_seqs=32, max_num_batched_tokens=2048, spec_tokens=3),
    dict(name="mtp1-seqs-above-platform-ceiling", mode="FULL_DECODE_ONLY", graph_mode="FULL_DECODE_ONLY", max_num_seqs=600, max_num_batched_tokens=8192, spec_tokens=1),
    dict(name="mtp1-seqs-above-blackwell-ceiling", mode="FULL_DECODE_ONLY", graph_mode="FULL_DECODE_ONLY", max_num_seqs=1100, max_num_batched_tokens=16384, spec_tokens=1, blackwell=True),
    dict(name="mtp3-recoverssm", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=64, max_num_batched_tokens=8192, spec_tokens=3, kda=True),
    dict(name="refused-stated-max-and-sizes-disagree", mode="PIECEWISE", graph_mode="PIECEWISE", max_num_seqs=8, max_num_batched_tokens=64, max_capture=5, sizes=[8, 16]),
    dict(name="refused-empty-stated-sizes", mode="PIECEWISE", graph_mode="PIECEWISE", max_num_seqs=8, max_num_batched_tokens=64, sizes=[]),
    dict(name="refused-no-batched-tokens", mode="PIECEWISE", graph_mode="PIECEWISE", max_num_seqs=8, max_num_batched_tokens=0),
    dict(name="stated-max-is-the-largest-stated-size", mode="PIECEWISE", graph_mode="PIECEWISE", max_num_seqs=8, max_num_batched_tokens=64, max_capture=16, sizes=[8, 16]),
    dict(name="interactivity", mode="PIECEWISE", graph_mode="PIECEWISE", max_num_seqs=256, max_num_batched_tokens=8192, performance_mode="interactivity"),
    # The values a deployment states for max_cudagraph_capture_size: 2048 (the most common), the
    # smallest the points state (2), a value off the stride, one equal to max_num_batched_tokens,
    # and the same off-stride value with speculation.
    dict(name="stated-max-2048", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=256, max_num_batched_tokens=8192, max_capture=2048),
    dict(name="stated-max-2", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=256, max_num_batched_tokens=8192, max_capture=2),
    dict(name="stated-max-1000", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=256, max_num_batched_tokens=8192, max_capture=1000),
    dict(name="stated-max-equals-batched-tokens", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=256, max_num_batched_tokens=1024, max_capture=1024),
    dict(name="mtp3-stated-max-1000", mode="FULL_AND_PIECEWISE", graph_mode="FULL_AND_PIECEWISE", max_num_seqs=128, max_num_batched_tokens=8192, spec_tokens=3, max_capture=1000),
]
# The configurations whose resolution reads the part, again on a data-center Blackwell part
# (platform.is_device_capability_family(100)), where the unstated ceiling is 1024 rather than 512,
# and the stated values the dashboard's points use. The refusals do not depend on the part.
BLACKWELL_TWINS = ("tiny-seqs-ceiling-8", "batched-tokens-off-stride", "seqs-off-stride", "small-seqs",
                   "piecewise", "full-decode-only", "none", "stated-max", "stated-max-above-batched",
                   "stated-sizes", "mtp3", "mtp1-small", "stated-max-is-the-largest-stated-size",
                   "stated-max-2048", "stated-max-2", "stated-max-1000", "stated-max-equals-batched-tokens",
                   "mtp3-stated-max-1000")
CAPTURE_CASES += [dict(c, name=c["name"] + "-blackwell", blackwell=True) for c in CAPTURE_CASES
                  if c["name"] in BLACKWELL_TWINS]


def cases(ns):
    out = []
    for c in CAPTURE_CASES:
        r = set_cudagraph_sizes(ns, c)
        row = {"case": c, "resolved": r, "dispatch": []}
        if r["error"] is None and r["sizes"] and c["graph_mode"] != "NONE":
            m = manager(ns, c, r)
            q = 1 + c.get("spec_tokens", 0)
            probes = set()
            for s in r["sizes"][:14] + r["sizes"][-14:]:
                probes.update({s - 1, s, s + 1})
            probes.update({1, 2, q, 2 * q, c["max_num_seqs"], r["max"] + 1})
            for n in sorted(p for p in probes if p >= 1):
                for kind in ("uniform", "mixed"):
                    if kind == "uniform":
                        if n % q:
                            continue
                        reqs, uni, mql = n // q, q, q
                    else:
                        reqs, uni, mql = min(n, c["max_num_seqs"]), None, None
                    d = m.dispatch(reqs, n, uni, num_active_loras=0, max_query_len=mql if c.get("varlen") else None)
                    row["dispatch"].append({"reqs": reqs, "tokens": n, "uniform": uni, "got": desc_json(d)})
        out.append(row)
    return out


# Data-parallel synchronisation. The one replacement is the all-reduce: a transport (FID-1). With
# every rank's inputs known, the sum of the per-rank tensors is what the collective returns, so the
# harness builds the global tensor from the same six assignments the function makes and hands it
# back. Everything the function decides afterwards is vLLM's own text.
DP_CASES = [
    dict(name="uniform-decode-equal", spec_tokens=0, mode="FULL_AND_PIECEWISE", seqs=256, mnbt=8192,
         ranks=[dict(reqs=8, tokens=8, mql=1, uniform=1), dict(reqs=8, tokens=8, mql=1, uniform=1)]),
    dict(name="uniform-decode-unequal", spec_tokens=0, mode="FULL_AND_PIECEWISE", seqs=256, mnbt=8192,
         ranks=[dict(reqs=3, tokens=3, mql=1, uniform=1), dict(reqs=20, tokens=20, mql=1, uniform=1), dict(reqs=9, tokens=9, mql=1, uniform=1)]),
    dict(name="mixed-with-prefill", spec_tokens=0, mode="FULL_AND_PIECEWISE", seqs=256, mnbt=8192,
         ranks=[dict(reqs=2, tokens=300, mql=None, uniform=None), dict(reqs=5, tokens=5, mql=1, uniform=1)]),
    dict(name="one-rank-over-ceiling", spec_tokens=0, mode="FULL_AND_PIECEWISE", seqs=256, mnbt=8192,
         ranks=[dict(reqs=2, tokens=2000, mql=None, uniform=None), dict(reqs=5, tokens=5, mql=1, uniform=1)]),
    dict(name="decode-only-mode-mixed-batch", spec_tokens=0, mode="FULL_DECODE_ONLY", seqs=128, mnbt=4096,
         ranks=[dict(reqs=2, tokens=60, mql=None, uniform=None), dict(reqs=4, tokens=4, mql=1, uniform=1)]),
    dict(name="spec-uniform", spec_tokens=3, mode="FULL_AND_PIECEWISE", seqs=64, mnbt=8192,
         ranks=[dict(reqs=3, tokens=12, mql=4, uniform=4), dict(reqs=10, tokens=40, mql=4, uniform=4)]),
    dict(name="spec-ranks-disagree-on-uniform", spec_tokens=3, mode="FULL_AND_PIECEWISE", seqs=64, mnbt=8192,
         ranks=[dict(reqs=3, tokens=12, mql=4, uniform=4), dict(reqs=6, tokens=30, mql=None, uniform=None)]),
    dict(name="idle-rank-runs-dummy", spec_tokens=0, mode="FULL_AND_PIECEWISE", seqs=256, mnbt=8192,
         ranks=[dict(reqs=0, tokens=0, mql=None, uniform=None, dummy=True), dict(reqs=12, tokens=12, mql=1, uniform=1)]),
    dict(name="idle-rank-runs-dummy-spec", spec_tokens=3, mode="FULL_AND_PIECEWISE", seqs=64, mnbt=8192,
         ranks=[dict(reqs=0, tokens=0, mql=None, uniform=None, dummy=True), dict(reqs=7, tokens=28, mql=4, uniform=4)]),
    dict(name="all-idle", spec_tokens=0, mode="FULL_AND_PIECEWISE", seqs=256, mnbt=8192,
         ranks=[dict(reqs=0, tokens=0, mql=None, uniform=None), dict(reqs=0, tokens=0, mql=None, uniform=None)]),
    dict(name="piecewise-only", spec_tokens=0, mode="PIECEWISE", seqs=256, mnbt=8192,
         ranks=[dict(reqs=4, tokens=70, mql=None, uniform=None), dict(reqs=9, tokens=9, mql=1, uniform=1)]),
    dict(name="none", spec_tokens=0, mode="NONE", seqs=256, mnbt=8192,
         ranks=[dict(reqs=4, tokens=70, mql=None, uniform=None), dict(reqs=9, tokens=9, mql=1, uniform=1)]),
    dict(name="full-mode-decodes", spec_tokens=0, mode="FULL", seqs=128, mnbt=4096,
         ranks=[dict(reqs=3, tokens=3, mql=1, uniform=1), dict(reqs=20, tokens=20, mql=1, uniform=1)]),
    dict(name="full-mode-mixed-batch-pads-to-the-request-bound", spec_tokens=0, mode="FULL", seqs=16, mnbt=4096,
         ranks=[dict(reqs=2, tokens=300, mql=None, uniform=None), dict(reqs=9, tokens=9, mql=1, uniform=1)]),
    dict(name="varlen-ranks-differ-in-query-bound", spec_tokens=3, mode="FULL_AND_PIECEWISE", varlen=True, seqs=64, mnbt=8192,
         ranks=[dict(reqs=3, tokens=15, mql=5, uniform=None), dict(reqs=8, tokens=16, mql=2, uniform=None)]),
    dict(name="varlen-ranks-within-the-graph-bound", spec_tokens=3, mode="FULL_AND_PIECEWISE", varlen=True, seqs=64, mnbt=8192,
         ranks=[dict(reqs=3, tokens=9, mql=4, uniform=None), dict(reqs=8, tokens=16, mql=2, uniform=None)]),
]


def dp_cases(ns):
    torch = ns["torch"]
    out = []
    for c in DP_CASES:
        base = dict(mode=c["mode"], graph_mode=c["mode"], max_num_seqs=c["seqs"], max_num_batched_tokens=c["mnbt"],
                    spec_tokens=c["spec_tokens"], varlen=c.get("varlen", False))
        r = set_cudagraph_sizes(ns, base)
        m = manager(ns, base, r) if (r["sizes"] and c["mode"] != "NONE") else None
        n = len(c["ranks"])
        dql = 1 + c["spec_tokens"]
        # A rank with no work runs the dummy batch: one uniform decode of decode_query_len tokens
        # (execute_dummy_batch, gpu_worker.py:1431-1437; _dummy_run with uniform_decode).
        c = dict(c, ranks=[dict(reqs=1, tokens=dql, mql=dql, uniform=dql) if k.get("dummy") else k for k in c["ranks"]])
        results = []
        # Build the global tensor once, from each rank's own descriptor as the function would.
        descs = []
        for k in c["ranks"]:
            if m is None:
                d = ns["BatchExecutionDescriptor"](cg_mode=ns["CUDAGraphMode"].NONE, num_tokens=k["tokens"], num_reqs=k["reqs"])
            else:
                d = m.dispatch(k["reqs"], k["tokens"], k["uniform"], num_active_loras=0, max_query_len=k["mql"])
            descs.append(d)
        g = torch.zeros(6, n, dtype=torch.int32)
        for i, k in enumerate(c["ranks"]):
            g[0][i] = k["tokens"]
            g[1][i] = descs[i].cg_mode.value
            g[2][i] = k["uniform"] or 0
            g[3][i] = k["mql"] or -1
            g[4][i] = 0
            g[5][i] = k["reqs"]
        ns["dist"] = types.SimpleNamespace(all_reduce=lambda t, group=None, _g=g: t.copy_(_g))
        ns["sync_cudagraph_and_dp_padding"].__globals__["dist"] = ns["dist"]
        for i, k in enumerate(c["ranks"]):
            if m is None and any(d.cg_mode.value for d in descs):
                continue
            desc, sync = ns["sync_cudagraph_and_dp_padding"](
                m, descs[i], k["tokens"], k["reqs"], k["uniform"], n, i, max_query_len=k["mql"])
            results.append({"rank": i, "mode": desc.cg_mode.name, "tokens": desc.num_tokens, "reqs": desc.num_reqs,
                            "sync": None if sync is None else {"eager": sync.eager, "reqs": sync.num_reqs,
                                                                "across": [int(x) for x in sync.num_tokens_across_dp]}})
        out.append({"case": c, "decode_query_len": dql, "results": results})
    return out


# Engram host tables. EngramConfig's resolution and checks, and the shard an EngramTable holds, are
# run as written. The engine's `EngramTable.__init__` cannot run (it needs a GPU), so the rows a
# shard holds are computed in core/lower from the head sizes and the shard it is given, and tested
# there against the source's formulas, not against an oracle.
ENGRAM_CASES = [
    dict(stated=None, cpu_offload=True, use_thp=False, dp=1, elastic=False),
    dict(stated=None, cpu_offload=True, use_thp=False, dp=2, elastic=False),
    dict(stated=None, cpu_offload=True, use_thp=False, dp=4, elastic=True),
    dict(stated=None, cpu_offload=True, use_thp=True, dp=2, elastic=False),
    dict(stated=None, cpu_offload=False, use_thp=False, dp=2, elastic=False),
    dict(stated=True, cpu_offload=True, use_thp=False, dp=2, elastic=False),
    dict(stated=True, cpu_offload=True, use_thp=False, dp=1, elastic=False),
    dict(stated=True, cpu_offload=False, use_thp=False, dp=2, elastic=False),
    dict(stated=True, cpu_offload=True, use_thp=False, dp=2, elastic=True),
    dict(stated=True, cpu_offload=True, use_thp=True, dp=2, elastic=False),
    dict(stated=False, cpu_offload=True, use_thp=False, dp=2, elastic=False),
    dict(stated=False, cpu_offload=False, use_thp=False, dp=1, elastic=False),
    dict(stated=None, cpu_offload=False, use_thp=True, dp=2, elastic=False),
    dict(stated=False, cpu_offload=True, use_thp=True, dp=2, elastic=False),
    dict(stated=True, cpu_offload=False, use_thp=True, dp=2, elastic=False),
    dict(stated=None, cpu_offload=True, use_thp=True, dp=1, elastic=False),
]
SHARD_CASES = [dict(tp=tp, tp_rank=r, edp=edp, dp_rank=d, shared=sh)
               for tp in (1, 4, 8) for edp in (1, 2, 4) for sh in (False, True)
               for r in sorted({0, tp - 1}) for d in sorted({0, edp - 1})]


def engram_cases(ns):
    out = {"resolve": [], "shard": [], "parallel_size": []}
    for c in ENGRAM_CASES:
        cfg = types.SimpleNamespace(dp_shared_memory=c["stated"], cpu_offload=c["cpu_offload"], use_thp=c["use_thp"],
                                    embedding_across_dp=False)
        pc = types.SimpleNamespace(data_parallel_size=c["dp"], enable_elastic_ep=c["elastic"], tensor_parallel_size=8)
        row = {"case": c}
        try:
            # The config's own validator runs on the stated values, when the config is built; the
            # parallel checks run once the parallel config is known.
            ns["engram__validate_shared_memory"](cfg)
            ns["engram_resolve_dp_shared_memory"](cfg, pc)
            row["resolved"] = bool(cfg.dp_shared_memory)
            ns["engram_verify_parallel_config"](cfg, pc)
            row["error"] = None
        except ValueError as e:
            row["resolved"] = bool(cfg.dp_shared_memory)
            row["error"] = str(e)
        out["resolve"].append(row)
    for c in SHARD_CASES:
        ns["_cur"] = c
        self_ = types.SimpleNamespace(dp_size=1 if c["shared"] else c["edp"], tp_size=c["tp"])
        out["shard"].append({"case": c, "got": list(ns["_get_shard_info"](self_))})
    for tp, dp, across in itertools.product((1, 4, 8), (1, 2, 4), (False, True)):
        cfg = types.SimpleNamespace(embedding_across_dp=across)
        pc = types.SimpleNamespace(tensor_parallel_size=tp, data_parallel_size=dp)
        out["parallel_size"].append({"tp": tp, "dp": dp, "across": across, "got": ns["engram_get_parallel_size"](cfg, pc)})
    return out


# The drafter pass of a parallel-block drafter (DFlash): the draft query rows are dispatched on the
# drafter's own manager, through the target's agreement rebuilt by _build_uniform_batch_dp_sync and
# dispatch_cg_and_sync_dp, both run as written. The branch that stores context rows eagerly sits
# in a method that needs a GPU; its inputs are recorded here (the descriptor and the rows the
# captured graph stores) and the rule over them is stated in core/lower and its test.
DRAFTER_CASES = [
    dict(name="one-rank", steps=3, reqs=3, target=12, sync=None),
    dict(name="one-rank-padded", steps=5, reqs=9, target=58, sync=None),
    dict(name="group-agreed-graph", steps=3, reqs=3, target=12, sync=dict(reqs=40, eager=False)),
    dict(name="group-agreed-eager", steps=3, reqs=3, target=12, sync=dict(reqs=40, eager=True)),
    dict(name="group-agreed-eager-narrow", steps=3, reqs=3, target=12, sync=dict(reqs=3, eager=True)),
    dict(name="group-agreed-graph-wide", steps=5, reqs=9, target=58, sync=dict(reqs=33, eager=False)),
]


def drafter_cases(ns):
    torch = ns["torch"]
    out = []
    for c in DRAFTER_CASES:
        base = dict(mode="FULL_DECODE_ONLY", graph_mode="FULL_DECODE_ONLY", max_num_seqs=64, max_num_batched_tokens=8192,
                    spec_tokens=c["steps"])
        r = set_cudagraph_sizes(ns, base)
        m = manager(ns, base, r)
        nq = 1 + c["steps"]
        if c["sync"] is None:
            desc, _ = ns["dispatch_cg_and_sync_dp"](m, c["reqs"], c["reqs"] * nq, nq, dp_size=1, dp_rank=0)
        else:
            target = ns["DPSyncState"](
                num_tokens_across_dp=torch.tensor([c["target"], c["target"]], dtype=torch.int32),
                uniform_token_count=None, eager=c["sync"]["eager"], num_reqs=c["sync"]["reqs"])
            batch_sync, num_batch_tokens = ns["_build_uniform_batch_dp_sync"](None, target, c["reqs"], nq)
            desc, _ = ns["dispatch_cg_and_sync_dp"](m, c["reqs"], num_batch_tokens, nq, dp_size=2, dp_rank=0,
                                                    dp_sync_state=batch_sync)
        ctx = None
        if desc.cg_mode.name == "FULL":
            s = types.SimpleNamespace(num_speculative_steps=c["steps"], max_num_tokens=8192)
            ctx = ns["_num_graph_context_tokens"](s, desc.num_reqs)
        out.append({"case": c, "desc": desc_json(desc), "graph_context_rows": ctx})
    return out


def dcp_preferences(root):
    """The classes of MODELS_CONFIG_MAP whose verify_and_update_config hook calls set_dcp_defaults,
    with the literal arguments it passes, read by syntax tree from vllm/model_executor/models/config.py.
    A model's preference is whatever its hook passes; the stock one (ag_rs, no replication) applies
    to every model that passes none."""
    tree = parse(root, "vllm/model_executor/models/config.py")
    mapped = {}
    for n in tree.body:
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "MODELS_CONFIG_MAP":
            for k, v in zip(n.value.keys, n.value.values):
                mapped[k.value] = v.id
    calls = {}
    for n in tree.body:
        if not isinstance(n, ast.ClassDef):
            continue
        for c in ast.walk(n):
            if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "set_dcp_defaults":
                args = {kw.arg: kw.value.value for kw in c.keywords}
                for i, a in enumerate(c.args):
                    args[["comm_backend", "q_replicate"][i]] = a.value
                calls[n.name] = args
    out = {}
    for arch, cls in sorted(mapped.items()):
        if cls in calls:
            out[arch] = {"comm_backend": calls[cls].get("comm_backend", "ag_rs"),
                         "q_replicate": calls[cls].get("q_replicate", False)}
    return out


def dbo_cases(ns):
    """Dual-batch overlap: whether a step splits, and into how many ubatches."""
    out = {"split": [], "defaults": []}
    for enable, ubatch_size, dec, pre in itertools.product((False, True), (0, 1, 3), (32, 0), (512, 100)):
        cfg = types.SimpleNamespace(enable_dbo=enable, ubatch_size=ubatch_size, dbo_decode_token_threshold=dec,
                                    dbo_prefill_token_threshold=pre)
        use = ns["pc_use_ubatching"](cfg)
        cfg.use_ubatching = use
        row = {"enable_dbo": enable, "ubatch_size": ubatch_size, "decode": dec, "prefill": pre, "use": use,
               "num": ns["pc_num_ubatches"](cfg), "steps": []}
        for tokens, uniform in itertools.product((1, 31, 32, 33, 99, 100, 101, 511, 512, 513), (False, True)):
            row["steps"].append({"tokens": tokens, "uniform": uniform, "split": ns["ub_check_ubatch_thresholds"](cfg, tokens, uniform)})
        out["split"].append(row)
    for stated_b, stated_q, mb, mq in itertools.product((None, "a2a", "ag_rs"), (None, True, False), ("ag_rs", "a2a"), (False, True)):
        cfg = types.SimpleNamespace(dcp_comm_backend=stated_b, dcp_q_replicate=stated_q)
        ns["pc_set_dcp_defaults"](cfg, comm_backend=mb, q_replicate=mq)
        out["defaults"].append({"stated_backend": stated_b, "stated_q": stated_q, "model_backend": mb, "model_q": mq,
                                "backend": cfg.dcp_comm_backend, "q": cfg.dcp_q_replicate})
    return out


def decode_len_cases(ns):
    out = []
    for spec, par, v2 in itertools.product([None, 0, 1, 3, 7], [False, True], [False, True]):
        sc = None if spec is None else types.SimpleNamespace(num_speculative_tokens=spec, parallel_drafting=par)
        cfg = types.SimpleNamespace(speculative_config=sc, use_v2_model_runner=v2)
        out.append({"spec": spec, "parallel_drafting": par, "v2": v2, "got": ns["max_decode_query_len"](cfg)})
    return out


def context_cases(ns):
    f = ns["_num_graph_context_tokens"]
    out = []
    for steps, mnt, reqs in itertools.product([1, 3, 5], [64, 8192], [0, 1, 9, 10, 100, 2000]):
        s = types.SimpleNamespace(num_speculative_steps=steps, max_num_tokens=mnt)
        out.append({"steps": steps, "max_num_tokens": mnt, "reqs": reqs, "got": f(s, reqs)})
    return out


def build(root):
    ns = load(root)
    return {
        "dcp_preferences": dcp_preferences(root),
        "commit": COMMIT,
        "breakable": sorted(ns["DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES"]),
        "capture": cases(ns),
        "max_decode_query_len": decode_len_cases(ns),
        "graph_context_tokens": context_cases(ns),
        "dp_sync": dp_cases(ns),
        "drafter": drafter_cases(ns),
        "engram": engram_cases(ns),
        "dbo": dbo_cases(ns),
    }


def main(argv):
    if len(argv) < 3 or argv[2] not in ("print", "check"):
        raise SystemExit(__doc__)
    out = json.dumps(build(argv[1]), indent=1, sort_keys=True) + "\n"
    if argv[2] == "print":
        sys.stdout.write(out)
        return
    with open(argv[3], encoding="utf-8") as f:
        if f.read() != out:
            raise SystemExit(f"{argv[3]} is not what vLLM at {COMMIT} gives; regenerate it with `print`")
    print("ok:", argv[3])


if __name__ == "__main__":
    main(sys.argv)
