"""Run spec: an InferenceX inventory point -> a BLIS scenario, workload and command.

Every deployment value comes from the run's own vLLM server log (the InferenceX DB, by the
inventory's server_log_ids) or, without a log, from the recipe at the run's commit
(repos/cov/recipe_recon.jsonl). Where both are silent, the value is the default OF THE RUN'S
OWN vLLM VERSION (IMAGE_DEFAULTS, read with git show <tag>:<file>), never v0.31.0's. Every
setting BLIS cannot state is a named MISMATCH, and every resolved default is recorded with its
source, so a point's provenance shows how each scenario field was decided.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys as _s
from pathlib import Path
_s.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402

# Defaults by vLLM release, each read from the tag (vllm repo, git show <tag>:<path>):
#   gmu: vllm/config/cache.py gpu_memory_utilization Field default (0.9 through v0.19.1, 0.92 from v0.20.0).
#   max_num_seqs / max_num_batched_tokens: vllm/engine/arg_utils.py get_batch_defaults, OPENAI_API_SERVER,
#     device >= 70 GiB and not A100: 1024 / 8192 at every tag here; v0.29.0+ adds >= 160 GiB -> 1024 / 16384.
#   block_size: vllm/config/cache.py DEFAULT_BLOCK_SIZE 16 (model/backends may override; the log is authoritative).
#   v2 runner: vllm/envs.py VLLM_USE_V2_MODEL_RUNNER default "0" (V1) through v0.21.0; None (auto, V2 unless
#     unsupported, vllm/config/vllm.py use_v2_model_runner) from v0.25.1 on.
#   swa_bounded_replay: introduced by vllm 1dc2d854c1 (2026-09-18, #56227); absent (off) before it.
VERSION_ORDER = ["0.11.2", "0.13.0", "0.15.1", "0.18.0", "0.20.1", "0.20.2", "0.21.0", "0.25.1", "0.27.1",
                 "0.29.0", "0.29.1", "0.31.0"]


# A "0.1.devN+g<sha>" build (a source build without tags) has N commits of history; its release line is the
# last tag whose `git rev-list --count <tag>` is <= N (counts from the vllm repo).
TAG_COUNTS = [((0, 11, 2), 11391), ((0, 13, 0), 12291), ((0, 15, 1), 13311), ((0, 16, 0), 13751), ((0, 17, 0), 14447),
              ((0, 18, 0), 14900), ((0, 19, 0), 15355), ((0, 20, 0), 16107), ((0, 21, 0), 16499), ((0, 22, 0), 16958),
              ((0, 23, 0), 17366), ((0, 24, 0), 17936), ((0, 25, 0), 18494), ((0, 26, 0), 18906), ((0, 27, 0), 19467),
              ((0, 28, 0), 20051), ((0, 29, 0), 20645), ((0, 30, 0), 21403), ((0, 31, 0), 22120)]


def vtuple(v: str) -> tuple:
    m = re.match(r"0\.1\.dev(\d+)", v or "")
    if m:
        n = int(m.group(1))
        return max((t for t, c in TAG_COUNTS if c <= n), default=(0, 0, 0))
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", v or "")
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def version_defaults(ver: str | None, device_gib: float) -> dict:
    v = vtuple(ver or "")
    big = device_gib >= 160 and v >= (0, 28, 0)  # #51726 (IMAGE_RESOLUTION.md row 7)
    return {
        "gpu_memory_utilization": 0.92 if v >= (0, 20, 0) else 0.9,
        "max_num_seqs": 1024,
        "max_num_batched_tokens": 16384 if big else 8192,
        "block_size": 16,
        "model_runner": ("v2" if v >= (0, 29, 0) else "allowlist(v0.22-v0.28)" if v >= (0, 22, 0) else "v1"),
        "_source": f"vllm v{'.'.join(map(str, v))} defaults (arg_utils.get_batch_defaults, config/cache.py, envs.py)",
    }


def git_ancestor(vllm_repo: str, a: str, b: str) -> bool | None:
    r = subprocess.run(["git", "-C", vllm_repo, "merge-base", "--is-ancestor", a, b], capture_output=True)
    return {0: True, 1: False}.get(r.returncode)


# ---------------------------------------------------------------- server log parsing
def _first(pat, text, g=1):
    m = re.search(pat, text)
    return m.group(g) if m else None


def log_block_size(text: str) -> str | None:
    """The engine's final cache_config.block_size as the log states it.

    vLLM v0.31.0 Platform.update_block_size_for_backend first picks the backends' common size
    ("Setting kv cache block size to N for X backend(s)", vllm/platforms/interface.py:693), then a
    hybrid model raises it so one attention page covers a Mamba/KDA state or a padded higher-precision
    page ("Setting attention block size to N tokens ...", :809 and :990). Each later line assigns
    cache_config.block_size again, and in mamba "align" mode mamba_block_size follows it (:996-997),
    so prefix hits land on that block (single_type_kv_cache_manager.py:1488-1560). The last
    attention-block line therefore wins; without one, the kv-cache line.
    """
    attn = re.findall(r"Setting attention block size to (\d+) tokens", text)
    return attn[-1] if attn else _first(r"Setting kv cache block size to (\d+)", text)


def parse_non_default(text: str) -> dict | None:
    nd = _first(r"non-default args: (\{.*\})", text)
    if not nd:
        return None
    nd = re.sub(r"EngramConfig\(([^)]*)\)", lambda m: "{" + ",".join(
        f"'{k}': {v}" for k, v in (kv.split("=") for kv in m.group(1).split(", "))) + "}", nd)
    nd = re.sub(r"<CompilationMode\.\w+: (\d+)>", r"\1", nd)
    nd = re.sub(r"<CUDAGraphMode\.(\w+): [^>]*>", r"'\1'", nd)
    nd = re.sub(r"<DynamicShapesType\.\w+: '(\w+)'>", r"'\1'", nd)
    nd = _quote_calls(nd)
    try:
        return ast.literal_eval(nd)
    except (ValueError, SyntaxError):
        return {"_unparsed": nd[:4000]}


def _quote_calls(nd: str) -> str:
    """Replace every Name(...) object repr (AttentionConfig(...), KVTransferConfig(...), ...) by its text as
    a string, matching parentheses, so the surrounding dict parses; the raw text stays inspectable."""
    out, i = [], 0
    for m in re.finditer(r"(?<![\w'\"])([A-Z][A-Za-z0-9_]*)\(", nd):
        if m.start() < i:
            continue
        depth, j = 0, m.end() - 1
        while j < len(nd):
            depth += {"(": 1, ")": -1}.get(nd[j], 0)
            if depth == 0:
                break
            j += 1
        out.append(nd[i:m.start()])
        out.append(repr(nd[m.start():j + 1]))
        i = j + 1
    out.append(nd[i:])
    return "".join(out)


def parse_log(text: str) -> dict:
    """Engine facts the log states (each None when the log is silent)."""
    caps = _first(r"'cudagraph_capture_sizes': (\[[0-9, ]*\])", text)
    kvt = _first(r"kv_connector_extra_config=(\{.*?\})(?:, kv_connector_module_path|\))", text)
    return {
        "non_default": parse_non_default(text),
        "vllm_version": _first(r"Initializing a V1 LLM engine \(v([^)]+)\)", text),
        "gmu": _first(r"memory utilization is \(([\d.]+),", text),
        "chunk_tokens": _first(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)", text),
        "block_size": log_block_size(text),
        "kv_tokens": [int(x.replace(",", "")) for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)],
        "cudagraph_mode": _first(r"'cudagraph_mode': <CUDAGraphMode\.(\w+)", text),
        "capture_sizes": ast.literal_eval(caps) if caps else None,
        "enforce_eager": _first(r"enforce_eager=(\w+)", text),
        "prefix_caching": _first(r"enable_prefix_caching=(\w+)", text),
        "kv_cache_dtype": _first(r"kv_cache_dtype=([a-z0-9_]+)", text),
        "kv_dtype_used": _first(r"Using ([a-z0-9_]+) data type to store kv cache", text)
                         or ("fp8_ds_mla" if "fp8_ds_mla KV cache" in text else None),
        "quantization": _first(r"quantization=([a-z0-9_]+)", text),
        "max_seq_len": _first(r"max_seq_len=(\d+)", text),
        "dp": _first(r"data_parallel_size=(\d+)", text),
        "rust_frontend": "Launching Rust frontend" in text,
        "async": True if "Asynchronous scheduling is enabled" in text else
                 (False if "Asynchronous scheduling is disabled" in text else None),
        "v2_runner_log": True if "Using V2 Model Runner" in text else None,
        "a2a_manager": _first(r"Using (\w+All2AllManager)", text),
        "kv_connector": _first(r"kv_connector='(\w+)'", text),
        "kv_connector_extra": kvt,
        "mean_al": [float(x) for x in re.findall(r"Mean acceptance length: ([\d.]+)", text)],
        "weight_offload_gb": _first(r"'cpu_offload_gb': ([\d.]+)", text),
    }


A2A_MANAGER = {"AgRsAll2AllManager": "allgather_reducescatter", "NaiveAll2AllManager": "naive",
               "DeepEPHTAll2AllManager": "deepep_high_throughput", "DeepEPLLAll2AllManager": "deepep_low_latency",
               "PplxAll2AllManager": "pplx", "FlashInferAllToAllManager": "flashinfer_all2allv"}


# ---------------------------------------------------------------- resolution
HW = {"h100": ("h100", 8, 79.19), "h200": ("h200", 8, 139.8), "b200": ("b200", 8, 178.4),
      "b300": ("b300", 8, 268.0), "gb200": ("gb200-nvl72", 4, 185.0), "gb300": ("gb300", 4, 268.0)}
COEFFS = ("cost-model-primitives, cost-model-collectives, cost-model-host-overheads, "
          "cost-model-attention, cost-model-recurrent, cost-model-memory")
SPEC_METHOD = {"mtp": "mtp", "dspark": "dspark", "eagle3": "eagle3", "eagle": "eagle", "draft_model": "draft_model",
               "deepseek_mtp": "mtp"}


def resolve(inv: dict, log: dict | None, recon: dict | None, conc: int, catalog_models: dict) -> dict:
    """Resolve every scenario field for one point, each with its source; collect MISMATCHes."""
    mism, src, d = [], {}, {}

    def put(k, v, s):
        d[k] = v
        src[k] = s

    gpu = inv["gpu"]
    if gpu not in HW:
        raise Unexpressible(f"gpu {gpu!r} has no catalog hardware entry")
    hw, gpn, dev_gib = HW[gpu]
    nd = (log or {}).get("non_default") or {}
    if "_unparsed" in nd:
        raise Unexpressible("server log non-default args did not parse")
    ver = (log or {}).get("vllm_version") or inv.get("engine_version") or _ver_from_image(inv.get("engine_image"))
    dflt = version_defaults(ver, dev_gib)
    put("vllm_version", ver, "server log" if (log or {}).get("vllm_version") else "image tag")
    put("engine_image", inv.get("engine_image"), "inventory")

    # model: the SERVED checkpoint selects the catalog entry (quantized variants are their own entries)
    served = nd.get("model") or inv.get("hf_repo_served") or inv.get("hf_repo")
    model = catalog_models.get(_repo_tail(served)) or catalog_models.get(_repo_tail(inv.get("hf_repo")))
    if not model and log is None and inv.get("model_slug") == "llama70b":
        raise Unexpressible("no server log; the 70b recipe at the run commit (.github/workflows/70b-tmpl.yml) serves "
                            "nvidia/Llama-3.3-70B-Instruct-FP8 (FP4/FP8 NVIDIA checkpoints), not the inventory's "
                            "meta-llama/Llama-3.1-70B: no catalog entry for the served checkpoint")
    if not model:
        raise Unexpressible(f"served checkpoint {served!r} has no catalog model")
    if _repo_tail(served) not in catalog_models and _repo_tail(served) != _repo_tail(inv.get("hf_repo")):
        mism.append(_mm("served checkpoint", served, model, False))
    put("model", model, f"catalog repo match ({served})")

    par = inv["parallel"]["decode"]
    if inv["parallel"].get("disagg") or inv.get("is_multinode"):
        raise Unexpressible("P/D or multinode serving (Dynamo / llm-d disagg) is not replayed by this harness yet")
    # A logged run omits a flag left at vLLM's default (tensor/data_parallel_size 1), so with a log the
    # default applies, never the inventory's TP (which InferenceX records as the attention width for
    # DP-attention runs: decode_tp 8 with dp_attention is DP8 x TP1).
    if log is not None:
        tp = int(nd.get("tensor_parallel_size", 1))
        dp = int(nd.get("data_parallel_size", (log or {}).get("dp") or 1))
    else:
        tp = int((recon or {}).get("tensor-parallel-size", "").replace("$TP", "") or par.get("tp") or 1)
        dp = int((recon or {}).get("data-parallel-size", "").replace("$TP", "") or 1)
    put("tp", tp, "server log (default 1 when unstated)" if log is not None else "recipe/inventory")
    put("dp", dp, "server log" if "data_parallel_size" in nd else "default 1")
    put("enable_expert_parallel", bool(nd.get("enable_expert_parallel", par.get("ep", 1) > 1)), "server log/inventory")
    nodes = -(-tp * dp // gpn)
    # InferenceX's single-node GPU count is its TP env (utils/process_result.py divides throughput by TP);
    # the inventory's decode.gpus is tp^2 for these rows and is not used.
    if tp * dp != int(par.get("tp") or tp * dp):
        mism.append(_mm("GPU count", f"{par.get('tp')} GPUs (InferenceX TP env)", f"tp {tp} x dp {dp}", False,
                        "layout from the log disagrees with InferenceX's GPU count"))
    if nodes > 1:
        raise Unexpressible(f"tp {tp} x dp {dp} spans {nodes} nodes: multinode layouts need a fabric the run does not state")
    put("nodes", nodes, "derived tp*dp/gpus_per_node")
    put("hardware", hw, f"inventory gpu {gpu}")

    eng = {}
    def e(k, v, s):
        eng[k] = v
        src[f"engine.{k}"] = s
    bs = (log or {}).get("block_size") or nd.get("block_size")
    if bs is None and recon:
        bs = recon.get("block-size")
    e("block_size", int(bs) if bs else dflt["block_size"], "server log" if bs else dflt["_source"])
    mns = nd.get("max_num_seqs") or (recon or {}).get("max-num-seqs")
    e("max_num_seqs", int(mns) if mns else dflt["max_num_seqs"], "non-default args / recipe" if mns else dflt["_source"])
    ch = (log or {}).get("chunk_tokens") or nd.get("max_num_batched_tokens") or (recon or {}).get("max-num-batched-tokens")
    e("max_num_batched_tokens", int(ch) if ch else dflt["max_num_batched_tokens"], "server log" if ch else dflt["_source"])
    mml = nd.get("max_model_len") or (log or {}).get("max_seq_len") or (recon or {}).get("max-model-len")
    if mml is None:
        raise Unexpressible("max_model_len unstated (no log, no recipe)")
    e("max_model_len", int(mml), "non-default args / log max_seq_len / recipe")
    gmu = (log or {}).get("gmu") or nd.get("gpu_memory_utilization") or (recon or {}).get("gpu-memory-utilization")
    e("gpu_memory_utilization", float(gmu) if gmu else dflt["gpu_memory_utilization"], "server log" if gmu else dflt["_source"])
    cgm = (log or {}).get("cudagraph_mode")
    if (log or {}).get("enforce_eager") == "True":
        cgm = "NONE"
    e("cudagraph_mode", cgm or "FULL_AND_PIECEWISE", "server log" if cgm else "default FULL_AND_PIECEWISE (v0.11+)")
    caps = (log or {}).get("capture_sizes")
    if caps:
        e("cudagraph_capture_sizes", caps, "server log (resolved sizes)")
        e("max_cudagraph_capture_size", max(caps), "server log (max of resolved sizes)")
    elif (recon or {}).get("cuda-graph-sizes"):
        vals = [int(x) for x in str(recon["cuda-graph-sizes"]).split()]
        vt = vtuple(ver or "")
        if vt <= (0, 11, 0):  # SchedulerConfig.cuda_graph_sizes: one N = ceiling on [1,2,4]+range(8,N+1,8)
            sizes = ([1, 2, 4] + list(range(8, vals[0] + 1, 8))) if len(vals) == 1 else vals
            how = "recipe cuda-graph-sizes as a ceiling (config/__init__.py:3652@v0.10.2, :593@v0.11.0)"
        elif vt <= (0, 11, 2):  # deprecated alias of --cudagraph-capture-sizes: exact list
            sizes, how = vals, "recipe cuda-graph-sizes as an exact list (arg_utils.py:1119,1754@v0.11.1)"
        else:
            raise Unexpressible("recipe passes --cuda-graph-sizes, which v0.13.0+ refuses (#30396)")
        cap = eng.get("max_num_batched_tokens")
        sizes = [x for x in sizes if not cap or x <= cap]
        e("cudagraph_capture_sizes", sizes, how)
        e("max_cudagraph_capture_size", max(sizes), how)
    elif nd.get("max_cudagraph_capture_size") or (recon or {}).get("max-cudagraph-capture-size"):
        e("max_cudagraph_capture_size", int(nd.get("max_cudagraph_capture_size") or recon["max-cudagraph-capture-size"]),
          "non-default args / recipe")
    pc = (log or {}).get("prefix_caching")
    if pc is None and recon:
        pc = "False" if recon.get("no-enable-prefix-caching") == "true" else None
    e("enable_prefix_caching", pc != "False" if pc is not None else True, "server log/recipe" if pc else "default on")
    kvd = (log or {}).get("kv_dtype_used") or (log or {}).get("kv_cache_dtype") or (recon or {}).get("kv-cache-dtype")
    if kvd and kvd not in ("auto",):
        e("cache_dtype", "fp8" if kvd.startswith("fp8") else kvd, f"server log ({kvd})")
    fe = "rust" if (log or {}).get("rust_frontend") else "python"
    if log is None and inv.get("frontend"):
        fe = inv["frontend"]
    e("frontend", fe, "server log (Launching Rust frontend)" if fe == "rust" and log else "server log/inventory")
    lpt = nd.get("long_prefill_token_threshold") or (recon or {}).get("long-prefill-token-threshold")
    if lpt and int(lpt) > 0:
        e("long_prefill_token_threshold", int(lpt), "non-default args / recipe")
    unstated_scheduler_args(nd, recon, mism, d)
    asy = (log or {}).get("async")
    if asy is None and recon and "async-scheduling" in recon:
        asy = recon["async-scheduling"] == "true"
    if asy is not None:
        e("async_scheduling", bool(asy), "server log / recipe")
    # version-dependent DSV4.1 behaviour: SWA bounded replay exists only from vllm 1dc2d854c1 (#56227)
    if model == "deepseek-v4.1-flash":
        commit = _commit_from_image(inv.get("engine_image"))
        has = None
        if commit:
            has = git_ancestor(VLLM_REPO, "1dc2d854c1", commit)
        if has is False:
            e("swa_bounded_replay", False, f"image commit {commit[:10]} predates vllm 1dc2d854c1 (#56227)")
        elif has is None:
            e("swa_bounded_replay", False, "image predates 2026-09-18 by date or commit unknown: feature absent")
    a2a_mgr = (log or {}).get("a2a_manager")
    a2a = A2A_MANAGER.get(a2a_mgr) if a2a_mgr else nd.get("all2all_backend")
    if a2a and (dp > 1 or eng.get("enable_expert_parallel")):
        e("all2all_backend", a2a, f"server log {a2a_mgr}" if a2a_mgr else "non-default args")
    d["a2a_resolved"] = a2a or ("allgather_reducescatter (default, every tag)" if dp > 1 else "none (no EP/DP)")

    # speculative decoding
    sp = nd.get("speculative_config")
    spec_method = None
    al = None
    if isinstance(sp, dict):
        spec_method = SPEC_METHOD.get(sp.get("method"), sp.get("method"))
        e("speculative", {"method": spec_method, "num_spec_tokens": int(sp.get("num_speculative_tokens", 1))},
          "non-default args speculative_config")
        if sp.get("rejection_sample_method") == "synthetic" and sp.get("synthetic_acceptance_length"):
            al = float(sp["synthetic_acceptance_length"])
            src["acceptance"] = "synthetic_acceptance_length (forced by the run)"
        else:
            src["acceptance"] = "registry inferencex-golden-al (model, method, K, coding)"
            mism.append(_mm("acceptance", "natural acceptance on the random/agentic prompts",
                            "registry golden AL (SPEED-Bench coding); B5: no per-request spread", False))
        if sp.get("model") and spec_method in ("eagle3", "eagle", "draft_model"):
            d["drafter"] = sp["model"]
    elif inv.get("spec_method") not in (None, "none"):
        if log is None:
            raise Unexpressible(f"spec {inv['spec_method']} without a log or recipe speculative_config")
    d["acceptance_length"] = al
    d["spec_method"] = spec_method

    # engram (DSV4.1)
    if isinstance(nd.get("engram_config"), dict):
        e("engram_config", {"cpu_offload": bool(nd["engram_config"].get("cpu_offload"))}, "non-default args")

    # KV offload
    conn = (log or {}).get("kv_connector")
    if inv.get("offload") == "on" or conn:
        raw = (log or {}).get("kv_connector_extra") or ""
        cpu_bytes = _first(r"'cpu_bytes_to_use': (\d+)", raw)
        sec = "'secondary_tiers'" in raw
        if conn != "OffloadingConnector":
            raise Unexpressible(f"kv connector {conn or inv.get('kv_connector')} is not simulated "
                                "(BLIS models vLLM's OffloadingConnector only)")
        if not cpu_bytes:
            raise Unexpressible("OffloadingConnector without a stated cpu_bytes_to_use")
        d["offload"] = {"connector": "OffloadingConnector",
                        "spec": _first(r"'spec_name': '(\w+)'", raw) or "CPUOffloadingSpec",
                        "eviction_policy": _first(r"'eviction_policy': '(\w+)'", raw), "cpu_bytes": int(cpu_bytes)}
        if sec:
            mism.append(_mm("offload secondary tier", "fs tier " + (_first(r"'root_dir': '([^']+)'", raw) or ""),
                            "not stated in the scenario (the fs tier needs root_dir/direct_io/threads blis-schemas "
                            "does not state)", False))
    wo = (log or {}).get("weight_offload_gb")
    if wo and float(wo) > 0:
        raise Unexpressible(f"cpu_offload_gb={wo} (UVA weight offload) is not modelled")
    if (log or {}).get("quantization") and catalog_models.get("_quant_" + model) is None:
        d["quantization_log"] = log["quantization"]
    # model runner: BLIS resolves v0.31.0's (V2 unless a feature forces V1); the schema has no field for it
    d["model_runner_image"] = "v2" if (log or {}).get("v2_runner_log") else dflt["model_runner"]
    apply_build(d, mism, inv.get("engine_image"), ver, model, spec_method, bool((log or {}).get("cudagraph_mode")))
    d["engine"] = eng
    d["sources"] = src
    d["mismatches"] = mism
    d["gpus_per_node"] = gpn
    if log is None:
        mism.append(_mm("server log", "absent", f"config from recipe {recon.get('_provenance') if recon else '?'}; "
                        "the logged KV pool and resolved defaults are unavailable", False))
    if ver and vtuple(ver) != (0, 31, 0):
        mism.append(_mm("vLLM engine", f"v{ver} ({inv.get('engine_image')})",
                        "BLIS kernel prices vLLM v0.31.0 mechanism; version-specific defaults stated above", False))
    return d


BEHAVIOUR_TABLE = str(PX.BEHAVIOUR)
# Cells blis-schemas d48d263 cannot state (IMAGE_RESOLUTION.md "Schema gaps"): a cell that differs from v0.31.0 is a
# named MISMATCH on the point, never silently v0.31.0's behaviour.
UNSTATEABLE = {"model_runner": "model runner V1/V2 (VLLM_USE_V2_MODEL_RUNNER)",
               "eagle_block_drop": "EAGLE/MTP/DSpark last-block drop (disable_eagle_block_drop)",
               "sample_methods": "draft/rejection sample methods", "padded_drafter_batch": "padded drafter batch"}
_BT = None


def build_for(image: str | None, ver: str | None):
    """The build-table entry for a run's image (or its +g<sha> version)."""
    global _BT
    if _BT is None:
        d = __import__("json").load(open(BEHAVIOUR_TABLE))
        _BT = {"by_image": {}, "by_build": {}}
        for b in d["builds"]:
            _BT["by_build"][b["build"]] = b
            for img in b["images"]:
                _BT["by_image"][img] = b
    if image and image in _BT["by_image"]:
        return _BT["by_image"][image]
    m = re.search(r"\+g([0-9a-f]{7,})", ver or "")
    if m:
        for k, b in _BT["by_build"].items():
            if k.startswith(m.group(1)) or m.group(1).startswith(k[:9]):
                return b
    return None


def apply_build(d: dict, mism: list, image, ver, model, spec_method, log_has_cgmode: bool):
    b = build_for(image, ver)
    if b is None:
        d["build"] = {"status": "NOT IN TABLE", "image": image}
        mism.append(_mm("build", f"{image} ({ver})", "not in behaviour_by_image.json", False))
        return
    st = b["resolution"]["status"]
    d["build"] = {"build": b["build"], "status": st, "ref": b["resolution"].get("ref"),
                  "nearest_release": b["resolution"].get("nearest_release")}
    if st == "UNRESOLVABLE":
        d["build"]["flag"] = "unresolvable build"
        mism.append(_mm("build", f"{b['build']} ({b['resolution'].get('reason', '')[:120]})",
                        "unresolvable build: version-dependent behaviour taken only from the run's own log", False))
        return
    beh = b.get("behaviours", {})
    cite = {k: v.get("citation") for k, v in beh.items()}
    d["build"]["citations"] = cite
    for k, label in UNSTATEABLE.items():
        c = beh.get(k)
        if not c or c.get("same_as_v0.31.0"):
            continue
        if k in ("eagle_block_drop", "sample_methods", "padded_drafter_batch") and not spec_method:
            continue
        if k == "model_runner":
            key = "dsv41_cuda" if model == "deepseek-v4.1-flash" else "dsv4_cuda" if model == "deepseek-v4-pro" else None
            if key and c.get(key) == "V2":
                continue
        mism.append(_mm(label, f"{c['value'][:160]} ({', '.join(c.get('citation', [])[:2])})",
                        "not stateable in blis-schemas d48d263: BLIS applies v0.31.0's behaviour", False))
    cg = beh.get("cudagraph_mode", {})
    if not log_has_cgmode and cg and not cg.get("same_as_v0.31.0") and "breakable" in cg.get("value", "") \
            and model in ("deepseek-v4-pro", "deepseek-v4.1-flash"):
        mism.append(_mm("breakable CUDA graphs / optimization_level", cg["value"][:160],
                        "not stateable; cudagraph_mode stated from the log where printed", False))


class Unexpressible(Exception):
    """The run has a setting BLIS cannot state: the point is blocked, with this reason."""


VLLM_REPO = str(PX.VLLM_REPO)


def unstated_scheduler_args(nd: dict, recon: dict | None, mism: list, d: dict) -> None:
    """Run args that change scheduling or client-visible streaming but have no blis-schemas field. Each is a
    named MISMATCH, never silently dropped. stream_interval > 1 also sets d["stream_interval"]: vLLM then sends
    one chunk per stream_interval tokens (vllm/v1/engine/output_processor.py:315-327@v0.31.0), so vllm bench's
    pooled per-chunk ITL is a gap of ~stream_interval tokens, which BLIS's per-step ITL is not."""
    rc = recon or {}
    si = nd.get("stream_interval") or rc.get("stream-interval")
    if si and int(si) > 1:
        d["stream_interval"] = int(si)
        mism.append(_mm("stream_interval", f"{si} tokens per streamed chunk",
                        "per-token streaming (no schema field)", False,
                        "pooled per-chunk ITL is not comparable and is not scored"))
    psi = nd.get("prefill_schedule_interval") or rc.get("prefill-schedule-interval")
    if psi and int(psi) > 1:
        mism.append(_mm("prefill_schedule_interval", f"waiting prefills admitted every {psi} steps",
                        "every step (no schema field)", False, "vllm/config/scheduler.py:204@v0.31.0"))
    asc = nd.get("api_server_count") or rc.get("api-server-count")
    if asc and int(asc) > 1:
        mism.append(_mm("api_server_count", f"{asc} API server processes", "one frontend (no schema field)", False))


def _mm(item, measured, blis, removable, note=""):
    return {"item": item, "measured": measured, "blis": blis, "removable": removable, "note": note}


def _repo_tail(r):
    return (r or "").rstrip("/").split("/")[-1].lower()


def _ver_from_image(img):
    m = re.search(r":v(\d+\.\d+\.\d+)", img or "")
    return m.group(1) if m else None


def _commit_from_image(img):
    m = re.search(r"nightly-(?:dev-[\w.-]*?-)?([0-9a-f]{12,40})", img or "")
    return m.group(1) if m else None


def scenario_yaml(name: str, d: dict, coeffs: str = COEFFS) -> str:
    eng = d["engine"]
    lines = []
    for k, v in eng.items():
        if isinstance(v, dict):
            lines.append(f"      {k}:")
            lines += [f"        {kk}: {_y(vv)}" for kk, vv in v.items()]
        elif isinstance(v, list):
            lines.append(f"      {k}: [{', '.join(map(str, v))}]")
        else:
            lines.append(f"      {k}: {_y(v)}")
    off = ""
    if d.get("offload"):
        o = d["offload"]
        off = (f"offload:\n  connector: {o['connector']}\n  spec: {o['spec']}\n"
               + (f"  eviction_policy: {o['eviction_policy']}\n" if o.get("eviction_policy") else "")
               + f"  tiers:\n    - tier: cpu_dram\n      bytes: {o['cpu_bytes']}\n")
    return f"""# GENERATED by validation/coverage/harness (spec.py) from the InferenceX run's own server log.
kind: Scenario
name: {name}
engine_version: "0.31.0"

model: {d['model']}
coefficients: [{coeffs}]

cluster:
  hardware: {d['hardware']}
  nodes: {d['nodes']}
  gpus_per_node: {d['gpus_per_node']}
---
kind: Deployment
name: {name}

pools:
  - role: colocated
    nodes: {d['nodes']}
    parallel:
      tp: {d['tp']}
      pp: 1
      dp: {d['dp']}
      enable_expert_parallel: {_y(d['enable_expert_parallel'])}
    engine:
{chr(10).join(lines)}
{off}"""


def _y(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
