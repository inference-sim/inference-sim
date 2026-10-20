"""Resolve version-dependent vLLM engine defaults for every vLLM build in the inventory.

Reads the inventory ($IX_INVENTORY), maps each run's image/version to a vLLM ref (release tag,
main commit, or the merge base of a fork commit), and evaluates each behaviour at that ref
from source (git show for local refs, the GitHub contents API for fork-only commits). Every
value carries file:line citations at the ref and a function-limited comparison with v0.31.0.

Usage: uv run --no-project python -I build_table.py   (writes $IX_BEHAVIOUR; needs $IX_VLLM_REPO)
"""

import collections
import difflib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PX  # noqa: E402

VLLM = str(PX.VLLM_REPO)
GH = "vllm-project/vllm"
BASE = "v0.31.0"
RELEASES = [
    "v0.10.1", "v0.10.2", "v0.11.0", "v0.11.1", "v0.11.2", "v0.12.0", "v0.13.0", "v0.14.0",
    "v0.15.1", "v0.16.0", "v0.17.0", "v0.18.0", "v0.19.0", "v0.19.1", "v0.20.0", "v0.20.1",
    "v0.20.2", "v0.21.0", "v0.22.0", "v0.23.0", "v0.24.0", "v0.25.0", "v0.25.1", "v0.26.0",
    "v0.27.0", "v0.27.1", "v0.28.0", "v0.29.0", "v0.30.0", "v0.31.0",
]


def git(*args, check=False):
    p = subprocess.run(["git", "-C", VLLM, *args], capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(p.stderr)
    return p.stdout.strip() if p.returncode == 0 else None


def gh(path, raw=False):
    cmd = ["gh", "api", path]
    if raw:
        cmd[2:2] = ["-H", "Accept: application/vnd.github.raw"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


class Src:
    """File reader at one ref: local git object, or GitHub contents for fork-only commits."""

    def __init__(self, ref, remote=False):
        self.ref, self.remote, self.cache = ref, remote, {}

    def get(self, path):
        if path not in self.cache:
            if self.remote:
                self.cache[path] = gh(f"repos/{GH}/contents/{path}?ref={self.ref}", raw=True)
            else:
                self.cache[path] = git("show", f"{self.ref}:{path}")
        return self.cache[path]

    def label(self):
        return self.ref if not re.fullmatch(r"[0-9a-f]{40}", self.ref) else self.ref[:10]

    def find(self, path, pattern, start=0, flags=0):
        """First (line_no, line) matching pattern at or after 1-based line start."""
        text = self.get(path)
        if text is None:
            return None
        for i, line in enumerate(text.split("\n"), 1):
            if i >= start and re.search(pattern, line, flags):
                return i, line
        return None

    def cite(self, path, pattern, start=0):
        hit = self.find(path, pattern, start)
        return f"{path}:{hit[0]}@{self.label()}" if hit else None


def enclosing_block(text, lineno):
    """The def/class (or top-level assignment) enclosing a 1-based line, as text."""
    lines = text.split("\n")
    i = lineno - 1
    ind = lambda s: len(s) - len(s.lstrip())
    start = i
    while start > 0 and not re.match(r"\s*(async\s+)?def |\s*class |[A-Z_]+ = ", lines[start]):
        start -= 1
    base = ind(lines[start])
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or ind(lines[end]) > base
                                or lines[end].strip().startswith((")", "]", "}"))):
        end += 1
    return "\n".join(lines[start:end]), lines[start].strip()


def statement_block(text, lineno):
    """The anchor statement and its indented body (for anchors inside very long functions)."""
    lines = text.split("\n")
    i = lineno - 1
    base = len(lines[i]) - len(lines[i].lstrip())
    end = i + 1
    while end < len(lines) and (not lines[end].strip()
                                or len(lines[end]) - len(lines[end].lstrip()) > base
                                or lines[end].strip().startswith((")", "]", "}", '"""'))):
        end += 1
    return "\n".join(lines[i:end]), lines[i].strip()


def scoped(text, lineno):
    block, name = enclosing_block(text, lineno)
    if block.count("\n") > 150 or name.startswith("class "):
        return statement_block(text, lineno)
    return block, name


def diff_evidence(src, base, path, pattern, path31=None, pattern31=None):
    a = src.find(path, pattern)
    b = base.find(path31 or path, pattern31 or pattern)
    if not a or not b:
        return {"status": "not comparable: the code lives in another file/function at "
                + (src.label() if not a else BASE) + " (pre-refactor layout); compare values"}
    ta, name = scoped(src.get(path), a[0])
    tb, _ = scoped(base.get(path31 or path), b[0])
    norm = lambda t: [l.strip() for l in t.split("\n") if l.strip() and not l.strip().startswith("#")]
    d = [l for l in difflib.unified_diff(norm(ta), norm(tb), lineterm="", n=0)
         if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    return {
        "status": "identical" if not d else f"differs (+{sum(l[0] == '+' for l in d)} "
                  f"-{sum(l[0] == '-' for l in d)} normalized lines)",
        "scope": f"{path} :: {name[:90]}",
        "command": f"git diff {src.label()} {BASE} -- {path31 or path}",
        "diff": d[:12],
    }


# ---------------------------------------------------------------- behaviours

def first(src, candidates):
    for path, pat in candidates:
        hit = src.find(path, pat)
        if hit:
            return path, pat, hit
    return None


def b_swa(src):
    cache = src.find("vllm/config/cache.py", r"swa_bounded_replay: bool = ")
    attn = "vllm/models/deepseek_v41/attention.py"
    reg = src.cite("vllm/model_executor/models/registry.py", r'"DeepseekV41ForCausalLM": \(')
    if not cache:
        has41 = reg is not None
        return {
            "value": "absent; DSV4.1 sliding-window cache always takes part in prefix caching"
            if has41 else "absent (no DeepSeek-V4.1 model in this build)",
            "citation": [c for c in [reg] if c] or ["vllm/config/cache.py (no swa_bounded_replay)"],
            "anchor": None,
        }
    default = cache[1].split("=")[1].strip()
    gates, cites = [], [f"vllm/config/cache.py:{cache[0]}@{src.label()}"]
    for pat, desc in [
        (r"swa_bounded_replay and not vllm_config.use_v2_model_runner", "needs V2 runner"),
        (r"prefill_context_parallel_size > 1", "off under PCP>1"),
        (r"swa_bounded_replay and current_platform.is_rocm\(\)", "off on ROCm (encoder side)"),
    ]:
        hit = src.find(attn, pat)
        if hit:
            gates.append(desc)
            cites.append(f"{attn}:{hit[0]}@{src.label()}")
    if src.get("vllm/models/deepseek_v41/decoder_replay_layers.py") is not None:
        gates.append("plus decoder-side replay (post-v0.31.0 #58132)")
        cites.append(f"vllm/models/deepseek_v41/decoder_replay_layers.py:1@{src.label()}")
    return {"value": f"present; default {default}; " + ", ".join(gates), "citation": cites,
            "anchor": ("vllm/config/cache.py", r"swa_bounded_replay: bool = ")}


def frozenset_after(src, path, name):
    text = src.get(path) or ""
    m = re.search(name + r" = frozenset\(\s*\{(.*?)\}\s*\)", text, re.S)
    return sorted(re.findall(r'"([A-Za-z0-9_]+)"', m.group(1))) if m else None


def b_runner(src):
    envs = src.find("vllm/envs.py", r"^\s+VLLM_USE_V2_MODEL_RUNNER: ")
    if not envs:
        return {"value": "V1 only (no V2 model runner)", "citation": ["vllm/envs.py (no VLLM_USE_V2_MODEL_RUNNER)"],
                "anchor": None, "dsv4_cuda": "V1", "dsv4_rocm": "V1", "dsv41_cuda": "V1", "dsv41_rocm": "V1"}
    lam = src.cite("vllm/envs.py", r'"VLLM_USE_V2_MODEL_RUNNER": lambda')
    cites = [f"vllm/envs.py:{envs[0]}@{src.label()}", lam]
    if "= False" in envs[1]:
        r = {"value": "V1 default; VLLM_USE_V2_MODEL_RUNNER=1 opts into V2 (bool(int(env or '0')))",
             "citation": cites, "anchor": ("vllm/envs.py", r'"VLLM_USE_V2_MODEL_RUNNER": lambda')}
        r.update(dsv4_cuda="V1", dsv4_rocm="V1", dsv41_cuda="V1", dsv41_rocm="V1")
        return r
    vp = "vllm/config/vllm.py"
    prop = src.find(vp, r"def use_v2_model_runner")
    cites.append(f"{vp}:{prop[0]}@{src.label()}")
    allow = frozenset_after(src, vp, "DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES")
    rocm_v1 = frozenset_after(src, vp, "ROCM_DEFAULT_MRV1_ARCHITECTURES")
    text = src.get(vp)
    res = {}
    if src.find(vp, r"def _is_default_v2_model_runner_model"):
        oracle = src.find(vp, r"def _is_default_v2_model_runner_model")
        cites.append(f"{vp}:{oracle[0]}@{src.label()}")
        body, _ = enclosing_block(text, oracle[0])
        non_moe = "or not model_config.is_moe" in body
        rocm_minus = re.search(r'DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES - \{"(\w+)"\}', text)
        rule = (f"env unset: V2 only for allowlist {allow}"
                + (" or any non-MoE" if non_moe else "")
                + (" (non-MoE, non-quantized only)" if "not model_config.is_quantized" in body else "")
                + (f"; ROCm drops {rocm_minus.group(1)}" if rocm_minus else "")
                + "; else V1; env=0/1 forces")
        for arch, key in [("DeepseekV4ForCausalLM", "dsv4"), ("DeepseekV41ForCausalLM", "dsv41")]:
            inl = arch in (allow or []) and "not model_config.is_quantized" not in body
            res[key + "_cuda"] = "V2" if inl else "V1"
            res[key + "_rocm"] = "V2" if inl and not (rocm_minus and rocm_minus.group(1) == arch) else "V1"
    else:
        rule = "env unset: V2 for all models unless an unsupported feature forces V1"
        if rocm_v1:
            rule += f"; ROCm defaults V1 for {rocm_v1}"
            if "_get_v1_model_runner_unsupported_features()" in text:
                rule += " (yields to V2 when V1 lacks a feature, e.g. PCP, DSpark)"
            cites.append(src.cite(vp, r"^ROCM_DEFAULT_MRV1_ARCHITECTURES = "))
        rule += "; env=0/1 forces"
        for arch, key in [("DeepseekV4ForCausalLM", "dsv4"), ("DeepseekV41ForCausalLM", "dsv41")]:
            res[key + "_cuda"] = "V2"
            res[key + "_rocm"] = "V1" if rocm_v1 and arch in rocm_v1 else "V2"
    r = {"value": rule, "citation": [c for c in cites if c], "anchor": (vp, r"def use_v2_model_runner")}
    r.update(res)
    return r


def b_async(src):
    sp = "vllm/config/scheduler.py"
    f = src.find(sp, r"^\s+async_scheduling: ")
    if f and "False" in f[1]:
        return {"value": "False (opt-in via --async-scheduling)", "citation": [f"{sp}:{f[0]}@{src.label()}"],
                "anchor": (sp, r"^\s+async_scheduling: ")}
    vp = "vllm/config/vllm.py"
    a = src.find(vp, r"elif self.scheduler_config.async_scheduling is None")
    text = src.get(vp).split("\n")
    end = a[0]
    while "self.scheduler_config.async_scheduling = True" not in text[end - 1]:
        end += 1
    block = "\n".join(text[a[0] - 1:end])
    off = []
    if 'runner_type == "pooling"' in block:
        off.append("pooling model")
    m = re.search(r"pipeline_parallel_size > 1(.{0,80})", block, re.S)
    if m:
        off.append("PP>1 under the V1 runner" if "use_v2_model_runner" in m.group(1) else "PP>1")
    ok = ["EAGLE/MTP"]
    for tok, name in [("NgramGPUTypes", "ngram_gpu"), ('"draft_model"', "draft_model"),
                      ('"dspark"', "dspark"), ('"dflash"', "dflash")]:
        if tok in block:
            ok.append(name)
    off.append("spec method not in {" + ", ".join(ok) + "}")
    if "disable_padded_drafter_batch" in block:
        off.append("disable_padded_drafter_batch=True")
    if "executor_supports_async_sched" in block:
        off.append("executor without async support (e.g. ray)")
    if "uses_rocm_deepep_ht_dbo" in block:
        off.append("ROCm DeepEP-HT + DBO")
    return {"value": "None -> True unless: " + "; ".join(off),
            "citation": [f"{sp}:{f[0]}@{src.label()}", f"{vp}:{a[0]}-{end}@{src.label()}"],
            "anchor": (vp, r"elif self.scheduler_config.async_scheduling is None")}


def b_frontend(src):
    lam = src.find("vllm/envs.py", r'"VLLM_USE_RUST_FRONTEND": lambda')
    if not lam:
        return {"value": "absent (Python API server only)", "citation": ["vllm/envs.py (no VLLM_USE_RUST_FRONTEND)"],
                "anchor": None}
    cites = [f"vllm/envs.py:{lam[0]}@{src.label()}"]
    s = src.find("vllm/entrypoints/cli/serve.py", r"envs.VLLM_USE_RUST_FRONTEND")
    if s:
        cites.append(f"vllm/entrypoints/cli/serve.py:{s[0]}@{src.label()}")
    else:
        s = src.find("vllm/entrypoints/cli/serve.py", r"envs.VLLM_RUST_FRONTEND_PATH")
        r = src.find("vllm/envs.py", r'use_rust = bool\(int\(os.environ.get\("VLLM_USE_RUST_FRONTEND"')
        cites += [c for c in [s and f"vllm/entrypoints/cli/serve.py:{s[0]}@{src.label()}",
                              r and f"vllm/envs.py:{r[0]}@{src.label()}"] if c]
    return {"value": "present; default off (Python frontend); VLLM_USE_RUST_FRONTEND=1 selects vllm-rs",
            "citation": cites, "anchor": ("vllm/envs.py", r'"VLLM_USE_RUST_FRONTEND": lambda')}


def b_cg_mode(src):
    vp = "vllm/config/vllm.py" if src.get("vllm/config/vllm.py") else "vllm/config/__init__.py"
    notes, cites = [], []
    o2 = src.find(vp, r"^OPTIMIZATION_LEVEL_02 = ")
    if o2:
        mode = src.find(vp, r'"cudagraph_mode": CUDAGraphMode\.', o2[0])
        lvl = src.find(vp, r"optimization_level: OptimizationLevel = ")
        val = mode[1].split("CUDAGraphMode.")[1].rstrip(",") + " (via default " + lvl[1].split("= ")[1] + ")"
        cites += [f"{vp}:{mode[0]}@{src.label()}", f"{vp}:{lvl[0]}@{src.label()}"]
        anchor = (vp, r"^OPTIMIZATION_LEVEL_02 = ")
    else:
        a = src.find(vp, r"if self.compilation_config.cudagraph_mode is None")
        hit = src.find(vp, r"CUDAGraphMode\.(FULL_AND_PIECEWISE|PIECEWISE)$", a[0])
        val = hit[1].strip().split(".")[1] + " (when compiled; NONE otherwise)"
        cites.append(f"{vp}:{hit[0]}@{src.label()}")
        anchor = (vp, r"if self.compilation_config.cudagraph_mode is None")
    cp = "vllm/config/compilation.py"
    r = src.find(cp, r"def resolve_cudagraph_mode_and_sizes")
    if r:
        notes.append("backend support downgrades it (FULL->FULL_AND_PIECEWISE/FULL_DECODE_ONLY; "
                     "spec decode below UNIFORM_BATCH->PIECEWISE)")
        cites.append(f"{cp}:{r[0]}@{src.label()}")
    d = src.find(vp, r"def _maybe_override_dynamic_sd_cudagraph_mode")
    if d:
        notes.append("dynamic SD on V1 -> PIECEWISE")
        cites.append(f"{vp}:{d[0]}@{src.label()}")
    brk = frozenset_after(src, vp, "DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES")
    if brk:
        has = [a for a in ("DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM") if a in brk]
        notes.append("breakable CUDA graphs auto-on for " + (", ".join(has) or "no DSV4 arch")
                     + " (compilation mode NONE)")
        cites.append(src.cite(vp, r"^DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = "))
    return {"value": val + ("; " + "; ".join(notes) if notes else ""), "citation": [c for c in cites if c],
            "anchor": anchor}


def b_cg_size(src):
    vp = "vllm/config/vllm.py"
    a = src.find(vp, r"^\s+if max_cudagraph_capture_size is None:")
    if a:
        lines = src.get(vp).split("\n")[a[0]:a[0] + 25]
        k = next(i for i, l in enumerate(lines) if "* 2" in l)
        expr = " ".join(x.strip() for x in lines[k:k + 2])
        sm100 = any("1024" in l for l in lines[:k + 2])
        q = "*decode_query_len" if "decode_query_len" in expr else ""
        val = f"min(max_num_seqs{q}*2, " + ("1024 on SM100 else 512)" if sm100 else "512)")
        cites = [f"{vp}:{a[0] + k + 1}@{src.label()}"]
        anchor = (vp, r"^\s+if max_cudagraph_capture_size is None:")
    else:
        sp = "vllm/config/scheduler.py"
        h = src.find(sp, r"self.cuda_graph_sizes = \[min\(self.max_num_seqs \* 2, 512\)\]")
        g = src.find("vllm/config/__init__.py", r"batch_size_capture_list = \[1, 2, 4\] \+ \[")
        return {"value": "max=min(max_num_seqs*2, 512); grid [1,2,4]+range(8,max+1,8); "
                         "filtered to <= max_num_batched_tokens",
                "citation": [f"{sp}:{h[0]}@{src.label()}", f"vllm/config/__init__.py:{g[0]}@{src.label()}"],
                "anchor": None}
    val = "max=" + val + ", capped at max_num_batched_tokens; grid [1,2,4]+8-step<256+16-step"
    if src.find(vp, r"ensure max_num_tokens is captured"):
        val += "; max_num_batched_tokens appended if within max"
    if src.find(vp, r"uniform_decode_sizes = sorted\("):
        val += "; uniform-decode sizes appended (spec decode)"
        cites.append(src.cite(vp, r"uniform_decode_sizes = sorted\("))
    return {"value": val, "citation": cites, "anchor": anchor}


def b_cg_flag(src):
    ap = "vllm/engine/arg_utils.py"
    flag = src.find(ap, r'"--cuda-graph-sizes"')
    if flag:
        cfg = src.find(ap, r"compilation_config.cudagraph_capture_sizes = self.cuda_graph_sizes")
        return {"value": "deprecated alias of --cudagraph-capture-sizes: the list is the exact capture "
                         "set (one value N captures only N)",
                "citation": [f"{ap}:{flag[0]}@{src.label()}", f"{ap}:{cfg[0]}@{src.label()}"], "anchor": None}
    sp = "vllm/config/scheduler.py"
    f = src.find(sp, r"^\s+cuda_graph_sizes: list\[int\]")
    if f:
        g = src.find("vllm/config/__init__.py", r"if len\(cuda_graph_sizes\) == 1:")
        return {"value": "SchedulerConfig.cuda_graph_sizes: one value N = max capture N on grid "
                         "[1,2,4]+range(8,N+1,8); several values = exact set",
                "citation": [f"{sp}:{f[0]}@{src.label()}", f"vllm/config/__init__.py:{g[0]}@{src.label()}"],
                "anchor": None}
    m = src.find(ap, r'"--max-cudagraph-capture-size"')
    return {"value": "flag removed (#30396): --cuda-graph-sizes is an unrecognized argument; "
                     "use --max-cudagraph-capture-size / --cudagraph-capture-sizes",
            "citation": [f"{ap}:{m[0]}@{src.label()} (successor flag)"] if m else [], "anchor": None}


def spec_path(src):
    return "vllm/config/speculative.py" if src.get("vllm/config/speculative.py") else "vllm/config/__init__.py"


def b_block_drop(src):
    p = spec_path(src)
    text = src.get(p)
    m = re.search(r"def use_eagle\(self\) -> bool:.*?return self\.method in \((.*?)\)", text, re.S)
    methods = sorted(re.findall(r'"(\w+)"', m.group(1)))
    hit = src.find(p, r"def use_eagle\(self\) -> bool")
    cites = [f"{p}:{hit[0]}@{src.label()}"]
    sch = src.find("vllm/v1/core/sched/scheduler.py", r"use_eagle=self\.use_eagle")
    if src.find(p, r"def use_eagle_block_drop"):
        d = src.find(p, r"disable_eagle_block_drop: bool = ")
        cites += [f"{p}:{d[0]}@{src.label()}", src.cite(p, r"def use_eagle_block_drop")]
        val = f"drop trailing prefix-cache block for {methods}; disable_eagle_block_drop default " \
              + d[1].split("= ")[1]
    else:
        val = f"always drop trailing prefix-cache block for {methods} (not configurable)"
    if sch:
        cites.append(f"vllm/v1/core/sched/scheduler.py:{sch[0]}@{src.label()}")
    return {"value": val, "citation": [c for c in cites if c], "anchor": (p, r"def use_eagle\(self\) -> bool")}


def b_nspec(src):
    p = spec_path(src)
    h = src.find(p, r"self.num_speculative_tokens = n_predict")
    val = "None -> draft hf_config n_predict (MTP: num_nextn_predict_layers)"
    cites = [f"{p}:{h[0]}@{src.label()}"]
    d = src.find(p, r"self.num_speculative_tokens = block_size")
    if d:
        val += "; DSpark: draft block_size"
        cites.append(f"{p}:{d[0]}@{src.label()}")
    return {"value": val, "citation": cites, "anchor": (p, r"self.num_speculative_tokens = n_predict")}


def b_sample(src):
    p = spec_path(src)
    out, cites = [], []
    for name in ("draft_sample_method", "rejection_sample_method"):
        h = src.find(p, rf"^\s+{name}: \w+ = ")
        if h:
            out.append(f"{name}={h[1].split('= ')[1].strip()}")
            cites.append(f"{p}:{h[0]}@{src.label()}")
        else:
            out.append(f"{name}=absent")
    return {"value": ", ".join(out), "citation": cites or [f"{p} (fields absent)"],
            "anchor": (p, r"^\s+rejection_sample_method: \w+ = ") if cites else None}


def b_padded(src):
    p = spec_path(src)
    h = src.find(p, r"disable_padded_drafter_batch: bool = ")
    if not h:
        return {"value": "absent: no padded drafter batch (introduced v0.11.0, #24539)",
                "citation": [f"{p} (no disable_padded_drafter_batch)"], "anchor": None}
    return {"value": "padded drafter batch on (disable_padded_drafter_batch=" + h[1].split("= ")[1] + ")",
            "citation": [f"{p}:{h[0]}@{src.label()}"], "anchor": (p, r"disable_padded_drafter_batch: bool = ")}


def b_batch(src):
    ap = "vllm/engine/arg_utils.py"
    h = src.find(ap, r"def get_batch_defaults") or src.find(ap, r"def _set_default_args(_v1)?\(")
    body, _ = enclosing_block(src.get(ap), h[0])
    body = body.split("# tpu specific")[0].split("if current_platform.is_tpu()")[0]
    body = re.sub(r"(el)?if \(\s*(.*?)\s*\):", lambda m: f"{m.group(1) or ''}if "
                  + " ".join(m.group(2).split()) + ":", body, flags=re.S)
    segs = re.split(r"\n\s*(?=(?:el)?if device_memory >=|else:)", body)
    tiers = []
    for s in segs:
        cond = re.match(r"\s*(?:el)?if device_memory >= (\d+) \* GiB_bytes(.*?):", s)
        srv = re.findall(r"UsageContext\.OPENAI_API_SERVER: (\d+)", s)
        llm = re.findall(r"UsageContext\.LLM_CLASS: (\d+)", s)
        if len(srv) >= 2:
            label = (f">={cond.group(1)}GiB" + (" non-A100" if "a100" in cond.group(2) else "")
                     + (" non-GB10" if "is_gb10" in cond.group(2) else "")) if cond else "other"
            tiers.append(f"{label}: server mnbt={srv[0]} mns={srv[1]}, LLM mnbt={llm[0]} mns={llm[1]}")
    t = src.find(ap, r"device_memory >= 70 \* GiB_bytes")
    return {"value": "; ".join(tiers),
            "citation": [f"{ap}:{h[0]}@{src.label()}"] + ([f"{ap}:{t[0]}@{src.label()}"] if t else []),
            "anchor": (ap, h[1].strip().split("(")[0].replace("def ", r"def "))}


def b_block_prefix(src):
    cp = "vllm/config/cache.py"
    ip = "vllm/platforms/interface.py"
    cites = []
    pref = src.find(ip, r"def _preferred_block_size_for_backends")
    upd = src.find(ip, r"def update_block_size_for_backend")
    dflt = src.find(cp, r"DEFAULT_BLOCK_SIZE: ClassVar\[int\] = ")
    if upd:
        bs = ("16 (CacheConfig.DEFAULT_BLOCK_SIZE) then the attention backend's preferred size"
              + (" common to every backend" if pref else ""))
        cites += [c for c in [dflt and f"{cp}:{dflt[0]}@{src.label()}", f"{ip}:{upd[0]}@{src.label()}",
                              pref and f"{ip}:{pref[0]}@{src.label()}"] if c]
    else:
        c16 = src.find("vllm/platforms/cuda.py", r"cache_config.block_size = 16")
        bs = "16 (CUDA platform default); MLA/sparse-MLA backends force 64"
        cites.append(f"vllm/platforms/cuda.py:{c16[0]}@{src.label()}")
    ap = "vllm/engine/arg_utils.py"
    text = src.get(ap)
    m = re.search(r"default_prefix_caching = (\(.*?\)|[^\n]+)", text, re.S)
    if m:
        expr = " ".join(m.group(1).split())
        pc = ("on for generative models" + (", except hybrid (opt-in)" if "is_hybrid" in expr else "")
              + f" [{expr.strip('()').strip()}]")
        h = src.find(ap, r"default_prefix_caching = ")
    else:
        h = src.find(ap, r"self.enable_prefix_caching = True")
        pc = "on for generative models (V1 default)"
    cites.append(f"{ap}:{h[0]}@{src.label()}")
    return {"value": f"block_size: {bs}; prefix caching: {pc}", "citation": cites,
            "anchor": (ip, r"def update_block_size_for_backend") if upd else None}


def b_gmu(src):
    h = src.find("vllm/config/cache.py", r"gpu_memory_utilization: float = ")
    v = re.search(r"(0\.\d+)", h[1]).group(1)
    return {"value": v, "citation": [f"vllm/config/cache.py:{h[0]}@{src.label()}"],
            "anchor": ("vllm/config/cache.py", r"gpu_memory_utilization: float = ")}


BEHAVIOURS = [
    ("swa_bounded_replay", b_swa), ("model_runner", b_runner), ("async_scheduling", b_async),
    ("frontend", b_frontend), ("cudagraph_mode", b_cg_mode), ("cudagraph_capture_size", b_cg_size),
    ("cuda_graph_sizes_flag", b_cg_flag), ("eagle_block_drop", b_block_drop),
    ("num_speculative_tokens", b_nspec), ("sample_methods", b_sample), ("padded_drafter_batch", b_padded),
    ("batch_defaults", b_batch), ("block_size_and_prefix_caching", b_block_prefix),
    ("gpu_memory_utilization", b_gmu),
]


def evaluate(src, base, base_vals):
    out = {}
    for name, fn in BEHAVIOURS:
        try:
            r = fn(src)
        except Exception as e:  # a build whose source lacks an anchor this evaluator expects
            out[name] = {"value": None, "error": f"{type(e).__name__}: {e}"}
            continue
        anchor = r.pop("anchor")
        if base_vals is not None:
            r["same_as_v0.31.0"] = r["value"] == base_vals[name]["value"]
            if anchor:
                r["diff_evidence"] = diff_evidence(src, base, *anchor)
        out[name] = r
    return out


# ---------------------------------------------------------------- builds

def server_versions():
    """image -> vllm_version seen in parsed server logs of this campaign (derived files)."""
    m = collections.defaultdict(set)
    pts = str(PX.ROOT / "results" / "fit" / "points.jsonl")
    if os.path.exists(pts):
        for line in open(pts):
            r = json.loads(line).get("resolved") or {}
            if r.get("engine_image") and r.get("vllm_version"):
                m[r["engine_image"]].add(r["vllm_version"])
    mx = os.environ.get("IX_MEASURED_IX", "")  # optional: DSV4.1 InferenceX rows {"rows": [{image, server}]}
    if mx and os.path.exists(mx):
        for r in json.load(open(mx)).get("rows", []):
            v = (r.get("server") or {}).get("vllm_version")
            if r.get("image") and v:
                m[r["image"]].add(v)
    return m


SERVER_VERSIONS = {}


def build_key(row):
    img = row.get("engine_image") or row.get("engine_image_from_recipe")
    ver = row.get("engine_version") or ""
    if not ver and img and len(SERVER_VERSIONS.get(img, ())) == 1:
        ver = next(iter(SERVER_VERSIONS[img]))
        row["engine_version"] = ver + " (from this campaign's parsed server logs for the image)"
    m = re.search(r"\+g([0-9a-f]{7,40})", ver)
    if m:
        return ("sha", m.group(1)), img
    if img:
        m = re.search(r"nightly-(?:rocm100-)?([0-9a-f]{40})", img)
        if m:
            return ("sha", m.group(1)), img
        tag = img.split("@")[0]
        m = re.search(r"-([0-9a-f]{7,12})$", tag)
        if m and re.search(r"[a-f]", m.group(1)):
            return ("sha", m.group(1)), img
        m = re.search(r":v(\d+\.\d+\.\d+)", tag) or re.search(r"vllm_(\d+\.\d+\.\d+)", tag)
        if m:
            return ("tag", "v" + m.group(1)), img
    if re.fullmatch(r"\d+\.\d+\.\d+", ver):
        return ("tag", "v" + ver), img
    return ("none", img or "(no image recorded)"), img


def fork_points():
    fp = {}
    for t in RELEASES:
        if git("rev-parse", "-q", "--verify", t + "^{commit}"):
            fp[t] = git("merge-base", t, "origin/main")
    return fp


def release_base(sha, fps):
    """Latest release on main's line before sha: the last tag whose main branch point is an
    ancestor of sha."""
    best = None
    for t in RELEASES:
        if t in fps and git("merge-base", "--is-ancestor", fps[t], sha) is not None:
            best = t
    return best


def tag_ancestor_local(sha, floor):
    best = None
    for t in RELEASES[RELEASES.index(floor):] if floor in RELEASES else RELEASES:
        if git("merge-base", "--is-ancestor", t, sha) is not None:
            best = t
    return best


def tag_ancestor_remote(sha, floor):
    best = None
    for t in RELEASES[RELEASES.index(floor):] if floor in RELEASES else RELEASES:
        c = gh(f"repos/{GH}/compare/{t}...{sha}")
        if c and json.loads(c)["behind_by"] == 0:
            best = t
    return best


def resolve(kind, ident, fps):
    if kind == "none" and SERVER_VERSIONS.get(ident):
        return {"status": "UNRESOLVABLE", "reason": "mutable image tag; server logs show several "
                f"builds {sorted(SERVER_VERSIONS[ident])} and this run records none"}
    if kind == "tag":
        if git("rev-parse", "-q", "--verify", ident + "^{commit}"):
            return {"status": "RELEASE", "ref": ident, "remote": False, "nearest_release": ident}
        return {"status": "UNRESOLVABLE", "reason": f"tag {ident} not in vllm-project/vllm"}
    if kind == "none":
        return {"status": "UNRESOLVABLE",
                "reason": "image tag names neither a vLLM release nor a commit, and the run records no "
                          "engine_version" if ident != "(no image recorded)"
                          else "the run records no engine image, recipe image or engine_version"}
    full = git("rev-parse", "-q", "--verify", ident + "^{commit}")
    if full:
        on_main = git("merge-base", "--is-ancestor", full, "origin/main") is not None
        rb = release_base(git("merge-base", full, "origin/main"), fps)
        ta = None if on_main else tag_ancestor_local(full, rb)
        return {"status": "MAIN_COMMIT" if on_main else "LOCAL_COMMIT", "ref": full, "remote": False,
                "date": git("log", "-1", "--format=%cs", full), "nearest_release": ta or rb,
                "relation": f"descends from tag {ta}" if ta else f"main commit after the {rb} branch point",
                "commits_after_release_fork": int(git("rev-list", "--count", f"{fps[rb]}..{full}")),
                "method": "git merge-base <release> origin/main is-ancestor of sha; git describe "
                          + (git("describe", "--tags", "--match", "v0.*", "--exclude", "*rc*", full) or "")}
    c = gh(f"repos/{GH}/compare/{BASE}...{ident}")
    if not c:
        return {"status": "UNRESOLVABLE",
                "reason": f"commit {ident} is not in vllm-project/vllm (gh api compare/commits: 404); "
                          "the build came from a private or fork branch"}
    c = json.loads(c)
    head = c["commits"][-1]["sha"] if c.get("commits") else None
    full = gh(f"repos/{GH}/commits/{ident}")
    full = json.loads(full)["sha"] if full else head
    mb = c["merge_base_commit"]["sha"]
    rb = release_base(mb, fps)
    ta = tag_ancestor_remote(full, rb)
    files = gh(f"repos/{GH}/compare/{mb}...{full}")
    touched = sorted(f["filename"] for f in json.loads(files)["files"]) if files else []
    return {"status": "FORK_COMMIT", "ref": full, "remote": True, "merge_base": mb,
            "merge_base_date": git("log", "-1", "--format=%cs", mb), "nearest_release": ta or rb,
            "relation": f"descends from tag {ta} (+{json.loads(gh(f'repos/{GH}/compare/{ta}...{full}'))['ahead_by']}"
                        " fork commits)" if ta else f"fork of main after the {rb} branch point",
            "commits_after_release_fork": int(git("rev-list", "--count", f"{fps[rb]}..{mb}")),
            "fork_ahead_by": c["ahead_by"], "fork_files_touched_config": [
                f for f in touched if re.match(r"vllm/(envs\.py|config/|engine/arg_utils\.py|v1/core/|"
                                               r"platforms/|entrypoints/cli/|models/deepseek_v41/)", f)],
            "method": f"gh api repos/{GH}/compare/{BASE}...{ident} (merge_base_commit); files read at the "
                      "fork commit via the GitHub contents API"}


def main_commit_at_depth(n):
    first = git("rev-list", "--first-parent", "--reverse", "origin/main").split()
    lo, hi = 0, len(first) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if int(git("rev-list", "--count", first[mid])) <= n:
            lo = mid
        else:
            hi = mid - 1
    return first[lo]


def dev_count_hint(versions, fps):
    """Non-authoritative placement of a non-public build from its setuptools-scm version.

    X.YrcZ.devN counts commits since that rc tag; 0.1.devN (no tag reachable) is the total
    history depth, matched here to the origin/main first-parent commit of that depth. Either
    places the build in time only; the fork's own commits, and so its behaviour, are unknown.
    """
    out = []
    for v in sorted(versions):
        m = re.fullmatch(r"0\.1\.dev(\d+)\+g[0-9a-f]+", v)
        if m:
            c = main_commit_at_depth(int(m.group(1)))
            out.append(f"{v}: history depth {m.group(1)} ~ main {git('log', '-1', '--format=%h %cs', c)}, "
                       f"after the {release_base(c, fps)} branch point")
            continue
        m = re.fullmatch(r"(\d+\.\d+\.\d+rc\d+)\.dev(\d+)\+g[0-9a-f]+", v)
        if m:
            out.append(f"{v}: {m.group(2)} commits after tag v{m.group(1)}")
    return out


def main():
    SERVER_VERSIONS.update(server_versions())
    out_dir = str(PX.BEHAVIOUR.parent)
    os.makedirs(out_dir, exist_ok=True)
    builds = collections.OrderedDict()
    with open(PX.INV) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("engine") not in ("vllm", "dynamo-vllm"):
                continue
            key, img = build_key(row)
            b = builds.setdefault(key, {"images": collections.Counter(), "runs": 0, "points": 0,
                                        "engine_versions": set()})
            b["images"][img or "(none)"] += 1
            b["runs"] += 1
            b["points"] += len(row.get("points") or [])
            if row.get("engine_version"):
                b["engine_versions"].add(row["engine_version"].split(" (")[0])
                if "(from" in row["engine_version"]:
                    b.setdefault("version_sources", set()).add("server log of a sibling run with the same image")
    fps = fork_points()
    base = Src(BASE)
    base_vals = evaluate(base, base, None)
    merged = collections.OrderedDict()
    for (kind, ident), b in builds.items():
        res = resolve(kind, ident, fps)
        k = res.get("ref") or ident
        if k in merged:
            m = merged[k][1]
            m["images"].update(b["images"])
            m["runs"] += b["runs"]
            m["points"] += b["points"]
            m["engine_versions"] |= b["engine_versions"]
            m.setdefault("version_sources", set()).update(b.get("version_sources", ()))
            merged[k][2].append(ident)
        else:
            merged[k] = [(kind, ident), b, [ident], res]
    table, memo = [], {}
    for (kind, ident), b, idents, res in sorted(merged.values(), key=lambda v: -v[1]["points"]):
        if res["status"] == "UNRESOLVABLE":
            hint = dev_count_hint(b["engine_versions"], fps)
            if hint:
                res["approximate_position_hint"] = hint
        entry = {"build": ident, "aliases": sorted(set(idents) - {ident}), "kind": kind,
                 "version_sources": sorted(b.get("version_sources", ())) or ["inventory engine_version / image tag"], "runs": b["runs"], "points": b["points"],
                 "images": dict(b["images"]), "engine_versions": sorted(b["engine_versions"]),
                 "resolution": res}
        if res["status"] != "UNRESOLVABLE":
            if res["ref"] not in memo:
                memo[res["ref"]] = evaluate(Src(res["ref"], res["remote"]), base, base_vals)
            entry["behaviours"] = memo[res["ref"]]
        table.append(entry)
        print(f"{ident:44s} {res['status']:13s} {b['points']:6d}", file=sys.stderr, flush=True)
    doc = {
        "generated_by": "validation/coverage/image_resolution/build_table.py",
        "vllm_repo_head": git("rev-parse", "origin/main"),
        "baseline": {"ref": BASE, "behaviours": base_vals},
        "dsv41_architecture": "DeepseekV41ForCausalLM (DSV4 = DeepseekV4ForCausalLM)",
        "builds": table,
    }
    with open(PX.BEHAVIOUR, "w") as fh:
        json.dump(doc, fh, indent=1, default=sorted)


if __name__ == "__main__":
    main()
