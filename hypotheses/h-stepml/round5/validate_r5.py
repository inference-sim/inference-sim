"""Shared BLIS validation for Round 5 experiments.

Uses R4's proven workload-spec format (explicit clients with constant distributions)
rather than the inference_perf expansion path, which produces ~10x E2E errors.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(_SCRIPT_DIR, "..", "shared")
_REPO_ROOT = os.path.abspath(os.path.join(_SHARED_DIR, "..", "..", ".."))
DATA_ROOT = os.path.join(_REPO_ROOT, "eval", "ground_truth")
BINARY = os.path.join(_REPO_ROOT, "simulation_worker")

sys.path.insert(0, _SHARED_DIR)
from data_loader import parse_experiment_metadata

BLOCK_SIZE = 16
MODEL_ALIASES = {"llama-2-70b-hf": "llama-2-70b"}


def normalize_model(name):
    return MODEL_ALIASES.get(name, name)


def parse_experiment_dir(dirname):
    pattern = re.compile(r"^(\d{8}-\d{6})-(.+)-tp(\d+)-(\w+)$")
    m = pattern.match(dirname)
    if m:
        return {"model": m.group(2), "tp": int(m.group(3)), "workload": m.group(4)}
    tp_matches = list(re.finditer(r"-tp(\d+)-", dirname))
    if not tp_matches:
        raise ValueError(f"Cannot parse: {dirname}")
    last = tp_matches[-1]
    return {"model": dirname[16:last.start()], "tp": int(last.group(1)), "workload": dirname[last.end():]}


def load_ground_truth(exp_dir):
    with open(os.path.join(exp_dir, "results", "summary_lifecycle_metrics.json")) as f:
        data = json.load(f)
    s = data.get("successes", {})
    lat = s.get("latency", {})
    return {
        "e2e_mean_s": lat.get("request_latency", {}).get("mean", 0),
        "ttft_mean_s": lat.get("time_to_first_token", {}).get("mean", 0),
        "itl_mean_s": lat.get("inter_token_latency", {}).get("mean", 0),
        "num_requests": data.get("load_summary", {}).get("count", 0),
        "throughput_rps": s.get("throughput", {}).get("requests_per_sec", 0),
        "prompt_len_mean": s.get("prompt_len", {}).get("mean", 0),
        "output_len_mean": s.get("output_len", {}).get("mean", 0),
    }


def load_exp_config(exp_dir):
    with open(os.path.join(exp_dir, "exp-config.yaml")) as f:
        return yaml.safe_load(f)


def load_profile(exp_dir):
    with open(os.path.join(exp_dir, "profile.yaml")) as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return yaml.safe_load(content)


def extract_kv_blocks(exp_dir):
    path = os.path.join(exp_dir, "vllm.log")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            m = re.search(r"GPU KV cache size:\s+([\d,]+)\s+tokens", line)
            if m:
                return int(m.group(1).replace(",", "")) // BLOCK_SIZE
    return None


def extract_cpu_kv_blocks(exp_dir):
    path = os.path.join(exp_dir, "vllm.log")
    if not os.path.isfile(path):
        return 0
    cpu_bytes = kv_shape = None
    with open(path) as f:
        for line in f:
            if cpu_bytes is None:
                m = re.search(r"cpu_bytes_to_use['\"]?:\s*([\d.]+)", line)
                if m:
                    cpu_bytes = float(m.group(1))
            if kv_shape is None:
                m = re.search(
                    r"cross layer KV cache of shape \((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", line)
                if m:
                    kv_shape = tuple(int(m.group(i)) for i in range(1, 7))
            if cpu_bytes is not None and kv_shape is not None:
                break
    if not cpu_bytes or not kv_shape:
        return 0
    _, nl, _, bs, nkv, hd = kv_shape
    return int(cpu_bytes) // (nl * 2 * bs * nkv * hd * 2)


def build_workload_spec(profile, gt):
    """Build workload spec with explicit clients (R4 proven format)."""
    data = profile.get("data", {})
    load_cfg = profile.get("load", {})
    sp = data.get("shared_prefix", {})

    stages = [{"rate": s["rate"], "duration": s["duration"]}
              for s in load_cfg.get("stages", [])]
    if not stages:
        n = gt.get("num_requests", 1000)
        rps = max(gt.get("throughput_rps", 10), 1)
        stages.append({"rate": rps, "duration": int(n / rps) + 60})

    total_dur = sum(s["duration"] for s in stages)
    agg_rate = sum(s["rate"] * s["duration"] for s in stages) / total_dur
    total_req = int(sum(s["rate"] * s["duration"] for s in stages))
    horizon = int(total_dur * 1_000_000) + 60_000_000

    np_ = sp.get("num_unique_system_prompts", 9)
    nu = sp.get("num_users_per_system_prompt", 5)
    nc = np_ * nu
    ql = sp.get("question_len", int(gt.get("prompt_len_mean", 500)))
    ol = sp.get("output_len", int(gt.get("output_len_mean", 250)))
    spl = sp.get("system_prompt_len", 100)

    rf = 1.0 / nc
    clients = []
    for p in range(np_):
        for u in range(nu):
            clients.append({
                "id": f"prompt-{p}-user-{u}", "tenant_id": f"prompt-{p}",
                "slo_class": "batch", "rate_fraction": rf,
                "arrival": {"process": "poisson"},
                "input_distribution": {"type": "constant", "params": {"value": float(ql)}},
                "output_distribution": {"type": "constant", "params": {"value": float(ol)}},
                "prefix_group": f"prompt-{p}", "prefix_length": spl, "streaming": False,
            })

    return {
        "version": "2", "seed": 42, "category": "language",
        "num_requests": total_req, "horizon": horizon,
        "aggregate_rate": agg_rate, "clients": clients,
    }


def run_blis(exp_dir, alpha, beta):
    """Run BLIS with alpha/beta coefficients. Returns metrics dict or None."""
    gt = load_ground_truth(exp_dir)
    cfg = load_exp_config(exp_dir)
    kv = extract_kv_blocks(exp_dir)
    if kv is None:
        return None
    cpu_kv = extract_cpu_kv_blocks(exp_dir)
    try:
        profile = load_profile(exp_dir)
    except Exception:
        return None

    spec = build_workload_spec(profile, gt)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(spec, f, default_flow_style=False)
        sp = f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        rp = f.name

    cmd = [
        BINARY, "run", "--model", cfg.get("model", "unknown"),
        "--workload-spec", sp,
        "--tp", str(cfg.get("tensor_parallelism", 1)),
        "--max-model-len", str(cfg.get("max_model_len", 4096)),
        "--max-num-running-reqs", str(cfg.get("max_num_seqs", 128)),
        "--max-num-scheduled-tokens", str(cfg.get("max_num_batched_tokens", 2048)),
        "--total-kv-blocks", str(kv), "--block-size-in-tokens", str(BLOCK_SIZE),
        "--alpha-coeffs=" + ",".join(str(c) for c in alpha),
        "--beta-coeffs=" + ",".join(str(c) for c in beta),
        "--results-path", rp, "--log", "error",
    ]
    if cpu_kv > 0:
        cmd.extend(["--kv-cpu-blocks", str(cpu_kv)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=_REPO_ROOT)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    finally:
        for p in [sp, rp]:
            try:
                os.unlink(p)
            except OSError:
                pass

    if result.returncode != 0:
        return None

    # Parse JSON from stdout
    lines = result.stdout.split("\n")
    jl, in_json, depth = [], False, 0
    for line in lines:
        if "Simulation Metrics" in line:
            in_json, jl = False, []
            continue
        if not in_json and line.strip().startswith("{"):
            in_json, depth = True, 0
        if in_json:
            jl.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                break

    if not jl:
        return None
    try:
        m = json.loads("\n".join(jl))
    except json.JSONDecodeError:
        return None
    return {
        "e2e_mean_ms": m.get("e2e_mean_ms", 0),
        "ttft_mean_ms": m.get("ttft_mean_ms", 0),
        "itl_mean_ms": m.get("itl_mean_ms", 0),
        "completed_requests": m.get("completed_requests", 0),
    }


def rel_error(pred, obs):
    if obs == 0:
        return float("inf") if pred != 0 else 0.0
    return abs(pred - obs) / obs


def list_experiments():
    """List all experiments in the ground truth directory."""
    exps = []
    for d in sorted(os.listdir(DATA_ROOT)):
        dp = os.path.join(DATA_ROOT, d)
        if not os.path.isdir(dp):
            continue
        if not os.path.isfile(os.path.join(dp, "results", "summary_lifecycle_metrics.json")):
            continue
        try:
            meta = parse_experiment_dir(d)
        except ValueError:
            continue
        exps.append({"dirname": d, "dirpath": dp, "model": meta["model"],
                      "tp": meta["tp"], "workload": meta["workload"]})
    return exps


def validate_all(experiments, coefficients, label=""):
    """Run BLIS validation for all experiments with given per-model coefficients.

    coefficients: dict mapping model_name → {"alpha": [a0, a1, a2], "beta": [b0, b1, b2]}
    """
    results = []
    for exp in experiments:
        model = exp["model"]
        norm = normalize_model(model)

        coeffs = None
        for k, v in coefficients.items():
            if k == model or normalize_model(k) == norm:
                coeffs = v
                break
        if coeffs is None:
            results.append({"experiment": exp["dirname"], "model": model,
                          "workload": exp["workload"], "status": "no_coeffs"})
            continue

        gt = load_ground_truth(exp["dirpath"])
        blis = run_blis(exp["dirpath"], coeffs["alpha"], coeffs["beta"])
        if blis is None:
            results.append({"experiment": exp["dirname"], "model": model,
                          "workload": exp["workload"], "status": "blis_failed"})
            continue

        e = rel_error(blis["e2e_mean_ms"], gt["e2e_mean_s"] * 1000)
        t = rel_error(blis["ttft_mean_ms"], gt["ttft_mean_s"] * 1000)
        i = rel_error(blis["itl_mean_ms"], gt["itl_mean_s"] * 1000)

        prefix = f"[{label}] " if label else ""
        print(f"  {prefix}{exp['dirname']}: E2E={e*100:.1f}% TTFT={t*100:.1f}% "
              f"(pred={blis['e2e_mean_ms']:.0f}ms gt={gt['e2e_mean_s']*1000:.0f}ms)")

        results.append({
            "experiment": exp["dirname"], "model": model,
            "workload": exp["workload"], "status": "ok",
            "gt_e2e_ms": gt["e2e_mean_s"] * 1000,
            "blis_e2e_ms": blis["e2e_mean_ms"],
            "e2e_error": e, "ttft_error": t, "itl_error": i,
        })

    return results
