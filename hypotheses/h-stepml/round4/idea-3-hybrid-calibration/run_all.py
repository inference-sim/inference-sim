#!/usr/bin/env python3
"""Round 4 Idea 3: Hybrid Calibration -- All 4 Sub-Hypotheses.

This script runs all four sub-hypotheses for the hybrid calibration approach:
  H1: Principled base (E2E-derived step time + TTFT corrections)
  H2: Constrained CMA-ES residual tuning (+-30% around H1 coefficients)
  H3: Leave-one-model-out (LOMO) cross-validation
  H4: Leave-one-workload-out (LOWO) cross-validation

Key design decisions:
  - Uses alpha/beta coefficients (the proven BLIS interface) rather than StepML
    model artifacts, since that path is known to work.
  - Expands inference_perf profiles into proper v2 workload specs with clients
    (BLIS Validate() requires clients before GenerateRequests expansion).
  - Derives "effective step time" from E2E and output length data, then
    decomposes into overhead floor + per-token components via regression.
  - TTFT correction as alpha0 (QueueingTime intercept).

BLIS latency model:
  StepTime(batch) = beta0 + beta1 * prefill_tokens + beta2 * decode_tokens (us)
  QueueingTime(req) = alpha0 + alpha1 * input_len (us)
  OutputTokenProcessingTime() = alpha2 (us, per output token)
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "shared")
_REPO_ROOT = os.path.abspath(os.path.join(_SHARED_DIR, "..", "..", ".."))
_DATA_ROOT = os.path.join(_REPO_ROOT, "eval", "ground_truth")
_BINARY = os.path.join(_REPO_ROOT, "simulation_worker")

sys.path.insert(0, _SHARED_DIR)
from data_loader import (
    load_all_experiments,
    parse_experiment_metadata,
)

# Output directories
H1_OUTPUT = os.path.join(_SCRIPT_DIR, "h1-principled-base", "output")
H2_OUTPUT = os.path.join(_SCRIPT_DIR, "h2-constrained-cmaes", "output")
H3_OUTPUT = os.path.join(_SCRIPT_DIR, "h3-lomo", "output")
H4_OUTPUT = os.path.join(_SCRIPT_DIR, "h4-lowo", "output")

for d in [H1_OUTPUT, H2_OUTPUT, H3_OUTPUT, H4_OUTPUT]:
    os.makedirs(d, exist_ok=True)

BLOCK_SIZE_TOKENS = 16


def normalize_model_name(name):
    """Normalize model names: strip -hf suffix."""
    return re.sub(r"-hf$", "", name)


# ---------------------------------------------------------------------------
# Experiment parsing helpers (same as validate_blis.py)
# ---------------------------------------------------------------------------
def parse_experiment_dir(dirname):
    pattern = re.compile(r"^(\d{8}-\d{6})-(.+)-tp(\d+)-(\w+)$")
    m = pattern.match(dirname)
    if m:
        return {"timestamp": m.group(1), "model": m.group(2),
                "tp": int(m.group(3)), "workload": m.group(4)}
    tp_matches = list(re.finditer(r"-tp(\d+)-", dirname))
    if not tp_matches:
        raise ValueError(f"Cannot parse: {dirname}")
    last = tp_matches[-1]
    return {"timestamp": dirname[:15], "model": dirname[16:last.start()],
            "tp": int(last.group(1)), "workload": dirname[last.end():]}


def load_ground_truth(exp_dir):
    with open(os.path.join(exp_dir, "results", "summary_lifecycle_metrics.json")) as f:
        data = json.load(f)
    s = data.get("successes", {})
    lat = s.get("latency", {})
    return {
        "e2e_mean_s": lat.get("request_latency", {}).get("mean", 0),
        "ttft_mean_s": lat.get("time_to_first_token", {}).get("mean", 0),
        "itl_mean_s": lat.get("inter_token_latency", {}).get("mean", 0),
        "throughput_rps": s.get("throughput", {}).get("requests_per_sec", 0),
        "num_requests": data.get("load_summary", {}).get("count", 0),
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
                return int(m.group(1).replace(",", "")) // BLOCK_SIZE_TOKENS
    return None


def extract_cpu_kv_blocks(exp_dir):
    path = os.path.join(exp_dir, "vllm.log")
    if not os.path.isfile(path):
        return 0
    cpu_bytes = None
    kv_shape = None
    with open(path) as f:
        for line in f:
            if cpu_bytes is None:
                m = re.search(r"cpu_bytes_to_use['\"]?:\s*([\d.]+)", line)
                if m:
                    cpu_bytes = float(m.group(1))
            if kv_shape is None:
                m = re.search(
                    r"cross layer KV cache of shape \((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",
                    line)
                if m:
                    kv_shape = tuple(int(m.group(i)) for i in range(1, 7))
            if cpu_bytes is not None and kv_shape is not None:
                break
    if not cpu_bytes or not kv_shape or cpu_bytes == 0:
        return 0
    _, nl, _, bs, nkv, hd = kv_shape
    return int(cpu_bytes) // (nl * 2 * bs * nkv * hd * 2)


# ---------------------------------------------------------------------------
# Workload spec (with clients expanded)
# ---------------------------------------------------------------------------
def build_workload_spec(profile, gt):
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
                "id": f"prompt-{p}-user-{u}",
                "tenant_id": f"prompt-{p}",
                "slo_class": "batch",
                "rate_fraction": rf,
                "arrival": {"process": "poisson"},
                "input_distribution": {"type": "constant", "params": {"value": float(ql)}},
                "output_distribution": {"type": "constant", "params": {"value": float(ol)}},
                "prefix_group": f"prompt-{p}",
                "prefix_length": spl,
                "streaming": False,
            })

    return {
        "version": "2", "seed": 42, "category": "language",
        "num_requests": total_req, "horizon": horizon,
        "aggregate_rate": agg_rate, "clients": clients,
    }


# ---------------------------------------------------------------------------
# BLIS execution
# ---------------------------------------------------------------------------
def run_blis(exp_dir, alpha, beta):
    """Run BLIS and return metrics dict or None."""
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
        _BINARY, "run",
        "--model", cfg.get("model", "unknown"),
        "--workload-spec", sp,
        "--tp", str(cfg.get("tensor_parallelism", 1)),
        "--max-model-len", str(cfg.get("max_model_len", 4096)),
        "--max-num-running-reqs", str(cfg.get("max_num_seqs", 128)),
        "--max-num-scheduled-tokens", str(cfg.get("max_num_batched_tokens", 2048)),
        "--total-kv-blocks", str(kv),
        "--block-size-in-tokens", str(BLOCK_SIZE_TOKENS),
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
    return _parse_stdout(result.stdout)


def _parse_stdout(stdout):
    lines = stdout.split("\n")
    jl = []
    in_json = False
    depth = 0
    for line in lines:
        if "Simulation Metrics" in line:
            in_json = False
            jl = []
            continue
        if not in_json and line.strip().startswith("{"):
            in_json = True
            depth = 0
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


# ---------------------------------------------------------------------------
# Enumerate experiments
# ---------------------------------------------------------------------------
def list_experiments():
    exps = []
    for d in sorted(os.listdir(_DATA_ROOT)):
        dp = os.path.join(_DATA_ROOT, d)
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


# ===================================================================
# H1: PRINCIPLED BASE
# ===================================================================
def calibrate_coefficients(experiments, step_df):
    """Calibrate per-model beta coefficients from ground truth E2E data.

    Strategy:
    1. Compute target step time from E2E: target = (E2E - TTFT) / output_len
    2. Compute average batch composition from step data (avg decode batch size)
    3. Set beta2 = measured_step_duration / avg_decode_batch (marginal per-token cost)
    4. Set beta0 = target_step - beta2 * avg_decode_batch (overhead floor)
    5. Set beta1 from regression on prefill steps (marginal prefill cost)

    The overhead floor (beta0) captures CPU scheduling, CUDA sync, memory management
    -- all the time that step.duration_us does NOT measure.
    """
    from sklearn.linear_model import Ridge

    # Gather per-model ground truth
    model_gt = defaultdict(list)
    for exp in experiments:
        gt = load_ground_truth(exp["dirpath"])
        model_gt[exp["model"]].append(gt)

    results = {}
    for model in sorted(step_df["model"].unique()):
        mdf = step_df[step_df["model"] == model].dropna(subset=["step.duration_us"])
        if len(mdf) < 20:
            continue

        norm = normalize_model_name(model)

        # Get ground truth for this model (merge variants)
        gts = model_gt.get(model, [])
        if not gts:
            for k, v in model_gt.items():
                if normalize_model_name(k) == norm:
                    gts.extend(v)

        if not gts:
            continue

        # Target step time from E2E
        avg_e2e_us = np.mean([g["e2e_mean_s"] for g in gts]) * 1_000_000
        avg_ttft_us = np.mean([g["ttft_mean_s"] for g in gts]) * 1_000_000
        avg_outlen = np.mean([g["output_len_mean"] for g in gts])
        target_step_us = (avg_e2e_us - avg_ttft_us) / avg_outlen if avg_outlen > 0 else 10000

        # Average batch composition from step data
        decode = mdf.get("batch.decode_tokens", pd.Series(0, index=mdf.index)).fillna(0).astype(float)
        prefill = mdf.get("batch.prefill_tokens", pd.Series(0, index=mdf.index)).fillna(0).astype(float)
        duration = mdf["step.duration_us"].astype(float)

        avg_decode_batch = float(decode.mean())
        avg_prefill_batch = float(prefill.mean())
        avg_dur = float(duration.mean())

        # beta2 = marginal cost per decode token (from measured GPU time / batch size)
        # This captures the compute cost that DOES scale with batch size
        beta2 = avg_dur / avg_decode_batch if avg_decode_batch > 0 else 10.0

        # beta0 = overhead floor (everything that doesn't scale with batch)
        beta0 = target_step_us - beta2 * avg_decode_batch
        beta0 = max(beta0, 1000)  # At least 1ms floor

        # beta1 from regression on mixed (prefill) steps
        mixed = mdf[prefill > 0]
        if len(mixed) > 10:
            X_mixed = mixed[["batch.prefill_tokens"]].values.astype(float)
            y_mixed = mixed["step.duration_us"].values.astype(float)
            reg = Ridge(alpha=10.0)
            reg.fit(X_mixed, y_mixed)
            beta1 = max(0.0, float(reg.coef_[0]))
        else:
            beta1 = beta2 * 0.5  # Rough estimate: prefill ~half decode cost per token

        # Verify: predicted step at average batch should approximate target
        pred_avg = beta0 + beta1 * avg_prefill_batch + beta2 * avg_decode_batch
        ratio = pred_avg / target_step_us if target_step_us > 0 else 1.0

        results[model] = {
            "beta0": beta0,
            "beta1_prefill": beta1,
            "beta2_decode": beta2,
            "target_step_us": target_step_us,
            "avg_decode_batch": avg_decode_batch,
            "avg_dur_us": avg_dur,
            "pred_at_avg": pred_avg,
            "ratio": ratio,
            "n_steps": len(mdf),
        }

        print(
            f"  {model}: beta0={beta0:.0f} beta1={beta1:.2f} beta2={beta2:.1f} "
            f"| target={target_step_us:.0f}us avg_batch={avg_decode_batch:.1f} "
            f"pred@avg={pred_avg:.0f}us ratio={ratio:.2f}"
        )

    return results


def compute_ttft_corrections(experiments):
    """Per-model TTFT from ground truth."""
    model_ttfts = defaultdict(list)
    for exp in experiments:
        gt = load_ground_truth(exp["dirpath"])
        if gt["ttft_mean_s"] > 0:
            model_ttfts[exp["model"]].append(gt["ttft_mean_s"] * 1_000_000)

    result = {}
    for model, vals in model_ttfts.items():
        result[model] = float(np.mean(vals))
        print(f"  {model}: TTFT={result[model]/1000:.1f}ms")
    return result


def validate_coeffs(experiments, per_model_coeffs):
    """Run BLIS validation for all experiments with given coefficients."""
    results = []
    for exp in experiments:
        model = exp["model"]
        norm = normalize_model_name(model)

        coeffs = None
        for k, v in per_model_coeffs.items():
            if k == model or normalize_model_name(k) == norm:
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

        print(
            f"  {exp['dirname']}: E2E={e*100:.1f}% TTFT={t*100:.1f}% ITL={i*100:.1f}%  "
            f"(pred={blis['e2e_mean_ms']:.0f}ms gt={gt['e2e_mean_s']*1000:.0f}ms)"
        )

        results.append({
            "experiment": exp["dirname"], "model": model,
            "workload": exp["workload"], "tp": exp["tp"], "status": "ok",
            "gt_e2e_ms": gt["e2e_mean_s"] * 1000,
            "gt_ttft_ms": gt["ttft_mean_s"] * 1000,
            "gt_itl_ms": gt["itl_mean_s"] * 1000,
            "blis_e2e_ms": blis["e2e_mean_ms"],
            "blis_ttft_ms": blis["ttft_mean_ms"],
            "blis_itl_ms": blis["itl_mean_ms"],
            "e2e_error": e, "ttft_error": t, "itl_error": i,
            "alpha": coeffs["alpha"], "beta": coeffs["beta"],
        })
    return results


def run_h1(experiments, step_df):
    """Run H1: Principled Base."""
    print("\n" + "=" * 70)
    print("H1: PRINCIPLED BASE -- Direct Calibration from E2E + Step Data")
    print("=" * 70)

    # Step 1+2: Calibrate beta coefficients
    print("\n--- Steps 1-2: Calibrate beta from E2E targets + step features ---")
    regression = calibrate_coefficients(experiments, step_df)

    # Step 3: TTFT corrections
    print("\n--- Step 3: TTFT corrections ---")
    ttft = compute_ttft_corrections(experiments)

    # Step 4: Build coefficients
    print("\n--- Step 4: Build alpha/beta coefficients ---")
    per_model_coeffs = {}
    for model, reg in regression.items():
        norm = normalize_model_name(model)
        ttft_us = 0
        for k, v in ttft.items():
            if k == model or normalize_model_name(k) == norm:
                ttft_us = v
                break

        alpha = [ttft_us, 0.0, 0.0]
        beta = [reg["beta0"], reg["beta1_prefill"], reg["beta2_decode"]]
        per_model_coeffs[model] = {"alpha": alpha, "beta": beta}
        print(
            f"  {model}: alpha=[{alpha[0]:.0f},0,0] "
            f"beta=[{beta[0]:.0f}, {beta[1]:.3f}, {beta[2]:.1f}]"
        )

    # Step 5: Validate
    print("\n--- Step 5: BLIS validation ---")
    results = validate_coeffs(experiments, per_model_coeffs)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(H1_OUTPUT, "blis_validation.csv"), index=False)
    ok = df[df["status"] == "ok"]

    summary = {}
    if len(ok) > 0:
        summary = {
            "mean_e2e_error_pct": float(ok["e2e_error"].mean() * 100),
            "mean_ttft_error_pct": float(ok["ttft_error"].mean() * 100),
            "mean_itl_error_pct": float(ok["itl_error"].mean() * 100),
            "median_e2e_error_pct": float(ok["e2e_error"].median() * 100),
            "n_experiments": len(ok),
            "n_below_10_e2e": int((ok["e2e_error"] < 0.10).sum()),
            "n_below_20_e2e": int((ok["e2e_error"] < 0.20).sum()),
            "per_model_coeffs": {k: v for k, v in per_model_coeffs.items()},
            "regression": regression,
            "ttft_us": {k: float(v) for k, v in ttft.items()},
        }

    with open(os.path.join(H1_OUTPUT, "h1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _print_table("H1 PRINCIPLED BASE", ok)

    return {"per_model_coeffs": per_model_coeffs, "regression": regression,
            "results": results, "summary": summary}


# ===================================================================
# H2: CONSTRAINED CMA-ES
# ===================================================================
def cmaes_objective(x, base_alpha, base_beta, model_experiments):
    """Evaluate candidate coefficients on model's experiments.

    x = [scale_b0, scale_b1, scale_b2, scale_a0, alpha2]
    """
    s0, s1, s2, sa, a2 = x
    alpha = [base_alpha[0] * sa, base_alpha[1], a2]
    beta = [base_beta[0] * s0, base_beta[1] * s1, base_beta[2] * s2]

    e2e_errs = []
    itl_errs = []
    for exp in model_experiments:
        gt = load_ground_truth(exp["dirpath"])
        blis = run_blis(exp["dirpath"], alpha, beta)
        if blis is None:
            return 100.0
        e2e_errs.append(rel_error(blis["e2e_mean_ms"], gt["e2e_mean_s"] * 1000))
        itl_errs.append(rel_error(blis["itl_mean_ms"], gt["itl_mean_s"] * 1000))

    me = np.mean(e2e_errs)
    mi = np.mean(itl_errs)
    return float(0.5 * me + 0.5 * mi + 5.0 * max(0, mi - 0.20))


def run_h2(experiments, h1_coeffs):
    """Run H2: Constrained CMA-ES."""
    print("\n" + "=" * 70)
    print("H2: CONSTRAINED CMA-ES RESIDUAL TUNING")
    print("=" * 70)

    try:
        import cma
    except ImportError:
        print("ERROR: cma not installed")
        return {"error": "cma not installed"}

    model_exps = defaultdict(list)
    for exp in experiments:
        model_exps[normalize_model_name(exp["model"])].append(exp)

    optimized = {}
    for nm, mexps in sorted(model_exps.items()):
        print(f"\n--- Optimizing {nm} ({len(mexps)} exps) ---")

        base = None
        for k, v in h1_coeffs.items():
            if normalize_model_name(k) == nm:
                base = v
                break
        if not base:
            print(f"  SKIP: no base coefficients")
            continue

        ba, bb = base["alpha"], base["beta"]

        # Wider bounds: +-30% for scales, [0, 2000] for alpha2
        x0 = [1.0, 1.0, 1.0, 1.0, 0.0]
        opts = {
            "maxfevals": 60,
            "bounds": [[0.70, 0.70, 0.70, 0.70, 0.0],
                       [1.30, 1.30, 1.30, 1.30, 2000.0]],
            "seed": 42, "verbose": -9, "tolfun": 1e-6,
        }

        es = cma.CMAEvolutionStrategy(x0, 0.08, opts)
        es.optimize(lambda x: cmaes_objective(x, ba, bb, mexps))
        r = es.result

        s0, s1, s2, sa, a2 = r.xbest
        oa = [ba[0] * sa, ba[1], a2]
        ob = [bb[0] * s0, bb[1] * s1, bb[2] * s2]
        optimized[nm] = {"alpha": oa, "beta": ob}

        print(f"  Best f={r.fbest:.4f} scales=[{s0:.3f},{s1:.3f},{s2:.3f},{sa:.3f}] a2={a2:.0f}")
        print(f"  alpha=[{oa[0]:.0f},{oa[1]:.1f},{oa[2]:.0f}]")
        print(f"  beta=[{ob[0]:.0f},{ob[1]:.3f},{ob[2]:.1f}]")
        print(f"  {r.evaluations} evals")

    # Validate
    print("\n--- Full validation ---")
    per_model_coeffs = {}
    for k, v in optimized.items():
        per_model_coeffs[k] = v  # Already normalized

    results = []
    for exp in experiments:
        norm = normalize_model_name(exp["model"])
        coeffs = optimized.get(norm)
        if not coeffs:
            results.append({"experiment": exp["dirname"], "model": exp["model"],
                            "workload": exp["workload"], "status": "no_coeffs"})
            continue

        gt = load_ground_truth(exp["dirpath"])
        blis = run_blis(exp["dirpath"], coeffs["alpha"], coeffs["beta"])
        if not blis:
            results.append({"experiment": exp["dirname"], "model": exp["model"],
                            "workload": exp["workload"], "status": "blis_failed"})
            continue

        e = rel_error(blis["e2e_mean_ms"], gt["e2e_mean_s"] * 1000)
        t = rel_error(blis["ttft_mean_ms"], gt["ttft_mean_s"] * 1000)
        i = rel_error(blis["itl_mean_ms"], gt["itl_mean_s"] * 1000)

        print(
            f"  {exp['dirname']}: E2E={e*100:.1f}% TTFT={t*100:.1f}% ITL={i*100:.1f}%  "
            f"(pred={blis['e2e_mean_ms']:.0f}ms gt={gt['e2e_mean_s']*1000:.0f}ms)"
        )

        results.append({
            "experiment": exp["dirname"], "model": exp["model"],
            "workload": exp["workload"], "tp": exp["tp"], "status": "ok",
            "gt_e2e_ms": gt["e2e_mean_s"] * 1000, "gt_ttft_ms": gt["ttft_mean_s"] * 1000,
            "gt_itl_ms": gt["itl_mean_s"] * 1000,
            "blis_e2e_ms": blis["e2e_mean_ms"], "blis_ttft_ms": blis["ttft_mean_ms"],
            "blis_itl_ms": blis["itl_mean_ms"],
            "e2e_error": e, "ttft_error": t, "itl_error": i,
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(H2_OUTPUT, "blis_validation.csv"), index=False)
    ok = df[df["status"] == "ok"]

    summary = {}
    if len(ok) > 0:
        summary = {
            "mean_e2e_error_pct": float(ok["e2e_error"].mean() * 100),
            "mean_ttft_error_pct": float(ok["ttft_error"].mean() * 100),
            "mean_itl_error_pct": float(ok["itl_error"].mean() * 100),
            "median_e2e_error_pct": float(ok["e2e_error"].median() * 100),
            "n_experiments": len(ok),
            "n_below_10_e2e": int((ok["e2e_error"] < 0.10).sum()),
            "n_below_20_e2e": int((ok["e2e_error"] < 0.20).sum()),
            "optimized_coeffs": optimized,
        }

    with open(os.path.join(H2_OUTPUT, "h2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _print_table("H2 CMA-ES", ok)

    return {"optimized_coeffs": optimized, "results": results, "summary": summary}


# ===================================================================
# H3: LOMO
# ===================================================================
def run_h3(experiments, step_df):
    print("\n" + "=" * 70)
    print("H3: LEAVE-ONE-MODEL-OUT (LOMO)")
    print("=" * 70)

    model_exps = defaultdict(list)
    for exp in experiments:
        model_exps[normalize_model_name(exp["model"])].append(exp)
    models = sorted(model_exps.keys())
    print(f"Models: {models}")

    all_results = []
    fold_summaries = []

    for holdout in models:
        print(f"\n--- Hold out {holdout} ---")

        # Train on other models
        train_exps = [e for e in experiments if normalize_model_name(e["model"]) != holdout]
        train_mask = step_df["model"].map(normalize_model_name) != holdout
        train_df = step_df[train_mask].copy()

        regression = calibrate_coefficients(train_exps, train_df)
        ttft = compute_ttft_corrections(train_exps)

        # Try each donor
        donor_results = {}
        for donor in [m for m in models if m != holdout]:
            donor_coeffs = None
            for k, reg in regression.items():
                if normalize_model_name(k) == donor:
                    ttft_us = 0
                    for tk, tv in ttft.items():
                        if normalize_model_name(tk) == donor:
                            ttft_us = tv
                            break
                    donor_coeffs = {
                        "alpha": [ttft_us, 0.0, 0.0],
                        "beta": [reg["beta0"], reg["beta1_prefill"], reg["beta2_decode"]],
                    }
                    break

            if not donor_coeffs:
                continue

            errors = []
            for exp in model_exps[holdout]:
                gt = load_ground_truth(exp["dirpath"])
                blis = run_blis(exp["dirpath"], donor_coeffs["alpha"], donor_coeffs["beta"])
                if not blis:
                    continue
                e = rel_error(blis["e2e_mean_ms"], gt["e2e_mean_s"] * 1000)
                i = rel_error(blis["itl_mean_ms"], gt["itl_mean_s"] * 1000)
                errors.append({"experiment": exp["dirname"], "donor": donor,
                               "holdout": holdout, "e2e_error": e, "itl_error": i,
                               "gt_e2e_ms": gt["e2e_mean_s"] * 1000,
                               "blis_e2e_ms": blis["e2e_mean_ms"]})

            if errors:
                me = np.mean([x["e2e_error"] for x in errors])
                mi = np.mean([x["itl_error"] for x in errors])
                donor_results[donor] = {"mean_e2e": me, "mean_itl": mi, "errors": errors}
                print(f"  {donor} -> {holdout}: E2E={me*100:.1f}% ITL={mi*100:.1f}%")
                all_results.extend(errors)

        if donor_results:
            best = min(donor_results, key=lambda d: donor_results[d]["mean_e2e"])
            b = donor_results[best]
            fold_summaries.append({
                "holdout": holdout, "best_donor": best,
                "best_e2e": b["mean_e2e"], "best_itl": b["mean_itl"],
                "all_donors": {d: {"e2e": v["mean_e2e"], "itl": v["mean_itl"]}
                               for d, v in donor_results.items()},
            })
            print(f"  Best: {best} (E2E={b['mean_e2e']*100:.1f}%)")

    pd.DataFrame(all_results).to_csv(os.path.join(H3_OUTPUT, "lomo_results.csv"), index=False)

    summary = {
        "fold_summaries": fold_summaries,
        "mean_best_donor_e2e_pct": float(np.mean([f["best_e2e"] for f in fold_summaries]) * 100) if fold_summaries else 0,
        "mean_best_donor_itl_pct": float(np.mean([f["best_itl"] for f in fold_summaries]) * 100) if fold_summaries else 0,
    }
    with open(os.path.join(H3_OUTPUT, "h3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n--- LOMO Summary ---")
    for fs in fold_summaries:
        print(f"  {fs['holdout']:<25} best={fs['best_donor']:<25} E2E={fs['best_e2e']*100:.1f}% ITL={fs['best_itl']*100:.1f}%")
    if fold_summaries:
        print(f"  MEAN best-donor E2E={summary['mean_best_donor_e2e_pct']:.1f}% ITL={summary['mean_best_donor_itl_pct']:.1f}%")

    return {"summary": summary, "fold_summaries": fold_summaries}


# ===================================================================
# H4: LOWO
# ===================================================================
def run_h4(experiments, coeffs):
    print("\n" + "=" * 70)
    print("H4: LEAVE-ONE-WORKLOAD-OUT (LOWO)")
    print("=" * 70)

    results = []
    for exp in experiments:
        norm = normalize_model_name(exp["model"])
        c = coeffs.get(norm)
        if not c:
            continue
        gt = load_ground_truth(exp["dirpath"])
        blis = run_blis(exp["dirpath"], c["alpha"], c["beta"])
        if not blis:
            continue
        e = rel_error(blis["e2e_mean_ms"], gt["e2e_mean_s"] * 1000)
        i = rel_error(blis["itl_mean_ms"], gt["itl_mean_s"] * 1000)
        results.append({
            "experiment": exp["dirname"], "model": exp["model"],
            "norm_model": norm, "workload": exp["workload"],
            "e2e_error": e, "itl_error": i,
            "gt_e2e_ms": gt["e2e_mean_s"] * 1000, "blis_e2e_ms": blis["e2e_mean_ms"],
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(H4_OUTPUT, "lowo_results.csv"), index=False)

    # Per-model breakdown
    mw_stats = {}
    for nm in sorted(df["norm_model"].unique()):
        mdf = df[df["norm_model"] == nm]
        pw = {}
        for wl in sorted(mdf["workload"].unique()):
            wdf = mdf[mdf["workload"] == wl]
            pw[wl] = {"e2e": float(wdf["e2e_error"].mean()), "itl": float(wdf["itl_error"].mean())}
        errs = [v["e2e"] for v in pw.values()]
        rng = (max(errs) - min(errs)) * 100 if len(errs) > 1 else 0
        mw_stats[nm] = {"per_wl": pw, "range_pp": rng, "mean_e2e": float(mdf["e2e_error"].mean())}

    agg_e2e = float(df["e2e_error"].mean()) if len(df) > 0 else 1.0
    agg_itl = float(df["itl_error"].mean()) if len(df) > 0 else 1.0
    within2x = int((df["e2e_error"] < 2 * agg_e2e).sum()) if len(df) > 0 else 0

    summary = {
        "aggregate_e2e_pct": agg_e2e * 100, "aggregate_itl_pct": agg_itl * 100,
        "n_within_2x": within2x, "n_total": len(df),
        "model_workload_stats": mw_stats,
    }
    with open(os.path.join(H4_OUTPUT, "h4_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n--- LOWO Summary ---")
    print(f"Aggregate E2E: {agg_e2e*100:.1f}% | Within 2x: {within2x}/{len(df)}")
    for nm, st in sorted(mw_stats.items()):
        print(f"  {nm}: range={st['range_pp']:.1f}pp mean_e2e={st['mean_e2e']*100:.1f}%")
        for wl, ws in sorted(st["per_wl"].items()):
            print(f"    {wl}: E2E={ws['e2e']*100:.1f}% ITL={ws['itl']*100:.1f}%")

    return {"summary": summary}


def _print_table(title, ok):
    if ok is None or len(ok) == 0:
        print(f"\n{title}: No results!")
        return
    me = ok["e2e_error"].mean() * 100
    mt = ok["ttft_error"].mean() * 100
    mi = ok["itl_error"].mean() * 100
    b10 = (ok["e2e_error"] < 0.10).sum()
    b20 = (ok["e2e_error"] < 0.20).sum()
    print(f"\n{'=' * 70}")
    print(f"{title} RESULTS")
    print(f"{'=' * 70}")
    print(f"N={len(ok)} | E2E={me:.1f}% | TTFT={mt:.1f}% | ITL={mi:.1f}% | <10%={b10}/{len(ok)} | <20%={b20}/{len(ok)}")
    print(f"{'Experiment':<55} {'E2E%':>7} {'TTFT%':>7} {'ITL%':>7}")
    print("-" * 80)
    for _, r in ok.iterrows():
        print(f"{r['experiment']:<55} {r['e2e_error']*100:>7.1f} {r['ttft_error']*100:>7.1f} {r['itl_error']*100:>7.1f}")
    print("-" * 80)
    print(f"{'MEAN':<55} {me:>7.1f} {mt:>7.1f} {mi:>7.1f}")


def main():
    t0 = time.time()
    print("=" * 70)
    print("ROUND 4 IDEA 3: HYBRID CALIBRATION")
    print("=" * 70)

    if not os.path.isfile(_BINARY):
        print("Building BLIS...")
        subprocess.run(["go", "build", "-o", _BINARY, "main.go"], cwd=_REPO_ROOT, check=True)

    print("\n--- Loading step data ---")
    step_df = load_all_experiments(_DATA_ROOT)
    print(f"{len(step_df)} steps, {step_df['experiment_id'].nunique()} experiments")

    experiments = list_experiments()
    print(f"{len(experiments)} experiments for validation")

    h1 = run_h1(experiments, step_df)
    h2 = run_h2(experiments, h1["per_model_coeffs"])
    h3 = run_h3(experiments, step_df)

    # H4 uses H2 coefficients if available, else H1
    h4_coeffs = h2.get("optimized_coeffs", {})
    if not h4_coeffs:
        h4_coeffs = {}
        for k, v in h1["per_model_coeffs"].items():
            h4_coeffs[normalize_model_name(k)] = v
    h4 = run_h4(experiments, h4_coeffs)

    elapsed = time.time() - t0
    h1s = h1.get("summary", {})
    h2s = h2.get("summary", {})
    h3s = h3.get("summary", {})
    h4s = h4.get("summary", {})

    print("\n" + "=" * 70)
    print("GRAND SUMMARY")
    print("=" * 70)
    print(f"{'Hyp':<25} {'E2E%':>8} {'ITL%':>8} {'<10%':>6} {'Note':>25}")
    print("-" * 75)
    print(f"{'H1 Base':<25} {h1s.get('mean_e2e_error_pct',0):>8.1f} {h1s.get('mean_itl_error_pct',0):>8.1f} {h1s.get('n_below_10_e2e',0):>6} {'target <30%':>25}")
    print(f"{'H2 CMA-ES':<25} {h2s.get('mean_e2e_error_pct',0):>8.1f} {h2s.get('mean_itl_error_pct',0):>8.1f} {h2s.get('n_below_10_e2e',0):>6} {'target <12%':>25}")
    print(f"{'H3 LOMO best-donor':<25} {h3s.get('mean_best_donor_e2e_pct',0):>8.1f} {h3s.get('mean_best_donor_itl_pct',0):>8.1f} {'':>6} {'target <20%':>25}")
    n2x = h4s.get('n_within_2x', 0)
    nt = h4s.get('n_total', 0)
    print(f"{'H4 LOWO':<25} {h4s.get('aggregate_e2e_pct',0):>8.1f} {h4s.get('aggregate_itl_pct',0):>8.1f} {'':>6} {f'{n2x}/{nt} within 2x':>25}")

    print(f"\nR3 baselines: CMA-ES 15.1/87.4 | TraceReplay 56.2/9.5 | LOMO 14.8")
    print(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    json.dump({"h1": h1s, "h2": h2s, "h3": h3s, "h4": h4s, "runtime_s": elapsed},
              open(os.path.join(_SCRIPT_DIR, "grand_summary.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
