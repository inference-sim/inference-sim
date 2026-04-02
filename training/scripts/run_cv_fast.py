#!/usr/bin/env python3
"""Fast parallel CV runner with incremental progress.

Runs CV-1, CV-2, CV-3 using inner_loop_optimize.py with:
- n_jobs=8 for parallel trials within each CV test
- SQLite storage (on disk, checkable)
- --patience 100 for early stopping
- Progress logged to per-CV log files every trial

Usage:
    python3.11 scripts/run_cv_fast.py --iteration 24 --cv-test CV-1
    python3.11 scripts/run_cv_fast.py --iteration 24 --cv-test CV-2
    python3.11 scripts/run_cv_fast.py --iteration 24 --cv-test CV-3
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Deterministic splits (from run_cv_tests.py)
CV_SPLITS = {
    "CV-1": {
        "train": [
            "20260217-155451-llama-2-7b-tp1-codegen",
            "20260217-162547-llama-2-7b-tp1-roleplay",
            "20260217-231439-llama-2-7b-tp1-general",
            "67-llama-2-7b-hf-tp1-reasoning-lite-1-1",
            "60-llama-3-1-70b-tp4-general-lite-4-1",
            "61-llama-3-1-70b-tp4-codegen-4-1",
            "64-qwen2-5-7b-instruct-tp1-roleplay-1-1",
            "66-qwen2-5-7b-instruct-tp1-reasoning-lite-1-1",
            "17-llama-4-scout-17b-16e-tp2-general-lite-2-1",
            "20-llama-4-scout-17b-16e-tp2-codegen-2",
            "21-llama-4-scout-17b-16e-tp2-roleplay-2",
            "48-llama-4-scout-17b-16e-tp2-reasoning-lite-2-1",
        ],
        "test": [
            "65-01-ai-yi-34b-tp2-general-lite-2-1",
            "62-mistral-nemo-12b-tp2-general-lite-2-1",
            "63-mistral-nemo-12b-tp1-codegen-1-1",
        ],
        "pass": "MAPE < 15%",
    },
    "CV-2": {
        "train": [
            "20260217-155451-llama-2-7b-tp1-codegen",
            "67-llama-2-7b-hf-tp1-reasoning-lite-1-1",
            "20-llama-4-scout-17b-16e-tp2-codegen-2",
            "48-llama-4-scout-17b-16e-tp2-reasoning-lite-2-1",
            "61-llama-3-1-70b-tp4-codegen-4-1",
            "63-mistral-nemo-12b-tp1-codegen-1-1",
            "66-qwen2-5-7b-instruct-tp1-reasoning-lite-1-1",
        ],
        "test": [
            "20260217-162547-llama-2-7b-tp1-roleplay",
            "20260217-231439-llama-2-7b-tp1-general",
            "17-llama-4-scout-17b-16e-tp2-general-lite-2-1",
            "21-llama-4-scout-17b-16e-tp2-roleplay-2",
            "60-llama-3-1-70b-tp4-general-lite-4-1",
            "62-mistral-nemo-12b-tp2-general-lite-2-1",
            "64-qwen2-5-7b-instruct-tp1-roleplay-1-1",
            "65-01-ai-yi-34b-tp2-general-lite-2-1",
        ],
        "pass": "MAPE < 15%, workload variance < 3%",
    },
    "CV-3": {
        "train": [
            "20260217-155451-llama-2-7b-tp1-codegen",
            "20260217-162547-llama-2-7b-tp1-roleplay",
            "67-llama-2-7b-hf-tp1-reasoning-lite-1-1",
            "20260217-231439-llama-2-7b-tp1-general",
            "60-llama-3-1-70b-tp4-general-lite-4-1",
            "61-llama-3-1-70b-tp4-codegen-4-1",
            "63-mistral-nemo-12b-tp1-codegen-1-1",
            "64-qwen2-5-7b-instruct-tp1-roleplay-1-1",
            "66-qwen2-5-7b-instruct-tp1-reasoning-lite-1-1",
        ],
        "test": [
            "17-llama-4-scout-17b-16e-tp2-general-lite-2-1",
            "20-llama-4-scout-17b-16e-tp2-codegen-2",
            "21-llama-4-scout-17b-16e-tp2-roleplay-2",
            "48-llama-4-scout-17b-16e-tp2-reasoning-lite-2-1",
            "62-mistral-nemo-12b-tp2-general-lite-2-1",
            "65-01-ai-yi-34b-tp2-general-lite-2-1",
        ],
        "pass": "MAPE < 15%",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--cv-test", choices=["CV-1", "CV-2", "CV-3"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="cv_results/iter24")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    args = parser.parse_args()

    cv = args.cv_test
    split = CV_SPLITS[cv]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    training_dir = Path(__file__).parent.parent
    log_file = out_dir / f"{cv.lower()}_progress.log"

    def log(msg):
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    log(f"=== {cv}: {len(split['train'])} train, {len(split['test'])} test ===")

    # Step 1: Create train data directory
    train_dir = out_dir / f"{cv.lower()}_train_data"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir()
    for exp in split["train"]:
        src = data_dir / exp
        if src.exists():
            shutil.copytree(src, train_dir / exp)
    log(f"Train data: {len(list(train_dir.iterdir()))} experiments copied")

    # Step 2: Create a per-CV iteration directory (avoids SQLite conflicts between parallel CVs)
    # Use 2401/2402/2403 as iteration numbers for CV-1/CV-2/CV-3
    cv_num = {"CV-1": 2401, "CV-2": 2402, "CV-3": 2403}[cv]
    cv_iter_dir = training_dir / "iterations" / f"iter{cv_num}"
    cv_iter_dir.mkdir(parents=True, exist_ok=True)
    # Copy bounds from the real iteration
    real_iter_dir = training_dir / "iterations" / f"iter{args.iteration}"
    shutil.copy2(real_iter_dir / "coefficient_bounds.yaml", cv_iter_dir / "coefficient_bounds.yaml")
    # Write a minimal manifest for the CV iteration
    (cv_iter_dir / "iteration_manifest.yaml").write_text(
        f'iteration: {cv_num}\nlatency_backend_name: "evolved"\n'
        f'modified_files:\n  - "sim/latency/evolved_model.go"\n'
        f'reasoning: "CV test {cv} for iteration {args.iteration}"\n'
    )

    # Use absolute paths to avoid cwd confusion
    train_results_file = (out_dir / f"{cv.lower()}_train_results.json").resolve()
    log(f"Training: {args.n_trials} trials, n_jobs={args.n_jobs}, patience={args.patience}")
    log(f"CV iteration dir: iter{cv_num}")

    cmd = [
        sys.executable,
        str(training_dir / "inner_loop_optimize.py"),
        f"--iteration={cv_num}",
        f"--n-trials={args.n_trials}",
        f"--n-jobs={args.n_jobs}",
        f"--patience={args.patience}",
        f"--timeout=200",
        f"--data-dir={train_dir.resolve()}",
        f"--seed=42",
        f"--output={train_results_file}",
        "--no-detailed-eval",
    ]

    result = subprocess.run(cmd, cwd=training_dir, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"TRAIN FAILED: {result.stderr[-500:]}")
        return

    # Step 3: Load trained coefficients
    with open(train_results_file) as f:
        train = json.load(f)
    alpha = train["best_params"]["alpha"]
    beta = train["best_params"]["beta"]
    train_loss = train["loss"]["overall_loss"]
    log(f"Train loss: {train_loss:.2f}%")

    # Step 4: Create test data directory
    test_dir = out_dir / f"{cv.lower()}_test_data"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    for exp in split["test"]:
        src = data_dir / exp
        if src.exists():
            shutil.copytree(src, test_dir / exp)

    # Step 5: Evaluate on test set
    alpha_str = ",".join(f"{x:.10e}" for x in alpha)
    beta_str = ",".join(f"{x:.10e}" for x in beta)

    eval_cmd = [
        sys.executable,
        str(training_dir / "run_blis_and_compute_loss.py"),
        "--latency-model", "evolved",
        "--alpha-coeffs", alpha_str,
        "--beta-coeffs", beta_str,
        "--blis-binary", str((training_dir / ".." / "blis").resolve()),
        "--data-dir", str(test_dir.resolve()),
        "--evaluate-per-experiment",
    ]

    eval_result = subprocess.run(eval_cmd, cwd=training_dir, capture_output=True, text=True, timeout=300)
    diag = json.loads(eval_result.stdout)

    test_loss = diag["overall_loss"]
    ttft_rmse = diag["ttft_rmse"]
    e2e_rmse = diag["e2e_rmse"]

    per_exp = diag.get("per_experiment", [])
    ttft_mape = sum(e["ttft_mean_ape"] for e in per_exp) / len(per_exp) if per_exp else ttft_rmse
    e2e_mape = sum(e["e2e_mean_ape"] for e in per_exp) / len(per_exp) if per_exp else e2e_rmse

    passed = ttft_mape < 15 and e2e_mape < 15
    status = "PASS" if passed else "FAIL"

    log(f"Test TTFT MAPE: {ttft_mape:.2f}%")
    log(f"Test E2E MAPE: {e2e_mape:.2f}%")
    log(f"Test loss: {test_loss:.2f}%")
    log(f"Status: {'✅' if passed else '❌'} {status}")

    # Save results
    out = {
        "cv_test": cv,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_ttft_mape": ttft_mape,
        "test_e2e_mape": e2e_mape,
        "status": status,
        "per_experiment": per_exp,
        "best_alpha": alpha,
        "best_beta": beta,
    }
    with open(out_dir / f"{cv.lower()}_results.json", "w") as f:
        json.dump(out, f, indent=2)

    log(f"Done. Results: {out_dir / f'{cv.lower()}_results.json'}")


if __name__ == "__main__":
    main()
