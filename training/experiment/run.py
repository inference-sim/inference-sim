"""Orchestrator — run all simulators against all experiments and generate report.

Usage::

    python -m experiment.run --data-dir vllm_data/ground_truth --output-dir results
"""

from __future__ import annotations

import argparse
import logging
import time

import yaml

logger = logging.getLogger(__name__)

from experiment.adapters.aiconfigurator_est import AIConfiguratorEstimateAdapter
from experiment.adapters.base import SimulatorAdapter
from experiment.adapters.blis_blackbox import BLISBlackboxAdapter
from experiment.adapters.blis_crossmodel import BLISCrossModelAdapter
from experiment.adapters.blis_roofline import BLISRooflineAdapter
from experiment.adapters.blis_trained_roofline import BLISTrainedRooflineAdapter
from experiment.adapters.llm_optimizer_est import LLMOptimizerEstimateAdapter
from experiment.adapters.llmservingsim import LLMServingSimAdapter
from experiment.adapters.vidur import VidurAdapter
from experiment.ground_truth import discover_experiments, parse_experiment
from experiment.metrics import ErrorRecord, RuntimeRecord, compute_errors
from experiment.report import generate_report

ALL_ADAPTER_NAMES = [
    "blis-blackbox",
    "blis-roofline",
    "blis-crossmodel",
    "blis-trained-roofline",
    "vidur",
    "llm-optimizer-estimate",
    "aiconfigurator-estimate",
    "llmservingsim",
]


def build_adapter_registry(
    blis_binary: str,
    vidur_dir: str,
    llmservingsim_dir: str,
    adapter_names: list[str] | None = None,
    no_docker: bool = False,
    max_requests_per_experiment: int | None = None,
) -> dict[str, SimulatorAdapter]:
    """Build a name → adapter instance mapping.

    Only instantiates adapters listed in *adapter_names* (default: all).
    """
    factories: dict[str, callable] = {
        "blis-blackbox": lambda: BLISBlackboxAdapter(blis_binary),
        "blis-roofline": lambda: BLISRooflineAdapter(blis_binary),
        "blis-crossmodel": lambda: BLISCrossModelAdapter(blis_binary),
        "blis-trained-roofline": lambda: BLISTrainedRooflineAdapter(blis_binary),
        "vidur": lambda: VidurAdapter(vidur_dir),
        "llm-optimizer-estimate": lambda: LLMOptimizerEstimateAdapter(),
        "aiconfigurator-estimate": lambda: AIConfiguratorEstimateAdapter(),
        "llmservingsim": lambda: LLMServingSimAdapter(
            llmservingsim_dir,
            use_docker=not no_docker,
            max_requests_per_experiment=max_requests_per_experiment,
        ),
    }
    if adapter_names is None:
        adapter_names = list(factories.keys())
    return {name: factories[name]() for name in adapter_names if name in factories}


def run_pipeline(
    data_dir: str,
    blis_binary: str,
    vidur_dir: str,
    llmservingsim_dir: str,
    output_dir: str,
    adapter_names: list[str] | None = None,
    no_dp_scaling: bool = False,
    no_docker: bool = False,
    max_requests_per_experiment: int | None = None,
) -> tuple[list[ErrorRecord], list[RuntimeRecord]]:
    """Core pipeline: discover → run → compute errors → report.

    Returns (error_records, runtime_records).
    """
    if adapter_names is None:
        adapter_names = ALL_ADAPTER_NAMES

    # 1. Discover experiments
    discovered = discover_experiments(data_dir)
    if not discovered:
        print(f"No experiments found in {data_dir}")
        return [], []

    print(f"Found {len(discovered)} experiments")

    # 2. Parse experiments
    experiments = []
    for manifest_entry, dir_path in discovered:
        try:
            experiments.append(parse_experiment(dir_path, manifest_entry=manifest_entry))
        except (OSError, KeyError, ValueError, yaml.YAMLError) as exc:
            logger.error("SKIP (parse error): %s — %s", dir_path, exc)

    print(f"Parsed {len(experiments)} experiments successfully")
    if discovered and not experiments:
        logger.warning("All %d experiments failed to parse", len(discovered))

    # Filter by DP if requested
    if no_dp_scaling:
        before_count = len(experiments)
        experiments = [exp for exp in experiments
                       if exp.dp is None or exp.dp <= 1]
        filtered_count = before_count - len(experiments)
        print(f"Filtered to {len(experiments)} single-replica experiments "
              f"(excluded {filtered_count} with DP > 1)")

    # 3. Build adapter registry (only requested adapters)
    adapters = build_adapter_registry(
        blis_binary, vidur_dir, llmservingsim_dir, adapter_names, no_docker, max_requests_per_experiment
    )

    # 4. Run all (experiment, adapter) pairs
    all_records: list[ErrorRecord] = []
    runtime_records: list[RuntimeRecord] = []
    fail_count = 0
    skip_count = 0
    for exp in experiments:
        for adapter_name, adapter in adapters.items():
            if not adapter.can_run(exp):
                skip_count += 1
                continue
            try:
                t0 = time.perf_counter()
                result = adapter.run(exp)
                elapsed = time.perf_counter() - t0
                result.wall_clock_seconds = elapsed
                records = compute_errors(exp, result)
                all_records.extend(records)
                runtime_records.append(RuntimeRecord(
                    simulator=adapter_name,
                    experiment_folder=exp.folder,
                    model=exp.model,
                    workload=exp.workload,
                    wall_clock_seconds=elapsed,
                    exp_id=exp.exp_id,
                    hardware=exp.hardware,
                    dp=exp.dp,
                    cpu_offload=exp.cpu_offload,
                    gpu_mem_util=exp.gpu_mem_util,
                    precision=exp.precision,
                    tp=exp.tp,
                    max_num_batched_tokens=exp.max_num_batched_tokens,
                ))
                print(f"  OK: {adapter_name} × {exp.model} ({exp.workload}) [{elapsed:.2f}s]")
            except Exception as exc:
                fail_count += 1
                logger.error(
                    "FAIL: %s × %s (%s): %s", adapter_name, exp.model, exp.workload, exc,
                )

    if skip_count:
        logger.info("Skipped %d (experiment, adapter) pairs via can_run()", skip_count)
    if fail_count:
        logger.warning("%d adapter runs failed", fail_count)

    # 5. Generate report
    generate_report(all_records, output_dir, runtime_records=runtime_records)

    return all_records, runtime_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sim-to-real accuracy validation."
    )
    parser.add_argument(
        "--data-dir",
        default="vllm_data/ground_truth",
        help="Directory containing ground-truth experiment folders.",
    )
    parser.add_argument(
        "--blis-binary",
        default="inference-sim/blis",
        help="Path to the BLIS simulator binary.",
    )
    parser.add_argument(
        "--vidur-dir",
        default="vidur",
        help="Path to the cloned Vidur repository.",
    )
    parser.add_argument(
        "--llmservingsim-dir",
        default="LLMServingSim",
        help="Path to LLMServingSim directory containing main.py",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Disable Docker for LLMServingSim (use native execution)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where reports and CSV will be saved.",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=ALL_ADAPTER_NAMES,
        choices=ALL_ADAPTER_NAMES,
        help="Which adapters to run.",
    )
    parser.add_argument(
        "--no-dp-scaling",
        action="store_true",
        help="Exclude experiments with data parallelism > 1 (multi-replica).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )
    parser.add_argument(
        "--max-requests-per-experiment",
        type=int,
        default=100,
        help="Limit number of requests per experiment (default: 100 for LLMServingSim). Set to 0 for unlimited.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Configure logging level based on --verbose flag
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    run_pipeline(
        data_dir=args.data_dir,
        blis_binary=args.blis_binary,
        vidur_dir=args.vidur_dir,
        llmservingsim_dir=args.llmservingsim_dir,
        output_dir=args.output_dir,
        adapter_names=args.adapters,
        no_dp_scaling=args.no_dp_scaling,
        no_docker=args.no_docker,
        max_requests_per_experiment=args.max_requests_per_experiment,
    )


if __name__ == "__main__":
    main()
