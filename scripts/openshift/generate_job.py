#!/usr/bin/env python3
"""Generate OpenShift Job YAML from template with GPU-specific parameters"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load_config(repo_root: Path) -> dict:
    """Load benchmark configuration"""
    config_path = repo_root / "config" / "benchmark_config.json"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return json.load(f)


def get_node_selector(gpu_type: str, default_selector: str) -> tuple:
    """Get node selector for GPU type - returns (label_key, label_value)"""
    # Allow override via environment variable
    import os
    if "NODE_SELECTOR" in os.environ:
        # For backward compatibility, assume it's instance-type if no = sign
        selector = os.environ["NODE_SELECTOR"]
        if "=" in selector:
            key, value = selector.split("=", 1)
            return key, value
        return "node.kubernetes.io/instance-type", selector

    # GPU type specific node selectors (using actual cluster labels)
    gpu_to_selector = {
        "H100": ("nvidia.com/gpu.product", "NVIDIA-H100-80GB-HBM3"),
        "A100": ("nvidia.com/gpu.product", "NVIDIA-A100-SXM4-80GB"),
        "H20": ("nvidia.com/gpu.product", "NVIDIA-H20"),
        "B100": ("nvidia.com/gpu.product", "NVIDIA-B100"),
    }

    selector = gpu_to_selector.get(gpu_type)
    if selector is None:
        print(f"Warning: Unknown GPU type {gpu_type}, using default from config")
        return "node.kubernetes.io/instance-type", default_selector

    return selector


def generate_job_yaml(
    gpu_type: str,
    job_suffix: str,
    config: dict,
    template_file: Path,
    output_file: Path,
    model: str = None,
    phase: str = None,
    tp: int = None,
    shape: str = None
):
    """Generate job YAML from template"""
    # Extract config values
    openshift_config = config["openshift"]
    namespace = openshift_config["namespace"]
    job_prefix = openshift_config["job_name_prefix"]
    container_image = openshift_config["container_image"]
    default_node_selector = openshift_config["node_selector"]

    # Generate job name with shape/model/phase/tp info
    gpu_type_lower = gpu_type.lower()
    name_parts = [job_prefix, gpu_type_lower]

    if shape:
        # Use shape directly for job name (e.g., 32-8-128)
        name_parts.append(shape)
    elif model:
        # Convert model name to job-friendly format (llama-2-7b)
        model_slug = model.replace("_", "-")
        name_parts.append(model_slug)

    if phase:
        name_parts.append(phase)

    if tp is not None:
        name_parts.append(f"tp{tp}")

    name_parts.append(job_suffix)
    job_name = "-".join(name_parts)

    # Build benchmark script arguments
    bench_args = [f"--gpu {gpu_type}"]

    if shape:
        # Pass shape parameters to benchmark scripts
        bench_args.append(f"--shape {shape}")
    elif model:
        bench_args.append(f"--model-filter {model}")

    if phase:
        bench_args.append(f"--phase-filter {phase}")
    if tp is not None:
        bench_args.append(f"--tp-filter {tp}")

    bench_args_str = " ".join(bench_args)

    # Get node selector
    node_selector_key, node_selector_value = get_node_selector(gpu_type, default_node_selector)

    # Read template
    template_content = template_file.read_text()

    # Substitute variables
    yaml_content = template_content
    yaml_content = yaml_content.replace("${JOB_NAME}", job_name)
    yaml_content = yaml_content.replace("${NAMESPACE}", namespace)
    yaml_content = yaml_content.replace("${GPU_TYPE}", gpu_type)
    yaml_content = yaml_content.replace("${GPU_TYPE_LOWER}", gpu_type_lower)
    yaml_content = yaml_content.replace("${NODE_SELECTOR_KEY}", node_selector_key)
    yaml_content = yaml_content.replace("${NODE_SELECTOR_VALUE}", node_selector_value)
    yaml_content = yaml_content.replace("${CONTAINER_IMAGE}", container_image)
    yaml_content = yaml_content.replace("${BENCH_ARGS}", bench_args_str)
    yaml_content = yaml_content.replace("${PHASE}", phase or "")
    yaml_content = yaml_content.replace("${TP}", str(tp) if tp is not None else "")

    # Handle optional S3_BUCKET for uploading results
    import os
    s3_bucket = os.environ.get("S3_BUCKET", "")
    yaml_content = yaml_content.replace("${S3_BUCKET}", s3_bucket)

    # Write output
    output_file.write_text(yaml_content)

    # Print summary
    print("Generating OpenShift Job YAML")
    print("=" * 60)
    print(f"GPU Type:       {gpu_type}")
    print(f"Job Name:       {job_name}")
    if shape:
        print(f"Shape:          {shape}")
    if model:
        print(f"Model:          {model}")
    if phase:
        print(f"Phase:          {phase}")
    if tp is not None:
        print(f"TP Value:       {tp}")
    print(f"Namespace:      {namespace}")
    print(f"Node Selector:  {node_selector_key}={node_selector_value}")
    print(f"Container:      {container_image}")
    print(f"Benchmark Args: {bench_args_str}")
    print(f"Output:         {output_file}")
    print()
    print("Job YAML generated successfully!")
    print()
    print("To submit the job, run:")
    print(f"  oc apply -f {output_file} -n {namespace}")
    print()
    print("To monitor the job, run:")
    print(f"  oc logs -f job/{job_name} -n {namespace}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenShift Job YAML for GPU benchmarking"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="H100",
        help="GPU type to benchmark (default: H100)"
    )
    parser.add_argument(
        "--shape",
        type=str,
        help="Attention shape in format nh-nkv-dh (e.g., 32-8-128). Alternative to --model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama-2-7b", "llama-2-70b", "mixtral-8x7b"],
        help="Model to benchmark (creates focused job for single model)"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["prefill", "decode", "gemm"],
        help="Benchmark phase (creates focused job for single phase)"
    )
    parser.add_argument(
        "--tp",
        type=int,
        choices=[1, 2, 4],
        help="Tensor parallelism value (only for decode phase)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Job name suffix (default: timestamp)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for job YAML (default: scripts/openshift)"
    )
    args = parser.parse_args()

    # Validation
    if args.tp is not None and args.phase != "decode":
        print("Error: --tp can only be used with --phase decode")
        sys.exit(1)

    # Validate shape format if provided
    if args.shape:
        parts = args.shape.split("-")
        if len(parts) != 3:
            print("Error: --shape must be in format nh-nkv-dh (e.g., 32-8-128)")
            sys.exit(1)
        try:
            nh, nkv, dh = map(int, parts)
        except ValueError:
            print("Error: --shape components must be integers")
            sys.exit(1)

        if args.model:
            print("Error: Cannot specify both --shape and --model")
            sys.exit(1)

    # Generate default suffix if not provided
    job_suffix = args.suffix or datetime.now().strftime("%Y%m%d-%H%M%S")

    # Find repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    # Load config
    config = load_config(repo_root)

    # Paths
    template_file = script_dir / "job-benchmarks-template.yaml"
    gpu_type_lower = args.gpu.lower()

    # Build output filename based on filters
    output_parts = ["job", gpu_type_lower]
    if args.shape:
        output_parts.append(args.shape)
    elif args.model:
        output_parts.append(args.model)
    if args.phase:
        output_parts.append(args.phase)
    if args.tp is not None:
        output_parts.append(f"tp{args.tp}")
    output_parts.append(job_suffix)

    output_dir = args.output_dir or script_dir
    output_file = output_dir / f"{'-'.join(output_parts)}.yaml"

    if not template_file.exists():
        print(f"Error: Template file not found: {template_file}")
        sys.exit(1)

    # Generate YAML
    generate_job_yaml(
        args.gpu,
        job_suffix,
        config,
        template_file,
        output_file,
        model=args.model,
        phase=args.phase,
        tp=args.tp,
        shape=args.shape
    )


if __name__ == "__main__":
    main()
