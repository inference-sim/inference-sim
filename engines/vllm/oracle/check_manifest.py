# Original to blis-vllm.
"""Cross-check docs/metrics-manifest.json against the families vLLM registers at runtime.

cmd/metricsmanifest reads the families from the release's source. This builds vLLM's own
PrometheusStatLogger (with the speculative-decoding and KV-cache metrics on) on CPU, collects
every vllm:* family the Prometheus registry then holds, and checks each against the
manifest: same type, and every label the registry shows is one the manifest lists (the
per-engine model_name and engine stand behind "labelnames"). Families the CPU oracle cannot
build here (connectors, offload tiers, the API server's) are checked by the source scan only;
the count of each kind is printed.

It also writes, with --buckets <path>, the bucket upper bounds of every histogram the
registry holds (max_model_len 256 for the request-token series), so the histogram buckets
the tier compares are vLLM's own, read from the objects it built rather than from its source.

Usage: python -I oracle/check_manifest.py <vllm checkout> docs/metrics-manifest.json [--buckets <path>]
"""
import json
import os
import sys


def main(vllm_root: str, manifest_path: str, buckets_path: str | None = None) -> int:
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    from prometheus_client import REGISTRY  # noqa: E402

    from tests.v1.core.utils import create_scheduler  # noqa: E402
    from vllm.v1.metrics.loggers import PrometheusStatLogger  # noqa: E402

    model = os.environ.get("ORACLE_MODEL", "facebook/opt-125m")
    scheduler = create_scheduler(model=model, max_num_seqs=4, max_num_batched_tokens=64,
                                 num_blocks=64, block_size=16, max_model_len=256,
                                 num_speculative_tokens=2)
    cfg = scheduler.vllm_config
    cfg.observability_config.kv_cache_metrics = True
    PrometheusStatLogger(cfg, [0])
    with open(manifest_path) as f:
        manifest = {fam["name"]: fam for fam in json.load(f)["families"]}

    bad = 0
    seen = 0
    buckets = {}
    for metric in REGISTRY.collect():
        if not metric.name.startswith("vllm:"):
            continue
        seen += 1
        # prometheus_client strips a counter's _total suffix from the family name.
        fam = manifest.get(metric.name) or manifest.get(metric.name + "_total")
        if fam is None:
            print(f"{metric.name}: registered at runtime, missing from the manifest")
            bad += 1
            continue
        if fam["type"] != metric.type:
            print(f"{metric.name}: the manifest says {fam['type']}, the registry {metric.type}")
            bad += 1
        if metric.type == "histogram":
            bounds = sorted({float(s.labels["le"]) for s in metric.samples
                             if s.name.endswith("_bucket") and s.labels["le"] != "+Inf"})
            buckets[fam["name"]] = bounds
        labels = {k for s in metric.samples for k in s.labels} - {"le"}
        allowed = set(fam["labels"])
        if "labelnames" in allowed:
            allowed |= {"model_name", "engine"}
        if not labels <= allowed and not any(lbl.startswith("(") for lbl in allowed):
            print(f"{metric.name}: labels {sorted(labels)} at runtime, the manifest lists {fam['labels']}")
            bad += 1
    print(f"{seen} families registered at runtime checked against {len(manifest)} in the manifest: {bad} problems")
    if buckets_path:
        with open(buckets_path, "w") as f:
            json.dump(dict(sorted(buckets.items())), f, indent=1)
            f.write("\n")
        print(f"wrote {len(buckets)} histograms' bucket bounds to {buckets_path}")
    return 1 if bad or seen == 0 else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    buckets_path = None
    if "--buckets" in args:
        i = args.index("--buckets")
        buckets_path = args[i + 1]
        del args[i:i + 2]
    sys.exit(main(args[0], args[1], buckets_path))
