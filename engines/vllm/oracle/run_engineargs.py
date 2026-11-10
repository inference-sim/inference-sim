# Original to blis-vllm.
"""Tier 1, the dynamic half: run engine blocks through vLLM's own
EngineArgs(...).create_engine_config() and record whether it accepts each and the values it
resolves (vllm/engine/arg_utils.py@5df63a72).

The rules packs are compared with the release's source text by their MatchSource tests; this
runs the same settings through the release's configuration classes, so a refusal or a default the
pack gets wrong, or states from a misreading of the source, shows as a difference. The Go side
(core/vllm TestEngineArgsAgainstVLLM) resolves the same cases with the factory.

async_scheduling is not recorded as a resolved value: the CPU platform turns it off after the
configuration resolves it (vllm/platforms/cpu.py:329-331@5df63a72, listed in oracle/mocks.json),
so the CPU answer is the platform's and not the engine's. Its refusals are recorded.

Usage: python -I oracle/run_engineargs.py <vllm checkout> <cases.json> > trace.json
cases.json holds {"cases": [{"name", "args"}]}: args are EngineArgs keyword arguments. The model is
facebook/opt-125m (a configuration only, no weights are read), the CPU platform.
"""
import json
import os
import sys


def main() -> None:
    vllm_root, cases_path = sys.argv[1], sys.argv[2]
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from vllm.engine.arg_utils import EngineArgs  # noqa: E402

    with open(cases_path) as f:
        cases = json.load(f)["cases"]
    model = os.environ.get("ORACLE_MODEL", "facebook/opt-125m")
    results = []
    for c in cases:
        args = dict(model=model, enforce_eager=True)
        args.update(c["args"])
        row = {"name": c["name"]}
        try:
            cfg = EngineArgs(**args).create_engine_config()
        except Exception as e:  # noqa: BLE001 - any refusal at start-up is the answer
            row["accepted"] = False
            # The first line of the message after the class name: pydantic's own wording of the rule.
            lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
            row["error"] = f"{type(e).__name__}: {' | '.join(lines[:3])}"[:400]
        else:
            s, k = cfg.scheduler_config, cfg.cache_config
            row["accepted"] = True
            row["resolved"] = {
                "max_num_seqs": s.max_num_seqs,
                "max_num_batched_tokens": s.max_num_batched_tokens,
                "max_model_len": cfg.model_config.max_model_len,
                "scheduling_policy": s.policy,
                "enable_chunked_prefill": bool(s.enable_chunked_prefill),
                "long_prefill_token_threshold": s.long_prefill_token_threshold,
                "enable_prefix_caching": bool(k.enable_prefix_caching),
                "block_size": k.block_size,
                "watermark": s.watermark,
                "max_loras": (cfg.lora_config.max_loras if cfg.lora_config is not None else 0),
            }
        results.append(row)
    from run_scheduler import provenance  # noqa: E402

    prov = provenance(vllm_root, cases_path)
    prov["tier"] = 1
    print(json.dumps({"provenance": prov, "cases": results}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
