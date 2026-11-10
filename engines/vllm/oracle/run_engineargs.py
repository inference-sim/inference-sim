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
       python -I oracle/run_engineargs.py --key-names <vllm checkout> <keys.json> > names.json
(the second form asks, for the engine block's keys, whether EngineArgs takes each name)
cases.json holds {"cases": [{"name", "args"}]}: args are EngineArgs keyword arguments. The model is
facebook/opt-125m (a configuration only, no weights are read), the CPU platform.
"""
import json
import os
import sys


def key_names(vllm_root: str, keys_path: str) -> dict:
    """Whether each engine-block key is a name vLLM's EngineArgs takes (blis-vllm#93).

    keys_path holds {"top": [...], "nested": {"compilation_config": ["cudagraph_mode", ...]}}.
    A top-level key is an EngineArgs field, or an environment variable (env's own keys are not
    checked here); a nested key is a field of the config class the EngineArgs field of that name
    is built from, which is read from the dataclass the field's annotation names.
    """
    import dataclasses
    import typing

    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    from vllm.engine.arg_utils import EngineArgs  # noqa: E402

    with open(keys_path) as f:
        want = json.load(f)
    fields = {f.name: f for f in dataclasses.fields(EngineArgs)}
    out = {"top": {k: k in fields for k in want["top"]}, "nested": {}}
    # The class behind each config-object argument is the one VllmConfig holds it as, found by the
    # name vLLM gives it: the config object's argument name, camel-cased ("kv_events_config" is
    # KVEventsConfig, "eplb_config" is EPLBConfig). Matching is by lower-cased letters.
    import vllm.config as vc  # noqa: E402

    by_letters = {n.lower(): getattr(vc, n) for n in dir(vc) if n.endswith("Config")}
    classes = {g: by_letters.get(g.replace("_", "")) for g in want["nested"]}
    for g, leaves in want["nested"].items():
        cls = classes.get(g)
        known = set()
        if cls is not None:
            known = set(getattr(cls, "model_fields", {}) or {}) | set(getattr(cls, "__dataclass_fields__", {}) or {})
        out["nested"][g] = {leaf: leaf in known for leaf in leaves}
    return out


def main() -> None:
    if len(sys.argv) > 3 and sys.argv[1] == "--key-names":
        print(json.dumps(key_names(sys.argv[2], sys.argv[3]), indent=1, sort_keys=True))
        return
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
