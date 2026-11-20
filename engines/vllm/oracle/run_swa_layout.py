# Original to blis-vllm.
"""Tier-2 oracle for the field-to-layout path of swa_bounded_replay.

oracle/run_scheduler.py is given DeepSeek-V4.1's KV layers already resolved: each sliding-window
MLA run states its own bounded_replay. This runner is given the deployment's STATEMENT instead,
`config.swa_bounded_replay` (absent: the field's default) and `config.use_v2_model_runner` (and
`config.prefill_context_parallel_size`), and lets vLLM resolve it into the layers. A scenario's
sliding_window_mla runs state no bounded_replay; the layout they get is the one vLLM's own code
decides.

vLLM's decision is the statements in DeepseekV4Attention.__init__ that read
CacheConfig.swa_bounded_replay and turn it off under the V1 model runner, under prefill context
parallelism and on ROCm (vllm/models/deepseek_v41/attention.py:529-556@5df63a72). That module
imports CUDA wheels, so, as oracle/lower/extract.py does, this parses it with `ast`, takes those
statements out as written and runs them unmodified in a namespace that holds only what they read:
the real CacheConfig and platform, and two stand-ins for the VllmConfig fields they read
(use_v2_model_runner, which the CPU wheel cannot select; parallel_config.prefill_context_parallel_size).
Nothing is rewritten. The resolved value is then written into the scenario's sliding-window runs
and handed to oracle/run_scheduler.py, which runs vLLM's Scheduler on the layers it builds, and the real SlidingWindowMLASpec reports
prefix_cacheable and prefix_replay_tokens, which the trace records under "swa_layout".

Usage: python -I oracle/run_swa_layout.py <vllm checkout> <scenario.json | dir> [out dir]
"""
import ast
import copy
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ATTENTION = "vllm/models/deepseek_v41/attention.py"


def resolution_statements(vllm_root: str) -> list:
    """The statements of DeepseekV4Attention.__init__ that resolve swa_bounded_replay, as written:
    the assignment from CacheConfig and the three guards that turn it off. Fails if the pinned
    source no longer has that shape, so a change to it cannot go unnoticed."""
    with open(os.path.join(vllm_root, ATTENTION), encoding="utf-8") as f:
        tree = ast.parse(f.read(), ATTENTION)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4Attention")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")

    def assigns_flag(n):
        return (isinstance(n, ast.Assign) and [getattr(t, "id", "") for t in n.targets] == ["swa_bounded_replay"]
                and ast.unparse(n.value) == "cache_config.swa_bounded_replay")

    start = [i for i, n in enumerate(init.body) if assigns_flag(n)]
    if len(start) != 1:
        raise SystemExit(f"{ATTENTION}: expected one `swa_bounded_replay = cache_config.swa_bounded_replay`")
    stmts = init.body[start[0]:start[0] + 4]
    guards = stmts[1:]
    ok = len(stmts) == 4 and all(isinstance(g, ast.If) and ast.unparse(g.test).startswith("swa_bounded_replay")
                                 for g in guards)
    ok = ok and all(ast.unparse(g.body[-1]) == "swa_bounded_replay = False" for g in guards)
    if not ok:
        raise SystemExit(f"{ATTENTION}: the swa_bounded_replay resolution is no longer an assignment and three guards")
    return stmts


def resolve(vllm_root: str, cfg: dict) -> dict:
    """What vLLM's own statements resolve the scenario's statement to."""
    from vllm.config import CacheConfig  # noqa: E402
    from vllm.platforms import current_platform  # noqa: E402

    stated = cfg.get("swa_bounded_replay")
    cache_config = CacheConfig() if stated is None else CacheConfig(swa_bounded_replay=stated)
    ns = {
        "cache_config": cache_config,
        "vllm_config": types.SimpleNamespace(
            use_v2_model_runner=bool(cfg.get("use_v2_model_runner", True)),
            parallel_config=types.SimpleNamespace(
                prefill_context_parallel_size=int(cfg.get("prefill_context_parallel_size", 1)))),
        "current_platform": current_platform,
        "logger": types.SimpleNamespace(warning_once=lambda *a, **k: None),
    }
    mod = ast.Module(body=copy.deepcopy(resolution_statements(vllm_root)), type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, ATTENTION, "exec"), ns)  # noqa: S102 - vLLM's own statements, run as written
    return {"stated": stated, "field_default": CacheConfig().swa_bounded_replay,
            "use_v2_model_runner": ns["vllm_config"].use_v2_model_runner,
            "prefill_context_parallel_size": ns["vllm_config"].parallel_config.prefill_context_parallel_size,
            "resolved_bounded_replay": ns["swa_bounded_replay"]}


def layout_of(resolved: bool, cfg: dict) -> list:
    """What the real SlidingWindowMLASpec says of each sliding-window run at that resolution."""
    import torch  # noqa: E402
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec  # noqa: E402

    out = []
    for run in cfg["kv_layers"]:
        if run["spec"] != "sliding_window_mla":
            continue
        spec = SlidingWindowMLASpec(
            sliding_window=run["sliding_window"], block_size=run.get("block_size", cfg["block_size"]),
            num_kv_heads=1, head_size=run.get("head_size", 1), dtype=torch.uint8, bounded_replay=resolved)
        out.append({"count": run["count"], "sliding_window": run["sliding_window"],
                    "bounded_replay": spec.bounded_replay, "prefix_cacheable": spec.prefix_cacheable,
                    "prefix_replay_tokens": spec.prefix_replay_tokens})
    return out


def run_one(vllm_root: str, scenario_path: str) -> dict:
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    import run_scheduler  # noqa: E402

    with open(scenario_path) as f:
        cfg = json.load(f)["config"]
    if any("bounded_replay" in r for r in cfg["kv_layers"]):
        sys.exit(f"{scenario_path}: a run states bounded_replay; this runner's scenarios state the field "
                 "(config.swa_bounded_replay) and let vLLM resolve the layers")
    swa = resolve(vllm_root, cfg)
    resolved = swa["resolved_bounded_replay"]
    swa["sliding_window_runs"] = layout_of(resolved, cfg)

    # oracle/run_scheduler.py runs the scenario it is given. Its input here is the scenario with the
    # sliding-window MLA runs stating the bounded_replay vLLM resolved, written to a temporary
    # file; nothing in the runner or in vLLM is patched. The trace's provenance is then recomputed
    # for the committed scenario, the file the golden stands for.
    with open(scenario_path) as f:
        sc = json.load(f)
    for r in sc["config"]["kv_layers"]:
        if r["spec"] == "sliding_window_mla":
            r["bounded_replay"] = resolved
    with tempfile.TemporaryDirectory() as tmp:
        resolved_path = os.path.join(tmp, os.path.basename(scenario_path))
        with open(resolved_path, "w") as f:
            json.dump(sc, f)
        trace = run_scheduler.run_one(vllm_root, resolved_path)
    trace["provenance"] = run_scheduler.provenance(vllm_root, scenario_path)
    trace["swa_layout"] = swa
    return trace


def main() -> None:
    vllm_root, scenario_path = sys.argv[1], sys.argv[2]
    if os.path.isdir(scenario_path):
        out_dir = sys.argv[3] if len(sys.argv) > 3 else scenario_path
        if "testdata/golden" in os.path.abspath(out_dir).replace(os.sep, "/") and \
                os.environ.get("BLIS_VLLM_ALLOW_GOLDEN_UPDATE") != "1":
            sys.exit("refusing to write golden traces: set BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1, and add "
                     "a golden-reviewed: <trace>: <reason> trailer for each trace the commit changes")
        for name in sorted(f for f in os.listdir(scenario_path)
                           if f.endswith(".json") and not f.endswith(".trace.json")):
            trace = run_one(vllm_root, os.path.join(scenario_path, name))
            with open(os.path.join(out_dir, name[:-5] + ".trace.json"), "w") as f:
                json.dump(trace, f, indent=1, sort_keys=True)
        return
    print(json.dumps(run_one(vllm_root, scenario_path), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
