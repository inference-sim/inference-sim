# Original to blis-sglang.
"""SGLang's own refusals the port quotes (oracle/refusal_quotes.json), executed through SGLang's own
code at the pinned commit, and the message each raises, recorded (SGL-6, SIM-9; issue #2349).

Each case of oracle/refusals/cases.json names the input of one refusal site and its accepted
partner (the same site, an input SGLang accepts):

  policy       SchedulePolicy._validate_and_adjust_policy(policy, tree_cache), where the tree cache
               states `disable` (managers/schedule_policy.py:318-344), called as SGLang's own function.
  tree_cache   mem_cache/registry.py create_tree_cache(ctx) with radix_eviction_policy set through
               SGLang's own override_server_args and a tree cache of the named real class, given by
               SGLang's own register_radix_cache_backend (mem_cache/registry.py:261-271).
  server_args  ServerArgs(...).resolve_once() and then .check_server_args(), the order
               entrypoints/engine.py:1165 launches in (arg_groups/validation_hook.py).

The answer is the exception SGLang raised (its type and its message) or "accepted". Nothing is
replaced beyond what oracle/mocks.json lists: the platform is the CPU's, no case reads it, and a
module of a CUDA-only package SGLang imports is stubbed as oracle/stubs.py stubs it. The model of a
server_args case is the CPU oracle's configuration-only model ($ORACLE_MODEL, as the scheduler
runners read it).

Usage: python -I oracle/run_refusals.py <sglang checkout> <cases.json> > answers.json
"""
import hashlib
import json
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def refused(e: BaseException) -> dict:
    msg = str(e)
    return {"refused": {"type": type(e).__name__, "message": msg.splitlines()[0] if msg else ""}}


def run_policy(case: dict) -> dict:
    from sglang.srt.managers.schedule_policy import SchedulePolicy

    tree_cache = types.SimpleNamespace(disable=case["tree_cache_disable"])
    try:
        policy = SchedulePolicy._validate_and_adjust_policy(None, case["policy"], tree_cache)
    except Exception as e:  # SGLang's own refusal is the answer
        return refused(e)
    return {"accepted": type(policy).__name__}


def run_tree_cache(case: dict, index: int) -> dict:
    from sglang.srt.mem_cache import registry
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.runtime_context import get_context

    classes = {"RadixCache": lambda: __import__("sglang.srt.mem_cache.radix_cache", fromlist=["RadixCache"]).RadixCache,
               "UnifiedRadixCache": lambda: __import__("sglang.srt.mem_cache.unified_radix_cache", fromlist=["UnifiedRadixCache"]).UnifiedRadixCache}
    cls = classes[case["cache_class"]]()

    def build(ctx):
        from sglang.srt.mem_cache.unified_cache.components import ComponentType

        params = CacheInitParams(disable=False, req_to_token_pool=None, token_to_kv_pool_allocator=None, page_size=1,
                                 tree_components=(ComponentType.FULL,) if case["cache_class"] == "UnifiedRadixCache" else None)
        return cls(params)

    name = "oracle-refusals-%d" % index
    registry.register_radix_cache_backend(name, build)
    ctx = registry.TreeCacheBuildContext(
        server_args=None, params=None, is_hybrid_swa=False, is_hybrid_ssm=False, enable_hierarchical_cache=False,
        disable_radix_cache=False, tp_worker=None, model_config=None, tp_size=1, tp_rank=0, tp_group=None)
    with get_context().override_server_args(model_path="dummy", radix_cache_backend=name,
                                            radix_eviction_policy=case["radix_eviction_policy"]):
        try:
            cache = registry.create_tree_cache(ctx)
        except Exception as e:  # SGLang's own refusal is the answer
            return refused(e)
    return {"accepted": type(cache).__name__}


def run_server_args(case: dict, model: str) -> dict:
    from sglang.srt.server_args import ServerArgs

    try:
        sa = ServerArgs(model_path=model, **case["server_args"])
        sa.resolve_once()
        sa.check_server_args()
    except Exception as e:  # SGLang's own refusal is the answer
        return refused(e)
    return {"accepted": "check_server_args"}


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    src, cases_path = argv
    source = os.path.realpath(os.path.join(src, "python"))
    sys.path.insert(0, source)
    sys.path.insert(0, HERE)
    with open(cases_path, "rb") as f:
        blob = f.read()
    spec = json.loads(blob)
    # SGLang reads its SGLANG_* variables while it resolves; none is an input of a case. The
    # process is run again without them, so no variable is written (oracle/mocks.json lists none).
    if any(k.startswith(("SGLANG_", "SGL_")) for k in os.environ):
        clean = {k: v for k, v in os.environ.items() if not k.startswith(("SGLANG_", "SGL_"))}
        return subprocess.run([sys.executable, "-I", os.path.abspath(__file__), src, cases_path], env=clean).returncode
    model = os.environ.get("ORACLE_MODEL", "")
    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    import stubs

    stubs.install()
    out = []
    for k, case in enumerate(spec["cases"]):
        if case["kind"] == "policy":
            ans = run_policy(case)
        elif case["kind"] == "tree_cache":
            ans = run_tree_cache(case, k)
        elif case["kind"] == "server_args":
            if not model:
                print("ORACLE_MODEL names no model for the case %s" % case["name"], file=sys.stderr)
                return 1
            ans = run_server_args(case, model)
        else:
            print("%s: unknown kind %r" % (case["name"], case["kind"]), file=sys.stderr)
            return 1
        out.append({"name": case["name"], "kind": case["kind"], "quote": case["quote"], **ans})
    doc = {"provenance": {"sglang_commit": commit, "cases_sha256": hashlib.sha256(blob).hexdigest()}, "cases": out}
    sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
