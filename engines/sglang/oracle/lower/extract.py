# Original to blis-sglang.
"""The lowering answers: SGLang's own decode CUDA-graph functions, run on CPU for each accepted case
of oracle/serverargs/cases.json, after SGLang resolves and publishes that case's ServerArgs.

For each case it records what the decode step lowers to:
  alignment          get_cuda_graph_batch_size_alignment() (python/sglang/srt/utils/common.py:4168)
  gather             require_mlp_tp_gather(), require_attn_tp_gather(), require_mlp_sync()
                     (utils/common.py:4067-4165)
  moe                the MoE all-to-all and runner backends initialize_moe_config() installs
                     (layers/moe/utils.py:479-495)
  attention          the (prefill, decode) attention backends the model runner builds
                     (attention_backends(), model_runner_components/attention_backend_setup.py:203)
  capture            for each request-pool size in POOL_SIZES and each captured request width
                     (1, and the case's draft tokens when it speculates), the capture batch sizes
                     get_batch_sizes_to_capture returns (model_executor/runner/
                     base_cuda_graph_runner.py:65-110), and for each raw batch size up to the
                     largest, the bucket BaseCudaGraphRunner._pad_to_bucket pads it to (:166-180)

Every replacement is in oracle/lower/replacements.json (oracle/lower/check_replacements.py): the CUDA
platform restore and stubs of oracle/run_serverargs.py, and a model runner that holds only the
request pool size and server_args, which is all get_batch_sizes_to_capture reads of it.

Usage: python -I oracle/lower/extract.py <sglang checkout> <cases.json> <models dir> [<extra cases.json>] > lower.json
The extra cases (oracle/lower/cases.json) use the chips and models of <cases.json>.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
POOL_SIZES = (8, 48, 100, 256, 1000, 4096)


class PoolOnlyRunner:
    """The model runner get_batch_sizes_to_capture reads: its request pool size and server_args."""

    class _Pool:
        def __init__(self, size):
            self.size = size

    def __init__(self, server_args, pool_size):
        self.server_args = server_args
        self.req_to_token_pool = self._Pool(pool_size)


def lower_case(case, chip, models):
    import run_serverargs

    model_path = os.path.join(models, case["model"])
    run_serverargs.restore_cuda(chip)
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs(model_path=model_path, **case.get("args", {}))
    try:
        sa.resolve_once()
    except Exception:
        return {"name": case["name"], "refused": True}
    from sglang.srt import runtime_context
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend, initialize_moe_config
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import BaseCudaGraphRunner, get_batch_sizes_to_capture
    from sglang.srt.utils import common

    runtime_context.publish(sa, role="scheduler")
    initialize_moe_config()
    from sglang.srt.runtime_context import attention_backends

    prefill, decode = attention_backends()
    out = {
        "name": case["name"],
        # The case's inputs, so the Go test resolves the same ServerArgs from one file.
        "args": case.get("args", {}),
        "chip": {k: chip[k] for k in ("capability", "nvidia_smi_memory_total_mib")},
        "model_facts": run_serverargs.model_facts(sa),
        "alignment": common.get_cuda_graph_batch_size_alignment(),
        "gather": {"mlp_tp": common.require_mlp_tp_gather(), "attn_tp": common.require_attn_tp_gather(),
                   "mlp_sync": common.require_mlp_sync()},
        "moe": {"a2a_backend": get_moe_a2a_backend().value, "runner_backend": str(getattr(get_moe_runner_backend(), "value", get_moe_runner_backend()))},
        "attention": {"prefill": prefill, "decode": decode},
        "capture": [],
    }
    widths = [1]
    k = sa.speculative_num_draft_tokens
    if sa.speculative_algorithm and k and k > 1:
        widths.append(k)
    for w in widths:
        for size in POOL_SIZES:
            bs, _ = get_batch_sizes_to_capture(PoolOnlyRunner(sa, size), captured_req_width=w)
            pads = [BaseCudaGraphRunner._pad_to_bucket(raw, bs) for raw in range(1, bs[-1] + 1)]
            out["capture"].append({"pool_size": size, "req_width": w, "capture_bs": bs, "padded": pads})
    return out


def main(argv):
    if len(argv) not in (3, 4):
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    src, cases_path, models = argv[:3]
    sys.path.insert(0, os.path.realpath(os.path.join(src, "python")))
    sys.path.insert(0, ORACLE)
    for k in sorted(os.environ):
        if k.startswith(("SGLANG_", "SGL_")):
            del os.environ[k]
    with open(cases_path) as f:
        spec = json.load(f)
    if len(argv) == 4:
        with open(argv[3]) as f:
            spec["cases"] = spec["cases"] + json.load(f)["cases"]
    results = []
    for case in spec["cases"]:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                import run_serverargs
                import stubs

                stubs.STUBBED = tuple(stubs.STUBBED) + run_serverargs.CUDA_STUBBED
                stubs.install()
                res = lower_case(case, spec["chips"][case["chip"]], models)
            except BaseException as e:
                import traceback

                res = {"name": case["name"], "harness_error": "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc())}
            with os.fdopen(w, "w") as fw:
                json.dump(res, fw)
            os._exit(0)
        os.close(w)
        with os.fdopen(r) as fr:
            data = fr.read()
        os.waitpid(pid, 0)
        res = json.loads(data) if data else {"name": case["name"], "harness_error": "no answer"}
        if "harness_error" in res:
            print("%s: %s" % (case["name"], res["harness_error"]), file=sys.stderr)
            return 1
        if not res.get("refused"):
            results.append(res)
    sys.stdout.write(json.dumps({"cases": results}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
