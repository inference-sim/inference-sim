# Original to blis-vllm.
"""Hold the field list's `valid_when` clauses to the upstream's own resolution (design 11.2: a stand-in is exact
only where the upstream resolves the stated value and the unset field to the same behaviour).

Run: python3 -I fields/test_stand_in.py                              (the reference evaluator against the vectors)
     VLLM_SOURCE=<vLLM v0.32.0 checkout> python3 -I fields/test_stand_in.py   (and the vectors against the upstream)
     VLLM_SOURCE=... python3 -I fields/test_stand_in.py --write-vectors       (rewrite fields/stand_in_vectors.json)

fields/stand_in_vectors.json is the truth table: for each field, value and fact, whether the stand-in is exact.
Its rows come from the upstream: the pinned functions are compiled from the vLLM source and run with their
globals stubbed (node count, device capability, environment). Two evaluators are held to it: this directory's
Python reference (stand_in.py) here, and blis-substrate's Go fields.Exact in the root package's
TestStandInVectorsAgreeWithTheGoEvaluator. UCX_TLS has no upstream function; its rows are the transport
semantics of the UCX FAQ and are marked "owner": "ucx-faq".
"""
import ast
import itertools
import json
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stand_in  # noqa: E402

SOURCE = os.environ.get("VLLM_SOURCE")
VECTORS = os.path.join(HERE, "stand_in_vectors.json")
# NCCL's own documentation (no vLLM function resolves them): cuMem is auto-enabled by default since 2.19 and 1 overrides
# the autodetection with the same choice; NVLS defaults to 2 and 1 differs only in failing when allocation fails.
CUMEM = [("1", True), ("0", False), ("", False), ("2", False)]
NVLS = [("1", True), ("2", True), ("0", False), ("", False)]
# NCCL_MNNVL_ENABLE (NCCL user guide): 2 is the default (use multi-node NVLink where the system has it, else the network);
# 1 differs only in failing communicator initialization when MNNVL is unsupported, which a completed run did not do.
MNNVL = [("1", True), ("2", True), ("0", False), ("", False)]
# The env vars an earlier release read (removed in v0.24.0, 03878d1c2): the rows are v0.33.0's own selection for the flags
# that replaced them, fused_moe/oracle/mxfp4.py at 4ba787ec (map_mxfp4_backend :352-392, _filter_by_activation :614-631,
# _get_requested_backends :634-647) and the device each kernel supports (TrtLlmMxfp4Experts: capability family 10,
# experts/trtllm_mxfp4_moe.py:121-123; MarlinExperts: capability 7.5 and up, experts/marlin_moe.py:592-594).
# "true" is how the dashboard renders a stated 1: both variables are read as bool(int(value)) (envs.py at 03878d1c2^), so a run
# that completed stated an integer, and a recorded boolean is its 1.
MXFP8 = [("true", "10.0", True), ("false", "10.0", False), ("1", "10.0", True), ("1", "10.3", True), ("1", "9.0", False), ("1", "12.0", False), ("0", "10.0", False), ("", "10.0", False)]
MARLIN_MOE = [("true", "9.0", True), ("false", "9.0", False), ("1", "9.0", True), ("1", "7.5", True), ("1", "10.0", True), ("1", "7.0", False), ("0", "9.0", False), ("", "9.0", False)]
TLS = [("cuda_copy,rc", True), ("cuda_copy,cuda_ipc,rc", True), ("all", True), ("cuda_copy,cuda_ipc,tcp", False),
       ("tcp", False), ("^rc", False), ("sm,self", False)]
# The transport and runtime variables of the Dynamo P/D recipes. Each row is what the cited source says; a value the
# list does not hold is "not shown exact", not "shown different", so only the stated value and a clearly different one
# are listed. Sources: Dynamo's from_config_value (lib/runtime/src/pipeline/network.rs:132-152 at ai-dynamo/dynamo
# 807aaaa0), NCCL's ncclTopoCheckP2p (src/graph/paths.cc:383-395 at v2.32.6-1 ad2d192c), PyTorch's
# ProcessGroupNCCL.cpp:1709 and cpp_extension.py:2577-2590 at v2.11.0, UCX's option tables (src/ucp/core/ucp_context.c
# and src/ucs/config/global_opts.c at v1.19.0 e4636149) and vLLM v0.33.0's own setdefault and reads.
DYN_CODEC = [("msgpack", True), ("json", False), ("", False)]
P2P_LEVEL = [("NVL", "10.0", True), ("NVL", "10.3", True), ("NVL", "9.0", False), ("NVL", "12.0", False),
             ("PXB", "10.0", False), ("SYS", "10.0", False), ("", "10.0", False)]
ARCH_LIST = [("10.0", "10.0", True), ("10.0", "10.3", False), ("10.0", "9.0", False), ("9.0", "10.0", False), ("", "10.0", False)]
INERT_ONE = {  # field: (the stated value that is exact, a value that is not shown exact)
    "NCCL_WATCHDOG_TIMEOUT": ("1800", "60"), "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": ("1800", "60"),
    "UCX_MAX_RMA_RAILS": ("1", "4"), "UCX_MAX_RNDV_RAILS": ("1", "4"), "UCX_RNDV_SCHEME": ("put_zcopy", "get_zcopy"),
    "UCX_MEMTYPE_CACHE": ("n", "y"), "UCX_MEMTYPE_REG_WHOLE": ("n", "y"), "TILELANG_CLEANUP_TEMP_FILES": ("1", "0"),
    "VLLM_SERVER_DEV_MODE": ("1", "0"), "VLLM_LOG_STATS_INTERVAL": ("1", "0"),
}


def function_from(path, name, env):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(env)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, "exec"), ns)
    return ns[name]


class Log:
    def debug_once(self, *a, **k): pass
    info_once = warning_once = error_once = debug_once


def fi_backend(value, nodes, cap):
    path = os.path.join(SOURCE, "vllm/distributed/device_communicators/flashinfer_all_reduce.py")
    envs = types.SimpleNamespace(VLLM_FLASHINFER_ALLREDUCE_BACKEND=value)
    plat = types.SimpleNamespace(has_device_capability=lambda c: stand_in._cap(cap)[0] * 10 >= c)
    f = function_from(path, "_resolve_fi_ar_backend",
                      {"ProcessGroup": object, "envs": envs, "_node_count": lambda g: nodes, "current_platform": plat, "logger": Log()})
    try:
        return f(object())
    except Exception:
        return "error"


def rcache_final():
    path = os.path.join(SOURCE, "vllm/distributed/nixl_utils.py")
    env = {}
    function_from(path, "_maybe_set_ucx_rcache_limit",
                  {"os": types.SimpleNamespace(environ=env), "sys": types.SimpleNamespace(modules={}), "logger": Log()})()
    return env["UCX_RCACHE_MAX_UNRELEASED"]


def vectors_from_upstream():
    rows = []
    for value, nodes, cap in itertools.product(["auto", "mnnvl", "trtllm", "nccl"], [1, 2, 4], ["8.0", "9.0", "9.10", "10.0", "10.3"]):
        got, auto = fi_backend(value, nodes, cap), fi_backend("auto", nodes, cap)
        row = {"field": "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "value": value, "tp_group_nodes": nodes,
               "compute_capability": cap, "owner": "_resolve_fi_ar_backend", "exact": got == auto}
        if nodes == 1 and stand_in._cap(cap)[0] >= 10:
            # Blackwell nodes (HGX B200/B300, the GB200/GB300 NVL72 trays) have NVSwitch multicast, so auto's trtllm
            # fallback is never taken and auto and a stated mnnvl run the same backend; the stand-in is exact
            # under that assumption, which the point's estimation text states. The fallback flag is the only
            # difference the upstream function shows.
            row["owner"] = "_resolve_fi_ar_backend+multicast"
            row["exact"] = got[0] == auto[0]
        rows.append(row)
    for value in ["1024", "0", "inf", "2048"]:
        rows.append({"field": "UCX_RCACHE_MAX_UNRELEASED", "value": value, "owner": "_maybe_set_ucx_rcache_limit",
                     "exact": value == rcache_final()})
    for value, exact in TLS:
        rows.append({"field": "UCX_TLS", "value": value, "owner": "ucx-faq", "exact": exact})
    for field, values in (("NCCL_CUMEM_ENABLE", CUMEM), ("NCCL_NVLS_ENABLE", NVLS), ("NCCL_MNNVL_ENABLE", MNNVL)):
        for value, exact in values:
            rows.append({"field": field, "value": value, "owner": "nccl-user-guide", "exact": exact})
    for field, cases in (("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", MXFP8), ("VLLM_MXFP4_USE_MARLIN", MARLIN_MOE)):
        for value, cap, exact in cases:
            rows.append({"field": field, "value": value, "compute_capability": cap, "owner": "mxfp4-oracle-v0.33.0", "exact": exact})
    for value, exact in DYN_CODEC:
        rows.append({"field": "DYN_REQUEST_PLANE_CODEC", "value": value, "owner": "dynamo-request-plane-codec", "exact": exact})
    for value, cap, exact in P2P_LEVEL:
        rows.append({"field": "NCCL_P2P_LEVEL", "value": value, "compute_capability": cap, "owner": "nccl-paths-p2p-level", "exact": exact})
    for value, cap, exact in ARCH_LIST:
        rows.append({"field": "TORCH_CUDA_ARCH_LIST", "value": value, "compute_capability": cap, "owner": "torch-cpp-extension-arch-flags", "exact": exact})
    for field, (stated, other) in INERT_ONE.items():
        rows.append({"field": field, "value": stated, "owner": "cited-source", "exact": True})
        rows.append({"field": field, "value": other, "owner": "cited-source", "exact": False})
    return rows


def committed():
    with open(VECTORS, encoding="utf-8") as f:
        return json.load(f)


class Vectors(unittest.TestCase):
    def test_the_reference_evaluator_agrees_with_every_vector(self):
        for r in committed():
            got = stand_in.exact(r["field"], r["value"], tp_group_nodes=r.get("tp_group_nodes"),
                                 compute_capability=r.get("compute_capability"))
            self.assertEqual(got, r["exact"], r)

    def test_a_fact_the_caller_did_not_give_makes_a_clause_false(self):
        self.assertFalse(stand_in.exact("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl"))
        self.assertTrue(stand_in.exact("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "auto"))

    def test_a_stated_mnnvl_on_one_node_is_not_the_default(self):
        # auto on one node is (mnnvl, may fall back to trtllm); a stated mnnvl is (mnnvl, no fallback). They differ on a
        # node without NVSwitch multicast, which the caller cannot state, so the stand-in is not exact there.
        for cap in ("8.0", "9.0"):
            self.assertFalse(stand_in.exact("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl", tp_group_nodes=1, compute_capability=cap))
        # On Blackwell nodes, which all have NVSwitch multicast, the fallback is never taken: exact under that assumption.
        for cap in ("10.0", "10.3"):
            self.assertTrue(stand_in.exact("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl", tp_group_nodes=1, compute_capability=cap))
            self.assertFalse(stand_in.exact("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "trtllm", tp_group_nodes=1, compute_capability=cap))
        self.assertTrue(stand_in.exact("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl", tp_group_nodes=2, compute_capability="10.0"))

    def test_an_unlisted_field_has_no_exact_stand_in(self):
        self.assertFalse(stand_in.exact("SOME_OTHER_VARIABLE", "1"))


@unittest.skipUnless(SOURCE, "VLLM_SOURCE names a vLLM v0.32.0 checkout")
class Upstream(unittest.TestCase):
    def test_the_vectors_are_what_the_upstream_resolves(self):
        self.assertEqual(committed(), vectors_from_upstream())


if __name__ == "__main__":
    if "--write-vectors" in sys.argv:
        assert SOURCE, "VLLM_SOURCE is required"
        with open(VECTORS, "w", encoding="utf-8") as f:
            json.dump(vectors_from_upstream(), f, indent=1)
            f.write("\n")
        sys.exit(0)
    unittest.main()
