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
TLS = [("cuda_copy,rc", True), ("cuda_copy,cuda_ipc,rc", True), ("all", True), ("cuda_copy,cuda_ipc,tcp", False),
       ("tcp", False), ("^rc", False), ("sm,self", False)]


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
        return f(object())[0]
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
        rows.append({"field": "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "value": value, "tp_group_nodes": nodes,
                     "compute_capability": cap, "owner": "_resolve_fi_ar_backend",
                     "exact": fi_backend(value, nodes, cap) == fi_backend("auto", nodes, cap)})
    for value in ["1024", "0", "inf", "2048"]:
        rows.append({"field": "UCX_RCACHE_MAX_UNRELEASED", "value": value, "owner": "_maybe_set_ucx_rcache_limit",
                     "exact": value == rcache_final()})
    for value, exact in TLS:
        rows.append({"field": "UCX_TLS", "value": value, "owner": "ucx-faq", "exact": exact})
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
