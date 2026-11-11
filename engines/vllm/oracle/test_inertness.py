# Original to blis-vllm.
"""Negative controls for oracle/inertness.py: the harness must find a field that is NOT inert, a read it
was not told about, a stale read, and a variable the source does name.

Run: VLLM_SOURCE=<pinned vLLM checkout> python3 -I oracle/test_inertness.py   (ORACLE_MODEL etc. as oracle/README.md)
"""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.environ.get("VLLM_SOURCE")


def load():
    spec = importlib.util.spec_from_file_location("inertness", os.path.join(HERE, "inertness.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def proof(name):
    with open(os.path.join(HERE, "inertness", name + ".json"), encoding="utf-8") as f:
        return json.load(f)


@unittest.skipUnless(ROOT, "VLLM_SOURCE is unset")
class Controls(unittest.TestCase):
    def setUp(self):
        self.h = load()

    def test_a_field_that_changes_the_schedule_is_not_inert(self):
        # enable_chunked_prefill off makes a waiting request whose prompt does not fit the budget wait
        # (vllm/v1/core/sched/scheduler.py:1161): the scheduler's decisions differ across the probes.
        scenario = {"name": "chunked-prefill-control", "config": {
            "max_num_seqs": 4, "max_num_batched_tokens": 64, "block_size": 4, "num_blocks": 120, "max_model_len": 64,
            "long_prefill_token_threshold": 0, "enable_prefix_caching": False, "scheduling_policy": "fcfs"},
            "requests": [{"id": "c0", "arrival_step": 0, "prompt": list(range(100, 140)), "output": [7, 8, 9]},
                         {"id": "c1", "arrival_step": 0, "prompt": list(range(200, 240)), "output": [7, 8]},
                         {"id": "c2", "arrival_step": 0, "prompt": list(range(300, 330)), "output": [7, 8]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "chunked-prefill-control.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(scenario, f)
            spec = {"field": "enable_chunked_prefill", "release": "x", "readers": [],
                    "probes": [{"label": "on", "value": True}, {"label": "off", "value": False}],
                    "runs": [{"id": "sched", "kind": "scheduler", "apply": {"scheduler_config": "enable_chunked_prefill"},
                              "scenarios": [path]}]}
            r = self.h.evaluate(ROOT, spec)
        self.assertEqual(r["verdict"], "not_inert")
        self.assertTrue([p for p in r["problems"] if "decisions of run sched differ" in p], r["problems"])

    def test_a_variable_the_source_names_is_not_unread(self):
        spec = {"kind": "unread", "release": "x", "names": ["VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD"]}
        r = self.h.evaluate_unread(ROOT, spec)
        self.assertEqual(r["verdict"], "not_inert")
        self.assertIn("vllm/envs.py", r["problems"][0])

    def test_a_read_with_no_disposition_fails(self):
        spec = copy.deepcopy(proof("stream_interval"))
        spec["runs"] = []
        spec["readers"] = spec["readers"][1:]
        r = self.h.evaluate(ROOT, spec)
        self.assertTrue([p for p in r["problems"] if "has no disposition: vllm/engine/arg_utils.py:2641" in p], r["problems"])

    def test_a_listed_read_that_is_not_in_the_source_fails(self):
        spec = copy.deepcopy(proof("stream_interval"))
        spec["runs"] = []
        spec["readers"].append({"at": "vllm/engine/arg_utils.py:1", "disposition": {"kind": "run", "run": "config"}})
        r = self.h.evaluate(ROOT, spec)
        self.assertTrue([p for p in r["problems"] if "is not in the source: vllm/engine/arg_utils.py:1" in p], r["problems"])

    def test_a_run_disposition_no_run_executed_fails(self):
        spec = copy.deepcopy(proof("stream_interval"))
        spec["runs"] = []
        r = self.h.evaluate(ROOT, spec)
        self.assertTrue([p for p in r["problems"] if "did not execute it" in p], r["problems"])

    def test_a_pass_through_that_is_not_one_fails(self):
        spec = copy.deepcopy(proof("stream_interval"))
        spec["runs"] = []
        for rd in spec["readers"]:
            if rd["at"] == "vllm/v1/engine/async_llm.py:174":
                rd["disposition"]["callee"] = "SomethingElse"
        r = self.h.evaluate(ROOT, spec)
        self.assertTrue([p for p in r["problems"] if "async_llm.py:174 is not a keyword" in p], r["problems"])

    def test_a_reviewed_read_the_source_no_longer_holds_fails(self):
        spec = copy.deepcopy(proof("stream_interval"))
        spec["runs"] = []
        for rd in spec["readers"]:
            if rd.get("reviewed") and rd["at"].endswith(":324"):
                rd["expr"] = "if self.stream_interval > 2:"
        r = self.h.evaluate(ROOT, spec)
        self.assertTrue([p for p in r["problems"] if "no longer holds" in p], r["problems"])


class CommittedProofs(unittest.TestCase):
    @unittest.skipUnless(ROOT, "VLLM_SOURCE is unset")
    def test_every_committed_proof_is_what_the_harness_computes_now(self):
        h = load()
        d = os.path.join(HERE, "inertness")
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                with self.subTest(proof=fn):
                    path = os.path.join(d, fn)
                    self.assertEqual(h.main([ROOT, path, "--check"]), 0)


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
