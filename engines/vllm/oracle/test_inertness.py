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

    def config_spec(self, observe, compare):
        # max_num_batched_tokens 4096 instead of 2048 changes the encoder cache size and the compile ranges
        # the resolved config holds, and, unless it is listed, nothing the observed values name.
        return {"field": "max_num_batched_tokens", "release": "x", "readers": [],
                "probes": [{"label": "2048", "value": 2048}, {"label": "4096", "value": 4096}],
                "runs": [{"id": "config", "kind": "config", "apply": {"engine_args": "max_num_batched_tokens"},
                          "observe": observe, "compare": compare, "echo": "scheduler_config.max_num_batched_tokens"}]}

    @staticmethod
    def run_problems(r):
        return [p for p in r["problems"] if "differ" in p or "resolves differently" in p]

    def test_a_config_run_sees_a_resolved_value_it_was_not_told_to_watch(self):
        watched = ["scheduler_config.max_num_seqs"]
        blind = self.h.evaluate(ROOT, self.config_spec(watched, []))
        self.assertEqual(self.run_problems(blind), [], "the control needs a value outside the watched list that changes")
        r = self.h.evaluate(ROOT, self.config_spec(watched, ["scheduler_config", "compilation_config"]))
        self.assertEqual(r["verdict"], "not_inert")
        self.assertTrue([p for p in r["problems"] if "config run config: " in p and "scheduler_config.encoder_cache_size" in p
                         and "resolves differently" in p], r["problems"])
        self.assertTrue([p for p in r["problems"] if "decisions of run config differ" in p], r["problems"])

    def test_a_config_run_sees_a_watched_value_change(self):
        r = self.h.evaluate(ROOT, self.config_spec(["scheduler_config.max_num_seqs", "scheduler_config.encoder_cache_size"], []))
        self.assertEqual(r["verdict"], "not_inert")
        self.assertTrue([p for p in r["problems"] if "decisions of run config differ" in p], r["problems"])

    def test_an_output_processor_run_sees_what_it_hands_back_to_the_engine(self):
        # The probe is the step at which the script aborts request a; the processor hands the abort back to
        # the engine core, which is engine-facing, so the run must report it.
        script = copy.deepcopy([r for r in proof("stream_interval")["runs"] if r["id"] == "outputproc"][0]["script"])
        spec = {"field": "stream_interval", "release": "x", "readers": [],
                "probes": [{"label": "abort at 3", "value": 3}, {"label": "abort at 7", "value": 7}],
                "runs": [{"id": "outputproc", "kind": "outputproc", "apply": {"abort_step": ["a"]}, "script": script}]}
        r = self.h.evaluate(ROOT, spec)
        self.assertEqual(r["verdict"], "not_inert")
        self.assertTrue([p for p in r["problems"] if "decisions of run outputproc differ" in p], r["problems"])

    def test_a_disposition_the_harness_does_not_check_is_refused(self):
        spec = copy.deepcopy(proof("stream_interval"))
        spec["runs"] = []
        for rd in spec["readers"]:
            if rd["at"] == "vllm/v1/engine/async_llm.py:174":
                rd["disposition"] = {"kind": "gpu_replaced"}
        r = self.h.evaluate(ROOT, spec)
        self.assertTrue([p for p in r["problems"] if "async_llm.py:174: unknown disposition 'gpu_replaced'" in p], r["problems"])

    def test_a_file_that_names_the_field_and_the_proof_does_not_list_fails(self):
        spec = copy.deepcopy(proof("stream_interval"))
        spec["runs"] = []
        spec["named_in"]["vllm"].remove("vllm/sampling_params.py")
        spec["named_in"]["rust"].remove("rust/src/cmd/src/cli.rs")
        spec["named_in"]["rust"].append("rust/src/cmd/src/not_a_file.rs")
        r = self.h.evaluate(ROOT, spec)
        for want in ("vllm/sampling_params.py names the field and the proof does not list it",
                     "rust/src/cmd/src/cli.rs names the field and the proof does not list it",
                     "rust/src/cmd/src/not_a_file.rs is listed in named_in and does not name the field"):
            self.assertTrue([p for p in r["problems"] if want in p], r["problems"])

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


    def test_a_run_disposition_names_the_run_that_executed_the_line(self):
        # arg_utils.py:2641 is executed by the config run (EngineArgs.create_engine_config) and by no other
        # run. A proof that names the outputproc run for it must fail, though the config run came first:
        # coverage is per run, not the sum of the runs so far.
        spec = copy.deepcopy(proof("stream_interval"))
        spec["runs"] = [r for r in spec["runs"] if r["id"] in ("config", "outputproc")]
        for rd in spec["readers"]:
            if rd["at"] == "vllm/engine/arg_utils.py:2641":
                rd["disposition"]["run"] = "outputproc"
        r = self.h.evaluate(ROOT, spec)
        self.assertTrue([p for p in r["problems"]
                         if "arg_utils.py:2641 is marked run in outputproc and that run did not execute it" in p], r["problems"])
        got = [rd for rd in r["readers"] if rd["at"] == "vllm/engine/arg_utils.py:2641"][0]
        self.assertEqual(got["ran_in"], ["config"])


class CoverageMatching(unittest.TestCase):
    """Coverage names a file by its path inside vLLM and matches the executed file by that path's suffix:
    CI installs vLLM from a wheel, so the code that runs is under site-packages, not under the checkout."""

    SRC = "def read(x):\n    y = x + 1\n    return y\n"

    def run_module(self, path, cov):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.SRC)
        spec = importlib.util.spec_from_file_location("covered_module", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with cov:
            mod.read(1)

    def test_a_module_run_from_site_packages_counts_for_the_checkouts_path(self):
        h = load()
        with tempfile.TemporaryDirectory() as tmp:
            checkout = os.path.join(tmp, "checkout")  # the path the proof's vllm_root names; nothing runs from it
            installed = os.path.join(tmp, "venv", "lib", "python3.12", "site-packages", "vllm", "v1", "engine", "tiny.py")
            cov = h.Coverage(checkout, ["vllm/v1/engine/tiny.py"])
            self.run_module(installed, cov)
            self.assertTrue(cov.ran("vllm/v1/engine/tiny.py:2", 2), cov.hit)

    def test_a_module_run_from_the_checkout_counts(self):
        h = load()
        with tempfile.TemporaryDirectory() as tmp:
            checkout = os.path.join(tmp, "checkout")
            cov = h.Coverage(checkout, ["vllm/v1/engine/tiny.py"])
            self.run_module(os.path.join(checkout, "vllm", "v1", "engine", "tiny.py"), cov)
            self.assertTrue(cov.ran("vllm/v1/engine/tiny.py:2", 2), cov.hit)

    def test_a_file_whose_directory_only_ends_in_the_same_letters_does_not_count(self):
        h = load()
        with tempfile.TemporaryDirectory() as tmp:
            other = os.path.join(tmp, "site-packages", "notvllm", "v1", "engine", "tiny.py")
            cov = h.Coverage(os.path.join(tmp, "checkout"), ["vllm/v1/engine/tiny.py"])
            self.run_module(other, cov)
            self.assertFalse(cov.ran("vllm/v1/engine/tiny.py:2", 2), cov.hit)

    def test_reset_forgets_the_earlier_run(self):
        h = load()
        with tempfile.TemporaryDirectory() as tmp:
            cov = h.Coverage(tmp, ["vllm/v1/engine/tiny.py"])
            self.run_module(os.path.join(tmp, "site-packages", "vllm", "v1", "engine", "tiny.py"), cov)
            self.assertTrue(cov.ran("vllm/v1/engine/tiny.py:2", 2))
            cov.reset()
            self.assertFalse(cov.ran("vllm/v1/engine/tiny.py:2", 2))


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
