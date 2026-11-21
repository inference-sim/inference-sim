# Original to blis-sglang.
"""Tests for gen_classes.py over a small synthetic SGLang tree.

Run: python3 -I fields/test_classes.py
"""

import gzip
import json
import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_classes  # noqa: E402

SRT = "python/sglang/srt"
FILES = {
    SRT + "/managers/scheduler.py": '''
        import os
        from sglang.srt.environ import envs

        def f(server_args):
            if server_args.page_size == 1:
                pass
            ps = server_args.page_size
            if ps > 64:
                pass
            if server_args.enable_mixed_chunk:
                pass
            if server_args.schedule_policy in ("lpm", "dfs-weight"):
                pass
            if envs.SGLANG_RATIO.get() < 0.5:
                pass
            other = something.page_size
            if other == 999:
                pass
    ''',
    "python/sglang/kernels/__init__.py": "",
    "python/sglang/utils.py": "",
    "python/sglang/launch_server.py": "",
    "python/sglang/cli/serve.py": "",
}

DECLS = {
    "server_args": {
        "page_size": {"type": "Optional[int]", "default": None},
        "enable_mixed_chunk": {"type": "bool", "default": False},
        "schedule_policy": {"type": "str", "default": "fcfs", "choices": ["lpm", "fcfs", "dfs-weight"]},
    },
    "env": {"SGLANG_RATIO": {"type": "EnvFloat", "default": 0.7}},
}


def classes(src, name, source="config", evidence=None, repo="."):
    decl = DECLS["server_args"].get(name) if source == "config" else DECLS["env"].get(name)
    return gen_classes.classes_for(src, repo, gen_classes.Field(name, source, decl), evidence or {})


class ClassesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = self.tmp.name
        for rel, text in FILES.items():
            p = os.path.join(self.src, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(textwrap.dedent(text))

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_numeric_field_is_cut_at_every_comparison_it_takes_part_in(self):
        vc, basis = classes(self.src, "page_size")
        self.assertEqual(vc["type"], "numeric")
        # page_size == 1 cuts at 1 and 2; ps > 64 (an alias the function assigned) cuts at 65; the
        # unrelated something.page_size == 999 cuts nothing.
        self.assertEqual([t["value"] for t in vc["thresholds"]], [1, 2, 65])
        self.assertEqual(len(basis["classes"]), 4)

    def test_a_boolean_field_has_both_classes_backed_by_its_truth_test(self):
        vc, basis = classes(self.src, "enable_mixed_chunk")
        self.assertEqual(vc["type"], "boolean")
        self.assertTrue(all(c["compared_at"] == [SRT + "/managers/scheduler.py:11"] for c in basis["classes"]))

    def test_an_enumerated_field_takes_its_values_from_the_declaration(self):
        vc, basis = classes(self.src, "schedule_policy")
        self.assertEqual(vc["values"], ["lpm", "fcfs", "dfs-weight"])
        backed = [c["values"][0] for c in basis["classes"] if c["compared_at"]]
        self.assertEqual(backed, ["lpm", "dfs-weight"])

    def test_an_environment_variable_read_through_envs(self):
        vc, _ = classes(self.src, "SGLANG_RATIO", source="env")
        self.assertEqual(vc["thresholds"], [{"value": 0.5, "closed": "above"}])

    def test_covered_comes_from_the_evidence_traces(self):
        with tempfile.TemporaryDirectory() as repo:
            os.makedirs(os.path.join(repo, "oracle/scenarios"))
            with open(os.path.join(repo, "oracle/scenarios/s.json"), "w") as f:
                json.dump({"engine": {"page_size": 128}}, f)
            vc, _ = classes(self.src, "page_size", evidence={"traces": ["oracle/scenarios/s.json"]}, repo=repo)
            self.assertEqual(vc["covered_classes"], [3], "128 lies in the class at and above 65")

    def test_covered_comes_from_a_trace_committed_only_as_gzip(self):
        # A golden of 16000 bytes or more is committed as <name>.gz; the evidence names the plain path.
        with tempfile.TemporaryDirectory() as repo:
            os.makedirs(os.path.join(repo, "oracle/scenarios"))
            with gzip.open(os.path.join(repo, "oracle/scenarios/s.json.gz"), "wt") as f:
                json.dump({"engine": {"page_size": 128}}, f)
            vc, _ = classes(self.src, "page_size", evidence={"traces": ["oracle/scenarios/s.json"]}, repo=repo)
            self.assertEqual(vc["covered_classes"], [3], "the gzip is read as the trace it holds")

    def test_known_bad_a_modelled_field_without_classes(self):
        with tempfile.TemporaryDirectory() as d:
            fj = os.path.join(d, "fields.json")
            with open(fj, "w") as f:
                json.dump({"fields": [{"name": "page_size", "source": "config", "status": "modelled",
                                       "evidence": {"traces": ["x"]}}]}, f)
            probs = gen_classes.check_document({"fields": {}}, fj)
            self.assertEqual(probs, ["page_size: a modelled or proven-inert field has no value classes"])
            _, basis = classes(self.src, "page_size")
            basis["classes"][1]["compared_at"] = ["nowhere.py:1"]
            probs = gen_classes.check_document({"fields": {"page_size": {"basis": basis}}}, fj)
            self.assertEqual(probs, ["page_size: class 1 cites nowhere.py:1, which is not a comparison of the field"])

class SetLiteralOrder(unittest.TestCase):
    """A set literal's values come out in one order whatever the process's hash seed (a set of strings
    iterates in an order that depends on it)."""

    def test_the_values_of_a_set_literal_are_sorted_under_any_seed(self):
        import ast
        import subprocess

        code = ("import ast, sys; sys.path.insert(0, %r); import gen_classes; "
                "print(gen_classes.const_value(ast.parse(\"{'fp8', 'modelopt_fp8', 'bf16', 'awq'}\", mode='eval').body))"
                % os.path.dirname(os.path.abspath(__file__)))
        outs = set()
        for seed in ("0", "1", "2", "3", "4", "5"):
            outs.add(subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONHASHSEED=seed),
                                    capture_output=True, text=True, check=True).stdout)
        self.assertEqual(outs, {"(True, ['awq', 'bf16', 'fp8', 'modelopt_fp8'])\n"})


if __name__ == "__main__":
    unittest.main()
