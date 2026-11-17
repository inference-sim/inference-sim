#!/usr/bin/env python3
"""The oracle-fuzz action records where each corpus it restored or generated came from.

The test runs the action's own "record where this run's fuzz traces came from" step on a
scratch tree holding one trace per corpus, and reads the record it writes, so it checks what a
run records and not how the step is written.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ACTION = os.path.join(HERE, "..", "actions", "oracle-fuzz", "action.yml")
DIRS = {"t2": ["fuzz"], "mamba": ["fuzz-mamba"], "t3": ["fuzz-t3", "fuzz-t3-dp"],
        "connector": ["fuzz-connector"], "offloading": ["fuzz-offloading"], "multi": ["fuzz-multi"]}


def step():
    """The record step's environment variable names (from its env block) and its Python."""
    text = open(ACTION).read()
    i = text.index("name: record where this run's fuzz traces came from")
    block = text[i:]
    code = re.search(r"python3 -I - <<'PY'\n(.*?)\n\s*PY\n", block, re.S).group(1)
    code = "\n".join(l[8:] if l.startswith(" " * 8) else l for l in code.split("\n"))
    env = re.findall(r"^\s+([A-Z0-9_]+): \$\{\{ (steps\.[^}]*|inputs\.corpora) \}\}", block.split("run: |")[0], re.M)
    return code, dict(env)


def record(corpora):
    code, env = step()
    with tempfile.TemporaryDirectory() as root:
        for c in corpora:
            for d in DIRS[c]:
                os.makedirs(os.path.join(root, "oracle", "out", d))
                with open(os.path.join(root, "oracle", "out", d, "s1.trace.json"), "w") as f:
                    json.dump({"provenance": {"blis_vllm_commit": "c0ffee"}}, f)
        e = {"PATH": os.environ["PATH"], "GITHUB_SHA": "run-sha", "CORPORA": " ".join(corpora)}
        for name, src in env.items():
            if name.endswith("_KEY"):
                e[name] = "key-" + name[:-4].lower()
            elif name.endswith("_HIT"):
                e[name] = "true"
        p = subprocess.run([sys.executable, "-I", "-c", code], cwd=root, env=e, capture_output=True, text=True)
        if p.returncode:
            raise AssertionError(p.stderr)
        with open(os.path.join(root, "oracle", "out", "fuzz-cache.json")) as f:
            return json.load(f)


class FuzzProvenance(unittest.TestCase):
    def test_every_corpus_a_run_restored_is_recorded_with_its_key_and_hit(self):
        out = record(list(DIRS))
        self.assertEqual(out["run_blis_vllm_commit"], "run-sha")
        for field, key in [("tier2", "t2"), ("mamba", "mamba"), ("tier3_and_dp", "t3"), ("connector", "connector"),
                           ("offloading", "offloading"), ("multi", "multi")]:
            self.assertIn(field, out, "the %s corpus ran but is not in the record" % key)
            self.assertEqual(out[field]["key"], "key-" + key)
            self.assertTrue(out[field]["cache_hit"])
            self.assertEqual(out[field]["generated_by_blis_vllm_commit"], ["c0ffee"])

    def test_the_mamba_job_records_its_corpus_not_only_the_run_commit(self):
        out = record(["mamba"])
        self.assertEqual(sorted(out), ["mamba", "run_blis_vllm_commit"])
        self.assertEqual(out["mamba"]["traces"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
