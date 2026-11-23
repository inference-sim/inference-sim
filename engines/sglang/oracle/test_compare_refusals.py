# Original to blis-sglang.
"""compare_refusals.py fails on what it guards: a changed message, a refusal that became an acceptance,
a missing or foreign case, a different SGLang commit or case file. Standard library only (SGLang is not
imported).

Usage: python3 -I oracle/test_compare_refusals.py
"""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_refusals as c  # noqa: E402

DOC = {
    "provenance": {"sglang_commit": "a" * 40, "cases_sha256": "b" * 64},
    "cases": [
        {"name": "x", "kind": "policy", "quote": "q", "refused": {"type": "ValueError", "message": "no"}},
        {"name": "y", "kind": "policy", "quote": "q", "accepted": "CacheAwarePolicy"},
    ],
}


def run(a, b) -> int:
    with tempfile.TemporaryDirectory() as d:
        pa, pb = os.path.join(d, "a.json"), os.path.join(d, "b.json")
        for p, v in ((pa, a), (pb, b)):
            with open(p, "w") as f:
                json.dump(v, f)
        with contextlib.redirect_stdout(io.StringIO()):
            return c.main(pa, pb)


class CompareRefusals(unittest.TestCase):
    def test_equal_answers_pass(self):
        self.assertEqual(run(DOC, copy.deepcopy(DOC)), 0)

    def test_changed_message_fails(self):
        b = copy.deepcopy(DOC)
        b["cases"][0]["refused"]["message"] = "reworded"
        self.assertEqual(run(DOC, b), 1)

    def test_refusal_turned_acceptance_fails(self):
        b = copy.deepcopy(DOC)
        b["cases"][0] = {"name": "x", "kind": "policy", "quote": "q", "accepted": "CacheAwarePolicy"}
        self.assertEqual(run(DOC, b), 1)

    def test_missing_and_foreign_cases_fail(self):
        b = copy.deepcopy(DOC)
        del b["cases"][1]
        self.assertEqual(run(DOC, b), 1)
        b = copy.deepcopy(DOC)
        b["cases"].append({"name": "z", "kind": "policy", "quote": "q", "accepted": "p"})
        self.assertEqual(run(DOC, b), 1)

    def test_other_commit_or_case_file_fails(self):
        for key in ("sglang_commit", "cases_sha256"):
            b = copy.deepcopy(DOC)
            b["provenance"][key] = "c" * 40
            self.assertEqual(run(DOC, b), 1, key)


if __name__ == "__main__":
    unittest.main()
