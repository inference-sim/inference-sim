# Original to blis-sglang.
"""check_refusal_quotes.py fails on what it guards: a message reworded upstream, a message moved out
of the cited lines, and a message reworded in the port or in its test. Standard library only.

Usage: python3 -I oracle/test_check_refusal_quotes.py
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_refusal_quotes as c  # noqa: E402

SRC = """\
def f(policy, cache):
    try:
        pass
    except ValueError:
        raise ValueError(f"Unknown schedule_policy: {policy=}")
    if not isinstance(cache, X):
        raise ValueError(
            "--p requires X, but "
            f"tree_cache is {type(cache).__name__}. Drop it."
        )
"""

QUOTES = {
    "release": "test",
    "quotes": [{
        "name": "q", "file": "m/a.py", "lines": [4, 5],
        "source_literals": ["Unknown schedule_policy: policy={policy!r}"],
        "source_code": ["except ValueError:"],
        "go": {"file": "g.go", "fragments": ["Unknown schedule_policy: policy="]},
        "test": {"file": "t.go", "fragments": ["Unknown schedule_policy: policy='x'"]},
    }, {
        "name": "joined", "file": "m/a.py", "lines": [7, 9],
        "source_literals": ["--p requires X, but tree_cache is {type(cache).__name__}. Drop it."],
        "source_code": [], "go": {"file": "g.go", "fragments": []}, "test": {"file": "t.go", "fragments": []},
    }],
}


class Guards(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.eng = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.addCleanup(shutil.rmtree, self.eng)
        self.write(os.path.join(self.root, "python/sglang/srt/m/a.py"), SRC)
        self.write(os.path.join(self.eng, "g.go"), 'fmt.Errorf("Unknown schedule_policy: policy=%s")')
        self.write(os.path.join(self.eng, "t.go"), "\"Unknown schedule_policy: policy='x'\"")

    @staticmethod
    def write(path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def errs(self, quotes=QUOTES):
        return c.check(self.root, quotes, self.eng)

    def test_a_faithful_quote_passes(self):
        self.assertEqual(self.errs(), [])

    def test_a_message_reworded_upstream_fails(self):
        self.write(os.path.join(self.root, "python/sglang/srt/m/a.py"), SRC.replace("Unknown", "Bad"))
        self.assertTrue(self.errs())

    def test_a_message_moved_out_of_the_cited_lines_fails(self):
        self.write(os.path.join(self.root, "python/sglang/srt/m/a.py"), "\n\n" + SRC)
        self.assertTrue(self.errs())

    def test_a_statement_the_quote_depends_on_removed_fails(self):
        self.write(os.path.join(self.root, "python/sglang/srt/m/a.py"), SRC.replace("except ValueError:", "except KeyError:"))
        self.assertTrue(self.errs())

    def test_a_message_reworded_in_the_port_or_its_test_fails(self):
        self.write(os.path.join(self.eng, "g.go"), 'fmt.Errorf("unknown schedule_policy %s")')
        self.assertTrue(self.errs())
        self.write(os.path.join(self.eng, "g.go"), 'fmt.Errorf("Unknown schedule_policy: policy=%s")')
        self.write(os.path.join(self.eng, "t.go"), "nothing")
        self.assertTrue(self.errs())

    def test_the_list_cites_a_file_that_exists(self):
        q = copy.deepcopy(QUOTES)
        q["quotes"][0]["file"] = "m/missing.py"
        self.assertTrue(self.errs(q))

    def test_the_shipped_list_is_well_formed(self):
        with open(c.LIST, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertGreaterEqual(len(doc["quotes"]), 5)
        for q in doc["quotes"]:
            self.assertEqual(len(q["lines"]), 2)
            self.assertTrue(q["source_literals"] and q["go"]["fragments"] and q["test"]["fragments"], q["name"])


if __name__ == "__main__":
    unittest.main()
