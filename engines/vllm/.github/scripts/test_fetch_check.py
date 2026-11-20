#!/usr/bin/env python3
"""fetch.py --check: the recorded server-log hashes are compared with the logs now served.

The API is replaced by a table of served logs; the check runs for real on a recorded file.
"""
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
FETCH = os.path.join(HERE, "..", "..", "core", "kernelmodel", "testdata", "recorded_block_sizes", "fetch.py")


def load():
    spec = importlib.util.spec_from_file_location("fetch_under_test", FETCH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Check(unittest.TestCase):
    def run_check(self, recorded, served):
        """Exit status and output of check() when the API serves `served` (row -> log text, absent = no log)."""
        m = load()

        def get(url):
            row = int(url.rsplit("=", 1)[1])
            if row not in served:
                raise SystemExit("not served")
            return gzip.compress(json.dumps({"serverLog": served[row]}).encode())

        m.get = get
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "recorded.json")
            with open(path, "w") as f:
                json.dump({"runs": [{"row": r, "log_sha256": sha(t)} for r, t in recorded.items()]}, f)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = m.check(path)
        return status, out.getvalue()

    def test_logs_that_still_match_pass(self):
        status, out = self.run_check({1: "alpha", 2: "beta"}, {1: "alpha", 2: "beta"})
        self.assertEqual(status, 0, out)

    def test_a_changed_log_fails_and_names_its_row_only(self):
        status, out = self.run_check({1: "alpha", 2: "beta"}, {1: "alpha", 2: "beta, edited"})
        self.assertEqual(status, 1)
        self.assertIn("row 2:", out)
        self.assertNotIn("row 1:", out)

    def test_a_withdrawn_log_fails_and_says_it_is_gone(self):
        status, out = self.run_check({1: "alpha", 2: "beta"}, {1: "alpha"})
        self.assertEqual(status, 1)
        self.assertIn("row 2: the log is gone", out)


if __name__ == "__main__":
    unittest.main(verbosity=1)
