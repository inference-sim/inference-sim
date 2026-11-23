# Original to blis-sglang.
"""run_serverargs.py's shards and compare_serverargs.py: the shards partition the cases, each shard is
compared with the committed cases at its positions only, and a changed answer, a missing case or a
foreign case fails. Standard library only (SGLang is not imported).

Usage: python3 -I oracle/test_compare_serverargs.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_serverargs as c  # noqa: E402
import run_serverargs as r  # noqa: E402

PROV = {"sglang_commit": "a" * 40, "cases_sha256": "b" * 64, "models": {}}


def case(k):
    return {"name": "c%d" % k, "resolved": {"x": k}}


def doc(ks, shard=None):
    d = {"provenance": PROV, "cases": [case(k) for k in ks]}
    if shard:
        d["shard"] = {"index": shard[0], "of": shard[1]}
    return d


def compare(committed, fresh):
    with tempfile.TemporaryDirectory() as t:
        paths = []
        for name, d in (("a.json", committed), ("b.json", fresh)):
            p = os.path.join(t, name)
            with open(p, "w") as f:
                json.dump(d, f)
            paths.append(p)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = c.main(*paths)
        return rc, out.getvalue()


class Shards(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(r.parse_shard("2/4"), (2, 4))
        for bad in ("4/4", "-1/4", "a/b", "1", "0/0"):
            with self.assertRaises(SystemExit):
                r.parse_shard(bad)

    def test_every_shard_matches_its_slice_and_the_slices_partition_the_cases(self):
        n, total = 4, 10
        seen = []
        for i in range(n):
            ks = [k for k in range(total) if k % n == i]
            seen += ks
            rc, _ = compare(doc(range(total)), doc(ks, (i, n)))
            self.assertEqual(rc, 0)
        self.assertEqual(sorted(seen), list(range(total)))

    def test_a_whole_answer_still_compares_whole(self):
        self.assertEqual(compare(doc(range(5)), doc(range(5)))[0], 0)
        self.assertEqual(compare(doc(range(5)), doc(range(4)))[0], 1)

    def test_a_shard_fails_on_a_changed_case(self):
        fresh = doc([1, 5, 9], (1, 4))
        fresh["cases"][1]["resolved"]["x"] = -1
        rc, out = compare(doc(range(10)), fresh)
        self.assertEqual(rc, 1)
        self.assertIn("c5", out)

    def test_a_shard_fails_on_a_missing_or_foreign_case(self):
        self.assertEqual(compare(doc(range(10)), doc([1, 5], (1, 4)))[0], 1)
        self.assertEqual(compare(doc(range(10)), doc([1, 5, 9, 2], (1, 4)))[0], 1)
        self.assertEqual(compare(doc(range(10)), doc([2, 6], (1, 4)))[0], 1)

    def test_a_shard_with_no_committed_case_fails(self):
        self.assertEqual(compare(doc(range(2)), doc([], (3, 4)))[0], 1)

    def test_provenance_is_still_compared(self):
        fresh = doc([1], (1, 4))
        fresh["provenance"] = dict(PROV, sglang_commit="c" * 40)
        self.assertEqual(compare(doc(range(10)), fresh)[0], 1)


if __name__ == "__main__":
    unittest.main()
