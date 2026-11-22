# Original to blis-sglang.
"""extract.py's shards and compare_lower.py: each shard is compared with the committed answers of the
cases it ran, a refused case (no answer) still counts as run, and a changed answer, a missing or
foreign case, a shard not run, or a shard that ran the wrong cases fails. Standard library only
(SGLang is not imported).

Usage: python3 -I oracle/lower/test_compare_lower.py
"""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_lower as c  # noqa: E402

NAMES = ["a", "b", "c", "d", "e", "f", "g"]
REFUSED = {"c"}


def answer(name):
    return {"name": name, "alignment": 1, "capture": []}


COMMITTED = {"cases": [answer(n) for n in NAMES if n not in REFUSED]}


def shard(i, n, names=NAMES, refused=REFUSED):
    ran = [x for k, x in enumerate(names) if k % n == i]
    return {"cases": [answer(x) for x in ran if x not in refused], "shard": {"index": i, "of": n}, "ran": ran}


class Shard(unittest.TestCase):
    def test_each_shard_matches_its_cases(self):
        for n in (1, 2, 3, 4):
            for i in range(n):
                self.assertEqual(c.check_shard(COMMITTED, shard(i, n)), [], (i, n))

    def test_changed_answer_fails(self):
        s = shard(0, 2)
        s["cases"][0]["alignment"] = 2
        self.assertEqual(len(c.check_shard(COMMITTED, s)), 1)

    def test_missing_answer_fails(self):
        s = shard(0, 2)
        del s["cases"][1]
        self.assertTrue(c.check_shard(COMMITTED, s))

    def test_new_answer_for_a_committed_refusal_fails(self):
        s = shard(2, 3, refused=set())
        self.assertTrue(c.check_shard(COMMITTED, s))

    def test_answer_outside_the_cases_run_fails(self):
        s = shard(0, 2)
        s["cases"].append(answer("b"))
        self.assertTrue(c.check_shard(COMMITTED, s))

    def test_not_a_shard_fails(self):
        self.assertTrue(c.check_shard(COMMITTED, {"cases": COMMITTED["cases"]}))


class Complete(unittest.TestCase):
    def shards(self, n):
        return [shard(i, n) for i in range(n)]

    def test_every_shard_passes(self):
        for n in (1, 3, 4):
            self.assertEqual(c.check_complete(COMMITTED, NAMES, self.shards(n)), [], n)

    def test_missing_shard_fails(self):
        self.assertTrue(c.check_complete(COMMITTED, NAMES, self.shards(3)[:2]))

    def test_repeated_shard_fails(self):
        s = self.shards(3)
        self.assertTrue(c.check_complete(COMMITTED, NAMES, [s[0], s[0], s[2]]))

    def test_mixed_n_fails(self):
        self.assertTrue(c.check_complete(COMMITTED, NAMES, [shard(0, 2), shard(1, 3)]))

    def test_shard_that_skipped_a_case_fails(self):
        s = self.shards(3)
        s[1]["ran"] = s[1]["ran"][:-1]
        self.assertTrue(c.check_complete(COMMITTED, NAMES, s))

    def test_shard_that_took_a_foreign_case_fails(self):
        s = self.shards(3)
        s[1]["ran"] = s[1]["ran"] + ["a"]
        self.assertTrue(c.check_complete(COMMITTED, NAMES, s))

    def test_committed_answer_for_no_case_fails(self):
        committed = {"cases": COMMITTED["cases"] + [answer("zz")]}
        self.assertTrue(c.check_complete(committed, NAMES, self.shards(3)))

    def test_cases_added_without_a_shard_run_fail(self):
        self.assertTrue(c.check_complete(COMMITTED, NAMES + ["h"], self.shards(3)))


class Cli(unittest.TestCase):
    def test_usage_is_an_error(self):
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "compare_lower.py")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
