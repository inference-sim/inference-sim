#!/usr/bin/env python3
"""Tests of shard.py: the shards partition the items, and a shard that did not run its share fails."""
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD = os.path.join(HERE, "shard.py")


def run(args, stdin=""):
    p = subprocess.run([sys.executable, "-I", SHARD, *args], input=stdin, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


class Partition(unittest.TestCase):
    def test_every_item_is_in_exactly_one_shard(self):
        rnd = random.Random(7)
        for trial in range(10):
            n = rnd.randint(8, 80)
            items = ["pkg%d Test%d" % (rnd.randint(0, 5), i) for i in range(n)]
            of = rnd.randint(1, min(6, n))
            w = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
            w.write("".join("%s\t%.2f\n" % (it, rnd.random() * 10) for it in items if rnd.random() < 0.7))
            w.close()
            seen = []
            for k in range(of):
                rc, out, err = run(["select", "-of", str(of), "-index", str(k), "-weights", w.name], "\n".join(items))
                self.assertEqual(rc, 0, err)
                seen += out.split("\n")[:-1]
            os.unlink(w.name)
            self.assertEqual(sorted(seen), sorted(items), "of=%d" % of)

    def test_a_new_item_without_a_weight_is_still_run(self):
        items = ["a", "b", "new-unweighted"]
        got = []
        for k in range(2):
            _, out, _ = run(["select", "-of", "2", "-index", str(k)], "\n".join(items))
            got += out.split()
        self.assertEqual(sorted(got), sorted(items))

    def test_the_heaviest_items_are_spread(self):
        w = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        w.write("big1\t10\nbig2\t10\nbig3\t10\ns1\t1\ns2\t1\ns3\t1\n")
        w.close()
        _, out, _ = run(["plan", "-of", "3", "-weights", w.name], "big1\nbig2\nbig3\ns1\ns2\ns3\n")
        os.unlink(w.name)
        shard_of = dict(l.split("\t")[::-1] for l in out.splitlines())
        self.assertEqual(len({shard_of["big1"], shard_of["big2"], shard_of["big3"]}), 3)

    def test_check_refuses_a_short_list_and_an_empty_shard(self):
        rc, _, err = run(["check", "-of", "2", "-min-items", "5"], "a\nb\n")
        self.assertNotEqual(rc, 0)
        self.assertIn("listing is broken", err)
        rc, _, err = run(["check", "-of", "4"], "a\nb\n")
        self.assertNotEqual(rc, 0)
        self.assertIn("has no item", err)

    def test_a_duplicate_item_is_refused(self):
        rc, _, _ = run(["select", "-of", "2", "-index", "0"], "a\na\n")
        self.assertNotEqual(rc, 0)


class Verify(unittest.TestCase):
    def results(self, events):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        f.write("\n".join(json.dumps(e) for e in events) + "\n")
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_package_mode(self):
        r = self.results([{"Action": "pass", "Package": "m/a"}, {"Action": "pass", "Package": "m/b"}])
        self.assertEqual(run(["verify", "-results", r, "-mode", "package", "-module", "m"], "a\nb\n")[0], 0)
        rc, out, _ = run(["verify", "-results", r, "-mode", "package", "-module", "m"], "a\nc\n")
        self.assertNotEqual(rc, 0)
        self.assertIn("no result: c", out)
        r = self.results([{"Action": "fail", "Package": "m/a"}])
        self.assertNotEqual(run(["verify", "-results", r, "-mode", "package", "-module", "m"], "a\n")[0], 0)

    def test_root_package_and_test_mode(self):
        r = self.results([{"Action": "pass", "Package": "m"},
                          {"Action": "pass", "Package": "m/a", "Test": "TestX"},
                          {"Action": "pass", "Package": "m/a", "Test": "TestX/sub"}])
        self.assertEqual(run(["verify", "-results", r, "-mode", "package", "-module", "m"], ".\n")[0], 0)
        self.assertEqual(run(["verify", "-results", r, "-mode", "test", "-module", "m"], "a TestX\n")[0], 0)
        rc, out, _ = run(["verify", "-results", r, "-mode", "test", "-module", "m"], "a TestX\na TestY\n")
        self.assertNotEqual(rc, 0)
        self.assertIn("no result: a TestY", out)

    def test_a_shard_with_nothing_to_verify_fails(self):
        r = self.results([])
        self.assertNotEqual(run(["verify", "-results", r, "-mode", "package"], "")[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
