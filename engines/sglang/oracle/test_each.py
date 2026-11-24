# Original to blis-sglang.
"""Tests for each.sh, which the oracle-run steps run their per-file commands with: every file's
command runs once, no more than ORACLE_JOBS at once and that many do, a failing command fails the
whole while the others still run, no file runs nothing, and a bad call is a usage error.

Usage: python3 -I oracle/test_each.py   (stdlib, bash and xargs only)
"""
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EACH = os.path.join(HERE, "each.sh")

# Marks its file as started, records how many marks it sees, then finishes.
WORK = 'touch "$D/run-$(basename "$1")"; n=$(ls "$D" | grep -c "^run-"); echo "$n" > "$D/seen-$(basename "$1")"; sleep 0.3; rm "$D/run-$(basename "$1")"; echo "$1" >> "$D/done"'


def each(args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(["bash", EACH] + args, capture_output=True, text=True, env=e)


class EachTest(unittest.TestCase):
    def test_every_file_once_at_most_and_at_least_jobs_at_once(self):
        with tempfile.TemporaryDirectory() as d:
            files = ["f%d" % i for i in range(7)]
            r = each([WORK] + files, {"D": d, "ORACLE_JOBS": "3"})
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(d, "done")) as f:
                self.assertEqual(sorted(f.read().split()), sorted(files))
            seen = []
            for name in files:
                with open(os.path.join(d, "seen-" + name)) as f:
                    seen.append(int(f.read()))
            self.assertLessEqual(max(seen), 3)
            self.assertEqual(max(seen), 3, "three were allowed and fewer ever ran together")

    def test_a_failing_command_fails_the_whole_and_the_others_still_run(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = 'if [ "$1" = bad ]; then exit 7; fi; echo "$1" >> "$D/done"'
            r = each([cmd, "a", "bad", "b", "c"], {"D": d, "ORACLE_JOBS": "2"})
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("a command failed", r.stderr)
            with open(os.path.join(d, "done")) as f:
                self.assertEqual(sorted(f.read().split()), ["a", "b", "c"])

    def test_no_file_runs_nothing(self):
        r = each(["exit 1"], {"ORACLE_JOBS": "2"})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_a_bad_call_is_a_usage_error(self):
        self.assertEqual(each([]).returncode, 2)
        for bad in ("0", "x", "-1"):
            self.assertEqual(each(["true", "f"], {"ORACLE_JOBS": bad}).returncode, 2, bad)

    def test_without_oracle_jobs_it_uses_forkpools_count(self):
        env = {k: v for k, v in os.environ.items() if k != "ORACLE_JOBS"}
        r = subprocess.run(["bash", EACH, "true", "f"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
