# Original to blis-vllm.
"""Tests for the oracle's CI guards: each one must fail on the violation it exists to catch.

check_goldens.py, check_mocks.py and the connector registry rules guard the evidence every
supported row rests on (docs/testing.md), so a guard that cannot fail would let that evidence
decay unseen. Each case here plants one violation in a scratch copy and asserts the guard
reports it, and that the same scratch without the violation passes.

Usage: python3 -I oracle/test_checks.py   (stdlib only; exits non-zero on a failure)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def git(cwd, *args):
    r = run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args], cwd)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


class CheckGoldens(unittest.TestCase):
    """oracle/check_goldens.py over a scratch repository holding one golden per directory."""

    DIRS = ["core/sched/testdata/golden", "core/engine/testdata/golden", "core/engine/testdata/golden-dp",
            "core/connector/nixl/testdata/golden", "core/metrics/testdata/golden"]

    def setUp(self):
        self.repo = tempfile.mkdtemp()
        git(self.repo, "init", "-q")
        for d in self.DIRS:
            os.makedirs(os.path.join(self.repo, d))
            self.write(f"{d}/g.json", {"provenance": {"v": 1}, "steps": [{"a": 1}], "finished": {}})
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")

    def tearDown(self):
        shutil.rmtree(self.repo)

    def write(self, rel, obj):
        with open(os.path.join(self.repo, rel), "w") as f:
            json.dump(obj, f)

    def check(self):
        return run([sys.executable, "-I", os.path.join(HERE, "check_goldens.py"), self.base, "HEAD"], self.repo)

    def commit(self, msg):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", msg)

    def test_every_directory_is_guarded(self):
        for d in self.DIRS:
            for field, value in [("steps", [{"a": 2}]), ("finished", {"x": 1}), ("batches", [[1]])]:
                with self.subTest(dir=d, field=field):
                    git(self.repo, "reset", "-q", "--hard", self.base)
                    self.write(f"{d}/g.json", {"provenance": {"v": 1}, "steps": [{"a": 1}], "finished": {},
                                               field: value})
                    self.commit("change a decision")
                    r = self.check()
                    self.assertEqual(r.returncode, 1, r.stdout)
                    self.assertIn("decisions changed without a golden-reviewed: trailer", r.stdout)

    def test_a_trailer_allows_the_change(self):
        self.write("core/engine/testdata/golden-dp/g.json", {"provenance": {"v": 1}, "steps": [{"a": 9}], "finished": {}})
        self.commit("change a decision\n\ngolden-reviewed: a reviewer")
        self.assertEqual(self.check().returncode, 0)

    def test_a_provenance_refresh_needs_no_trailer(self):
        self.write("core/metrics/testdata/golden/g.json", {"provenance": {"v": 2}, "steps": [{"a": 1}], "finished": {}})
        self.commit("refresh provenance")
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_deletion_or_rename_needs_a_trailer(self):
        for how in ("delete", "rename"):
            with self.subTest(how=how):
                git(self.repo, "reset", "-q", "--hard", self.base)
                p = os.path.join(self.repo, "core/connector/nixl/testdata/golden/g.json")
                if how == "delete":
                    os.remove(p)
                else:
                    os.rename(p, p[:-5] + "2.json")
                self.commit(how)
                r = self.check()
                self.assertEqual(r.returncode, 1, r.stdout)

    def test_an_unlisted_golden_directory_fails(self):
        os.makedirs(os.path.join(self.repo, "core/newtier/testdata/golden"))
        self.write("core/newtier/testdata/golden/g.json", {"steps": []})
        self.commit("a new tier")
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("core/newtier/testdata/golden/", r.stdout)


class CheckMocks(unittest.TestCase):
    """oracle/check_mocks.py over a scratch oracle directory."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        shutil.copy(os.path.join(HERE, "mocks.json"), self.dir)
        for f in os.listdir(HERE):
            if f.startswith("run_") and f.endswith(".py"):
                shutil.copy(os.path.join(HERE, f), self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def check(self):
        return run([sys.executable, "-I", os.path.join(HERE, "check_mocks.py"), os.path.join(self.dir, "mocks.json")],
                   self.dir)

    def plant(self, runner, code):
        p = os.path.join(self.dir, runner)
        src = open(p).read()
        marker = "\ndef run_one("
        self.assertIn(marker, src)
        with open(p, "w") as f:
            f.write(src.replace(marker, "\n" + code + "\n" + marker, 1))

    def test_the_committed_runners_pass(self):
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_an_unlisted_patch_of_vllm_fails(self):
        for code in [
            "from vllm.v1.core.sched.scheduler import Scheduler\nScheduler.schedule = lambda self: None",
            "import vllm.v1.core.kv_cache_utils as u\nu.hash_block_tokens = None",
            "from vllm.v1.core.sched import scheduler\nsetattr(scheduler.Scheduler, 'schedule', None)",
            "import vllm.v1.engine.core as c\nc.DPEngineCoreProc._has_global_unfinished_reqs = staticmethod(lambda s, u: True)",
        ]:
            with self.subTest(code=code):
                self.setUp()
                self.plant("run_scheduler.py", code)
                r = self.check()
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("not listed in mocks.json", r.stdout)

    def test_a_listed_entry_no_runner_makes_fails(self):
        with open(os.path.join(self.dir, "mocks.json")) as f:
            spec = json.load(f)
        spec["mocks"].append({"symbol": "vllm.some.never_patched", "kind": "gpu", "how": "x",
                              "sites": ["run_scheduler.py:vllm_mod.never_patched"]})
        with open(os.path.join(self.dir, "mocks.json"), "w") as f:
            json.dump(spec, f)
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("which no runner makes", r.stdout)

    def test_a_mock_list_change_without_a_new_version_fails(self):
        with open(os.path.join(self.dir, "mocks.json")) as f:
            spec = json.load(f)
        spec["mocks"][0]["how"] += " (edited)"
        with open(os.path.join(self.dir, "mocks.json"), "w") as f:
            json.dump(spec, f)
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("does not name this mock list", r.stdout)

    def test_a_decision_symbol_fails(self):
        with open(os.path.join(self.dir, "mocks.json")) as f:
            spec = json.load(f)
        spec["mocks"].append({"symbol": "vllm.v1.core.sched.scheduler.Scheduler.schedule", "kind": "gpu", "how": "x",
                              "sites": []})
        with open(os.path.join(self.dir, "mocks.json"), "w") as f:
            json.dump(spec, f)
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("replaces decision code", r.stdout)


if __name__ == "__main__":
    unittest.main()
