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
        os.makedirs(os.path.join(self.repo, "oracle"))
        with open(os.path.join(self.repo, "oracle/requirements.lock"), "w") as f:
            f.write("pkg==1\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")

    def tearDown(self):
        shutil.rmtree(self.repo)

    def prov(self, **kw):
        """The provenance of a trace recorded by CI: its platform and the committed lock's hash."""
        import hashlib
        p = {"platform": "linux-x86_64",
             "requirements_lock_sha256": hashlib.sha256(b"pkg==1\n").hexdigest()}
        p.update(kw)
        return p

    def write(self, rel, obj):
        with open(os.path.join(self.repo, rel), "w") as f:
            json.dump(obj, f)

    def check(self):
        return run([sys.executable, "-I", os.path.join(HERE, "check_goldens.py"), self.base, "HEAD"], self.repo)

    def commit(self, msg):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", msg)

    def test_the_block_hash_fixture_is_guarded(self):
        os.makedirs(os.path.join(self.repo, "core/kv/testdata"), exist_ok=True)
        a, b = {"name": "a", "hashes": ["aa"]}, {"name": "b", "hashes": ["bb"]}
        self.write("core/kv/testdata/block_hashes.json", {"provenance": {"v": 1}, "none_hash": "00", "cases": [a]})
        self.commit("add the fixture")
        base = git(self.repo, "rev-parse", "HEAD")
        # Changing a case, removing one, or changing what every case rests on needs a trailer.
        for name, doc in [("change a hash", {"provenance": {"v": 1}, "none_hash": "00", "cases": [{**a, "hashes": ["ab"]}]}),
                          ("remove a case", {"provenance": {"v": 1}, "none_hash": "00", "cases": []}),
                          ("change none_hash", {"provenance": {"v": 1}, "none_hash": "01", "cases": [a, b]})]:
            with self.subTest(name):
                git(self.repo, "reset", "-q", "--hard", base)
                self.write("core/kv/testdata/block_hashes.json", doc)
                self.commit(name)
                r = run([sys.executable, "-I", os.path.join(HERE, "check_goldens.py"), base, "HEAD"], self.repo)
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("block_hashes.json decisions changed without a golden-reviewed: trailer", r.stdout)
        # Appending a case adds a golden, and a provenance-only refresh changes none: no trailer.
        for name, doc in [("append a case", {"provenance": {"v": 1}, "none_hash": "00", "cases": [a, b]}),
                          ("refresh provenance", {"provenance": {"v": 2}, "none_hash": "00", "cases": [a]})]:
            with self.subTest(name):
                git(self.repo, "reset", "-q", "--hard", base)
                self.write("core/kv/testdata/block_hashes.json", doc)
                self.commit(name)
                r = run([sys.executable, "-I", os.path.join(HERE, "check_goldens.py"), base, "HEAD"], self.repo)
                self.assertEqual(r.returncode, 0, r.stdout)

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

    GOLDEN = "core/engine/testdata/golden-dp/g.json"

    def change(self, trailer):
        """Change GOLDEN's decisions in one commit with the given message trailer (or none)."""
        self.write(self.GOLDEN, {"provenance": {"v": 1}, "steps": [{"a": 9}], "finished": {}})
        self.commit("change a decision" + (f"\n\n{trailer}" if trailer else ""))
        return self.check()

    def test_a_trailer_naming_the_trace_and_a_reason_allows_the_change(self):
        for name, trailer in [("path", f"golden-reviewed: {self.GOLDEN}: step 0 now schedules 9 tokens, as vLLM does"),
                              ("file name", "golden-reviewed: g.json: step 0 now schedules 9 tokens, as vLLM does")]:
            with self.subTest(name):
                git(self.repo, "reset", "-q", "--hard", self.base)
                r = self.change(trailer)
                self.assertEqual(r.returncode, 0, r.stdout)

    def test_no_trailer_fails(self):
        r = self.change(None)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("names the trace and gives the reason", r.stdout)

    def test_a_trailer_naming_a_different_trace_fails(self):
        r = self.change("golden-reviewed: core/engine/testdata/golden-dp/other.json: the schedule moved")
        self.assertEqual(r.returncode, 1, r.stdout)

    def test_a_trailer_with_a_name_and_no_reason_fails(self):
        for name, trailer in [("a person", "golden-reviewed: a reviewer"),
                              ("the trace, no reason", f"golden-reviewed: {self.GOLDEN}"),
                              ("the trace, empty reason", f"golden-reviewed: {self.GOLDEN}:  ")]:
            with self.subTest(name):
                git(self.repo, "reset", "-q", "--hard", self.base)
                r = self.change(trailer)
                self.assertEqual(r.returncode, 1, r.stdout)

    def test_each_changed_trace_needs_its_own_trailer(self):
        a, b = "core/engine/testdata/golden-dp/a.json", "core/engine/testdata/golden-dp/b.json"
        for p in (a, b):
            self.write(p, {"provenance": {"v": 1}, "steps": [{"a": 1}], "finished": {}})
        self.commit("add two goldens")
        base = git(self.repo, "rev-parse", "HEAD")
        for p in (a, b):
            self.write(p, {"provenance": {"v": 1}, "steps": [{"a": 2}], "finished": {}})
        one = f"golden-reviewed: {a}: the step moved"
        self.commit(f"change both\n\n{one}")
        r = run([sys.executable, "-I", os.path.join(HERE, "check_goldens.py"), base, "HEAD"], self.repo)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("b.json", r.stdout)
        self.assertNotIn("a.json decisions", r.stdout)
        git(self.repo, "commit", "-q", "--amend", "-m", f"change both\n\n{one}\ngolden-reviewed: {b}: the step moved")
        r = run([sys.executable, "-I", os.path.join(HERE, "check_goldens.py"), base, "HEAD"], self.repo)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_provenance_refresh_needs_no_trailer(self):
        self.write("core/metrics/testdata/golden/g.json", {"provenance": {"v": 2}, "steps": [{"a": 1}], "finished": {}})
        self.commit("refresh provenance")
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_deletion_or_rename_needs_a_trailer(self):
        path = "core/connector/nixl/testdata/golden/g.json"
        for how in ("delete", "rename"):
            for trailered in (False, True):
                with self.subTest(how=how, trailered=trailered):
                    git(self.repo, "reset", "-q", "--hard", self.base)
                    p = os.path.join(self.repo, path)
                    if how == "delete":
                        os.remove(p)
                    else:
                        os.rename(p, p[:-5] + "2.json")
                    self.commit(how + (f"\n\ngolden-reviewed: {path}: the scenario was dropped" if trailered else ""))
                    r = self.check()
                    self.assertEqual(r.returncode, 0 if trailered else 1, r.stdout)

    def test_a_new_golden_recorded_elsewhere_than_on_ci_fails(self):
        for what, prov in (("platform", self.prov(platform="darwin-arm64")),
                           ("lock", self.prov(requirements_lock_sha256="0" * 64)),
                           ("no platform", {"requirements_lock_sha256": self.prov()["requirements_lock_sha256"]})):
            with self.subTest(what=what):
                git(self.repo, "reset", "-q", "--hard", self.base)
                self.write("core/sched/testdata/golden/new.json", {"provenance": prov, "steps": [], "finished": {}})
                self.commit("add a golden")
                r = self.check()
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("new.json", r.stdout)

    def test_a_new_golden_recorded_on_ci_passes_and_an_old_one_is_not_held_to_it(self):
        self.write("core/sched/testdata/golden/new.json", {"provenance": self.prov(), "steps": [], "finished": {}})
        self.commit("add a golden")
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_golden_naming_a_commit_outside_the_history_fails(self):
        self.write("core/sched/testdata/golden/side.json", {"steps": []})
        self.commit("a commit that will not be in the history")
        lost = git(self.repo, "rev-parse", "HEAD")
        for commit in (lost, lost + "+dirty", "unknown"):
            with self.subTest(commit=commit):
                git(self.repo, "reset", "-q", "--hard", self.base)
                self.write("core/sched/testdata/golden/g.json",
                           {"provenance": {"blis_vllm_commit": commit}, "steps": [{"a": 1}], "finished": {}})
                self.commit("refresh provenance")
                r = self.check()
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("not a commit in HEAD's history", r.stdout)

    def test_a_golden_naming_a_commit_in_the_history_passes(self):
        self.write("core/sched/testdata/golden/g.json",
                   {"provenance": {"blis_vllm_commit": self.base}, "steps": [{"a": 1}], "finished": {}})
        self.commit("refresh provenance")
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

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


class CheckLockHashed(unittest.TestCase):
    """oracle/check_lock.py --hashed: the hashed install file pins exactly what the lock pins."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def check(self, lock, hashed):
        for name, body in [("lock", lock), ("hashed", hashed)]:
            with open(os.path.join(self.dir, name), "w") as f:
                f.write(body)
        return run([sys.executable, "-I", os.path.join(HERE, "check_lock.py"), "--hashed",
                    os.path.join(self.dir, "lock"), os.path.join(self.dir, "hashed")], self.dir)

    LOCK = "# pins\ndocstring_parser==0.18.0\nvllm @ https://x/vllm.whl#sha256=ab\n"
    HASHED = ("# generated\ndocstring-parser==0.18.0 \\\n    --hash=sha256:01\n"
              "vllm @ https://x/vllm.whl#sha256=ab \\\n    --hash=sha256:ab\n")

    def test_the_committed_files_agree(self):
        r = run([sys.executable, "-I", os.path.join(HERE, "check_lock.py"), "--hashed",
                 os.path.join(HERE, "requirements.lock"), os.path.join(HERE, "requirements-hashed.txt")], HERE)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_the_same_pins_pass(self):
        r = self.check(self.LOCK, self.HASHED)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_different_version_fails(self):
        r = self.check(self.LOCK, self.HASHED.replace("0.18.0", "0.17.0"))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("docstring-parser==0.18.0 is in the lock but not hashed", r.stdout)

    def test_a_package_missing_from_the_hashed_file_fails(self):
        r = self.check(self.LOCK + "torch==2.13.0+cpu\n", self.HASHED)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("torch==2.13.0+cpu is in the lock but not hashed", r.stdout)


if __name__ == "__main__":
    unittest.main()
