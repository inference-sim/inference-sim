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


def git_bytes(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout


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

    # --- a golden recorded at a commit a rebase has since rewritten (oracle_inputs) ---

    GONE = "ab" * 20  # a full commit id that is not in this repository

    def oracle_file(self, text):
        os.makedirs(os.path.join(self.repo, "oracle"), exist_ok=True)
        with open(os.path.join(self.repo, "oracle/run_x.py"), "w") as f:
            f.write(text)

    def inputs_of(self, rev="HEAD"):
        """oracle_inputs as a run at rev records them: sha256 of the oracle files it used."""
        import hashlib
        return {p: hashlib.sha256(git_bytes(self.repo, "show", f"{rev}:{p}")).hexdigest()
                for p in ("oracle/run_x.py", "oracle/requirements.lock")}

    def add_recorded_golden(self, commit, inputs, name="rec.json"):
        prov = self.prov(blis_vllm_commit=commit)
        if inputs is not None:
            prov["oracle_inputs"] = inputs
        self.write("core/sched/testdata/golden/" + name, {"provenance": prov, "steps": [], "finished": {}})
        self.commit("record a golden")

    def start_with_runner(self):
        self.oracle_file("v1\n")
        self.commit("the runner")
        self.base = git(self.repo, "rev-parse", "HEAD")

    def test_a_golden_whose_commit_was_rewritten_passes_when_its_oracle_files_are_unchanged(self):
        self.start_with_runner()
        inputs = self.inputs_of()
        # Go-side and golden-side commits after the run change nothing the run used.
        os.makedirs(os.path.join(self.repo, "core/x"))
        with open(os.path.join(self.repo, "core/x/x.go"), "w") as f:
            f.write("package x\n")
        self.commit("unrelated code")
        self.add_recorded_golden(self.GONE, inputs)
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)
        # The same golden without the record is what the old rule refused.
        git(self.repo, "reset", "-q", "--hard", "HEAD~1")
        self.add_recorded_golden(self.GONE, None)
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("not a commit in", r.stdout)

    def test_a_golden_whose_oracle_files_differ_at_head_fails(self):
        self.start_with_runner()
        inputs = self.inputs_of()
        self.oracle_file("v2\n")
        self.commit("the runner changes")
        self.add_recorded_golden(self.GONE, inputs)
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("oracle/run_x.py", r.stdout)
        self.assertNotIn("requirements.lock", r.stdout)

    def test_a_golden_that_records_no_files_or_a_file_head_lacks_fails(self):
        self.start_with_runner()
        for what, inputs in (("empty", {}), ("missing file", {**self.inputs_of(), "oracle/gone.py": "0" * 64})):
            with self.subTest(what=what):
                git(self.repo, "reset", "-q", "--hard", self.base)
                self.add_recorded_golden(self.GONE, inputs)
                r = self.check()
                self.assertEqual(r.returncode, 1, r.stdout)

    def test_a_tampered_record_fails(self):
        self.start_with_runner()
        good = self.inputs_of()
        flipped = dict(good)
        flipped["oracle/run_x.py"] = ("0" if good["oracle/run_x.py"][0] != "0" else "1") + good["oracle/run_x.py"][1:]
        git(self.repo, "reset", "-q", "--hard", self.base)
        self.add_recorded_golden(self.GONE, flipped)
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout)

    def test_a_record_that_its_own_commit_does_not_bear_out_fails(self):
        # The run's commit is readable here (it was rewritten away but its object is kept), and it holds
        # an older runner than the record claims: head's runner is copied into the record.
        self.start_with_runner()
        self.oracle_file("v0 as the run saw it\n")
        self.commit("the commit the run was at")
        run_commit = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "reset", "-q", "--hard", self.base)
        self.oracle_file("v1\n")  # same bytes as base: head's runner
        inputs = self.inputs_of("HEAD")
        self.add_recorded_golden(run_commit, inputs)
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("does not have", r.stdout)

    def test_a_record_does_not_excuse_a_commit_that_is_not_a_commit_id(self):
        self.start_with_runner()
        inputs = self.inputs_of()
        for commit in (self.GONE + "+dirty", "unknown", "abc"):
            with self.subTest(commit=commit):
                git(self.repo, "reset", "-q", "--hard", self.base)
                self.add_recorded_golden(commit, inputs)
                r = self.check()
                self.assertEqual(r.returncode, 1, r.stdout)

    def test_a_golden_at_an_ancestor_is_not_held_to_later_oracle_changes(self):
        # The old rule is unchanged: a commit in head's history stands, whatever the oracle did since.
        self.start_with_runner()
        inputs = self.inputs_of()
        run_commit = git(self.repo, "rev-parse", "HEAD")
        self.oracle_file("v2\n")
        self.commit("the runner changes")
        self.add_recorded_golden(run_commit, inputs)
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

    def stack_base(self, head="HEAD"):
        r = run([sys.executable, "-I", os.path.join(HERE, "check_goldens.py"), "--stack-base", head], self.repo)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def test_a_run_started_by_hand_is_judged_against_the_layer_below(self):
        # base: a golden. Layer 1 changes its decisions with its reviewer's trailer missing: a defect of layer 1,
        # judged when layer 1 was. Layer 2 adds a golden. Judged from the root, layer 2 fails for layer 1's defect.
        p = "core/sched/testdata/golden/g.json"
        self.write(p, {"provenance": {"v": 1}, "steps": [{"a": 2}], "finished": {}})
        self.commit("layer 1 changes a decision")
        layer1 = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "update-ref", "refs/remotes/origin/m2/1", layer1)
        git(self.repo, "update-ref", "refs/remotes/origin/main", self.base)
        self.add_recorded_golden(git(self.repo, "rev-parse", "HEAD"), None, name="layer2.json")
        self.assertEqual(self.check_range("").returncode, 1)  # BASE empty: from the root
        base = self.stack_base()
        self.assertEqual(base, layer1)
        r = self.check_range(base)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_the_base_of_a_hand_started_run_is_the_nearest_strict_ancestor_branch(self):
        self.write("core/sched/testdata/golden/a.json", {"steps": [1]})
        self.commit("one")
        one = git(self.repo, "rev-parse", "HEAD")
        self.write("core/sched/testdata/golden/b.json", {"steps": [2]})
        self.commit("two")
        two = git(self.repo, "rev-parse", "HEAD")
        self.write("core/sched/testdata/golden/c.json", {"steps": [3]})
        self.commit("three")
        git(self.repo, "update-ref", "refs/remotes/origin/main", self.base)
        git(self.repo, "update-ref", "refs/remotes/origin/m2/1", one)
        git(self.repo, "update-ref", "refs/remotes/origin/m2/2", two)
        git(self.repo, "update-ref", "refs/remotes/origin/scratch/other", two)  # a scratch branch is no layer
        git(self.repo, "update-ref", "refs/remotes/origin/m2/3", "HEAD")  # the same commit: not a strict ancestor
        self.assertEqual(self.stack_base(), two)
        git(self.repo, "update-ref", "-d", "refs/remotes/origin/m2/2")
        git(self.repo, "update-ref", "-d", "refs/remotes/origin/m2/1")
        self.assertEqual(self.stack_base(), self.base)  # none left: the merge-base with origin/main

    def check_range(self, base):
        return run([sys.executable, "-I", os.path.join(HERE, "check_goldens.py"), base, "HEAD"], self.repo)

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


class WallClockGauge(unittest.TestCase):
    """A gauge vLLM sets to the current time (vllm:lora_requests_info) is compared by its labels.

    Its value is when the sample was made, which differs between two runs of one scenario, so the
    golden records it as 1.0 (set) and compare_goldens_metrics.py compares the rest. The family
    stays in the golden with every label, and a changed label set still fails.
    """

    FAM = "vllm:lora_requests_info"

    @staticmethod
    def load(name):
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def series(self, key, value=1.0):
        return {"provenance": {"scenario_sha256": "s"}, "steps": 1,
                "per_step": [{key: value}], "series": {key: value}}

    def test_a_timestamp_is_recorded_as_set(self):
        m = self.load("run_metrics")
        self.assertEqual(m.sample_value(self.FAM, 1794345891.12), 1.0)
        self.assertEqual(m.sample_value(self.FAM, 0.0), 0.0)
        self.assertEqual(m.sample_value("vllm:num_requests_running", 3.0), 3.0)

    GOLDEN = os.path.join(HERE, "..", "core", "metrics", "testdata", "golden", "lora-max-loras-skip.metrics.json")

    def test_the_committed_trace_carries_the_adapter_label_sets(self):
        # vLLM's frontend fills running_lora_adapters and waiting_lora_adapters from
        # update_scheduler_stats (stats.py:696-700@5df63a72); a trace recorded without that call has
        # the empty set alone and compares no adapter label.
        with open(self.GOLDEN) as f:
            keys = [k for k in json.load(f)["series"] if k.startswith(self.FAM + "{")]
        self.assertGreaterEqual(len(keys), 3, keys)
        self.assertTrue(any("running_lora_adapters=A," in k for k in keys), keys)
        self.assertTrue(any("running_lora_adapters=B," in k for k in keys), keys)

    def test_same_labels_pass_and_changed_labels_fail(self):
        c = self.load("compare_goldens_metrics")
        with open(self.GOLDEN) as f:
            keys = sorted(k for k in json.load(f)["series"] if k.startswith(self.FAM + "{"))
        key = next(k for k in keys if "running_lora_adapters=A," in k)
        self.assertEqual(c.first_difference(self.series(key), self.series(key)), "")
        # the same sample under the other adapter's labels is a different label set
        other = key.replace("=A", "=B")
        self.assertIn(other, keys)
        self.assertNotEqual(c.first_difference(self.series(key), self.series(other)), "")
        self.assertNotEqual(c.first_difference(self.series(key, 1.0), self.series(key, 0.0)), "")

class MutantsStillApply(unittest.TestCase):
    """Each committed mutant of a Go source names its text exactly once, so a refactor that moves
    the text fails here and not by silently turning a sweep into BROKEN lines in CI."""

    ROOT = os.path.join(HERE, "..")

    def check(self, script):
        import importlib.util
        spec = importlib.util.spec_from_file_location(script, os.path.join(HERE, script + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertGreaterEqual(len(mod.MUTANTS), 2)
        for mutant in mod.MUTANTS:
            mid, path, old, new = mutant[0], mutant[-3], mutant[-2], mutant[-1]
            with open(os.path.join(self.ROOT, path)) as f:
                text = f.read()
            self.assertEqual(text.count(old), 1, f"{mid}: {path}")
            self.assertNotEqual(old, new, mid)

    def test_engineargs_mutants(self):
        self.check("mutate_engineargs")

    def test_sched_alloc_mutants(self):
        self.check("mutate_sched_alloc")


if __name__ == "__main__":
    unittest.main()
