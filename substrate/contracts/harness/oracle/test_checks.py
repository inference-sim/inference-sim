"""Tests for the oracle checks of contracts/harness: each one must fail on the violation it exists
to catch (X18).

check_goldens.py, check_mocks.py and check_lock.py guard the evidence every engine repository's
oracle rests on, so a guard that cannot fail would let that evidence decay unseen. Each case here
plants one violation in a scratch copy and asserts the guard reports it, and that the same scratch
without the violation passes. The engine repositories run the same checks over their own files.

Usage: python3 -I contracts/harness/oracle/test_checks.py   (stdlib and git only; exits non-zero on a failure)
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK_GOLDENS = os.path.join(HERE, "check_goldens.py")
CHECK_MOCKS = os.path.join(HERE, "check_mocks.py")
CHECK_LOCK = os.path.join(HERE, "check_lock.py")


def run(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def git(cwd, *args):
    r = run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
             "-c", "init.defaultBranch=main", *args], cwd)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


def git_bytes(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout


def write_config(directory, **over):
    """A scratch oracle/harness.json in directory; its path."""
    cfg = {"commit_field": "engine_commit", "ci_platform": "linux-x86_64",
           "golden_dirs": ["core/x/testdata/golden/"], "golden_files": [], "harness_files": ["run_*.py"],
           "decision_prefixes": ["upstream.sched."], "upstream_names": ["event_loop_normal"]}
    cfg.update(over)
    path = os.path.join(directory, "harness.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


class CheckGoldens(unittest.TestCase):
    """oracle/check_goldens.py over a scratch repository holding one golden per directory, with a
    configuration that names those directories and one golden file."""

    DIRS = ["core/sched/testdata/golden", "core/engine/testdata/golden", "core/engine/testdata/golden-dp",
            "core/connector/x/testdata/golden", "core/metrics/testdata/golden"]

    def setUp(self):
        self.repo = tempfile.mkdtemp()
        self.config = write_config(tempfile.mkdtemp(), golden_dirs=[d + "/" for d in self.DIRS],
                                   golden_files=["core/kv/testdata/block_hashes.json"])
        git(self.repo, "init", "-q")
        # Every golden written below names this commit, which is in every later head's history,
        # unless a test names another (write()).
        git(self.repo, "commit", "-q", "--allow-empty", "-m", "root")
        self.root = git(self.repo, "rev-parse", "HEAD")
        for d in self.DIRS:
            os.makedirs(os.path.join(self.repo, d))
            self.write(f"{d}/g.json", {"provenance": {"v": 1}, "steps": [{"a": 1}], "finished": {}})
        os.makedirs(os.path.join(self.repo, "core/kv/testdata"))
        self.write("core/kv/testdata/block_hashes.json", {"provenance": {"v": 1}, "none_hash": "00", "cases": []})
        os.makedirs(os.path.join(self.repo, "oracle"))
        with open(os.path.join(self.repo, "oracle/requirements.lock"), "w") as f:
            f.write("pkg==1\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")

    def tearDown(self):
        shutil.rmtree(self.repo)
        shutil.rmtree(os.path.dirname(self.config))

    def goldens(self, *args):
        return run([sys.executable, "-I", CHECK_GOLDENS, "--config", self.config, *args], self.repo)

    def prov(self, **kw):
        """The provenance of a trace recorded by CI: its platform and the committed lock's hash."""
        import hashlib
        p = {"platform": "linux-x86_64",
             "requirements_lock_sha256": hashlib.sha256(b"pkg==1\n").hexdigest()}
        p.update(kw)
        return p

    def write(self, rel, obj, raw=False):
        """Write a golden; unless raw, its provenance names self.root when it names no commit."""
        if not raw and isinstance(obj, dict):
            obj = {**obj, "provenance": {"engine_commit": self.root, **obj.get("provenance", {})}}
        with open(os.path.join(self.repo, rel), "w") as f:
            json.dump(obj, f)

    def check(self):
        return self.goldens(self.base, "HEAD")

    def test_a_golden_that_records_no_commit_fails(self):
        for name, doc in [("no field", {"provenance": {"v": 1}, "steps": []}), ("no provenance", {"steps": []})]:
            with self.subTest(name):
                git(self.repo, "reset", "-q", "--hard", self.base)
                self.write("core/sched/testdata/golden/n.json", doc, raw=True)
                self.commit("a golden that names no commit")
                r = self.check()
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("records no engine_commit (commit_field)", r.stdout)

    def test_without_config_the_configuration_is_heads(self):
        # The working tree's harness.json guards nothing; head's guards every directory.
        shutil.copy(self.config, os.path.join(self.repo, "oracle/harness.json"))
        self.commit("commit the configuration")
        self.base = git(self.repo, "rev-parse", "HEAD")
        with open(self.config) as f:
            cfg = json.load(f)
        with open(os.path.join(self.repo, "oracle/harness.json"), "w") as f:
            json.dump({**cfg, "golden_dirs": ["elsewhere/testdata/golden/"]}, f)
        self.write(self.GOLDEN, {"provenance": {"v": 1}, "steps": [{"a": 9}], "finished": {}})
        git(self.repo, "add", self.GOLDEN)
        git(self.repo, "commit", "-qm", "change a decision")
        r = run([sys.executable, "-I", CHECK_GOLDENS, self.base, "HEAD"], self.repo)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("decisions changed without a golden-reviewed: trailer", r.stdout)

    def test_an_unknown_option_is_refused(self):
        r = self.goldens("--confg", "x", self.base, "HEAD")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown option --confg", r.stderr)

    def test_a_named_golden_file_head_lacks_fails(self):
        os.remove(os.path.join(self.repo, "core/kv/testdata/block_hashes.json"))
        self.commit("drop the golden file")
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("core/kv/testdata/block_hashes.json: a golden file oracle/harness.json names", r.stdout)

    def test_a_configuration_with_a_missing_unknown_empty_or_mistyped_key_fails(self):
        with open(self.config) as f:
            cfg = json.load(f)
        for name, broken, key in config_breakages(cfg):
            with self.subTest(name):
                with open(self.config, "w") as f:
                    json.dump(broken, f)
                r = self.check()
                self.assertNotEqual(r.returncode, 0, r.stdout)
                self.assertIn(key, r.stderr)

    def commit(self, msg):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", msg)

    def test_the_block_hash_fixture_is_guarded(self):
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
                r = self.goldens(base, "HEAD")
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("block_hashes.json decisions changed without a golden-reviewed: trailer", r.stdout)
        # Appending a case adds a golden, and a provenance-only refresh changes none: no trailer.
        for name, doc in [("append a case", {"provenance": {"v": 1}, "none_hash": "00", "cases": [a, b]}),
                          ("refresh provenance", {"provenance": {"v": 2}, "none_hash": "00", "cases": [a]})]:
            with self.subTest(name):
                git(self.repo, "reset", "-q", "--hard", base)
                self.write("core/kv/testdata/block_hashes.json", doc)
                self.commit(name)
                r = self.goldens(base, "HEAD")
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
        for name, trailer in [("path", f"golden-reviewed: {self.GOLDEN}: step 0 now schedules 9 tokens, as the upstream does"),
                              ("file name", "golden-reviewed: g.json: step 0 now schedules 9 tokens, as the upstream does")]:
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
        r = self.goldens(base, "HEAD")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("b.json", r.stdout)
        self.assertNotIn("a.json decisions", r.stdout)
        git(self.repo, "commit", "-q", "--amend", "-m", f"change both\n\n{one}\ngolden-reviewed: {b}: the step moved")
        r = self.goldens(base, "HEAD")
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_provenance_refresh_needs_no_trailer(self):
        self.write("core/metrics/testdata/golden/g.json", {"provenance": {"v": 2}, "steps": [{"a": 1}], "finished": {}})
        self.commit("refresh provenance")
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_deletion_or_rename_needs_a_trailer(self):
        path = "core/connector/x/testdata/golden/g.json"
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
                           {"provenance": {"engine_commit": commit}, "steps": [{"a": 1}], "finished": {}})
                self.commit("refresh provenance")
                r = self.check()
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("not a commit in HEAD's history", r.stdout)

    def test_a_golden_naming_a_commit_in_the_history_passes(self):
        self.write("core/sched/testdata/golden/g.json",
                   {"provenance": {"engine_commit": self.base}, "steps": [{"a": 1}], "finished": {}})
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
        prov = self.prov(engine_commit=commit)
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
        r = self.goldens("--stack-base", head)
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
        return self.goldens(base, "HEAD")

    def test_an_unlisted_golden_directory_fails(self):
        os.makedirs(os.path.join(self.repo, "core/newtier/testdata/golden"))
        self.write("core/newtier/testdata/golden/g.json", {"steps": []})
        self.commit("a new tier")
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("core/newtier/testdata/golden/", r.stdout)


def config_breakages(cfg):
    """Each way a configuration can be wrong, with the key the refusal must name: every key
    missing, an unknown key, every list key empty (golden_files may be) or holding a non-string,
    every key the wrong type, an empty string, and a golden directory without its slash."""
    out = []
    for k in sorted(cfg):
        out.append((f"{k} missing", {x: v for x, v in cfg.items() if x != k}, k))
        if isinstance(cfg[k], list):
            if k != "golden_files":
                out.append((f"{k} empty", {**cfg, k: []}, k))
            out.append((f"{k} holds a number", {**cfg, k: [1]}, k))
            out.append((f"{k} a string", {**cfg, k: "x"}, k))
        else:
            out.append((f"{k} empty", {**cfg, k: ""}, k))
            out.append((f"{k} a list", {**cfg, k: [cfg[k]]}, k))
    out.append(("an unknown key", {**cfg, "golden_dir": []}, "golden_dir"))
    out.append(("a golden dir without its slash", {**cfg, "golden_dirs": ["core/x/testdata/golden"]}, "golden_dirs"))
    return out


def digest(mocks):
    canonical = json.dumps(mocks, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


RUNNER = '''import upstream.platform as p
import upstream.worker as w

p.current = "cuda"
w.Worker.forward = lambda self, batch: None


def run_one(scenario):
    return scenario
'''

MOCKS = [
    {"symbol": "upstream.platform.current", "kind": "platform", "how": "the CUDA platform restored",
     "sites": ["run_x.py:p.current"]},
    {"symbol": "upstream.worker.Worker.forward", "kind": "gpu", "how": "tokens from the scenario",
     "sites": ["run_x.py:w.Worker.forward"]},
]


class CheckMocks(unittest.TestCase):
    """oracle/check_mocks.py over a scratch oracle directory: one runner that makes two listed
    replacements, a manifest that lists them, and a configuration naming the engine's decision
    modules and upstream function names."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.config = write_config(self.dir, harness_files=["run_*.py", "harness.py"],
                                   upstream_names=["event_loop_normal"])
        self.files = {"run_x.py": RUNNER}
        self.mocks = [dict(m) for m in MOCKS]
        self.version = None

    def tearDown(self):
        shutil.rmtree(self.dir)

    def check(self):
        for name, body in self.files.items():
            with open(os.path.join(self.dir, name), "w") as f:
                f.write(body)
        version = self.version or f"gpu-mock/1+{digest(self.mocks)}"
        with open(os.path.join(self.dir, "mocks.json"), "w") as f:
            json.dump({"version": version, "mocks": self.mocks}, f)
        return run([sys.executable, "-I", CHECK_MOCKS, "--config", self.config,
                    os.path.join(self.dir, "mocks.json")], self.dir)

    def plant(self, code, file="run_x.py"):
        self.files[file] = self.files.get(file, "") + "\n" + code + "\n"

    def assertFails(self, r, text):
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn(text, r.stdout)

    def test_the_listed_replacements_pass(self):
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_an_unlisted_patch_fails(self):
        for code in [
            "import upstream.sched.scheduler as s\ns.Scheduler.get_next_batch = lambda self: None",
            "import upstream.cache as c\nc.hash_page = None",
            "import upstream.sched.scheduler as s\nsetattr(s.Scheduler, 'step', None)",
            "import upstream.sched.scheduler as s\nx = object.__new__(s.Scheduler)",
            "import upstream.cache as c\nc.TABLE['k'] = 1",
            "import upstream.cache as c\ndel c.hash_page",
        ]:
            with self.subTest(code=code):
                self.files = {"run_x.py": RUNNER}
                self.plant(code)
                self.assertFails(self.check(), "not listed in mocks.json")

    def test_a_patch_in_any_named_harness_file_is_read(self):
        # harness.py is not a run_*.py file; the configuration names it, so its writes count.
        self.files["harness.py"] = "import upstream.cache as c\nc.hash_page = None\n"
        r = self.check()
        self.assertFails(r, "harness.py:2: writes c.hash_page")
        self.mocks.append({"symbol": "upstream.cache.hash_page", "kind": "environment", "how": "x",
                           "sites": ["harness.py:c.hash_page"]})
        self.assertEqual(self.check().returncode, 0)

    def test_an_import_hook_is_a_site(self):
        self.plant("import sys\nsys.meta_path.insert(0, object())")
        self.assertFails(self.check(), "writes sys.meta_path, which is not listed")
        self.mocks.append({"symbol": "a stub finder", "kind": "environment", "how": "x", "sites": ["run_x.py:sys.meta_path"]})
        self.assertEqual(self.check().returncode, 0)

    def test_config_after_the_manifest_is_read(self):
        for name in ("run_x.py",):
            with open(os.path.join(self.dir, name), "w") as f:
                f.write(self.files[name])
        version = f"gpu-mock/1+{digest(self.mocks)}"
        with open(os.path.join(self.dir, "mocks.json"), "w") as f:
            json.dump({"version": version, "mocks": self.mocks}, f)
        broken = os.path.join(self.dir, "broken.json")
        with open(broken, "w") as f:
            json.dump({}, f)
        r = run([sys.executable, "-I", CHECK_MOCKS, os.path.join(self.dir, "mocks.json"), "--config", broken], self.dir)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("decision_prefixes: missing", r.stderr)

    def test_a_file_the_configuration_does_not_name_is_not_read(self):
        self.files["gen_fuzz.py"] = "import upstream.cache as c\nc.hash_page = None\n"
        self.assertEqual(self.check().returncode, 0)

    def test_no_harness_file_fails(self):
        self.config = write_config(self.dir, harness_files=["nothing_*.py"])
        self.assertFails(self.check(), "matches harness_files")

    def test_a_listed_entry_no_file_makes_fails(self):
        self.mocks.append({"symbol": "upstream.some.never_patched", "kind": "gpu", "how": "x",
                           "sites": ["run_x.py:m.never_patched"]})
        self.assertFails(self.check(), "which no harness file makes")

    def test_a_mock_list_change_without_a_new_version_fails(self):
        self.version = f"gpu-mock/1+{digest(self.mocks)}"
        self.mocks[0]["how"] += " (edited)"
        self.assertFails(self.check(), "does not name this mock list")

    def test_a_decision_symbol_fails(self):
        self.mocks.append({"symbol": "upstream.sched.scheduler.Scheduler.get_next_batch", "kind": "gpu",
                           "how": "x", "sites": []})
        self.assertFails(self.check(), "replaces decision code")

    def test_a_clock_inside_a_decision_module_passes(self):
        self.mocks.append({"symbol": "upstream.sched.scheduler.time", "kind": "clock", "how": "the step number",
                           "sites": []})
        self.assertEqual(self.check().returncode, 0)

    def test_every_design_kind_is_allowed_and_no_other(self):
        for kind in ("gpu", "clock", "transport", "environment", "platform", "external-service"):
            with self.subTest(kind=kind):
                self.mocks[0]["kind"] = kind
                self.assertEqual(self.check().returncode, 0, kind)
        self.mocks[0]["kind"] = "decision"
        self.assertFails(self.check(), "kind 'decision' is not one of")

    # --- FID-8: a copy of upstream code is declared as a transcription ---

    def transcription(self, **over):
        t = {"symbol": "upstream.sched.scheduler.Scheduler.event_loop_normal", "kind": "transcription",
             "how": "the loop blocks on a socket", "upstream": "sched/scheduler.py:10-40",
             "transcribed_in": ["run_x.py:event_loop_normal"], "sites": []}
        t.update(over)
        return t

    def test_a_function_named_like_an_upstream_function_fails_unless_declared(self):
        self.plant("def event_loop_normal(s):\n    while s.more():\n        s.step()")
        self.assertFails(self.check(), "run_x.py:event_loop_normal: defines event_loop_normal")
        self.mocks.append(self.transcription())
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_nested_function_named_like_an_upstream_function_fails(self):
        self.plant("def drive(s):\n    def event_loop_normal():\n        pass\n    event_loop_normal()")
        self.assertFails(self.check(), "defines event_loop_normal")

    def test_a_derived_from_header_fails_unless_declared(self):
        self.files["run_x.py"] = "# Derived from upstream/sched/scheduler.py at 0123abcd.\n" + RUNNER
        self.assertFails(self.check(), "run_x.py: has a Derived from header")
        self.mocks.append(self.transcription(transcribed_in=["run_x.py"]))
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_derived_from_line_below_the_header_is_not_a_header(self):
        self.files["run_x.py"] = RUNNER + "\n# Derived from nothing: a comment in the body.\n"
        self.assertEqual(self.check().returncode, 0)

    def test_a_transcription_names_its_upstream_and_an_existing_copy(self):
        self.plant("def event_loop_normal(s):\n    pass")
        for name, entry, text in [
            ("no upstream", self.transcription(upstream=""), "names the upstream code it copies"),
            ("no copy", self.transcription(transcribed_in=[]), "names where the harness holds the copy"),
            ("a missing function", self.transcription(transcribed_in=["run_x.py:event_loop_overlap"]),
             "which no harness file defines"),
            ("a missing file", self.transcription(transcribed_in=["run_y.py"]), "which no harness file defines"),
        ]:
            with self.subTest(name):
                self.mocks = [dict(m) for m in MOCKS] + [entry]
                self.assertFails(self.check(), text)

    def test_only_a_transcription_declares_a_copy(self):
        self.plant("def event_loop_normal(s):\n    pass")
        self.mocks.append(self.transcription(kind="environment", symbol="upstream.x.event_loop_normal"))
        self.assertFails(self.check(), "only a transcription has transcribed_in")

    def test_a_configuration_with_a_missing_unknown_empty_or_mistyped_key_fails(self):
        with open(self.config) as f:
            cfg = json.load(f)
        for name, broken, key in config_breakages(cfg):
            with self.subTest(name):
                with open(self.config, "w") as f:
                    json.dump(broken, f)
                r = self.check()
                self.assertNotEqual(r.returncode, 0, r.stdout)
                self.assertIn(key, r.stderr)

    # --- a transcription installs its declared copy, and a declaration names a real copy ---

    def test_a_transcription_site_must_install_its_declared_copy(self):
        self.plant("import upstream.sched.scheduler as s\n"
                   "def event_loop_normal(x):\n    pass\n"
                   "def run_one2(x):\n    pass\n"
                   "s.Scheduler.schedule = run_one2")
        entry = self.transcription(symbol="upstream.sched.scheduler.Scheduler.schedule",
                                   sites=["run_x.py:s.Scheduler.schedule"])
        self.mocks.append(entry)
        self.assertFails(self.check(), "binds run_one2, not a function its transcribed_in lists")
        self.files["run_x.py"] = self.files["run_x.py"].replace("s.Scheduler.schedule = run_one2",
                                                                "s.Scheduler.schedule = event_loop_normal")
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_declaration_with_no_copy_behind_it_fails(self):
        self.plant("def drive(x):\n    pass")
        self.mocks.append(self.transcription(transcribed_in=["run_x.py:drive"]))
        self.assertFails(self.check(), "run_x.py:drive: declared in transcribed_in, but check_mocks finds no copy")
        self.mocks[-1] = self.transcription(transcribed_in=["run_x.py"])
        self.assertFails(self.check(), "run_x.py: declared in transcribed_in, but check_mocks finds no copy")


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
        return run([sys.executable, "-I", CHECK_LOCK, "--hashed",
                    os.path.join(self.dir, "lock"), os.path.join(self.dir, "hashed")], self.dir)

    LOCK = "# pins\ndocstring_parser==0.18.0\nengine @ https://x/engine.whl#sha256=ab\n"
    HASHED = ("# generated\ndocstring-parser==0.18.0 \\\n    --hash=sha256:01\n"
              "engine @ https://x/engine.whl#sha256=ab \\\n    --hash=sha256:ab\n")

    def test_the_same_pins_pass(self):
        r = self.check(self.LOCK, self.HASHED)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_different_version_fails(self):
        r = self.check(self.LOCK, self.HASHED.replace("0.18.0", "0.17.0"))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("docstring-parser==0.18.0 is in the lock but not hashed", r.stdout)

    def test_a_package_missing_from_the_hashed_file_fails(self):
        r = self.check(self.LOCK + "torch==9.9.9+cpu\n", self.HASHED)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("torch==9.9.9+cpu is in the lock but not hashed", r.stdout)


    def test_an_empty_lock_fails(self):
        for args in (["--hashed"], []):
            with self.subTest(args=args):
                for name in ("lock", "other"):
                    with open(os.path.join(self.dir, name), "w") as f:
                        f.write("# only comments\n")
                r = run([sys.executable, "-I", CHECK_LOCK, *args, os.path.join(self.dir, "lock"),
                         os.path.join(self.dir, "other")], self.dir)
                self.assertNotEqual(r.returncode, 0, r.stdout)
                self.assertIn("pins no package", r.stderr)

    def test_the_installed_set_is_compared_by_canonical_name(self):
        with open(os.path.join(self.dir, "lock"), "w") as f:
            f.write("docstring_parser==0.18.0\n")
        with open(os.path.join(self.dir, "frozen"), "w") as f:
            f.write("docstring-parser==0.18.0\n")
        r = run([sys.executable, "-I", CHECK_LOCK, os.path.join(self.dir, "lock"), os.path.join(self.dir, "frozen")],
                self.dir)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_the_installed_set_must_equal_the_lock(self):
        for name, frozen, text in [("missing", "docstring_parser==0.18.0\n", "is not installed"),
                                   ("extra", self.LOCK + "six==1.0\n", "is installed but not in the lock")]:
            with self.subTest(name):
                with open(os.path.join(self.dir, "lock"), "w") as f:
                    f.write(self.LOCK)
                with open(os.path.join(self.dir, "frozen"), "w") as f:
                    f.write(frozen)
                r = run([sys.executable, "-I", CHECK_LOCK, os.path.join(self.dir, "lock"),
                         os.path.join(self.dir, "frozen")], self.dir)
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn(text, r.stdout)
        with open(os.path.join(self.dir, "frozen"), "w") as f:
            f.write(self.LOCK)
        r = run([sys.executable, "-I", CHECK_LOCK, os.path.join(self.dir, "lock"), os.path.join(self.dir, "frozen")],
                self.dir)
        self.assertEqual(r.returncode, 0, r.stdout)


if __name__ == "__main__":
    unittest.main()
