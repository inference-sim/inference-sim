# Original to blis-vllm.
"""Tests for the oracle's CI guards over this repository's own files: each one must fail on the
violation it exists to catch.

check_goldens.py, check_mocks.py and check_lock.py are copies of blis-substrate's
contracts/harness, which holds their known-bad tests over scratch inputs. The cases here run them
on blis-vllm's committed runners, manifest, lock and configuration (oracle/harness.json): the
committed files pass, and a patch of vLLM's decision code planted in a scratch copy of the runners
is reported, so the configuration names vLLM's decision modules. The other guards (the metrics
manifest, the mutants) are blis-vllm's own.

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
sys.path.insert(0, HERE)
import goldengz  # noqa: E402


def run(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


class CheckGoldensOverThisLayout(unittest.TestCase):
    """oracle/check_goldens.py with blis-vllm's own oracle/harness.json: every golden directory the
    repository holds is guarded, so is core/kv/testdata/block_hashes.json (vLLM's hash_block_tokens
    output), and a commit that changes a recorded case of it without a trailer fails."""

    BLOCK_HASHES = "core/kv/testdata/block_hashes.json"

    def config(self):
        with open(os.path.join(HERE, "harness.json")) as f:
            return json.load(f)

    def test_every_golden_directory_and_the_block_hashes_are_guarded(self):
        cfg = self.config()
        found = set()
        for top, dirs, _ in os.walk(ROOT):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            rel = os.path.relpath(top, ROOT).replace(os.sep, "/")
            parts = rel.split("/")
            if len(parts) > 1 and parts[-2] == "testdata" and parts[-1].startswith("golden"):
                found.add(rel + "/")
        self.assertTrue(found, "no testdata/golden* directory found; the walk is broken")
        self.assertEqual(sorted(found - set(cfg["golden_dirs"])), [], "golden directories oracle/harness.json does not guard")
        self.assertIn(self.BLOCK_HASHES, cfg["golden_files"])
        self.assertTrue(goldengz.exists(os.path.join(ROOT, self.BLOCK_HASHES)))

    def test_a_block_hash_change_without_a_trailer_fails(self):
        repo = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo)

        def git(*args):
            r = run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args], repo)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout.strip()

        git("init", "-q")
        git("commit", "-q", "--allow-empty", "-m", "root")
        root = git("rev-parse", "HEAD")
        doc = goldengz.load(os.path.join(ROOT, self.BLOCK_HASHES))
        self.assertGreater(len(doc["cases"]), 0)
        doc["provenance"] = {**doc.get("provenance", {}), "blis_vllm_commit": root}
        path = os.path.join(repo, self.BLOCK_HASHES)
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(doc, f)
        os.makedirs(os.path.join(repo, "oracle"))
        shutil.copy(os.path.join(HERE, "requirements.lock"), os.path.join(repo, "oracle/requirements.lock"))
        git("add", "-A")
        git("commit", "-qm", "the fixture")
        base = git("rev-parse", "HEAD")
        first = doc["cases"][0]
        key = next(k for k in first if k != "name")
        first[key] = "changed"
        with open(path, "w") as f:
            json.dump(doc, f)
        git("commit", "-qam", "change a recorded case")
        r = run([sys.executable, "-I", os.path.join(HERE, "check_goldens.py"), "--config", os.path.join(HERE, "harness.json"),
                 base, "HEAD"], repo)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("block_hashes.json decisions changed without a golden-reviewed: trailer", r.stdout)


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
        self.assertIn("which no harness file makes", r.stdout)

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
    """oracle/check_lock.py --hashed over the committed files: the hashed install file pins exactly
    what the lock pins. The check's own known-bad cases are blis-substrate's (contracts/harness)."""

    def test_the_committed_files_agree(self):
        r = run([sys.executable, "-I", os.path.join(HERE, "check_lock.py"), "--hashed",
                 os.path.join(HERE, "requirements.lock"), os.path.join(HERE, "requirements-hashed.txt")], HERE)
        self.assertEqual(r.returncode, 0, r.stdout)


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
        keys = [k for k in goldengz.load(self.GOLDEN)["series"] if k.startswith(self.FAM + "{")]
        self.assertGreaterEqual(len(keys), 3, keys)
        self.assertTrue(any("running_lora_adapters=A," in k for k in keys), keys)
        self.assertTrue(any("running_lora_adapters=B," in k for k in keys), keys)

    def test_same_labels_pass_and_changed_labels_fail(self):
        c = self.load("compare_goldens_metrics")
        keys = sorted(k for k in goldengz.load(self.GOLDEN)["series"] if k.startswith(self.FAM + "{"))
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


class RecordedTracesDoNotDependOnSetOrder(unittest.TestCase):
    """A script that walks a set of bytes (vLLM's complete_store walks a set of OffloadKey) records the same
    order in every run once it starts through fix_hash_seed, and without it records a different order from
    one run to the next. The seed is not left in the environment the script's code sees (vLLM derives
    NONE_HASH from it, which would change every key)."""

    PROBE = (
        "import os, sys\n"
        "sys.path.insert(0, %r)\n"
        "if %%s:\n"
        "    from fixed_hash_seed import fix_hash_seed\n"
        "    fix_hash_seed(os.path.abspath(__file__), sys.argv[1:])\n"
        "keys = {bytes([i]) * 8 + b'\\0\\0\\0\\0' for i in range(32)}\n"
        "print(','.join(k.hex()[:2] for k in keys), os.environ.get('PYTHONHASHSEED'), sys.argv[1:])\n"
    ) % HERE

    def order(self, tmp, fixed, n=6):
        path = os.path.join(tmp, "probe_%s.py" % fixed)
        with open(path, "w") as f:
            f.write(self.PROBE % fixed)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        out = []
        for _ in range(n):
            r = run([sys.executable, "-I", path, "a", "--b"], tmp, env)
            self.assertEqual(r.returncode, 0, r.stderr)
            out.append(r.stdout)
        return out

    def test_fixed_seed_gives_one_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self.order(tmp, True)
            self.assertEqual(len(set(runs)), 1, runs)
            self.assertTrue(runs[0].rstrip().endswith("None ['a', '--b']"), runs[0])

    def test_unfixed_seed_gives_different_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertGreater(len(set(self.order(tmp, False))), 1)


class CheckFid5Names(unittest.TestCase):
    """oracle/check_fid5_names.py: an offload scenario is committed with its recorded trace, in ci-light,
    not first found missing by oracle-scenarios after the merge (the eleven OFFLOAD-MTP traces)."""

    SCRIPT = os.path.join(HERE, "check_fid5_names.py")

    def tree(self, tmp, scenarios, traces):
        for d, names in (("oracle/scenarios/offloading", scenarios), ("oracle/scenarios/multiconnector", []),
                         ("oracle/scenarios-fid5", []), ("core/connector/offloading/testdata/golden-fid5", traces)):
            os.makedirs(os.path.join(tmp, d))
            for n in names:
                with open(os.path.join(tmp, d, n), "w") as f:
                    f.write("{}")

    def check(self, tmp):
        return run([sys.executable, "-I", self.SCRIPT, tmp], tmp)

    def test_the_committed_traces_are_the_scenarios_traces(self):
        r = run([sys.executable, "-I", self.SCRIPT, ROOT], ROOT)
        self.assertEqual((r.returncode, r.stdout), (0, ""), r.stdout)

    def test_a_scenario_with_no_committed_trace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(tmp, ["offloading-a.json", "offloading-b.json"], ["recorded-offloading-a.fid5.json"])
            r = self.check(tmp)
            self.assertEqual(r.returncode, 1)
            self.assertIn("recorded-offloading-b.fid5.json has no committed trace", r.stdout)
            self.assertNotIn("recorded-offloading-a", r.stdout)

    def test_a_trace_no_scenario_produces_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(tmp, ["offloading-a.json"], ["recorded-offloading-a.fid5.json", "recorded-offloading-gone.fid5.json"])
            r = self.check(tmp)
            self.assertEqual(r.returncode, 1)
            self.assertIn("recorded-offloading-gone.fid5.json has no scenario", r.stdout)

    def test_pending_goldens_are_not_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(tmp, ["offloading-a.json"], [])
            with open(os.path.join(tmp, "core/connector/offloading/testdata/golden-fid5.pending.json"), "w") as f:
                f.write("{}")
            self.assertEqual(self.check(tmp).returncode, 0)


class CompareGoldensDerived(unittest.TestCase):
    """A golden derived from the pinned source is compared as such, never as the release's own trace."""

    def compare(self, golden_prov, golden_steps, fresh_steps):
        with tempfile.TemporaryDirectory() as tmp:
            g, f = os.path.join(tmp, "g"), os.path.join(tmp, "f")
            os.mkdir(g)
            os.mkdir(f)
            base = {"scenario_sha256": "s"}
            gold = {"provenance": {**base, **golden_prov}, "steps": golden_steps, "finished": {}}
            fresh = {"provenance": base, "steps": fresh_steps, "finished": {}}
            with open(os.path.join(g, "x.json"), "w") as fh:
                json.dump(gold, fh)
            with open(os.path.join(f, "x.trace.json"), "w") as fh:
                json.dump(fresh, fh)
            return run([sys.executable, "-I", os.path.join(HERE, "compare_goldens.py"), g, f], HERE)

    def test_a_derived_golden_that_differs_from_the_release_passes(self):
        r = self.compare({"derived_from_source": "why", "oracle_run": False}, [{"a": 1}], [{"a": 2}])
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("1 derived from source", r.stdout)

    def test_a_derived_golden_the_release_also_produces_is_stale(self):
        r = self.compare({"derived_from_source": "why", "oracle_run": False}, [{"a": 1}], [{"a": 1}])
        self.assertEqual(r.returncode, 1)
        self.assertIn("replace it with the recording", r.stdout)

    def test_a_derived_golden_must_record_oracle_run_false(self):
        r = self.compare({"derived_from_source": "why", "oracle_run": True}, [{"a": 1}], [{"a": 2}])
        self.assertEqual(r.returncode, 1)
        self.assertIn("oracle_run false", r.stdout)

    def test_a_recorded_golden_that_differs_still_fails(self):
        r = self.compare({"oracle_run": True}, [{"a": 1}], [{"a": 2}])
        self.assertEqual(r.returncode, 1)
        self.assertIn("step 0", r.stdout)


if __name__ == "__main__":
    unittest.main()
