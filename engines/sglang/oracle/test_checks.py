# Original to blis-sglang.
"""Tests for the oracle's guards over this repository's own files: each must fail on what it
guards (X18). check_mocks.py, check_goldens.py and check_lock.py are copies of blis-substrate's
contracts/harness, which holds their scratch-input tests; the cases here run them, and this
repository's own checks, on blis-sglang's configuration.

Usage: python3 -I oracle/test_checks.py <sglang checkout>   (stdlib and git only)
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SGLANG = None  # the checkout, from argv


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def harness_config():
    with open(os.path.join(HERE, "harness.json")) as f:
        return json.load(f)


def digest(mocks):
    canonical = json.dumps(mocks, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# A provenance block of the trace format (oracle/traces.py), for scratch traces.
PROVENANCE = {
    "tier": 2, "sglang_commit": "297e54fd105e974cdf401e5b4d20253a22c16f90", "requirements_lock_sha256": "1" * 64,
    "lockfile_sha256": "2" * 64, "scenario_sha256": "3" * 64, "seed": 0, "mock_version": "gpu-mock/1+x",
    "model": "m", "model_revision": "r", "python": "3.12.0", "platform": "linux-x86_64",
    "blis_sglang_commit": "4" * 40, "oracle_inputs": {"oracle/scenarios/a.json": "3" * 64},
}


class CheckModules(unittest.TestCase):
    """oracle/check_modules.py: every sglang module the scheduler's files import is classified."""

    def test_the_checkout_is_classified(self):
        r = run([sys.executable, "-I", os.path.join(HERE, "check_modules.py"), SGLANG], ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def scratch(self, imports, prefixes, not_decision, extra_modules=()):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        mgr = os.path.join(root, "python/sglang/srt/managers")
        os.makedirs(os.path.join(mgr, "scheduler_components"))
        with open(os.path.join(mgr, "scheduler.py"), "w") as f:
            f.write("".join(f"import {m}\n" for m in imports) + "def f():\n    from sglang.srt.lazy import x\n")
        for m in list(imports) + ["sglang.srt.lazy"] + list(extra_modules):
            p = os.path.join(root, "python", *m.split(".")) + ".py"
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "a").close()
        cfg = os.path.join(root, "harness.json")
        with open(cfg, "w") as f:
            json.dump({"decision_prefixes": prefixes}, f)
        mods = os.path.join(root, "modules.json")
        with open(mods, "w") as f:
            json.dump({"not_decision": not_decision}, f)
        return run([sys.executable, "-I", os.path.join(HERE, "check_modules.py"), root, cfg, mods], root)

    def test_an_unclassified_import_fails(self):
        # sglang.srt.lazy is imported inside a function; it still counts.
        r = self.scratch(["sglang.srt.managers.scheduler"], ["sglang.srt.managers.scheduler."], {})
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("sglang.srt.lazy (imported by", r.stdout)
        r = self.scratch(["sglang.srt.managers.scheduler"], ["sglang.srt.managers.scheduler."],
                         {"sglang.srt.lazy": "a reason"})
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_stale_prefix_or_entry_fails(self):
        r = self.scratch([], ["sglang.srt.gone."], {"sglang.srt.lazy": "x", "sglang.srt.also_gone": "y"})
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("decision prefix sglang.srt.gone.: no such module", r.stdout)
        self.assertIn("modules.json entry sglang.srt.also_gone: no such module", r.stdout)

    def test_an_entry_without_a_reason_fails(self):
        r = self.scratch([], [], {"sglang.srt.lazy": " "})
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("gives no reason", r.stdout)


class DecisionPrefixes(unittest.TestCase):
    """check_mocks.py with this repository's harness.json refuses a replacement under each decision
    module the pre-review named."""

    NAMED = ["sglang.srt.managers.prefill_delayer.PrefillDelayer.should_delay",
             "sglang.srt.managers.min_free_slots_delayer.MinFreeSlotsDelayer.should_delay",
             "sglang.srt.model_executor.pool_configurator.MemoryPoolConfigurator.calculate_pool_sizes",
             "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver.recv_requests",
             "sglang.srt.managers.scheduler_components.kv_events_publisher.KvEventsPublisher.publish",
             "sglang.srt.managers.scheduler_components.metrics_reporter.SchedulerMetricsReporter.report",
             "sglang.srt.observability.metrics_collector.SchedulerMetricsCollector.log_stats",
             "sglang.srt.observability.req_time_stats.ReqTimeStats.set",
             "sglang.srt.managers.data_parallel_controller.DataParallelController.round_robin_scheduler"]

    def test_a_replacement_under_each_named_decision_module_fails(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        with open(os.path.join(d, "run_x.py"), "w") as f:
            f.write("x = 1\n")
        for sym in self.NAMED:
            with self.subTest(sym):
                mocks = [{"symbol": sym, "kind": "environment", "how": "x", "sites": []}]
                with open(os.path.join(d, "mocks.json"), "w") as f:
                    json.dump({"version": f"gpu-mock/1+{digest(mocks)}", "mocks": mocks}, f)
                r = run([sys.executable, "-I", os.path.join(HERE, "check_mocks.py"),
                         "--config", os.path.join(HERE, "harness.json"), os.path.join(d, "mocks.json")], d)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn(f"{sym}: replaces decision code", r.stdout)


class Stubs(unittest.TestCase):
    """oracle/stubs.py: a stubbed module must be named by an environment entry of mocks.json."""

    def test_an_unlisted_stub_fails_and_a_listed_one_passes(self):
        sys.path.insert(0, HERE)
        try:
            import stubs
        finally:
            sys.path.remove(HERE)
        self.assertEqual(stubs.problems(("sgl_kernel",), {"mocks": []}),
                         ["stubs.py stubs sgl_kernel, which no environment entry of oracle/mocks.json names (FID-1)"])
        self.assertTrue(stubs.problems(("sgl_kernel",), {"mocks": [{"symbol": "sgl_kernel", "kind": "gpu"}]}))
        self.assertEqual(stubs.problems(("sgl_kernel",), {"mocks": [{"symbol": "sgl_kernel", "kind": "environment"}]}), [])

    def test_the_stub_finder_is_a_site_of_the_manifest(self):
        # check_mocks.py counts the import hook as a site: drop its entry and the check fails.
        with open(os.path.join(HERE, "mocks.json")) as f:
            spec = json.load(f)
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        for name in os.listdir(HERE):
            if name.endswith(".py"):
                shutil.copy(os.path.join(HERE, name), d)
        spec["mocks"] = [m for m in spec["mocks"] if "stubs.py:sys.meta_path" not in m.get("sites", [])]
        spec["version"] = f"gpu-mock/9+{digest(spec['mocks'])}"
        with open(os.path.join(d, "mocks.json"), "w") as f:
            json.dump(spec, f)
        r = run([sys.executable, "-I", os.path.join(HERE, "check_mocks.py"), "--config",
                 os.path.join(HERE, "harness.json"), os.path.join(d, "mocks.json")], d)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("stubs.py", r.stdout)
        self.assertIn("writes sys.meta_path", r.stdout)


class RunImport(unittest.TestCase):
    """oracle/run_import.py fails unless the scheduler module it imports is the checkout's."""

    def fake_checkout(self, link):
        top = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, top)
        pkg = os.path.join(top, "elsewhere" if link else "checkout/python", "sglang")
        mgr = os.path.join(pkg, "srt", "managers")
        os.makedirs(mgr)
        for d in (pkg, os.path.join(pkg, "srt"), mgr):
            open(os.path.join(d, "__init__.py"), "w").close()
        with open(os.path.join(mgr, "scheduler.py"), "w") as f:
            f.write("class Scheduler:\n" + "".join(f"    def {n}(self): pass\n" for n in (
                "event_loop_normal", "event_loop_overlap", "get_next_batch_to_run", "process_batch_result",
                "ingest_requests")))
        if link:
            os.makedirs(os.path.join(top, "checkout/python"))
            os.symlink(pkg, os.path.join(top, "checkout/python/sglang"))
        return os.path.join(top, "checkout")

    def test_the_checkouts_scheduler_passes(self):
        r = run([sys.executable, "-I", os.path.join(HERE, "run_import.py"), self.fake_checkout(False)], HERE)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_scheduler_from_outside_the_checkout_fails(self):
        r = run([sys.executable, "-I", os.path.join(HERE, "run_import.py"), self.fake_checkout(True)], HERE)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("was imported from", r.stderr)


class CheckPeers(unittest.TestCase):
    """oracle/check_peers.py: a peer in the lock is at its peers.json version."""

    def check(self, lock):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        with open(os.path.join(d, "lock"), "w") as f:
            f.write(lock)
        return run([sys.executable, "-I", os.path.join(HERE, "check_peers.py"), os.path.join(HERE, "peers.json"),
                    os.path.join(d, "lock")], d)

    def test_a_peer_at_another_version_fails(self):
        r = self.check("numpy==2.0.0\nLMCache==0.5.7\n")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("LMCache==0.5.7; oracle/peers.json pins it at 0.5.6", r.stdout)

    def test_a_peer_at_its_pin_or_absent_passes(self):
        self.assertEqual(self.check("numpy==2.0.0\nlmcache==0.5.6\n").returncode, 0)
        self.assertEqual(self.check("numpy==2.0.0\n").returncode, 0)


def import_oracle(name):
    sys.path.insert(0, HERE)
    try:
        return __import__(name)
    finally:
        sys.path.remove(HERE)


class CopiesOfTheLoopAreRefused(unittest.TestCase):
    """check_mocks.py with this repository's configuration: a harness copy of an event loop, or a
    write of one into the Scheduler, fails (FID-8)."""

    def check_with(self, extra):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        for name in os.listdir(HERE):
            if name.endswith(".py") or name.endswith(".json"):
                shutil.copy(os.path.join(HERE, name), d)
        with open(os.path.join(d, "harness.py"), "a") as f:
            f.write(extra)
        return run([sys.executable, "-I", os.path.join(HERE, "check_mocks.py"), "--config",
                    os.path.join(HERE, "harness.json"), os.path.join(d, "mocks.json")], d)

    def test_the_committed_harness_passes(self):
        self.assertEqual(self.check_with("").returncode, 0)

    def test_a_copied_loop_fails(self):
        r = self.check_with("\n\ndef event_loop_normal(scheduler):\n    while not scheduler.gracefully_exit:\n        pass\n")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("harness.py:event_loop_normal: defines event_loop_normal", r.stdout)

    def test_a_write_into_the_scheduler_fails(self):
        r = self.check_with("\n\ndef f(s):\n    pass\n\n\nimport sglang.srt.managers.scheduler as sm\nsm.Scheduler.event_loop_normal = f\n")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("writes sm.Scheduler.event_loop_normal, which is not listed", r.stdout)


class CompareGoldens(unittest.TestCase):
    """oracle/compare_goldens.py over scratch scenario, golden and fresh directories."""

    TRACE = {"provenance": {**PROVENANCE, "sglang_commit": "c" * 40},
             "steps": [{"forwards": [{"extend_prefix_lens": [6]}]}]}

    def dirs(self, scenarios=("a",), goldens=("a",), fresh=None):
        top = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, top)
        out = {}
        for d in ("scenarios", "golden", "fresh"):
            out[d] = os.path.join(top, d)
            os.makedirs(out[d])
        def put(path, doc):
            with open(path, "w") as f:
                json.dump(doc, f)

        for n in scenarios:
            put(os.path.join(out["scenarios"], n + ".json"), {})
            put(os.path.join(out["fresh"], n + ".trace.json"), fresh or self.TRACE)
        for n in goldens:
            put(os.path.join(out["golden"], n + ".trace.json"), self.TRACE)
        return out

    def compare(self, d):
        return run([sys.executable, "-I", os.path.join(HERE, "compare_goldens.py"), d["fresh"], "c" * 40,
                    d["scenarios"], d["golden"]], HERE)

    def test_the_same_trace_passes(self):
        self.assertEqual(self.compare(self.dirs()).returncode, 0)

    def test_a_changed_prefix_length_fails(self):
        fresh = json.loads(json.dumps(self.TRACE))
        fresh["steps"][0]["forwards"][0]["extend_prefix_lens"] = [0]
        r = self.compare(self.dirs(fresh=fresh))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("/steps/0/forwards/0/extend_prefix_lens/0: 6 != 0", r.stdout)

    def test_a_missing_golden_an_orphan_golden_and_no_scenario_fail(self):
        for name, d, text in [("no golden", self.dirs(scenarios=("a", "b")), "b: no golden"),
                              ("orphan", self.dirs(goldens=("a", "z")), "z.trace.json: no scenario"),
                              ("no scenario", self.dirs(scenarios=(), goldens=()), "no scenario in")]:
            with self.subTest(name):
                r = self.compare(d)
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn(text, r.stdout)

    def test_a_stale_provenance_fails(self):
        fresh = json.loads(json.dumps(self.TRACE))
        fresh["provenance"]["mock_version"] = "gpu-mock/2+y"
        r = self.compare(self.dirs(fresh=fresh))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("mock_version", r.stdout)

    def test_a_provenance_key_one_side_lacks_fails(self):
        for key in ("seed", "mock_version", "tier"):
            with self.subTest(key):
                fresh = json.loads(json.dumps(self.TRACE))
                del fresh["provenance"][key]
                r = self.compare(self.dirs(fresh=fresh))
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn(f"this run's trace: provenance.{key}: missing", r.stdout)

    def test_another_seed_fails(self):
        fresh = json.loads(json.dumps(self.TRACE))
        fresh["provenance"]["seed"] = 7
        r = self.compare(self.dirs(fresh=fresh))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("seed", r.stdout)


class Traces(unittest.TestCase):
    """oracle/traces.py: the trace format, and the update flag on every write into a golden."""

    def setUp(self):
        self.t = import_oracle("traces")
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        os.makedirs(os.path.join(self.root, "oracle"))
        with open(os.path.join(self.root, "oracle", "harness.json"), "w") as f:
            json.dump({"golden_dirs": ["core/cache/goldens/"], "golden_files": ["oracle/cases.json"]}, f)
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(self.t.UPDATE_FLAG, None)

    def test_a_complete_provenance_is_of_the_format(self):
        self.assertEqual(self.t.trace_problems({"provenance": dict(PROVENANCE), "steps": []}), [])

    def test_each_missing_malformed_or_unknown_key_fails(self):
        cases = [(k, None, f"provenance.{k}: missing") for k in PROVENANCE]
        cases += [("sglang_commit", "297e54fd", "not a full SGLang commit id"),
                  ("requirements_lock_sha256", "l", "oracle/requirements.lock"),
                  ("scenario_sha256", "s", "the scenario file"),
                  ("seed", "0", "the scenario's seed"), ("seed", True, "the scenario's seed"),
                  ("mock_version", "", "mocks.json's version"),
                  ("blis_sglang_commit", "unknown", "full commit id"),
                  ("oracle_inputs", {}, "each oracle/ file"),
                  ("oracle_inputs", {"/etc/x": "3" * 64}, "each oracle/ file"),
                  ("extra", 1, "provenance.extra: not a key of the trace format")]
        for key, value, text in cases:
            with self.subTest(key=key, value=value):
                prov = dict(PROVENANCE)
                if value is None:
                    del prov[key]
                else:
                    prov[key] = value
                got = self.t.trace_problems({"provenance": prov, "steps": []})
                self.assertTrue(any(text in p for p in got), got)
        self.assertIn("the trace has no provenance block", self.t.trace_problems({"steps": []}))
        self.assertIn("the trace records nothing beside its provenance",
                      self.t.trace_problems({"provenance": dict(PROVENANCE)}))

    def test_a_write_into_any_golden_needs_the_update_flag(self):
        trace = {"provenance": dict(PROVENANCE), "steps": []}
        for rel in ("core/cache/goldens/a.json", "core/cache/goldens/sub/a.json", "oracle/cases.json",
                    "core/sched/testdata/golden-t3/a.json"):
            with self.subTest(rel):
                path = os.path.join(self.root, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with self.assertRaises(SystemExit) as e:
                    self.t.write(path, trace, self.root)
                self.assertIn("refusing to write a golden trace", str(e.exception))
                self.assertFalse(os.path.exists(path))
                os.environ[self.t.UPDATE_FLAG] = "1"
                self.t.write(path, trace, self.root)
                del os.environ[self.t.UPDATE_FLAG]
                with open(path) as f:
                    self.assertEqual(json.load(f), trace)

    def test_a_write_elsewhere_needs_no_flag_and_a_malformed_trace_is_never_written(self):
        path = os.path.join(self.root, "out.trace.json")
        self.t.write(path, {"provenance": dict(PROVENANCE), "steps": []}, self.root)
        self.assertTrue(os.path.exists(path))
        bad = os.path.join(self.root, "bad.trace.json")
        with self.assertRaises(SystemExit) as e:
            self.t.write(bad, {"provenance": {**PROVENANCE, "seed": None}, "steps": []}, self.root)
        self.assertIn("provenance.seed", str(e.exception))
        self.assertFalse(os.path.exists(bad))


class CheckProvenance(unittest.TestCase):
    """oracle/check_provenance.py, the trace provenance check (FID-2), on this tree and on scratch trees."""

    SCENARIO = {"model": "m", "seed": 5, "requests": []}

    def check(self, root):
        return run([sys.executable, "-I", os.path.join(HERE, "check_provenance.py"), root], ROOT)

    def test_the_committed_goldens_pass(self):
        r = self.check(ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def tree(self):
        """A scratch tree whose one golden records exactly the tree's environment."""
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        os.makedirs(os.path.join(root, "oracle", "scenarios"))
        os.makedirs(os.path.join(root, "oracle", "testdata", "golden"))
        shutil.copy(os.path.join(ROOT, "provenance.go"), root)
        with open(os.path.join(root, "oracle", "harness.json"), "w") as f:
            json.dump({"golden_dirs": ["oracle/testdata/golden/"], "golden_files": []}, f)
        self.put(root, "oracle/mocks.json", {"version": "gpu-mock/9+abc"})
        with open(os.path.join(root, "oracle", "requirements.lock"), "w") as f:
            f.write("torch==2.14.1\n")
        self.put(root, "oracle/scenarios/a.json", self.SCENARIO)
        prov = {**PROVENANCE, "sglang_commit": "297e54fd105e974cdf401e5b4d20253a22c16f90",
                "requirements_lock_sha256": self.sha(root, "oracle/requirements.lock"),
                "scenario_sha256": self.sha(root, "oracle/scenarios/a.json"), "seed": 5,
                "mock_version": "gpu-mock/9+abc",
                "oracle_inputs": {"oracle/scenarios/a.json": self.sha(root, "oracle/scenarios/a.json"),
                                  "oracle/mocks.json": self.sha(root, "oracle/mocks.json")}}
        self.put(root, "oracle/testdata/golden/a.trace.json", {"provenance": prov, "steps": [1]})
        return root

    @staticmethod
    def put(root, rel, doc):
        with open(os.path.join(root, rel), "w") as f:
            json.dump(doc, f)

    @staticmethod
    def sha(root, rel):
        with open(os.path.join(root, rel), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def edit_golden(self, root, **prov):
        path = os.path.join(root, "oracle/testdata/golden/a.trace.json")
        with open(path) as f:
            g = json.load(f)
        g["provenance"].update(prov)
        with open(path, "w") as f:
            json.dump(g, f)

    def test_a_golden_of_this_tree_passes(self):
        r = self.check(self.tree())
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_each_stale_record_fails(self):
        def other_pin(root):
            self.edit_golden(root, sglang_commit="c" * 40)

        def new_lock(root):
            with open(os.path.join(root, "oracle", "requirements.lock"), "a") as f:
                f.write("numpy==2.3.0\n")

        def new_mocks(root):
            self.put(root, "oracle/mocks.json", {"version": "gpu-mock/10+def"})

        def edited_scenario(root):
            self.put(root, "oracle/scenarios/a.json", {**self.SCENARIO, "requests": [{"rid": "r0"}]})

        def removed_scenario(root):
            os.remove(os.path.join(root, "oracle/scenarios/a.json"))

        def other_seed(root):
            self.edit_golden(root, seed=6)

        def scenario_not_an_input(root):
            self.edit_golden(root, oracle_inputs={"oracle/mocks.json": self.sha(root, "oracle/mocks.json")})

        def malformed(root):
            self.edit_golden(root, requirements_lock_sha256="l")

        def no_golden(root):
            os.remove(os.path.join(root, "oracle/testdata/golden/a.trace.json"))

        for edit, text in [(other_pin, "records sglang_commit"), (new_lock, "records requirements_lock_sha256"),
                           (new_mocks, "records mock_version"), (edited_scenario, "has changed since it was recorded"),
                           (removed_scenario, "the file is gone"), (other_seed, "has seed 5"),
                           (scenario_not_an_input, "not a file of this tree"),
                           (malformed, "provenance.requirements_lock_sha256"), (no_golden, "no golden trace")]:
            with self.subTest(edit.__name__):
                root = self.tree()
                edit(root)
                r = self.check(root)
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn(text, r.stdout)


class RefreshProvenance(unittest.TestCase):
    """oracle/refresh_provenance.py copies provenance only, and only with the update flag."""

    GOLDEN = {"provenance": dict(PROVENANCE), "steps": [{"forwards": [{"extend_prefix_lens": [6]}]}]}

    def dirs(self, fresh):
        top = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, top)
        g, f = os.path.join(top, "golden"), os.path.join(top, "fresh")
        os.makedirs(g)
        os.makedirs(f)
        with open(os.path.join(g, "a.trace.json"), "w") as h:
            json.dump(self.GOLDEN, h)
        with open(os.path.join(f, "a.trace.json"), "w") as h:
            json.dump(fresh, h)
        return g, f

    def refresh(self, g, f, flag=True):
        env = {k: v for k, v in os.environ.items() if k != "BLIS_SGLANG_ALLOW_GOLDEN_UPDATE"}
        if flag:
            env["BLIS_SGLANG_ALLOW_GOLDEN_UPDATE"] = "1"
        return subprocess.run([sys.executable, "-I", os.path.join(HERE, "refresh_provenance.py"), g, f],
                              cwd=HERE, env=env, capture_output=True, text=True)

    def golden(self, g):
        with open(os.path.join(g, "a.trace.json")) as h:
            return json.load(h)

    def fresh(self, **prov):
        fresh = json.loads(json.dumps(self.GOLDEN))
        fresh["provenance"].update(prov)
        return fresh

    def test_a_new_environment_is_copied_into_the_golden(self):
        g, f = self.dirs(self.fresh(mock_version="gpu-mock/2+y", requirements_lock_sha256="9" * 64))
        r = self.refresh(g, f)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.golden(g)["provenance"]["mock_version"], "gpu-mock/2+y")
        self.assertEqual(self.golden(g)["steps"], self.GOLDEN["steps"])

    def test_it_refuses_without_the_flag(self):
        g, f = self.dirs(self.fresh(mock_version="gpu-mock/2+y"))
        r = self.refresh(g, f, flag=False)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("refusing to write a golden trace", r.stderr)
        self.assertEqual(self.golden(g), self.GOLDEN)

    def test_it_refuses_changed_decisions_another_scenario_and_a_malformed_trace(self):
        changed = self.fresh(mock_version="gpu-mock/2+y")
        changed["steps"][0]["forwards"][0]["extend_prefix_lens"] = [0]
        for name, fresh, text in [("decisions", changed, "decisions differ (steps)"),
                                  ("scenario", self.fresh(scenario_sha256="8" * 64), "ran another scenario"),
                                  ("seed", self.fresh(seed=3), "ran another scenario"),
                                  ("format", self.fresh(seed="0"), "the fresh trace: provenance.seed")]:
            with self.subTest(name):
                g, f = self.dirs(fresh)
                r = self.refresh(g, f)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn(text, r.stdout)
                self.assertEqual(self.golden(g), self.GOLDEN)



class ValueCoverage(unittest.TestCase):
    """oracle/value_coverage.py, the value-coverage report (FID-10)."""

    def check(self, root):
        return run([sys.executable, "-I", os.path.join(HERE, "value_coverage.py"), root], ROOT)

    def test_the_committed_goldens_are_covered_or_held(self):
        r = self.check(ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def tree(self, goldens, held=None, maps=("finished",)):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        os.makedirs(os.path.join(root, "oracle", "testdata", "golden"))
        with open(os.path.join(root, "oracle", "harness.json"), "w") as f:
            json.dump({"golden_dirs": ["oracle/testdata/golden/"], "golden_files": []}, f)
        with open(os.path.join(root, "oracle", "value_coverage.json"), "w") as f:
            json.dump({"description": "x", "maps": list(maps), "held": held or {}}, f)
        for i, g in enumerate(goldens):
            with open(os.path.join(root, "oracle", "testdata", "golden", f"g{i}.trace.json"), "w") as f:
                json.dump({"provenance": {"seed": i}, **g}, f)
        return root

    def test_a_field_at_one_value_fails_and_two_values_pass(self):
        held = {"steps": {"kind": "fixed", "reason": "r"}}
        r = self.check(self.tree([{"steps": [{"batch_size": 2}]}, {"steps": [{"batch_size": 2}]}], held))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("steps[].batch_size: takes only 2", r.stdout)
        r = self.check(self.tree([{"steps": [{"batch_size": 2}, {"batch_size": 0}]}], held))
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_list_is_covered_only_when_both_empty_and_not(self):
        r = self.check(self.tree([{"arrivals": ["r0"]}, {"arrivals": ["r1"]}]))
        self.assertIn("arrivals: takes only [...]", r.stdout)
        r = self.check(self.tree([{"arrivals": ["r0"]}, {"arrivals": []}], held={"arrivals[]": {"kind": "fixed", "reason": "x"}}))
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_map_keyed_by_request_is_one_field(self):
        goldens = [{"finished": {"r0": {"step": 2}}}, {"finished": {"r1": {"step": 2}}}]
        held = {"finished.*.step": {"kind": "gap", "reason": "no scenario moves it (#32)"}}
        r = self.check(self.tree(goldens, held))
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_the_held_list_stays_true(self):
        two = [{"x": 1}, {"x": 0}]
        for name, goldens, held, text in [
                ("now covered", two, {"x": {"kind": "fixed", "reason": "r"}}, "x: is held as fixed but is now covered"),
                ("no such field", two, {"y": {"kind": "fixed", "reason": "r"}}, "y: is held, but no golden has it"),
                ("a gap without its issue", [{"x": 1}], {"x": {"kind": "gap", "reason": "later"}}, "cites the open issue"),
                ("no reason", [{"x": 1}], {"x": {"kind": "fixed", "reason": ""}}, "x: no reason"),
                ("no kind", [{"x": 1}], {"x": {"reason": "r"}}, "kind must be one of")]:
            with self.subTest(name):
                r = self.check(self.tree(goldens, held))
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn(text, r.stdout)

    def test_no_golden_fails(self):
        r = self.check(self.tree([]))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("no golden trace", r.stdout)


class MutationSweep(unittest.TestCase):
    """oracle/mutate.py, the mutation sweep (T7), on scratch repositories."""

    LIB = "def positive(x):\n    return x > 0\n\n\ndef untested():\n    return 1\n"
    TEST = "import lib\nassert lib.positive(1) and not lib.positive(0)\n"
    GO = ("// SPDX-License-Identifier: Apache-2.0\n// Derived from python/sglang/srt/x.py at SGLang v0.5.23.\n\n"
          "package core\n\n// Fits says a < b.\nfunc Fits(a, b int) bool {\n\treturn a < b && b != 0 // a > b\n}\n\n"
          "func Name() string {\n\treturn \"a >= b\"\n}\n")

    def repo(self, mutants, gaps=None, equivalent=None):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        os.makedirs(os.path.join(root, "oracle"))
        os.makedirs(os.path.join(root, "core"))
        files = {"lib.py": self.LIB, "check.py": self.TEST, "other.py": "X = 1\n", "core/x.go": self.GO}
        for rel, text in files.items():
            with open(os.path.join(root, rel), "w") as f:
                f.write(text)
        cfg = {"description": "x", "mutants": mutants, "known_gaps": gaps or {}, "equivalent": equivalent or {},
               "suites": {"unit": {"command": [sys.executable, "check.py"], "tests": ["check.py"]},
                          "go": {"command": ["false"], "tests": []}}}
        with open(os.path.join(root, "oracle", "mutants.json"), "w") as f:
            json.dump(cfg, f)
        for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false",
                                                     "commit", "-q", "-m", "base"]):
            subprocess.run(["git", "-C", root, *cmd], check=True, capture_output=True)
        return root

    def sweep(self, root, *extra):
        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, True)
        return run([sys.executable, "-I", os.path.join(HERE, "mutate.py"), "--root", root, *extra, "nowhere",
                    os.path.join(scratch, "tree")], root)

    @staticmethod
    def mutant(i, file="lib.py", old="return x > 0", new="return x >= 0"):
        return {"id": i, "guards": "g", "file": file, "old": old, "new": new, "suite": "unit"}

    def test_a_caught_mutant_passes(self):
        r = self.sweep(self.repo([self.mutant("A")]), "--only", "A")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("CAUGHT     A", r.stdout)

    def test_a_survivor_a_stale_gap_and_a_broken_mutant_fail(self):
        survivor = self.mutant("B", old="return 1", new="return 2")
        for name, mutants, gaps, text in [
                ("survivor", [survivor], {}, "SURVIVED   B  guards g"),
                ("gap now caught", [self.mutant("A")], {"A": "a gap (#32)"}, "NOW CAUGHT A"),
                ("broken", [self.mutant("C", old="return y")], {}, "BROKEN     C"),
                ("gap without its issue", [survivor], {"B": "later"}, "cites the open issue"),
                ("stale gap", [], {"Z": "gone (#32)"}, "STALE      Z")]:
            with self.subTest(name):
                r = self.sweep(self.repo(mutants, gaps), "--only", ",".join(m["id"] for m in mutants) or "none") \
                    if name != "stale gap" else self.sweep(self.repo(mutants, gaps), "--list")
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn(text, r.stdout)

    def test_a_listed_survivor_with_its_issue_passes(self):
        r = self.sweep(self.repo([self.mutant("B", old="return 1", new="return 2")], {"B": "no test yet (#32)"}), "--only", "B")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def package_list(self, root, entries, key="pending"):
        os.makedirs(os.path.join(root, "core", "testdata"))
        with open(os.path.join(root, "core", "testdata", "pending_mutants.json"), "w") as f:
            json.dump({key: entries}, f)

    def test_a_package_lists_its_pending_survivor(self):
        survivor = self.mutant("B", old="return 1", new="return 2")
        for name, key, entry, want in [
                ("pending, with its issue", "pending", {"id": "B", "stage": "S17", "issue": "x/y#53"}, 0),
                ("pending, without an issue", "pending", {"id": "B", "stage": "S17", "issue": "soon"}, 1),
                ("equivalent, with a reason", "equivalent", {"id": "B", "why": "the value is unused"}, 0)]:
            with self.subTest(name):
                root = self.repo([survivor])
                self.package_list(root, [entry], key)
                r = self.sweep(root, "--only", "B")
                self.assertEqual(r.returncode, want, r.stdout + r.stderr)

    def test_a_package_entry_that_a_test_now_catches_fails(self):
        root = self.repo([self.mutant("A")])
        self.package_list(root, [{"id": "A", "stage": "S17", "issue": "x/y#53"}])
        r = self.sweep(root, "--only", "A")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("NOW CAUGHT A", r.stdout)

    def test_generated_mutants_flip_code_operators_only(self):
        mu = import_oracle("mutate")
        got = mu.generated_mutants("core/x.go", self.GO)
        self.assertEqual(sorted(m["new"].strip() for m in got),
                         ["return a < b && b == 0 // a > b", "return a < b || b != 0 // a > b",
                          "return a <= b && b != 0 // a > b"])
        self.assertEqual({m["function"] for m in got}, {"Fits"})
        self.assertEqual({m["package"] for m in got}, {"./core"})

    def changed_list(self, edit):
        root = self.repo([self.mutant("A"), self.mutant("O", file="other.py", old="X = 1", new="X = 2")])
        edit(root)
        subprocess.run(["git", "-C", root, "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false",
                        "commit", "-qam", "change"], check=True, capture_output=True)
        r = self.sweep(root, "--changed", "HEAD~1", "--list")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return {line.split()[1].split(":")[0] if line.startswith("generated") else line.split()[1]
                for line in r.stdout.splitlines() if line.startswith(("listed", "generated"))}, r.stdout

    def test_a_pull_request_sweeps_what_it_changed(self):
        def edit(path, old, new):
            def f(root):
                with open(os.path.join(root, path)) as h:
                    text = h.read()
                with open(os.path.join(root, path), "w") as h:
                    h.write(text.replace(old, new))
            return f
        for name, change, want in [
                ("a changed file", edit("lib.py", "return 1", "return 3"), {"A"}),
                ("a changed test", edit("check.py", "lib.positive(1)", "lib.positive(2)"), {"A", "O"}),
                ("a changed ported function", edit("core/x.go", "b != 0 //", "b != 1 //"), {"core/x.go"}),
                ("an unchanged ported function", edit("core/x.go", "\"a >= b\"", "\"a\""), set())]:
            with self.subTest(name):
                got, out = self.changed_list(change)
                self.assertEqual(got, want, out)


class Runner(unittest.TestCase):
    """oracle/run_scheduler.py's own guards."""

    def test_it_refuses_to_write_a_golden_without_the_flag(self):
        env = {k: v for k, v in os.environ.items() if k != "BLIS_SGLANG_ALLOW_GOLDEN_UPDATE"}
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_scheduler.py"), "nowhere",
                            os.path.join(HERE, "scenarios", "tiny.json"), os.path.join(HERE, "testdata/golden/x.json")],
                           cwd=HERE, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("refusing to write a golden trace", r.stderr)

    def result(self, **over):
        fwd = {"loop": {"function": "event_loop_normal", "file": "python/sglang/srt/managers/scheduler.py",
                        "upstream_code": True}}
        res = {"steps": [{"step": 0, "forwards": [fwd]}], "finished": {"r0": {}}, "shutdown_sent": True,
               "truncated": False, "classes": {"tree_core": "UnifiedTreeCore"}, "harness_problems": []}
        res.update(over)
        return res

    SCENARIO = {"requests": [{"rid": "r0"}], "overlap": False, "max_steps": 8}

    def test_a_good_result_has_no_problem(self):
        rs = import_oracle("run_scheduler")
        self.assertEqual(rs.trace_problems(self.result(), self.SCENARIO), [])

    def test_each_guard_fails_on_what_it_guards(self):
        rs = import_oracle("run_scheduler")
        cases = [
            ("no loop", {"steps": [{"step": 0, "forwards": [{"loop": {"function": None}}]}]}, "not from SGLang's"),
            ("a copy", {"steps": [{"step": 0, "forwards": [{"loop": {"function": "event_loop_normal",
                       "file": "python/sglang/srt/managers/scheduler.py", "upstream_code": False}}]}]}, "not from SGLang's"),
            ("the other loop", {"steps": [{"step": 0, "forwards": [{"loop": {"function": "event_loop_overlap",
                       "file": "python/sglang/srt/managers/scheduler.py", "upstream_code": True}}]}]}, "not from SGLang's"),
            ("no forward", {"steps": [{"step": 0, "forwards": []}]}, "no forward pass ran"),
            ("no shutdown", {"shutdown_sent": False}, "before the harness sent its ShutdownReq"),
            ("truncated", {"truncated": True}, "did not finish within max_steps"),
            ("unfinished", {"finished": {}}, "expected ['r0']"),
            ("the Rust core", {"classes": {"tree_core": "RustUnifiedTreeCore"}}, "not the Python UnifiedTreeCore"),
            ("a harness problem", {"harness_problems": ["x reads the real clock"]}, "x reads the real clock"),
        ]
        for name, over, text in cases:
            with self.subTest(name):
                got = rs.trace_problems(self.result(**over), self.SCENARIO)
                self.assertTrue(any(text in p for p in got), got)


class Harness(unittest.TestCase):
    """oracle/harness.py's guards that need no SGLang import."""

    def test_a_loop_is_upstream_only_at_the_checkouts_definition(self):
        h = import_oracle("harness")
        top = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, top)
        path = os.path.join(top, "python/sglang/srt/managers/scheduler.py")
        os.makedirs(os.path.dirname(path))
        src = "class Scheduler:\n    @staticmethod\n    def event_loop_normal(run, record):\n        record.append(record_loop(run))\n"
        with open(path, "w") as f:
            f.write(src)
        run_ = h.Run({"requests": [], "max_steps": 1}, top)
        for name, filename, want in [("the checkout's", path, True), ("a copy elsewhere", os.path.join(top, "copy.py"), False)]:
            with self.subTest(name):
                scope = {"record_loop": h.running_loop}
                exec(compile(src, filename, "exec"), scope)
                got = []
                scope["Scheduler"].event_loop_normal(run_, got)
                self.assertEqual(got[0]["function"], "event_loop_normal")
                self.assertIs(got[0]["upstream_code"], want)

    def test_arrivals_left_when_an_iteration_ends_are_a_problem(self):
        h = import_oracle("harness")

        class Two(h.Run):
            def begin_step(self):
                self.step += 1
                return ["a", "b"]

        run_ = Two({"requests": [], "max_steps": 9}, HERE)
        tok = h.TokenizerSocket(run_)
        rpc = h.SilentSocket(tok)
        self.assertEqual(tok.recv_pyobj(), "a")
        try:
            rpc.recv_pyobj()
        except Exception:
            pass  # zmq.Again, or no zmq here: the iteration has ended either way
        self.assertEqual(tok.recv_pyobj(), "a")
        self.assertTrue(any("left undelivered" in p for p in run_.problems), run_.problems)

    def test_a_clocked_module_on_the_real_clock_is_found_and_replaced(self):
        h = import_oracle("harness")
        import time as real
        import types as ty
        mod = ty.ModuleType("sglang.srt.mem_cache.scratch_clock")
        mod.time = real
        mod.monotonic = real.monotonic
        sys.modules[mod.__name__] = mod
        self.addCleanup(sys.modules.pop, mod.__name__)
        found = [p for p in h.clock_problems() if "scratch_clock still reads the real clock through" in p]
        self.assertEqual(len(found), 1, h.clock_problems())
        self.assertIn("monotonic", found[0])
        self.assertIn("time", found[0])
        run_ = h.Run({"requests": [], "max_steps": 1}, HERE)
        h.install_clock(h.VirtualTime(run_, 0.5))
        self.assertFalse(any("scratch_clock" in p for p in h.clock_problems()))
        self.assertEqual(mod.monotonic(), 1_000_000.0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__.split("Usage:", 1)[1].strip())
    SGLANG = os.path.abspath(sys.argv.pop(1))
    unittest.main()
