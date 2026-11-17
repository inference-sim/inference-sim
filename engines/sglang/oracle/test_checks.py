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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__.split("Usage:", 1)[1].strip())
    SGLANG = os.path.abspath(sys.argv.pop(1))
    unittest.main()
