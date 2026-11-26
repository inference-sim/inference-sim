# Original to blis-sglang.
"""Tests for forkpool.py, the runner run_serverargs.py and lower/extract.py fork their cases with:
the answers come back in case order whatever order the children finish in, no more than n run at
once, each case runs in a process of its own (so one case's state never reaches another), a child
that fails or gives no answer is reported at its own position, and an answer larger than a pipe's
buffer comes back whole.

Usage: python3 -I oracle/test_forkpool.py   (stdlib only)
"""
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import forkpool  # noqa: E402

LEAK = []  # a case that appends here must not be seen by a later case


class ForkpoolTest(unittest.TestCase):
    def test_answers_are_in_case_order_whatever_order_the_children_end_in(self):
        cases = [{"name": "c%d" % i, "sleep": (5 - i) * 0.05} for i in range(6)]

        def child(case):
            time.sleep(case["sleep"])
            return {"name": case["name"]}

        got = [a["name"] for a in forkpool.run(cases, child, n=6)]
        self.assertEqual(got, [c["name"] for c in cases])

    def test_no_more_than_n_run_at_once_and_n_do(self):
        with tempfile.TemporaryDirectory() as d:
            def child(case):
                mark = os.path.join(d, case["name"])
                open(mark, "w").close()
                seen = len(os.listdir(d))
                time.sleep(0.2)
                os.remove(mark)
                return {"name": case["name"], "seen": seen}

            answers = forkpool.run([{"name": "c%d" % i} for i in range(8)], child, n=3)
        most = max(a["seen"] for a in answers)
        self.assertLessEqual(most, 3)
        self.assertEqual(most, 3, "three children were allowed and fewer ever ran together")

    def test_each_case_runs_in_a_process_of_its_own(self):
        def child(case):
            before = list(LEAK)
            LEAK.append(case["name"])
            return {"name": case["name"], "pid": os.getpid(), "before": before}

        answers = forkpool.run([{"name": "c%d" % i} for i in range(4)], child, n=1)
        pids = [a["pid"] for a in answers]
        self.assertEqual(len(set(pids)), 4)
        self.assertNotIn(os.getpid(), pids)
        for a in answers:
            self.assertEqual(a["before"], [], "%s saw an earlier case's state" % a["name"])
        self.assertEqual(LEAK, [])

    def test_a_failing_child_is_reported_at_its_own_position(self):
        def child(case):
            if case["name"] == "raises":
                raise RuntimeError("broken harness")
            if case["name"] == "silent":
                os._exit(0)
            return {"name": case["name"], "ok": True}

        cases = [{"name": n} for n in ("a", "raises", "b", "silent", "c")]
        answers = forkpool.run(cases, child, n=2)
        self.assertEqual([a["name"] for a in answers], ["a", "raises", "b", "silent", "c"])
        self.assertIn("RuntimeError: broken harness", answers[1]["harness_error"])
        self.assertEqual(answers[3]["harness_error"], "no answer")
        for i in (0, 2, 4):
            self.assertTrue(answers[i]["ok"])

    def test_an_answer_larger_than_a_pipe_buffer_comes_back_whole(self):
        big = "x" * (1 << 20)
        answers = forkpool.run([{"name": "big"}, {"name": "small"}], lambda c: {"name": c["name"], "v": big}, n=2)
        self.assertEqual([len(a["v"]) for a in answers], [len(big), len(big)])

    def test_no_cases_is_no_answers(self):
        self.assertEqual(forkpool.run([], lambda c: {}), [])

    def test_jobs_reads_oracle_jobs_or_what_the_job_can_afford(self):
        with mock.patch.dict(os.environ, {"ORACLE_JOBS": "3"}):
            self.assertEqual(forkpool.jobs(), 3)
        env = {k: v for k, v in os.environ.items() if k != "ORACLE_JOBS"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(forkpool, "affordable", return_value=5):
            self.assertEqual(forkpool.jobs(), 5)
        for bad in ("0", "-1", "x", "2.5"):
            with mock.patch.dict(os.environ, {"ORACLE_JOBS": bad}):
                with self.assertRaises(SystemExit, msg="ORACLE_JOBS=%r" % bad):
                    forkpool.jobs()

    def test_cpus_is_the_cgroup_quota_not_the_nodes_count(self):
        # A Kubernetes CPU limit is a quota: the node may have 64 CPUs and the pod 4.
        with mock.patch.object(os, "sched_getaffinity", return_value=set(range(64)), create=True):
            for files, want in (
                ({"cpu.max": "400000 100000\n"}, 4),                       # cgroup v2, 4 CPUs
                ({"cpu.max": "250000 100000\n"}, 3),                       # 2.5 CPUs, rounded up
                ({"cpu.max": "50000 100000\n"}, 1),                        # half a CPU is one
                ({"cpu.max": "max 100000\n"}, 64),                         # no limit
                ({"cpu/cpu.cfs_quota_us": "200000\n", "cpu/cpu.cfs_period_us": "100000\n"}, 2),  # v1
                ({"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, 64),     # v1, none
                ({}, 64),                                                   # no cgroup files
            ):
                with tempfile.TemporaryDirectory() as d:
                    for name, body in files.items():
                        os.makedirs(os.path.dirname(os.path.join(d, name)), exist_ok=True)
                        with open(os.path.join(d, name), "w") as f:
                            f.write(body)
                    self.assertEqual(forkpool.cpus(d), want, files)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "cpu.max"), "w") as f:
                f.write("1600000 100000\n")
            with mock.patch.object(os, "sched_getaffinity", return_value={0, 1}, create=True):
                self.assertEqual(forkpool.cpus(d), 2, "the affinity is fewer than the quota")

    def test_the_command_prints_cpus_with_the_flag_and_affordable_without(self):
        with mock.patch.object(forkpool, "cpus", return_value=4), \
                mock.patch.object(forkpool, "affordable", return_value=3), \
                mock.patch("builtins.print") as out:
            self.assertEqual(forkpool.main(["--cpus"]), 0)
            self.assertEqual(forkpool.main([]), 0)
            self.assertEqual([c.args for c in out.call_args_list], [(4,), (3,)])
            self.assertEqual(forkpool.main(["--jobs"]), 2)

    def test_the_memory_limit_caps_the_children(self):
        # A pod with many CPUs and little memory forks no more children than its memory holds.
        gib = 1 << 30
        env = {k: v for k, v in os.environ.items() if k != "ORACLE_CHILD_MIB"}
        with mock.patch.object(os, "sched_getaffinity", return_value=set(range(64)), create=True):
            for files, extra, want in (
                ({"cpu.max": "1600000 100000\n", "memory.max": "%d\n" % (16 * gib)}, {}, 4),   # 16 CPUs, 16 GiB
                ({"cpu.max": "400000 100000\n", "memory.max": "%d\n" % (64 * gib)}, {}, 4),    # the CPUs are fewer
                ({"cpu.max": "1600000 100000\n", "memory.max": "max\n"}, {}, 16),               # no memory limit
                ({"cpu.max": "1600000 100000\n", "memory.max": "%d\n" % gib}, {}, 1),          # less than one child
                ({"memory/memory.limit_in_bytes": "%d\n" % (8 * gib)}, {}, 2),                  # v1
                ({"memory/memory.limit_in_bytes": "%d\n" % (1 << 63)}, {}, 64),                 # v1, none
                ({"cpu.max": "1600000 100000\n", "memory.max": "%d\n" % (16 * gib)}, {"ORACLE_CHILD_MIB": "2048"}, 8),
            ):
                with tempfile.TemporaryDirectory() as d:
                    for name, body in files.items():
                        os.makedirs(os.path.dirname(os.path.join(d, name)), exist_ok=True)
                        with open(os.path.join(d, name), "w") as f:
                            f.write(body)
                    with mock.patch.dict(os.environ, dict(env, **extra), clear=True):
                        self.assertEqual(forkpool.affordable(d), want, (files, extra))
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "memory.max"), "w") as f:
                f.write("%d\n" % (16 * gib))
            for bad in ("0", "x", "-1"):
                with mock.patch.dict(os.environ, {"ORACLE_CHILD_MIB": bad}):
                    with self.assertRaises(SystemExit, msg="ORACLE_CHILD_MIB=%r" % bad):
                        forkpool.affordable(d)


if __name__ == "__main__":
    unittest.main()
