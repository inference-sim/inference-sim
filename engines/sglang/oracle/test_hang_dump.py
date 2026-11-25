# Original to blis-sglang.
"""Tests for run_scheduler.arm_hang_dump: a scheduler process still running HANG_DUMP_LEAD_S before
JOIN_TIMEOUT_S prints every thread's stack, naming the call it is blocked in, and keeps running;
one that finishes first prints nothing.

Usage: python3 -I oracle/test_hang_dump.py   (stdlib only)
"""
import faulthandler
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_scheduler  # noqa: E402


def blocked_in_rendezvous(seconds: float) -> None:
    time.sleep(seconds)


def child(stderr_path: str, run_for: float, done_path: str) -> None:
    with open(stderr_path, "w") as f:
        os.dup2(f.fileno(), 2)
    with mock.patch.object(run_scheduler, "JOIN_TIMEOUT_S", 2), mock.patch.object(run_scheduler, "HANG_DUMP_LEAD_S", 1):
        run_scheduler.arm_hang_dump()
    blocked_in_rendezvous(run_for)
    faulthandler.cancel_dump_traceback_later()
    open(done_path, "w").close()


def run_child(run_for: float) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory() as tmp:
        stderr_path, done_path = os.path.join(tmp, "stderr"), os.path.join(tmp, "done")
        proc = multiprocessing.get_context("fork").Process(target=child, args=(stderr_path, run_for, done_path))
        proc.start()
        proc.join(30)
        if proc.is_alive():
            proc.kill()
            raise AssertionError("the child did not finish")
        with open(stderr_path) as f:
            return f.read(), os.path.exists(done_path)


class HangDumpTest(unittest.TestCase):
    def test_a_process_still_running_prints_the_call_it_is_blocked_in_and_keeps_running(self):
        stderr, finished = run_child(run_for=3)
        self.assertIn("blocked_in_rendezvous", stderr)
        self.assertTrue(finished, "the dump must not end the process")

    def test_a_process_that_finishes_first_prints_nothing(self):
        stderr, finished = run_child(run_for=0)
        self.assertEqual(stderr, "")
        self.assertTrue(finished)


if __name__ == "__main__":
    unittest.main()
