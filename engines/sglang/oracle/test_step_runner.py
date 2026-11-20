# Original to blis-sglang.
"""The step-driven runner (oracle/step_runner.py) and the step hook it hangs on
(oracle/harness.py Run.on_step) do what the manifest says: a thread body runs one iteration of its
own loop at each step boundary, in start order; a device event completes at the boundary after its
record; and the boundary falls at the first receive of each event-loop iteration, before that
iteration's arrivals. Standard library only; no SGLang is imported.

Usage: python3 -I oracle/test_step_runner.py
"""
import json
import os
import queue
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness  # noqa: E402
import step_runner  # noqa: E402


def fresh_runner():
    step_runner.RUNNER.__init__()
    return step_runner.RUNNER


def body(stop, q, seen, name):
    """A thread body of HiCache's shape (managers/cache_controller.py:1446-1461 at SGLang
    297e54fd): loop until stopped, taking one item per iteration with a timed get."""
    def run():
        while not stop.is_set():
            try:
                item = q.get(block=True, timeout=1)
            except queue.Empty:
                continue
            seen.append((name, item))
    return run


class Runner(unittest.TestCase):
    def test_a_started_thread_runs_only_at_a_boundary_one_iteration_each(self):
        runner = fresh_runner()
        stop, q, seen = step_runner.StepEvent(), step_runner.StepQueue(), []
        for i in range(3):
            q.put(i)
        t = step_runner.StepThread(target=body(stop, q, seen, "b"), daemon=True)
        t.start()
        self.assertEqual(seen, [], "a start runs nothing")
        runner.boundary()
        self.assertEqual(seen, [("b", 0)])
        runner.boundary()
        self.assertEqual(seen, [("b", 0), ("b", 1)])
        self.assertEqual(runner.boundaries, 2)

    def test_threads_run_in_start_order_and_a_joined_thread_stops(self):
        runner = fresh_runner()
        stop, seen = step_runner.StepEvent(), []
        qa, qb = step_runner.StepQueue(), step_runner.StepQueue()
        qa.put("x")
        qb.put("y")
        a = step_runner.StepThread(target=body(stop, qa, seen, "a"))
        b = step_runner.StepThread(target=body(stop, qb, seen, "b"))
        b.start()
        a.start()
        runner.boundary()
        self.assertEqual(seen, [("b", "y"), ("a", "x")])
        a.join()
        self.assertFalse(a.is_alive())
        qa.put("z")
        runner.boundary()
        self.assertEqual(seen, [("b", "y"), ("a", "x")], "a joined thread no longer runs")

    def test_an_empty_queue_never_waits_and_a_set_stop_event_holds(self):
        runner = fresh_runner()
        stop, q, seen = step_runner.StepEvent(), step_runner.StepQueue(), []
        step_runner.StepThread(target=body(stop, q, seen, "b")).start()
        runner.boundary()  # an empty get raises Empty at once: no 1 s wait
        stop.set()
        q.put(1)
        runner.boundary()
        self.assertEqual(seen, [], "a body whose stop event is set does not iterate")
        self.assertTrue(stop.is_set())

    def test_a_loop_that_drains_past_its_stop_event_drains(self):
        # prefetch_thread_func's condition (cache_controller.py:1337) goes on while its queue
        # holds work, as the runner's rule says it may.
        runner = fresh_runner()
        stop, q, seen = step_runner.StepEvent(), step_runner.StepQueue(), []

        def drain():
            while (not stop.is_set()) or not q.empty():
                try:
                    seen.append(q.get(block=True, timeout=1))
                except queue.Empty:
                    continue
        for i in range(3):
            q.put(i)
        step_runner.StepThread(target=drain).start()
        runner.boundary()
        self.assertEqual(seen, [0, 1, 2])

    def test_a_device_event_completes_at_the_next_boundary(self):
        runner = fresh_runner()
        e = step_runner.StepDeviceEvent()
        self.assertTrue(e.query(), "an event never recorded is complete")
        e.record()
        self.assertFalse(e.query())
        runner.boundary()
        self.assertTrue(e.query())
        e.record()
        self.assertFalse(e.query())
        e.synchronize()
        self.assertTrue(e.query(), "a synchronize completes it at once")

    def test_a_device_event_takes_cudas_keywords_and_times_zero(self):
        # l2_transfer.py:18-28 at SGLang 297e54fd probes Event(enable_timing=True); a TypeError
        # there turns timing off for every ack, which CUDA does not.
        fresh_runner()
        start = step_runner.StepDeviceEvent(enable_timing=True, blocking=False, interprocess=False)
        end = step_runner.StepDeviceEvent(enable_timing=True)
        start.record()
        end.record()
        self.assertEqual(start.elapsed_time(end), 0.0)

    def test_the_pp_ticket_thread_is_refused_by_name(self):
        fresh_runner()

        def pp_prefetch_command_thread_func():
            pass
        with self.assertRaises(NotImplementedError):
            step_runner.StepThread(target=pp_prefetch_command_thread_func).start()


class OneRule(unittest.TestCase):
    """The scheduling rule is stated once, in RULE: every manifest entry for a replacement the
    runner makes cites it rather than restating it."""

    def test_every_runner_entry_of_the_manifest_cites_rule(self):
        with open(os.path.join(HERE, "mocks.json")) as f:
            mocks = json.load(f)["mocks"]
        runner_entries = [e for e in mocks if "step_runner" in e["how"] or "run synchronously at step k" in e["how"]
                          or any(s.startswith("step_runner.py:") for s in e["sites"])]
        self.assertGreaterEqual(len(runner_entries), 3, "the manifest lists the runner's thread, queue and event entries")
        for e in runner_entries:
            self.assertIn("oracle/step_runner.py RULE", e["how"], e["symbol"])
            # A restatement would name the rule's own terms: the start order and the boundary an
            # event completes at.
            for term in ("in start order", "recorded after step boundary k-1", "raises Empty at once"):
                self.assertNotIn(term, e["how"], e["symbol"])


SCENARIO = {"requests": [{"rid": "r0", "arrival_step": 3, "prompt": [1]}], "max_steps": 10}


class StepHook(unittest.TestCase):
    def test_on_step_runs_at_the_first_receive_of_each_iteration_before_its_record(self):
        calls = []
        run = harness.Run(SCENARIO, HERE, on_step=lambda: calls.append((run.step, len(run.steps))))
        for _ in range(3):
            run.begin_step()
        # Called once per iteration, after the step count advanced and before the iteration's
        # record was started: boundary k precedes everything iteration k does.
        self.assertEqual(calls, [(0, 0), (1, 1), (2, 2)])

    def test_with_no_hook_an_iteration_is_recorded_as_before(self):
        a, b = harness.Run(SCENARIO, HERE), harness.Run(SCENARIO, HERE, on_step=lambda: None)
        for _ in range(3):
            self.assertEqual(a.begin_step(), b.begin_step())
        self.assertEqual(a.steps, b.steps)


if __name__ == "__main__":
    unittest.main()
