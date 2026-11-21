# Original to blis-sglang.
"""The inertness verdict (oracle/inertness.py) on observed runs: what it takes for a field to be proven
inert, and that each part of it can fail. Standard library only; no SGLang is run.

Usage: python3 -I oracle/test_inertness.py
"""
import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inertness  # noqa: E402

A = "python/sglang/srt/managers/scheduler.py:508"
B = "python/sglang/srt/managers/scheduler_components/output_streamer.py:189"
C = "python/sglang/srt/managers/scheduler_components/output_streamer.py:456"
READS = [{"at": A, "to": 508, "expr": "get_serving().stream_interval"},
         {"at": B, "to": 190, "expr": "get_serving().stream_interval"}]
CONSUMERS = [{"at": C, "expr": "self.default_stream_interval"}]
PROBES = [1, 10]


def run(scenario, stream, value, digest, executed=(A, B), consumers=(C,)):
    return {"scenario": scenario, "stream": stream, "value": value, "digest": digest,
            "executed": list(executed), "consumers": list(consumers) if stream else []}


def good():
    """Two scenarios, each run non-streaming (the whole trace is the golden's) and streaming (the frames
    projection), at the default and at each probe."""
    runs = []
    for sc in ("s1", "s2"):
        for stream, d in ((False, "whole-" + sc), (True, "frames-" + sc)):
            for v in (None, 1, 10):
                runs.append(run(sc, stream, v, d))
    return {"golden": {"s1": "whole-s1", "s2": "whole-s2"}, "runs": runs}


def problems(observed, reads=READS, scanned=READS):
    return inertness.problems(reads, scanned, CONSUMERS, PROBES, observed)


class Verdict(unittest.TestCase):
    def test_runs_that_read_the_field_and_agree_prove_it_inert(self):
        self.assertEqual(problems(good()), [])

    def test_a_run_whose_result_differs_at_a_probe_is_refused(self):
        for i in range(len(good()["runs"])):
            o = good()
            o["runs"][i]["digest"] += "x"
            self.assertTrue(problems(o), "run %d differing was not found" % i)

    def test_a_non_streaming_trace_that_is_not_the_golden_is_refused(self):
        o = good()
        o["golden"]["s2"] = "other"
        self.assertTrue(any("golden" in p for p in problems(o)))

    def test_a_read_no_run_executed_is_refused(self):
        o = good()
        for r in o["runs"]:
            r["executed"] = [A]
        self.assertTrue(any(B in p for p in problems(o)))

    def test_a_probe_value_at_which_no_run_executed_a_read_is_refused(self):
        o = good()
        for r in o["runs"]:
            if r["value"] == 10:
                r["executed"] = []
        self.assertTrue(any("10" in p for p in problems(o)))

    def test_a_consumer_no_streaming_run_executed_is_refused(self):
        o = good()
        for r in o["runs"]:
            r["consumers"] = []
        self.assertTrue(any(C in p for p in problems(o)))

    def test_a_proof_whose_reads_are_not_what_the_scan_finds_is_refused(self):
        moved = copy.deepcopy(READS)
        moved[0]["at"] = moved[0]["at"].replace(":508", ":509")
        self.assertTrue(problems(good(), reads=moved))
        self.assertTrue(problems(good(), reads=READS[:1]))

    def test_a_run_set_missing_a_probe_is_refused(self):
        o = good()
        o["runs"] = [r for r in o["runs"] if r["value"] != 10]
        self.assertTrue(problems(o))


class Projection(unittest.TestCase):
    def trace(self, outputs):
        return {"provenance": {"scenario_sha256": "x"}, "config": {"a": 1, "inertness": {"executed": []}},
                "finished": {"q": {"step": 2}},
                "steps": [{"n": 0, "outputs": outputs[0]}, {"n": 1, "outputs": outputs[1]}]}

    def frame(self, lens, fin=()):
        return [{"type": "BatchTokenIDOutput", "rids": ["q"] * len(lens), "output_lens": lens, "finished": list(fin)}]

    def test_the_whole_projection_drops_the_provenance_and_the_coverage_only(self):
        t = self.trace([self.frame([1]), self.frame([1], ["q"])])
        p = inertness.projection(t, False)
        self.assertNotIn("provenance", p)
        self.assertEqual(p["config"], {"a": 1})
        self.assertIn("outputs", p["steps"][0])

    def test_the_frames_projection_keeps_what_was_delivered_and_when_it_finished_not_how_it_was_split(self):
        one = self.trace([self.frame([1]), self.frame([1], ["q"])])
        two = self.trace([[], self.frame([2], ["q"])])
        self.assertEqual(inertness.digest(inertness.projection(one, True)),
                         inertness.digest(inertness.projection(two, True)))

    def test_the_frames_projection_sees_a_token_delivered_less_or_a_finish_that_moved(self):
        base = inertness.digest(inertness.projection(self.trace([self.frame([1]), self.frame([1], ["q"])]), True))
        short = self.trace([self.frame([1]), self.frame([2], ["q"])])
        late = self.trace([self.frame([1]), self.frame([1])])
        late["steps"].append({"n": 2, "outputs": self.frame([0], ["q"])})
        for t in (short, late):
            self.assertNotEqual(base, inertness.digest(inertness.projection(t, True)))

    def test_the_frames_projection_still_compares_the_scheduling(self):
        a = self.trace([self.frame([1]), self.frame([1], ["q"])])
        b = copy.deepcopy(a)
        b["steps"][1]["n"] = 7
        self.assertNotEqual(inertness.digest(inertness.projection(a, True)), inertness.digest(inertness.projection(b, True)))

    def test_the_digest_does_not_depend_on_key_order(self):
        self.assertEqual(inertness.digest({"a": 1, "b": [1, 2]}), inertness.digest({"b": [1, 2], "a": 1}))


if __name__ == "__main__":
    unittest.main()
