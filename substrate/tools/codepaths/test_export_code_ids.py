"""export_code_ids.py: the CodeIDs engines state, as the registry's engine-neutral input."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_code_ids as exp  # noqa: E402

TABLE = {"kind": "CodePathTable",
         "engine": {"kind": "e", "version": "1.0", "commit": "c" * 40},
         "paths": [{"path": "host", "code_id": "1" * 63 + "a"}, {"path": "graphs", "code_id": "2" * 63 + "b"}]}


class ExportTests(unittest.TestCase):
    def test_the_code_ids_follow_the_tables(self):
        other = copy.deepcopy(TABLE)
        other["paths"][0]["code_id"] = "3" * 63 + "c"
        doc = exp.export({"b": other, "a": TABLE})
        self.assertEqual(list(doc["engines"]), ["a", "b"])
        self.assertEqual(doc["engines"]["a"]["host"], "1" * 63 + "a")
        self.assertEqual(doc["engines"]["b"]["host"], "3" * 63 + "c")
        self.assertEqual(doc["engines"]["b"]["graphs"], "2" * 63 + "b")

    def test_a_table_that_omits_a_path_or_is_not_a_table_is_refused(self):
        short = copy.deepcopy(TABLE)
        short["paths"].pop()
        with self.assertRaises(SystemExit):
            exp.export({"a": short})
        wrong = dict(TABLE, kind="Other")
        with self.assertRaises(SystemExit):
            exp.export({"a": wrong})

    def test_check_fails_on_a_stale_file_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            t, out = os.path.join(d, "t.json"), os.path.join(d, "out.json")
            json.dump(TABLE, open(t, "w"))
            run = lambda *a: subprocess.run([sys.executable, "-I", os.path.join(HERE, "export_code_ids.py"),
                                             "--out", out, *a], capture_output=True, text=True)
            self.assertEqual(run("--check", f"e={t}").returncode, 1)   # absent is stale
            self.assertFalse(os.path.exists(out))
            self.assertEqual(run(f"e={t}").returncode, 0)
            self.assertEqual(run("--check", f"e={t}").returncode, 0)
            changed = copy.deepcopy(TABLE)
            changed["paths"][0]["code_id"] = "9" * 64
            json.dump(changed, open(t, "w"))
            self.assertEqual(run("--check", f"e={t}").returncode, 1)


if __name__ == "__main__":
    unittest.main()
