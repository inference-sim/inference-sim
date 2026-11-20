"""export_kernel_impls.py: each engine's kernel-implementation table, as the registry's engine-neutral input."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_kernel_impls as exp  # noqa: E402

TABLE = {"kind": "KernelImplTable",
         "engine": {"kind": "e", "version": "1.0", "commit": "c" * 40},
         "primitives": ["attention.decode"], "archs": ["sm_90a"], "gaps": [], "libraries": [],
         "implementations": [{"primitive": "attention.decode", "code_hash": "1" * 64}]}


class ExportTests(unittest.TestCase):
    def test_the_copy_is_the_table_as_it_stands(self):
        docs = exp.export({"e": TABLE})
        self.assertEqual(list(docs), ["kernel_impls.e.json"])
        self.assertEqual(json.loads(docs["kernel_impls.e.json"]), TABLE)
        self.assertTrue(docs["kernel_impls.e.json"].endswith("}\n"))

    def test_a_table_that_is_not_one_or_is_another_engines_is_refused(self):
        for bad in (dict(TABLE, kind="CodePathTable"),
                    dict(TABLE, engine=dict(TABLE["engine"], kind="other")),
                    {k: v for k, v in TABLE.items() if k != "implementations"}):
            with self.assertRaises(SystemExit):
                exp.export({"e": bad})

    def test_check_fails_on_a_stale_copy_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            t, out = os.path.join(d, "t.json"), os.path.join(d, "out")
            os.mkdir(out)
            json.dump(TABLE, open(t, "w"))
            run = lambda *a: subprocess.run([sys.executable, "-I", os.path.join(HERE, "export_kernel_impls.py"),
                                             "--out-dir", out, *a], capture_output=True, text=True)
            self.assertEqual(run("--check", f"e={t}").returncode, 1)   # absent is stale
            self.assertEqual(os.listdir(out), [])
            self.assertEqual(run(f"e={t}").returncode, 0)
            self.assertEqual(run("--check", f"e={t}").returncode, 0)
            changed = copy.deepcopy(TABLE)
            changed["implementations"][0]["code_hash"] = "2" * 64
            json.dump(changed, open(t, "w"))
            self.assertEqual(run("--check", f"e={t}").returncode, 1)
            self.assertEqual(json.load(open(os.path.join(out, "kernel_impls.e.json"))), TABLE)


if __name__ == "__main__":
    unittest.main()
