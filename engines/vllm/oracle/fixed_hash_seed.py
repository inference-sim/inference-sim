# Original to blis-vllm.
"""Run an oracle script with the string and bytes hash seed fixed, so a trace does not depend on set order.

vLLM's offload scheduler keeps a job's keys in a set of OffloadKey (a bytes subclass,
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py, class TransferJob.keys) and hands the
set to OffloadingManager.complete_store and complete_load, which walk it. A bytes object hashes under the
interpreter's hash seed, so the walk order, and with it every manager call after it and the ids the policy
hands out, changes from one process to the next. A recorded trace of those calls then differs between two
runs of the same code on the same vLLM.

The oracle scripts run as `python -I`, and -I ignores PYTHONHASHSEED, so the seed cannot be set from outside.
fix_hash_seed() starts the same script again under a fixed seed, with every other PYTHON* variable removed
(what -I removes) and -s -P (what -I adds besides -E), and leaves the seed out of the environment the child's
code sees: vLLM derives NONE_HASH from PYTHONHASHSEED when it is set (resolve_none_hash_seed,
vllm/v1/core/kv_cache_utils.py), and a recorded key must not change with the way the script was started.
"""

import os
import subprocess
import sys

SEED = "0"


def fix_hash_seed(script: str, argv: list) -> None:
    """Return when the interpreter's hash randomization is already off; otherwise run `script argv` in a child with
    the seed fixed and exit with its status."""
    if not sys.flags.hash_randomization:
        os.environ.pop("PYTHONHASHSEED", None)
        return
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONHASHSEED"] = SEED
    child = subprocess.run([sys.executable, "-s", "-P", "-c", _BOOT, script, *argv], env=env)
    sys.exit(child.returncode)


# The child drops PYTHONHASHSEED from its environment once the interpreter has read it, then runs the script as __main__.
_BOOT = (
    "import os, runpy, sys\n"
    "os.environ.pop('PYTHONHASHSEED', None)\n"
    "script = sys.argv[1]\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(script, run_name='__main__')\n"
)
