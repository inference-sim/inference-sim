# Original to blis-sglang.
"""The oracle environment's check: SGLang's scheduler module, read from a checkout of the pinned
commit, imports in the locked environment on CPU, with no module stubbed but those
oracle/stubs.py lists.

It imports sglang.srt.managers.scheduler, the module every oracle runner drives, and fails if
the import raises or if the Scheduler class and its event loops are not there. A CUDA-only
package the lock leaves out (oracle/requirements.txt) that the import reaches fails it, unless
oracle/stubs.py stubs it and oracle/mocks.json lists the stub.

Usage: python -I oracle/run_import.py <sglang checkout>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    source = os.path.realpath(os.path.join(sys.argv[1], "python"))
    sys.path.insert(0, source)
    sys.path.insert(0, HERE)
    import json

    import stubs

    with open(os.path.join(HERE, "mocks.json")) as f:
        unlisted = stubs.problems(stubs.STUBBED, json.load(f))
    if unlisted:
        print("\n".join(unlisted), file=sys.stderr)
        return 1
    stubs.install()
    import sglang.srt.managers.scheduler as scheduler_module

    # SGL-1: the code imported is the checkout's, not an installed sglang.
    where = os.path.realpath(scheduler_module.__file__)
    if not where.startswith(source + os.sep):
        print(f"sglang.srt.managers.scheduler was imported from {where}, not from {source}", file=sys.stderr)
        return 1

    missing = [name for name in ("event_loop_normal", "event_loop_overlap", "get_next_batch_to_run",
                                 "process_batch_result", "ingest_requests")
               if not callable(getattr(getattr(scheduler_module, "Scheduler", None), name, None))]
    if missing:
        print(f"sglang.srt.managers.scheduler.Scheduler has no {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"imported {scheduler_module.__file__}; stubbed: {', '.join(stubs.STUBBED) or 'nothing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
