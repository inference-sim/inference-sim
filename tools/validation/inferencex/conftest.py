import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402

NEEDS_BEHAVIOUR = {"test_version_defaults_not_v031", "test_unresolvable_build_flagged",
                   "test_legacy_cuda_graph_sizes_is_a_ceiling_on_v0_10", "test_stream_interval_named_and_chunk_itl_not_scored",
                   "test_long_prefill_threshold_stated_and_unstateable_args_named"}


def pytest_collection_modifyitems(config, items):
    if PX.BEHAVIOUR.exists():
        return
    skip = pytest.mark.skip(reason=f"needs {PX.BEHAVIOUR} (build_table.py output; set IX_BEHAVIOUR)")
    for it in items:
        if it.name in NEEDS_BEHAVIOUR:
            it.add_marker(skip)
