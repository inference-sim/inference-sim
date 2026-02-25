# H28: Chunked Prefill Improves ITL for Concurrent Decode Requests

**Status**: Refuted
**Date**: 2026-02-25

## Hypothesis

> Enabling chunked prefill (--long-prefill-token-threshold=512) improves mean ITL by >15% for concurrent decode requests when large-input (2048-token) prefills are present, at the cost of >20% higher TTFT for those large-input requests compared to disabled chunking (threshold=0).

**Refuted if:** Mean ITL improvement for concurrent decode requests is <10% or TTFT increase for large-input requests is <10% when comparing threshold=512 vs threshold=0 under mixed prefill-decode workload with 4+ concurrent requests.
