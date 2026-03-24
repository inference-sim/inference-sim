# Hypothesis Catalog

| ID | Title | Status | Date |
|----|-------|--------|------|
| H1 | [Arrival Burstiness Elevates TTFT and E2E Latency at Equal Throughput](h1-arrival-burstiness/HYPOTHESIS.md) | **Confirmed** | 2026-03-24 |

## Summary of H1 Findings

Bursty (Gamma CV=3) arrivals produce **4–5× higher TTFT p99** and **3–4× higher TTFT mean**
compared to smooth (Poisson CV=1) arrivals at equivalent total throughput, across all tested
utilization levels (ρ = 0.22–0.93). Scheduling delay p99 is 8–9× higher, confirming queueing
as the root cause. Effect magnitude exceeds Kingman's G/G/1 prediction (3.46×) due to
multi-modal LLM service time distribution. TTFT mean ratio grows monotonically with ρ.

See [FINDINGS.md](h1-arrival-burstiness/FINDINGS.md) for full details.
