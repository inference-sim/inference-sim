// metrics.go
//
// Tracks simulation-wide and per-request performance metrics such as:
// - Latency
// - Time To First Token (TTFT)
// - Time Per Output Token (TPOT)
// - KV cache usage stats
// - Prefix cache reuse

package sim

import "fmt"

// Metrics aggregates statistics about the simulation
// for final reporting. Useful for evaluating system performance
// and debugging behavior over time.
type Metrics struct {
	CompletedRequests int   // Number of requests completed
	TotalLatency      int64 // Sum of total latencies (completion - arrival)
	KVBlocksUsed      int   // Cumulative number of KV blocks used
	PeakKVBlocksUsed  int   // Max number of simultaneously used KV blocks
	PrefixCacheHits   int   // Count of prefix reuse operations

	TTFTSum int64   // Total time-to-first-token sum
	TPOTSum float64 // Total TPOT sum across requests

	RequestLatencies map[string]int64 // Optional: map of request ID -> latency
}

// Print displays aggregated metrics at the end of the simulation.
// Includes average latency, TTFT, TPOT, KV usage, and prefix cache behavior.
func (m *Metrics) Print() {
	fmt.Println("=== Simulation Metrics ===")
	fmt.Printf("Completed Requests   : %d\n", m.CompletedRequests)
	if m.CompletedRequests > 0 {
		avgLatency := float64(m.TotalLatency) / float64(m.CompletedRequests)
		avgTTFT := float64(m.TTFTSum) / float64(m.CompletedRequests)
		avgTPOT := m.TPOTSum / float64(m.CompletedRequests)

		fmt.Printf("Average Latency      : %.2f ticks\n", avgLatency)
		fmt.Printf("Average TTFT         : %.2f ticks\n", avgTTFT)
		fmt.Printf("Average TPOT         : %.2f ticks\n", avgTPOT)
		fmt.Printf("Total KV Blocks Used : %d\n", m.KVBlocksUsed)
		fmt.Printf("Peak KV Usage        : %d blocks\n", m.PeakKVBlocksUsed)
		fmt.Printf("Prefix Cache Hits    : %d\n", m.PrefixCacheHits)
	}
}
