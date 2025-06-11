// Tracks simulation-wide and per-request performance metrics such as:

package sim

import "fmt"

// Metrics aggregates statistics about the simulation
// for final reporting. Useful for evaluating system performance
// and debugging behavior over time.
type Metrics struct {
	CompletedRequests int   // Number of requests completed
	TotalOutputTokens int   // Total number of output tokens
	TotalLatency      int64 // Sum of total latencies (completion - arrival)
	KVBlocksUsed      int   // Integral of KVBlockUsage over time
	PeakKVBlocksUsed  int   // Max number of simultaneously used KV blocks

	TTFTSum int64 // Total time-to-first-token sum (in ticks)
	TPOTSum int64 // Total TPOT sum across requests (in ticks)

	RequestLatencies map[string]int64 // map of request ID -> latency
}

// Print displays aggregated metrics at the end of the simulation.
// Includes average latency, TTFT, TPOT, KV usage, and prefix cache behavior.
func (m *Metrics) Print(step int64) {
	fmt.Println("=== Simulation Metrics ===")
	fmt.Printf("Completed Requests   : %d\n", m.CompletedRequests)
	if m.CompletedRequests > 0 {
		avgLatency := float64(m.TotalLatency) / float64(m.CompletedRequests)
		avgTTFT := float64(m.TTFTSum) / float64(m.CompletedRequests)
		avgTPOT := float64(m.TPOTSum) / float64(m.CompletedRequests)

		fmt.Printf("Average Latency      : %.2f ticks\n", avgLatency)
		fmt.Printf("Average TTFT         : %.2f ticks\n", avgTTFT)
		fmt.Printf("Average TPOT         : %.2f ticks\n", avgTPOT)
		fmt.Printf("Average KV Blocks Usage : %.2f\n", float64(m.KVBlocksUsed)/float64(step))
		fmt.Printf("Peak KV Usage        : %d blocks\n", m.PeakKVBlocksUsed)
	}
}
