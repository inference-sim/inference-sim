// Tracks simulation-wide and per-request performance metrics such as:

package sim

import (
	"fmt"
	"time"
)

// Metrics aggregates statistics about the simulation
// for final reporting. Useful for evaluating system performance
// and debugging behavior over time.
type Metrics struct {
	CompletedRequests int     // Number of requests completed
	TotalInputTokens  int     // Total number of input tokens
	TotalOutputTokens int     // Total number of output tokens
	RequestRate       float64 // Incoming request rate
	TotalLatency      int64   // Sum of total latencies (completion - arrival)
	SimEndedTime      int64   // Sim clock time in ticks when simulation ends
	KVBlocksUsed      int     // Integral of KVBlockUsage over time
	PeakKVBlocksUsed  int     // Max number of simultaneously used KV blocks

	TTFTSum int64 // Total time-to-first-token sum (in ticks)
	TPOTSum int64 // Total TPOT sum across requests (in ticks)

	RequestTTFTs []float64 // list of all requests' TTFT
	RequestTPOTs []float64 // list of all requests' TPOT
}

// Print displays aggregated metrics at the end of the simulation.
// Includes average latency, TTFT, TPOT, KV usage, and prefix cache behavior.
func (m *Metrics) Print(horizon int64, totalBlocks int, startTime time.Time) {
	fmt.Println("=== Simulation Metrics ===")
	fmt.Printf("Completed Requests   : %d\n", m.CompletedRequests)
	fmt.Printf("Total Input Tokens   : %d\n", m.TotalInputTokens)
	fmt.Printf("Total Output Tokens  : %d\n", m.TotalOutputTokens)
	if m.CompletedRequests > 0 {
		avgTTFT := float64(m.TTFTSum) / float64(m.CompletedRequests)
		avgTPOT := float64(m.TPOTSum) / float64(m.TotalOutputTokens)

		fmt.Printf("Mean TTFT(ms)     : %.3f\n", avgTTFT/1000)
		fmt.Printf("Mean TPOT(ms)     : %.3f\n", avgTPOT/1000)
		fmt.Printf("Avg KV Blocks Usage : %.3f\n", float64(m.KVBlocksUsed)/float64(m.SimEndedTime))
		fmt.Printf("Peak KV Usage       : %d blocks\n", m.PeakKVBlocksUsed)
	}
}
