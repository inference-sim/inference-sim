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
	KVBlocksUsed      float64 // Integral of KVBlockUsage over time
	PeakKVBlocksUsed  int64   // Max number of simultaneously used KV blocks

	TTFTSum int64 // Total time-to-first-token sum (in ticks)
	TPOTSum int64 // Total TPOT sum across requests (in ticks)

	RequestTTFTs           map[string]float64 // list of all requests' TTFT
	RequestTPOTs           map[string]float64 // list of all requests' TPOT
	RequestE2Es            map[string]float64 // list of all requests' latencies
	RequestCompletionTimes map[string]float64 // list of all requests' completion times in ticks
	RequestStepCounters    []int              // list of all requests' num of steps between scheduled and finished

	NumWaitQRequests        []int // number of requests in waitQ over different steps
	NumRunningBatchRequests []int // number of request in runningBatch over different steps
}

func NewMetrics() *Metrics {
	return &Metrics{
		CompletedRequests:       0,
		RequestTTFTs:            make(map[string]float64),
		RequestTPOTs:            make(map[string]float64),
		RequestE2Es:             make(map[string]float64),
		RequestCompletionTimes:  make(map[string]float64),
		NumWaitQRequests:        []int{},
		NumRunningBatchRequests: []int{},
	}
}

// calculateMean computes the average of a slice of integers.
func calculateMean(numbers []int) float64 {
	if len(numbers) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, number := range numbers {
		sum += float64(number)
	}

	return sum / float64(len(numbers))
}

// Print displays aggregated metrics at the end of the simulation.
// Includes average latency, TTFT, TPOT, KV usage, and prefix cache behavior.
func (m *Metrics) Print(horizon int64, totalBlocks int64, startTime time.Time) {
	vllmRuntime := float64(m.SimEndedTime) / float64(1e6)
	fmt.Println("=== Simulation Metrics ===")
	fmt.Printf("Completed Requests   : %v\n", m.CompletedRequests)
	fmt.Printf("Request Rate(req/s)  : %v\n", int(m.RequestRate*1e6))
	fmt.Printf("Total Input Tokens   : %v\n", m.TotalInputTokens)
	fmt.Printf("Total Output Tokens  : %v\n", m.TotalOutputTokens)
	fmt.Printf("Simulation Duration(s): %.3f\n", time.Since(startTime).Seconds())
	fmt.Printf("vLLM estimated Duration(s): %.3f\n", vllmRuntime)
	if m.CompletedRequests > 0 {
		// avgTTFT := float64(m.TTFTSum) / float64(m.CompletedRequests)
		// sortedTTFTs := SortRequestMetrics(m.RequestTTFTs)
		// sortedTPOTs := SortRequestMetrics(m.RequestTPOTs)
		sortedE2Es := SortRequestMetrics(m.RequestE2Es)
		// medianTTFT := CalculatePercentile(sortedTTFTs, 50)
		// p99TTFT := CalculatePercentile(sortedTTFTs, 99)
		// avgTPOT := float64(m.TPOTSum) / float64(m.TotalOutputTokens)
		// medianTPOT := CalculatePercentile(sortedTPOTs, 50)
		// p99TPOT := CalculatePercentile(sortedTPOTs, 99)
		avgE2E := float64(m.TotalLatency) / float64(m.CompletedRequests)
		medianE2E := CalculatePercentile(sortedE2Es, 50)
		p99E2E := CalculatePercentile(sortedE2Es, 99)
		reqThroughput := float64(m.CompletedRequests) / vllmRuntime
		meanActiveSteps := calculateMean(m.RequestStepCounters)

		fmt.Printf("Request throughput (req/s):  : %.3f\n", reqThroughput)
		// fmt.Printf("TTFTs             :[")
		// for _, ttft := range sortedTTFTs {
		// 	fmt.Printf("%.6f, ", ttft/1000)
		// }
		// fmt.Printf("]\n")
		// fmt.Printf("Mean TTFT(ms)     : %.3f\n", avgTTFT/1000)
		// fmt.Printf("Median TTFT(ms)   : %.3f\n", medianTTFT)
		// fmt.Printf("P99 TTFT(ms)      : %.3f\n", p99TTFT)
		// fmt.Printf("TPOTs             : [")
		// for _, tpot := range sortedTPOTs {
		// 	fmt.Printf("%.6f, ", tpot/1000)
		// }
		// fmt.Printf("]\n")
		// fmt.Printf("Mean TPOT(ms)     : %.3f\n", avgTPOT/1000)
		// fmt.Printf("Median TPOT(ms)   : %.3f\n", medianTPOT)
		// fmt.Printf("P99 TPOT(ms)      : %.3f\n", p99TPOT)
		// fmt.Printf("E2Es             : [")
		// for _, e2e := range sortedE2Es {
		// 	fmt.Printf("%.6f, ", e2e/1000)
		// }
		// fmt.Printf("]\n")
		fmt.Printf("Mean E2E(ms)     : %.3f\n", avgE2E/1000)
		fmt.Printf("Median E2E(ms)   : %.3f\n", medianE2E)
		fmt.Printf("P99 E2E(ms)      : %.3f\n", p99E2E)
		fmt.Printf("P99 E2E(ms)      : %.3f\n", p99E2E)
		fmt.Printf("Mean Active Steps     : %.3f\n", meanActiveSteps)
		fmt.Printf("KV Blocks Used : %v\n", m.KVBlocksUsed)
		fmt.Printf("Sim Ended Time : %v\n", m.SimEndedTime)
		fmt.Printf("Avg KV Blocks Usage : %.3f\n", float64(m.KVBlocksUsed)/float64(m.SimEndedTime))
		fmt.Printf("Peak KV Usage       : %v blocks\n", m.PeakKVBlocksUsed)

		fmt.Println("=== Saturation Metrics ===")
		fmt.Printf("Throughput to arrival rate ratio:  : %.3f\n", reqThroughput/(m.RequestRate*1e6))
	}

	// sanity checks
	// m.SavetoFile(m.NumWaitQRequests, "../inference-sim-analysis/waitQ_lengths_med.txt")
	// m.SavetoFile(m.NumRunningBatchRequests, "../inference-sim-analysis/runBatch_lengths_med.txt")

}
