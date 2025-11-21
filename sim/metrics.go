// Tracks simulation-wide and per-request performance metrics such as:

package sim

import (
	"fmt"
	"sort"
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

		// TTFT metrics
		sortedTTFTs := make([]float64, 0, len(m.RequestTTFTs))

		for _, value := range m.RequestTTFTs {
			sortedTTFTs = append(sortedTTFTs, value)
		}

		sort.Float64s(sortedTTFTs)
		avgTTFT := CalculateMean(sortedTTFTs)
		p90TTFT := CalculatePercentile(sortedTTFTs, 90)
		p95TTFT := CalculatePercentile(sortedTTFTs, 95)
		p99TTFT := CalculatePercentile(sortedTTFTs, 99)

		// ITL metrics
		sortedTPOTs := make([]float64, 0, len(m.RequestTPOTs))

		for _, value := range m.RequestTPOTs {
			sortedTPOTs = append(sortedTPOTs, value)
		}

		sort.Float64s(sortedTPOTs)
		avgTPOT := CalculateMean(sortedTPOTs)
		p90TPOT := CalculatePercentile(sortedTPOTs, 90)
		p95TPOT := CalculatePercentile(sortedTPOTs, 95)
		p99TPOT := CalculatePercentile(sortedTPOTs, 99)

		// E2E metrics
		sortedE2Es := make([]float64, 0, len(m.RequestE2Es))

		for _, value := range m.RequestE2Es {
			sortedE2Es = append(sortedE2Es, value)
		}

		sort.Float64s(sortedE2Es)
		avgE2E := CalculateMean(sortedE2Es)
		p90E2E := CalculatePercentile(sortedE2Es, 90)
		p95E2E := CalculatePercentile(sortedE2Es, 95)
		p99E2E := CalculatePercentile(sortedE2Es, 99)

		fmt.Printf("Mean E2E(ms)     : %.3f\n", avgE2E)
		fmt.Printf("P90 E2E(ms)   : %.3f\n", p90E2E)
		fmt.Printf("P95 E2E(ms)   : %.3f\n", p95E2E)
		fmt.Printf("P99 E2E(ms)      : %.3f\n", p99E2E)
		fmt.Printf("Mean TTFT(ms)     : %.3f\n", avgTTFT)
		fmt.Printf("P90 TTFT(ms)   : %.3f\n", p90TTFT)
		fmt.Printf("P95 TTFT(ms)   : %.3f\n", p95TTFT)
		fmt.Printf("P99 TTFT(ms)      : %.3f\n", p99TTFT)
		fmt.Printf("Mean TPOT(ms)     : %.3f\n", avgTPOT)
		fmt.Printf("P90 TPOT(ms)   : %.3f\n", p90TPOT)
		fmt.Printf("P95 TPOT(ms)   : %.3f\n", p95TPOT)
		fmt.Printf("P99 TPOT(ms)      : %.3f\n", p99TPOT)
	}
}
