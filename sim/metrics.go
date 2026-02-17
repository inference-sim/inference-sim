// Tracks simulation-wide and per-request performance metrics such as:

package sim

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
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
	PreemptionCount   int64   // Total preemption events (PR12)

	TTFTSum int64 // Total time-to-first-token sum (in ticks)
	ITLSum  int64 // Total ITL sum across requests (in ticks)

	RequestTTFTs            map[string]float64 // list of all requests' TTFT
	RequestITLs             map[string]float64 // list of all requests' ITL
	RequestSchedulingDelays map[string]int64   // list of all requests' scheduling delays
	AllITLs                 []int64            // list of all requests' ITL
	RequestE2Es             map[string]float64 // list of all requests' latencies
	RequestCompletionTimes  map[string]float64 // list of all requests' completion times in ticks
	RequestStepCounters     []int              // list of all requests' num of steps between scheduled and finished

	NumWaitQRequests        []int                     // number of requests in waitQ over different steps
	NumRunningBatchRequests []int                     // number of request in runningBatch over different steps
	Requests                map[string]RequestMetrics // request metrics list
}

func NewMetrics() *Metrics {
	return &Metrics{
		CompletedRequests:       0,
		RequestTTFTs:            make(map[string]float64),
		RequestITLs:             make(map[string]float64),
		AllITLs:                 []int64{},
		RequestE2Es:             make(map[string]float64),
		RequestCompletionTimes:  make(map[string]float64),
		RequestSchedulingDelays: make(map[string]int64),
		NumWaitQRequests:        []int{},
		NumRunningBatchRequests: []int{},
		Requests:                make(map[string]RequestMetrics),
	}
}

func (m *Metrics) SaveResults(instanceID string, horizon int64, totalBlocks int64, startTime time.Time, outputFilePath string) {
	vllmRuntime := float64(m.SimEndedTime) / float64(1e6)

	// Create an instance of our output struct to populate
	output := MetricsOutput{
		InstanceID:            instanceID,
		SimStartTimestamp:     startTime.Format("2006-01-02 15:04:05"),
		SimEndTimestamp:       time.Now().Format("2006-01-02 15:04:05"),
		CompletedRequests:     m.CompletedRequests,
		TotalInputTokens:      int(m.TotalInputTokens),
		TotalOutputTokens:     int(m.TotalOutputTokens),
		VllmDurationSec:       vllmRuntime,
		SimulationDurationSec: time.Since(startTime).Seconds(),
	}

	if m.CompletedRequests > 0 {
		// --- TTFT Calculations ---
		sortedTTFTs := make([]float64, 0, len(m.RequestTTFTs))
		for _, value := range m.RequestTTFTs {
			sortedTTFTs = append(sortedTTFTs, value)
		}
		sort.Float64s(sortedTTFTs)
		output.TTFTMeanMs = CalculateMean(sortedTTFTs)
		output.TTFTP90Ms = CalculatePercentile(sortedTTFTs, 90)
		output.TTFTP95Ms = CalculatePercentile(sortedTTFTs, 95)
		output.TTFTP99Ms = CalculatePercentile(sortedTTFTs, 99)

		// --- E2E Calculations ---
		sortedE2Es := make([]float64, 0, len(m.RequestE2Es))
		for _, value := range m.RequestE2Es {
			sortedE2Es = append(sortedE2Es, value)
		}
		sort.Float64s(sortedE2Es)
		output.E2EMeanMs = CalculateMean(sortedE2Es)
		output.E2EP90Ms = CalculatePercentile(sortedE2Es, 90)
		output.E2EP95Ms = CalculatePercentile(sortedE2Es, 95)
		output.E2EP99Ms = CalculatePercentile(sortedE2Es, 99)

		// --- ITL Calculations ---
		slices.Sort(m.AllITLs)
		output.ITLMeanMs = CalculateMean(m.AllITLs)
		output.ITLP90Ms = CalculatePercentile(m.AllITLs, 90)
		output.ITLP95Ms = CalculatePercentile(m.AllITLs, 95)
		output.ITLP99Ms = CalculatePercentile(m.AllITLs, 99)

		// --- P99 Scheduling Delay ---
		sortedSchedulingDelays := make([]float64, 0, len(m.RequestSchedulingDelays))
		for _, value := range m.RequestSchedulingDelays {
			sortedSchedulingDelays = append(sortedSchedulingDelays, float64(value))
		}
		sort.Float64s(sortedSchedulingDelays)
		output.SchedulingDelayP99Ms = CalculatePercentile(sortedSchedulingDelays, 99)

		output.ResponsesPerSec = float64(m.CompletedRequests) / vllmRuntime
		output.TokensPerSec = float64(m.TotalOutputTokens) / vllmRuntime

		// Print to Stdout
		fmt.Println("=== Simulation Metrics ===")
		data, err := json.MarshalIndent(output, "", "  ")
		if err != nil {
			fmt.Println("Error marshalling:", err)
			return
		}
		fmt.Println(string(data))
	}

	// --- Write to JSON File ---
	if outputFilePath != "" {
		// request-level metrics for detailed output in file
		for id, ttft := range m.RequestTTFTs {
			detail := m.Requests[id]
			detail.TTFT = ttft / 1e3
			detail.E2E = m.RequestE2Es[id] / 1e3
			detail.ITL = m.RequestITLs[id]
			detail.SchedulingDelay = float64(m.RequestSchedulingDelays[id])
			output.Requests = append(output.Requests, detail)
		}

		// 2. Sort by ArrivedAt (Ascending)
		sort.Slice(output.Requests, func(i, j int) bool {
			return output.Requests[i].ArrivedAt < output.Requests[j].ArrivedAt
		})

		data, err := json.MarshalIndent(output, "", "  ")
		if err != nil {
			fmt.Printf("Error marshalling metrics to JSON: %v\n", err)
			return
		}

		writeErr := os.WriteFile(outputFilePath, data, 0644)
		if writeErr != nil {
			fmt.Printf("Error writing JSON file: %v\n", writeErr)
			return
		}
		fmt.Printf("\nMetrics written to: %s\n", outputFilePath)
	}
}
