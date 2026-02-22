// Tracks simulation-wide and per-request performance metrics such as:

package sim

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"sort"

	"github.com/sirupsen/logrus"
)

// Metrics aggregates statistics about the simulation
// for final reporting. Useful for evaluating system performance
// and debugging behavior over time.
type Metrics struct {
	CompletedRequests int     // Number of requests completed
	TotalInputTokens  int     // Total number of input tokens
	TotalOutputTokens int   // Total number of output tokens
	SimEndedTime      int64 // Sim clock time in ticks when simulation ends
	KVBlocksUsed      float64 // Integral of KVBlockUsage over time
	PeakKVBlocksUsed  int64   // Max number of simultaneously used KV blocks
	PreemptionCount      int64   // Total preemption events (PR12)
	KVAllocationFailures int64   // KV allocation failures for the final decode token at completion; non-zero indicates a cache accounting anomaly (#183)
	CacheHitRate         float64 // Cumulative cache hit rate at finalization (PR12). Intentional observability signal: set by cluster/instance.go Finalize() from KVStore.CacheHitRate(). Read-only statistic — does not feed back into state evolution.
	KVThrashingRate      float64 // KV thrashing rate at finalization (PR12)
	StillQueued          int     // Requests still in wait queue at sim end
	StillRunning         int     // Requests still in running batch at sim end

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

func (m *Metrics) SaveResults(instanceID string, horizon int64, totalBlocks int64, outputFilePath string) {
	vllmRuntime := float64(m.SimEndedTime) / float64(1e6)

	// Create an instance of our output struct to populate
	output := MetricsOutput{
		InstanceID:           instanceID,
		CompletedRequests:    m.CompletedRequests,
		StillQueued:          m.StillQueued,
		StillRunning:         m.StillRunning,
		InjectedRequests:     m.CompletedRequests + m.StillQueued + m.StillRunning,
		TotalInputTokens:     int(m.TotalInputTokens),
		TotalOutputTokens:    int(m.TotalOutputTokens),
		VllmDurationSec:      vllmRuntime,
		KVAllocationFailures: m.KVAllocationFailures,
		PreemptionCount:      m.PreemptionCount,
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

		// Print to stdout (results are primary output, not log messages)
		fmt.Println("=== Simulation Metrics ===")
		data, err := json.MarshalIndent(output, "", "  ")
		if err != nil {
			logrus.Errorf("Error marshalling metrics: %v", err)
			return
		}
		fmt.Println(string(data))
	}

	// --- Write to JSON File ---
	if outputFilePath != "" {
		// request-level metrics for detailed output in file
		// Iterate over all registered requests (not just completed prefill)
		// so incomplete requests appear with zero-valued metrics.
		for _, id := range sortedRequestIDs(m.Requests) {
			detail := m.Requests[id]
			detail.TTFT = m.RequestTTFTs[id] / 1e3   // zero if not in map
			detail.E2E = m.RequestE2Es[id] / 1e3      // zero if not in map
			detail.ITL = m.RequestITLs[id]             // zero if not in map
			detail.SchedulingDelay = float64(m.RequestSchedulingDelays[id])
			output.Requests = append(output.Requests, detail)
		}

		// 2. Sort by ArrivedAt (Ascending)
		sort.Slice(output.Requests, func(i, j int) bool {
			return output.Requests[i].ArrivedAt < output.Requests[j].ArrivedAt
		})

		data, err := json.MarshalIndent(output, "", "  ")
		if err != nil {
			logrus.Errorf("Error marshalling metrics to JSON: %v", err)
			return
		}

		writeErr := os.WriteFile(outputFilePath, data, 0644)
		if writeErr != nil {
			logrus.Errorf("Error writing JSON file: %v", writeErr)
			return
		}
		logrus.Infof("Metrics written to: %s", outputFilePath)
	}
}

// sortedRequestIDs returns request IDs from the Requests map in sorted order.
// Ensures deterministic output ordering for JSON serialization.
func sortedRequestIDs(requests map[string]RequestMetrics) []string {
	ids := make([]string, 0, len(requests))
	for id := range requests {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}
