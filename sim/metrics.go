// Tracks simulation-wide and per-request performance metrics such as:

package sim

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"github.com/sirupsen/logrus"
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

	NumWaitQRequests        []int // number of requests in waitQ over different steps
	NumRunningBatchRequests []int // number of request in runningBatch over different steps
}

// CalculatePercentile is a util function that calculates the p-th percentile of a data list
func CalculatePercentile(data []float64, p float64) float64 {
	n := len(data)

	sortedData := make([]float64, n)
	copy(sortedData, data)

	sort.Float64s(sortedData)

	rank := p / 100.0 * float64(n-1)
	lowerIdx := int(math.Floor(rank))
	upperIdx := int(math.Ceil(rank))

	if lowerIdx == upperIdx {
		return sortedData[lowerIdx]
	} else {
		lowerVal := sortedData[lowerIdx]
		upperVal := sortedData[upperIdx]
		if upperIdx >= n {
			return sortedData[n-1]
		}
		return lowerVal + (upperVal-lowerVal)*(rank-float64(lowerIdx))
	}
}

func (m *Metrics) SavetoFile(data []int, fileName string) {
	file, err := os.OpenFile(fileName, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0777)
	if err != nil {
		logrus.Fatalf("Error creating file %s: %v\n", fileName, err)
		return
	}
	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			logrus.Fatalf("Error closing file %s: %v\n", fileName, closeErr)
		}
	}()

	writer := bufio.NewWriter(file)

	defer func() {
		if flushErr := writer.Flush(); flushErr != nil {
			logrus.Fatalf("Error flushing writer for file %s: %v\n", fileName, flushErr)
		}
	}()

	for _, f := range data {
		_, writeErr := fmt.Fprint(writer, f, ", ")
		if writeErr != nil {
			logrus.Fatalf("Error writing int %d to file: %v\n", f, writeErr)
			return // Stop writing on first error
		}
	}

	logrus.Debugf("Successfully wrote to '%s'\n", fileName)
}

// Print displays aggregated metrics at the end of the simulation.
// Includes average latency, TTFT, TPOT, KV usage, and prefix cache behavior.
func (m *Metrics) Print(horizon int64, totalBlocks int, startTime time.Time) {
	fmt.Println("=== Simulation Metrics ===")
	fmt.Printf("Completed Requests   : %d\n", m.CompletedRequests)
	fmt.Printf("Request Rate(req/s)  : %d\n", int(m.RequestRate*1e6))
	fmt.Printf("Total Input Tokens   : %d\n", m.TotalInputTokens)
	fmt.Printf("Total Output Tokens  : %d\n", m.TotalOutputTokens)
	fmt.Printf("Simulation Duration(s): %.3f\n", time.Since(startTime).Seconds())
	fmt.Printf("vLLM estimated Duration(s): %d\n", m.SimEndedTime)
	if m.CompletedRequests > 0 {
		avgTTFT := float64(m.TTFTSum) / float64(m.CompletedRequests)
		medianTTFT := CalculatePercentile(m.RequestTTFTs, 50)
		p99TTFT := CalculatePercentile(m.RequestTTFTs, 99)
		avgTPOT := float64(m.TPOTSum) / float64(m.TotalOutputTokens)
		medianTPOT := CalculatePercentile(m.RequestTPOTs, 50)
		p99TPOT := CalculatePercentile(m.RequestTPOTs, 99)
		reqThroughput := float64(m.CompletedRequests) / float64(m.SimEndedTime/1e6)

		fmt.Printf("Request throughput (req/s):  : %.3f\n", reqThroughput)
		fmt.Printf("Mean TTFT(ms)     : %.3f\n", avgTTFT/1000)
		fmt.Printf("Median TTFT(ms)   : %.3f\n", medianTTFT/1000)
		fmt.Printf("P99 TTFT(ms)      : %.3f\n", p99TTFT/1000)
		fmt.Printf("Mean TPOT(ms)     : %.3f\n", avgTPOT/1000)
		fmt.Printf("Median TPOT(ms)   : %.3f\n", medianTPOT/1000)
		fmt.Printf("P99 TPOT(ms)      : %.3f\n", p99TPOT/1000)
		fmt.Printf("Avg KV Blocks Usage : %.3f\n", float64(m.KVBlocksUsed)/float64(m.SimEndedTime))
		fmt.Printf("Peak KV Usage       : %d blocks\n", m.PeakKVBlocksUsed)

		fmt.Println("=== Saturation Metrics ===")
		fmt.Printf("Throughput to arrival rate ratio:  : %.3f\n", reqThroughput/(m.RequestRate*1e6))
	}

	// sanity checks
	m.SavetoFile(m.NumWaitQRequests, "../inference-sim-analysis/waitQ_lengths_med.txt")
	m.SavetoFile(m.NumRunningBatchRequests, "../inference-sim-analysis/runBatch_lengths_med.txt")

}
