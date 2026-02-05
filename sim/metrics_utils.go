// sim/metrics_utils.go
package sim

import (
	"bufio"
	"fmt"
	"math"
	"os"

	"github.com/sirupsen/logrus"
)

// Bin represents a single completion time integral bin with its integer key and count.
type Bin struct {
	Key   int
	Count int
}

type IntOrFloat64 interface {
	int | int64 | float64
}

// Individual request metrics for the JSON log
type RequestMetrics struct {
	ArrivedAt        float64 `json:"arrived_at"`
	ID               string  `json:"requestID"`
	NumPrefillTokens int     `json:"num_prefill_tokens"`
	NumDecodeTokens  int     `json:"num_decode_tokens"`
	TTFT             float64 `json:"ttft_ms"`
	ITL              float64 `json:"itl_ms"`
	E2E              float64 `json:"e2e_ms"`
	SchedulingDelay  float64 `json:"scheduling_delay_ms"`
	ReplicaIndex     int     `json:"replica_index"`
}

// MetricsOutput defines the JSON structure for the saved metrics
type MetricsOutput struct {
	SimStartTimestamp     string           `json:"sim_start_timestamp"`
	SimEndTimestamp       string           `json:"sim_end_timestamp"`
	CompletedRequests     int              `json:"completed_requests"`
	TotalInputTokens      int              `json:"total_input_tokens"`
	TotalOutputTokens     int              `json:"total_output_tokens"`
	VllmDurationSec       float64          `json:"vllm_estimated_duration_s"`
	SimulationDurationSec float64          `json:"simulation_duration_s"`
	ResponsesPerSec       float64          `json:"responses_per_sec"`
	TokensPerSec          float64          `json:"tokens_per_sec"`
	E2EMeanMs             float64          `json:"e2e_mean_ms"`
	E2EP90Ms              float64          `json:"e2e_p90_ms"`
	E2EP95Ms              float64          `json:"e2e_p95_ms"`
	E2EP99Ms              float64          `json:"e2e_p99_ms"`
	TTFTMeanMs            float64          `json:"ttft_mean_ms"`
	TTFTP90Ms             float64          `json:"ttft_p90_ms"`
	TTFTP95Ms             float64          `json:"ttft_p95_ms"`
	TTFTP99Ms             float64          `json:"ttft_p99_ms"`
	ITLMeanMs             float64          `json:"itl_mean_ms"`
	ITLP90Ms              float64          `json:"itl_p90_ms"`
	ITLP95Ms              float64          `json:"itl_p95_ms"`
	ITLP99Ms              float64          `json:"itl_p99_ms"`
	SchedulingDelayP99Ms  float64          `json:"scheduling_delay_p99_ms"`
	Requests              []RequestMetrics `json:"requests,omitempty"`
}

// MultiReplicaMetricsOutput defines the JSON structure for multi-replica simulations
type MultiReplicaMetricsOutput struct {
	ReplicaMetrics []MetricsOutput `json:"replica_metrics"`
	GlobalMetrics  MetricsOutput   `json:"global_metrics"`
}

// CalculatePercentile is a util function that calculates the p-th percentile of a data list
// return values are in milliseconds
func CalculatePercentile[T IntOrFloat64](data []T, p float64) float64 {
	n := len(data)

	rank := p / 100.0 * float64(n-1)
	lowerIdx := int(math.Floor(rank))
	upperIdx := int(math.Ceil(rank))

	if lowerIdx == upperIdx {
		return float64(data[lowerIdx]) / 1000
	} else {
		lowerVal := data[lowerIdx]
		upperVal := data[upperIdx]
		if upperIdx >= n {
			return float64(data[n-1]) / 1000
		}
		return float64(lowerVal)/1000 + float64(upperVal-lowerVal)*(rank-float64(lowerIdx))/1000
	}
}

// CalculateMean is a util function that calculates the mean of a data list
// return values are in milliseconds
func CalculateMean[T IntOrFloat64](numbers []T) float64 {
	if len(numbers) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, number := range numbers {
		sum += float64(number)
	}

	return (sum / float64(len(numbers))) / 1000
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
