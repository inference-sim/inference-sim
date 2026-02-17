package cluster

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// generateRequests creates the request list from pre-generated requests, CSV traces, or distribution config.
// Precedence: preGenerated → traces → distribution.
func (c *ClusterSimulator) generateRequests() []*sim.Request {
	if len(c.preGeneratedRequests) > 0 {
		return c.preGeneratedRequests
	}
	if c.tracesPath != "" && c.workload == nil {
		return c.generateRequestsFromCSV()
	}
	return c.generateRequestsFromDistribution()
}

// generateRequestsFromDistribution generates requests using the same RNG sequence
// as sim.Simulator.generateWorkloadDistribution, ensuring workload parity.
func (c *ClusterSimulator) generateRequestsFromDistribution() []*sim.Request {
	rng := c.rng.ForSubsystem(sim.SubsystemWorkload)
	cfg := c.workload
	horizon := c.config.Horizon

	var requests []*sim.Request
	currentTime := int64(0)
	reqIdx := 0

	prefix := sim.GenerateRandomTokenIDs(rng, cfg.PrefixTokens)

	for currentTime < horizon && reqIdx < cfg.MaxPrompts {
		promptLen := sim.GenerateLengthGauss(rng, cfg.PromptTokens,
			cfg.PromptTokensStdDev, cfg.PromptTokensMin, cfg.PromptTokensMax)
		prompt := sim.GenerateRandomTokenIDs(rng, promptLen)
		input := append(prefix, prompt...) // intentionally matches original behavior

		outputLen := sim.GenerateLengthGauss(rng, cfg.OutputTokens,
			cfg.OutputTokensStdDev, cfg.OutputTokensMin, cfg.OutputTokensMax)
		output := sim.GenerateRandomTokenIDs(rng, outputLen)

		req := &sim.Request{
			ID:               fmt.Sprintf("request_%v", reqIdx),
			ArrivalTime:      currentTime,
			InputTokens:      input,
			OutputTokens:     output,
			State:            "queued",
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
		}
		requests = append(requests, req)

		currentTime += int64(1 / cfg.Rate)
		reqIdx++
		if currentTime > horizon {
			break
		}
	}
	return requests
}

// generateRequestsFromCSV loads requests from a CSV trace file.
func (c *ClusterSimulator) generateRequestsFromCSV() []*sim.Request {
	file, err := os.Open(c.tracesPath)
	if err != nil {
		logrus.Fatalf("failed to open csv file: %v", err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			logrus.Warnf("failed to close csv file %q: %v", c.tracesPath, err)
		}
	}()

	reader := csv.NewReader(file)
	if _, err := reader.Read(); err != nil {
		logrus.Fatalf("failed to read csv header: %v", err)
	}

	var requests []*sim.Request
	reqIdx := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			logrus.Fatalf("error reading csv at row %d: %v", reqIdx, err)
		}
		if len(record) < 5 {
			logrus.Fatalf("csv row %d has %d columns, expected at least 5", reqIdx, len(record))
		}
		arrivalFloat, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			logrus.Fatalf("invalid arrival time at row %d: %v", reqIdx, err)
		}
		arrivalTime := int64(arrivalFloat * 1e6)
		if arrivalTime > c.config.Horizon {
			break
		}

		var inputTokens, outputTokens []int
		if err := json.Unmarshal([]byte(record[3]), &inputTokens); err != nil {
			logrus.Fatalf("failed to parse prefill_tokens at row %d: %v", reqIdx, err)
		}
		if err := json.Unmarshal([]byte(record[4]), &outputTokens); err != nil {
			logrus.Fatalf("failed to parse decode_tokens at row %d: %v", reqIdx, err)
		}

		requests = append(requests, &sim.Request{
			ID:               fmt.Sprintf("request_%d", reqIdx),
			ArrivalTime:      arrivalTime,
			InputTokens:      inputTokens,
			OutputTokens:     outputTokens,
			State:            "queued",
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
		})
		reqIdx++
	}
	return requests
}
