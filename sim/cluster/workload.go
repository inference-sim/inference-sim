package cluster

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/inference-sim/inference-sim/sim"
)

// generateRequests creates the request list from pre-generated requests, CSV traces, or distribution config.
// Precedence: preGenerated → traces → distribution.
func (c *ClusterSimulator) generateRequests() ([]*sim.Request, error) {
	if len(c.preGeneratedRequests) > 0 {
		return c.preGeneratedRequests, nil
	}
	if c.tracesPath != "" && c.workload == nil {
		return c.generateRequestsFromCSV()
	}
	return c.generateRequestsFromDistribution(), nil
}

// generateRequestsFromDistribution generates requests using the same RNG sequence
// as sim.Simulator.generateWorkloadDistribution, ensuring workload parity.
func (c *ClusterSimulator) generateRequestsFromDistribution() []*sim.Request {
	rng := c.rng.ForSubsystem(sim.SubsystemWorkload)
	cfg := c.workload
	if cfg.Rate <= 0 {
		panic("generateRequestsFromDistribution: Rate must be > 0 (validate at CLI level)")
	}
	horizon := c.config.Horizon

	var requests []*sim.Request
	currentTime := int64(0)
	reqIdx := 0

	prefix := sim.GenerateRandomTokenIDs(rng, cfg.PrefixTokens)

	for currentTime < horizon && reqIdx < cfg.NumRequests {
		promptLen := sim.GenerateLengthGauss(rng, cfg.PromptTokens,
			cfg.PromptTokensStdDev, cfg.PromptTokensMin, cfg.PromptTokensMax)
		prompt := sim.GenerateRandomTokenIDs(rng, promptLen)
		input := append(prefix, prompt...) // intentionally matches original behavior

		outputLen := sim.GenerateLengthGauss(rng, cfg.OutputTokens,
			cfg.OutputTokensStdDev, cfg.OutputTokensMin, cfg.OutputTokensMax)
		output := sim.GenerateRandomTokenIDs(rng, outputLen)

		req := sim.NewRequest(fmt.Sprintf("request_%v", reqIdx), currentTime, input, output)
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
func (c *ClusterSimulator) generateRequestsFromCSV() ([]*sim.Request, error) {
	file, err := os.Open(c.tracesPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open csv file: %w", err)
	}
	defer file.Close() //nolint:errcheck // read-only file; close error is not actionable

	reader := csv.NewReader(file)
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("failed to read csv header: %w", err)
	}

	var requests []*sim.Request
	reqIdx := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading csv at row %d: %w", reqIdx, err)
		}
		if len(record) < 5 {
			return nil, fmt.Errorf("csv row %d has %d columns, expected at least 5", reqIdx, len(record))
		}
		arrivalFloat, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return nil, fmt.Errorf("invalid arrival time at row %d: %w", reqIdx, err)
		}
		arrivalTime := int64(arrivalFloat * 1e6)
		if arrivalTime > c.config.Horizon {
			break
		}

		var inputTokens, outputTokens []int
		if err := json.Unmarshal([]byte(record[3]), &inputTokens); err != nil {
			return nil, fmt.Errorf("failed to parse prefill_tokens at row %d: %w", reqIdx, err)
		}
		if err := json.Unmarshal([]byte(record[4]), &outputTokens); err != nil {
			return nil, fmt.Errorf("failed to parse decode_tokens at row %d: %w", reqIdx, err)
		}

		requests = append(requests, sim.NewRequest(fmt.Sprintf("request_%d", reqIdx), arrivalTime, inputTokens, outputTokens))
		reqIdx++
	}
	return requests, nil
}
