package sim

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
)

func (sim *Simulator) generateWorkloadFromCSV() error {
	file, err := os.Open(sim.tracesWorkloadFilePath)
	if err != nil {
		return fmt.Errorf("failed to open csv file: %w", err)
	}
	defer file.Close() //nolint:errcheck // read-only file; close error is not actionable

	reader := csv.NewReader(file)

	// Skip header row
	if _, err := reader.Read(); err != nil {
		return fmt.Errorf("failed to read csv header: %w", err)
	}

	reqIdx := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("error reading csv at row %d: %w", reqIdx, err)
		}
		if len(record) < 5 {
			return fmt.Errorf("csv row %d has %d columns, expected at least 5", reqIdx, len(record))
		}

		// 1. Parse Arrival Time
		arrivalFloat, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return fmt.Errorf("invalid arrival time at row %d: %w", reqIdx, err)
		}
		arrivalTime := int64(arrivalFloat * 1e6)

		if arrivalTime > sim.Horizon {
			break
		}

		// 2. Parse Token Lists from record[3] and record[4]
		var inputTokens []int
		var outputTokens []int

		// Unmarshal JSON-style list: "[1, 2, 3]" -> []int{1, 2, 3}
		if err := json.Unmarshal([]byte(record[3]), &inputTokens); err != nil {
			return fmt.Errorf("failed to parse prefill_tokens at row %d: %w", reqIdx, err)
		}
		if err := json.Unmarshal([]byte(record[4]), &outputTokens); err != nil {
			return fmt.Errorf("failed to parse decode_tokens at row %d: %w", reqIdx, err)
		}

		// 3. Create the request object
		reqID := fmt.Sprintf("request_%d", reqIdx)
		req := &Request{
			ID:               reqID,
			ArrivalTime:      arrivalTime,
			InputTokens:      inputTokens,
			OutputTokens:     outputTokens,
			State:            StateQueued,
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
		}

		// 4. Inject via canonical path (handles both event scheduling and metrics registration)
		sim.InjectArrival(req)

		reqIdx++
	}
	return nil
}

// GenerateLengthGauss samples a length from a clamped Gaussian distribution.
// RNG calls: 1 × NormFloat64() (or 0 if min == max).
func GenerateLengthGauss(rng *rand.Rand, mean, std, min, max int) int {
	if min == max {
		return min
	}
	val := rng.NormFloat64()*float64(std) + float64(mean)
	clampedVal := math.Min(float64(max), val)
	clampedVal = math.Max(float64(min), clampedVal)
	return int(math.Round(clampedVal))
}

// GenerateRandomTokenIDs creates a slice of random token IDs in [0, MaxTokenID).
// RNG calls: length × Intn(MaxTokenID).
func GenerateRandomTokenIDs(rng *rand.Rand, length int) []int {
	tokens := make([]int, length)
	for i := range tokens {
		tokens[i] = rng.Intn(MaxTokenID)
	}
	return tokens
}

func (sim *Simulator) generateLengthGauss(lengthMean, lengthStd, lengthMin, lengthMax int) int {
	return GenerateLengthGauss(sim.WorkloadRNG(), lengthMean, lengthStd, lengthMin, lengthMax)
}

func (sim *Simulator) generateRandomTokenIDs(length int) []int {
	return GenerateRandomTokenIDs(sim.WorkloadRNG(), length)
}

// generateWorkloadDistribution generates request arrivals according to gen config
func (sim *Simulator) generateWorkloadDistribution() {
	if sim.Metrics.RequestRate <= 0 {
		panic("generateWorkloadDistribution: RequestRate must be > 0 (validate at CLI level)")
	}

	currentTime := int64(0)
	// keep track of how many requests have been generated
	reqIdx := 0

	// generate prefix here; this is a random sequence of tokens of prefix len
	prefix := sim.generateRandomTokenIDs(sim.guideLLMConfig.PrefixTokens)

	// create request arrivals iteratively
	for currentTime < sim.Horizon && reqIdx < sim.guideLLMConfig.MaxPrompts {
		// In a Poisson process, the arrival rate is inversely proportional
		// to the mean interarrival time
		// go through the workload requests one by one
		// ToDo: create flags for max input and output lengths

		// get input token length given DataConfig distribution
		promptLen := sim.generateLengthGauss(sim.guideLLMConfig.PromptTokens, sim.guideLLMConfig.PromptTokensStdDev, sim.guideLLMConfig.PromptTokensMin, sim.guideLLMConfig.PromptTokensMax)
		// generate random input tokens of above promptLen
		prompt := sim.generateRandomTokenIDs(promptLen)
		// combine prefix and prompt
		input := append(prefix, prompt...)

		// get output token len given DataConfig distribution
		outputLen := sim.generateLengthGauss(sim.guideLLMConfig.OutputTokens, sim.guideLLMConfig.OutputTokensStdDev, sim.guideLLMConfig.OutputTokensMin, sim.guideLLMConfig.OutputTokensMax)
		// generate random output tokens of above outputLen
		output := sim.generateRandomTokenIDs(outputLen)

		// form the request; it will be in the "queued" state when it arrives
		reqID := fmt.Sprintf("request_%v", reqIdx)

		req := &Request{
			ID:               reqID,
			ArrivalTime:      currentTime,
			InputTokens:      input,
			OutputTokens:     output,
			State:            StateQueued,
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
		}

		sim.InjectArrival(req)

		// estimate arrivalTime based on constant RPS
		currentTime += int64(1 / sim.Metrics.RequestRate)

		// move on to the next request
		reqIdx++

		if currentTime > sim.Horizon {
			break
		}
	}

}
