package sim

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"

	"github.com/sirupsen/logrus"
)

func (sim *Simulator) generateWorkloadFromCSV() {
	file, err := os.Open(sim.TracesWorkloadFilePath)
	if err != nil {
		logrus.Fatalf("failed to open csv file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Skip header row
	if _, err := reader.Read(); err != nil {
		logrus.Fatalf("failed to read csv header: %v", err)
	}

	reqIdx := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			logrus.Fatalf("error reading csv at row %d: %v", reqIdx, err)
		}

		// 1. Parse Arrival Time
		arrivalFloat, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			logrus.Fatalf("invalid arrival time at row %d: %v", reqIdx, err)
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
			logrus.Fatalf("failed to parse prefill_tokens at row %d: %v", reqIdx, err)
		}
		if err := json.Unmarshal([]byte(record[4]), &outputTokens); err != nil {
			logrus.Fatalf("failed to parse decode_tokens at row %d: %v", reqIdx, err)
		}

		// 3. Create the request object
		reqID := fmt.Sprintf("request_%d", reqIdx)
		req := &Request{
			ID:               reqID,
			ArrivalTime:      arrivalTime,
			InputTokens:      inputTokens,
			OutputTokens:     outputTokens,
			State:            "queued",
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
		}

		// 4. Push to schedule and metrics
		sim.Schedule(&ArrivalEvent{
			time:    arrivalTime,
			Request: req,
		})

		sim.GlobalMetrics.Requests[reqID] = RequestMetrics{
			ID:               reqID,
			ArrivedAt:        arrivalFloat,
			NumPrefillTokens: len(inputTokens),
			NumDecodeTokens:  len(outputTokens),
			ReplicaIndex:     -1, // Will be set when request is enqueued to a replica
		}

		reqIdx++
	}
}

// generateLengthGauss generates input or output length satisfying DataConfig distribution
// The generated length is sampled from a Gaussian distribution with mean=lengthMean, std=lengthStd
// and is clamped between (lengthMin, lengthMax)
func (sim *Simulator) generateLengthGauss(lengthMean, lengthStd, lengthMin, lengthMax int) int {
	if lengthMin == lengthMax {
		return lengthMin
	}
	val := sim.randomNumberGenerator.NormFloat64()*float64(lengthStd) + float64(lengthMean)
	clampedVal := math.Min(float64(lengthMax), val)
	clampedVal = math.Max(float64(lengthMin), clampedVal)
	roundedVal := math.Round(clampedVal)
	return int(roundedVal)
}

// generateRandomTokenIDs creates a slice of 'length' random integers.
// each token ID ranges between 0 to 32000.
func (sim *Simulator) generateRandomTokenIDs(length int) []int {

	tokens := make([]int, length)

	for i := 0; i < length; i++ {
		tokens[i] = sim.randomNumberGenerator.Intn(MaxTokenID)
	}
	return tokens
}

// generateWorkloadDistribution generates request arrivals according to gen config
func (sim *Simulator) generateWorkloadDistribution() {

	currentTime := int64(0)
	// keep track of how many requests have been generated
	reqIdx := 0

	// generate prefix here; this is a random sequence of tokens of prefix len
	prefix := sim.generateRandomTokenIDs(sim.GuideLLMConfig.PrefixTokens)

	// create request arrivals iteratively
	for currentTime < sim.Horizon && reqIdx < sim.GuideLLMConfig.MaxPrompts {
		// In a Poisson process, the arrival rate is inversely proportional
		// to the mean interarrival time
		// go through the workload requests one by one
		// ToDo: create flags for max input and output lengths

		// get input token length given DataConfig distribution
		promptLen := sim.generateLengthGauss(sim.GuideLLMConfig.PromptTokens, sim.GuideLLMConfig.PromptTokensStdDev, sim.GuideLLMConfig.PromptTokensMin, sim.GuideLLMConfig.PromptTokensMax)
		// generate random input tokens of above promptLen
		prompt := sim.generateRandomTokenIDs(promptLen)
		// combine prefix and prompt
		input := append(prefix, prompt...)

		// get output token len given DataConfig distribution
		outputLen := sim.generateLengthGauss(sim.GuideLLMConfig.OutputTokens, sim.GuideLLMConfig.OutputTokensStdDev, sim.GuideLLMConfig.OutputTokensMin, sim.GuideLLMConfig.OutputTokensMax)
		// generate random output tokens of above outputLen
		output := sim.generateRandomTokenIDs(outputLen)

		// form the request; it will be in the "queued" state when it arrives
		reqID := fmt.Sprintf("request_%v", reqIdx)

		req := &Request{
			ID:               reqID,
			ArrivalTime:      currentTime,
			InputTokens:      input,
			OutputTokens:     output,
			State:            "queued",
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
		}

		// push the request for arrival
		sim.Schedule(&ArrivalEvent{time: currentTime, Request: req})

		// Add to metrics.Requests
		detail := RequestMetrics{
			ID:               reqID,
			ArrivedAt:        float64(currentTime) / 1e6,
			NumPrefillTokens: len(input),
			NumDecodeTokens:  len(output),
			ReplicaIndex:     -1, // Will be set when request is enqueued to a replica
		}
		sim.GlobalMetrics.Requests[reqID] = detail

		// estimate arrivalTime based on constant RPS
		currentTime += int64(1 / sim.GlobalMetrics.RequestRate)

		// move on to the next request
		reqIdx++

		if currentTime > sim.Horizon {
			break
		}
	}

}
