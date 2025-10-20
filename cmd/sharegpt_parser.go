package cmd

import (
	"encoding/json"
	"fmt"
	"os"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// Prompt corresponds to each request inside the "prompts" JSON array.
type Prompt struct {
	InputText     []int   `json:"input_text"`
	GeneratedText []int   `json:"generated_text"`
	InputLen      int     `json:"input_len"`
	PrefixLen     int     `json:"prefix_len"`
	OutputLen     int     `json:"output_len"`
	E2ELatency    float64 `json:"e2e_latency"`
	ArrivalTime   int64   `json:"arrival_time"`
}

// DataEntry corresponds to the root JSON object.
type DataEntry struct {
	NumPrompts  int      `json:"num_prompts"`
	RequestRate float64  `json:"request_rate"`
	Prompts     []Prompt `json:"prompts"` // A slice of the struct defined above
}

// Process input tokenized requests JSON file to extract only the first human-gpt from-value pair
// from conversations
func ProcessInputShareGPT(requestsFilePath string) []*sim.Request {

	// Read the content of the JSON file
	fileContent, err := os.ReadFile(requestsFilePath)
	if err != nil {
		logrus.Fatalf("Error reading file %s: %v", requestsFilePath, err)
	}

	// Declare a slice of DataEntry to unmarshal the raw JSON into
	var rawData DataEntry

	// Unmarshal the JSON data into the rawData slice
	err = json.Unmarshal(fileContent, &rawData)
	if err != nil {
		logrus.Fatalf("Error unmarshaling raw JSON: %v", err)
	}

	var requests []*sim.Request

	for i, prompt := range rawData.Prompts {
		// We need to find the first human-gpt pair

		reqID := fmt.Sprintf("request_%d", i)
		inputTokens := prompt.InputText
		outputTokens := prompt.GeneratedText

		req := sim.Request{
			ID:           reqID,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			ArrivalTime:  prompt.ArrivalTime,
		}
		requests = append(requests, &req)
	}

	// Print the parsed requests to verify
	logrus.Infof("Successfully parsed requests file %s. Total number of requests: %d\n", requestsFilePath, len(requests))
	return requests
}
